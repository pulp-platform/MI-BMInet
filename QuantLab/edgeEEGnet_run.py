import os
import shutil
import json
import sys
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import progress
from tqdm import tqdm
import pickle

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from main import main as quantlab_main

import torch

from sklearn.model_selection import KFold


PROBLEM = "BCI-CompIV-2a"
TOPOLOGY = "edgeEEGNet"
EXP_FOLDER = "logs/exp{}"
MEAS_ID = 9
INQ_CONFIG = f"measurement/M{MEAS_ID:02}.json"
BAK_CONFIG = ".config_backup.json"
MAIN_CONFIG = "config.json"
EXP_BASE = MEAS_ID * 100
EXPORT_FILE = f"logs/measurement_{MEAS_ID:02}" + "_{}.npz"
EXPORT_GRID_FILE = 'logs/grid_{}.npz'

BENCHMARK = True
GRID_MEASUREMENT = False
CV = False
N_ITER = 1

#GLOBAL_MODEL = False

NUM_SUBJECTS = 9 #109 # for physionet MMMI
#if GLOBAL_MODEL:
#    NUM_SUBJECTS = 1

SILENT = False


def cross_validate(bar=None, silent=False):

    subjects_list = np.arange(1,110)
    # Define subjects whose data is not taken, for details see data tester added 106 again to analyze it, deleted from the exluded list
    excluded_subjects = [88,92,100,104]
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects = np.asarray([x for x in subjects_list if (x not in excluded_subjects)])

    stats = np.zeros((N_ITER, 1, 4))

    kf = KFold(n_splits = N_ITER)

    with tqdm(desc=f'Benchmarking Measurement {MEAS_ID:02}', total=N_ITER, ascii=True) as bar:
        with TestEnvironment():
            for fold, (train_index, test_index) in enumerate(kf.split(subjects)):
                train_index = subjects[train_index]
                test_index = subjects[test_index]
            #for fold in range(N_ITER):
                fold_stats = _do_one_fold(fold, subj_indexes=(train_index, test_index), bar=bar, silent=SILENT)
                if not silent:
                    print(f"Fold {fold}: quantized accuracy: {fold_stats['valid_acc']:.4f}                 ")
                stats[fold, :, :] = np.array([fold_stats['train_loss'],
                                              fold_stats['train_acc'],
                                              fold_stats['valid_loss'],
                                              fold_stats['valid_acc']])

                # store the data to make sure not to loose it
                np.savez(file=os.path.join(PROBLEM, EXPORT_FILE.format("runs")),
                         train_loss=stats[fold, :, 0],
                         train_acc=stats[fold, :, 1],
                         valid_loss=stats[fold, :, 2],
                         valid_acc=stats[fold, :, 3])

    # compute statistics
    avg_stats = stats.mean(axis=0) # average over all N_ITER experiments
    std_stats = stats.std(axis=0) # std over all N_ITER experiments

    # For the overall score, first average along all subjects.
    mean_avg_stats = avg_stats[:].mean(axis=0) # average over all subjects
    # For standard deviation, average all standard deviations of all subjects
    mean_std_stats = std_stats[:].mean(axis=0) # std over all subjects

    print(f"Total Average Accuracy: {mean_avg_stats[3]:.4f} +- {mean_std_stats[3]:.4f}\n")


def _do_one_fold(fold, subj_indexes, bar=None, silent=False, n_weights=None, n_activ=None):
    exp_id = EXP_BASE + fold

    if not silent:
        print(f"Subject {fold}: training quantized model (exp{exp_id})...\r", end='',
              flush=True)

    modification = {'treat.data.subject': fold}
    if n_weights is not None:
        modification['indiv.net.params.weightInqNumLevels'] = n_weights
        modification["indiv.net.params.first_layer_only"] = True
    if n_activ is not None:
        modification['indiv.net.params.actSTENumLevels'] = n_activ
    valid_stats, train_stats = _execute_quantlab(INQ_CONFIG, exp_id, fold=fold, subj_indexes=subj_indexes, modify_keys=modification)

    if bar is not None:
        bar.update()

    # accumulate log files
    if BENCHMARK or GRID_MEASUREMENT:
        # _accumulate_logs(subject, exp_id)
        _just_store_anything(fold, exp_id, n_weights=n_weights, n_activ=n_activ)

    return _format_all_stats(train_stats, valid_stats)


def single_iter(bar=None, silent=False, n_weights=None, n_activ=None):
    iter_stats = np.zeros((NUM_SUBJECTS, 4))
    with TestEnvironment():
        for i in range(NUM_SUBJECTS):
            subject = i + 1
            stats = _do_subject(subject, bar, silent, n_weights=n_weights, n_activ=n_activ)
            if not silent:
                print(f"Subject {subject}: quantized accuracy: {stats['valid_acc']:.4f}                 ")
            iter_stats[i] = np.array([stats['train_loss'],
                                      stats['train_acc'],
                                      stats['valid_loss'],
                                      stats['valid_acc']])

    if not silent:
        print(f"Average quantized accuracy = {iter_stats.mean(axis=0)[3]}")

    return iter_stats


def grid_measurement():
    stats = {}
    cases = [
        (255, 255),
        (255, 127),
        (255, 63),
        (255, 31),
        (255, 15),
        (127, 255),
        (127, 127),
        (127, 63),
        (127, 31),
        (127, 15),
        (63, 255),
        (63, 127),
        (63, 63),
        (63, 31),
        (63, 15),
        (31, 255),
        (31, 127),
        (31, 63),
        (31, 31),
        (31, 15),
        (15, 255),
        (15, 127),
        (15, 63),
        (15, 31),
        (15, 15),
    ]
    with tqdm(desc=f'Grid Searching on measurement {MEAS_ID:02}', total=N_ITER * NUM_SUBJECTS * len(cases),
              ascii=True) as bar:
        for n_weights, n_activ in cases:
            stats[(n_weights, n_activ)] = np.zeros((N_ITER, NUM_SUBJECTS, 4))
            for i in range(N_ITER):
                iter_stats = single_iter(bar=bar, silent=True, n_weights=n_weights, n_activ=n_activ)
                stats[(n_weights, n_activ)][i, :, :] = iter_stats
                legend = ["train_loss", "train_acc", "valid_loss", "valid_acc"]

                # store it
                filename = os.path.join(PROBLEM, 'grid_results.pkl')
                with open(filename, 'wb') as _f:
                    pickle.dump({"stats": stats, "legend": legend}, _f)


def benchmark():
    stats = np.zeros((N_ITER, NUM_SUBJECTS, 4))

    with tqdm(desc=f'Benchmarking Measurement {MEAS_ID:02}', total=N_ITER * NUM_SUBJECTS, ascii=True) as bar:
        for i in range(N_ITER):
            iter_stats = single_iter(bar=bar, silent=SILENT)
            stats[i, :, :] = iter_stats

            # store the data to make sure not to loose it
            np.savez(file=os.path.join(PROBLEM, EXPORT_FILE.format("runs")),
                     train_loss=stats[i, :, 0],
                     train_acc=stats[i, :, 1],
                     valid_loss=stats[i, :, 2],
                     valid_acc=stats[i, :, 3])

    # compute statistics
    avg_stats = stats.mean(axis=0) # average over all N_ITER experiments
    std_stats = stats.std(axis=0) # std over all N_ITER experiments

    # For the overall score, first average along all subjects.
    mean_avg_stats = avg_stats[:].mean(axis=0) # average over all subjects
    # For standard deviation, average all standard deviations of all subjects
    mean_std_stats = std_stats[:].mean(axis=0) # std over all subjects
    # For standard deviation, calculate the mean over subjects for every run, then the std of the mean
    std_mean_stats = stats.mean(axis=1).std(axis=0)

    print(f"Total Average Accuracy: {mean_avg_stats[3]:.4f} +- {mean_std_stats[3]:.4f}\n")
    print(f"Total Average Accuracy: {mean_avg_stats[3]:.4f} +- {std_mean_stats[3]:.4f}\n")
    # std of the mean, i.e.
    std_mean_sub_stats = avg_stats[:].std(axis=0) # std of mean over all subjects
    print(f"Std. dev. of mean accuracy of subjects: {std_mean_sub_stats[3]:.4f}\n")
    for i in range(0, NUM_SUBJECTS):
        print(f"subject {i+1}: quantized model = {avg_stats[i,3]:.4f} +- {std_stats[i,3]:.4f}")


def _do_subject(subject, bar=None, silent=False, n_weights=None, n_activ=None):
    exp_id = EXP_BASE + subject

    if not silent:
        print(f"Subject {subject}: training quantized model (exp{exp_id})...\r", end='',
              flush=True)

    modification = {'treat.data.subject': subject}
    if n_weights is not None:
        modification['indiv.net.params.weightInqNumLevels'] = n_weights
        modification["indiv.net.params.first_layer_only"] = True
    if n_activ is not None:
        modification['indiv.net.params.actSTENumLevels'] = n_activ
    valid_stats, train_stats = _execute_quantlab(INQ_CONFIG, exp_id, modify_keys=modification)

    if bar is not None:
        bar.update()

    # accumulate log files
    if BENCHMARK or GRID_MEASUREMENT:
        # _accumulate_logs(subject, exp_id)
        _just_store_anything(subject, exp_id, n_weights=n_weights, n_activ=n_activ)

    return _format_all_stats(train_stats, valid_stats)


def _execute_quantlab(config_file, exp_id, fold=None, subj_indexes=None, modify_keys=None):
    # remove all the logs of the previous quantized training experiment
    log_folder = os.path.join(PROBLEM, EXP_FOLDER.format(exp_id))
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder, ignore_errors=True)

    # load configuration
    config = {}
    with open(os.path.join(PROBLEM, config_file)) as _fp:
        config = json.load(_fp)

    # modify keys
    for path, value in modify_keys.items():
        _set_dict_value(config, path, value)

    # store the configuration back as config.json
    if os.path.exists(os.path.join(PROBLEM, MAIN_CONFIG)):
        os.remove(os.path.join(PROBLEM, MAIN_CONFIG))
    with open(os.path.join(PROBLEM, MAIN_CONFIG), "w") as _fp:
        json.dump(config, _fp)

    # execute quantlab without output
    with open(os.devnull, 'w') as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
        train_stats, stats = quantlab_main(PROBLEM, TOPOLOGY, exp_id, 'best', 'train', 10, 1, False,
                                           True, cv_fold=fold, subj_indexes=subj_indexes)
    #train_stats, stats = quantlab_main(PROBLEM, TOPOLOGY, exp_id, 'best', 'train', 10, 1, False,
    #                                   True, cv_fold=fold, subj_indexes=subj_indexes)

    return stats, train_stats


def _format_all_stats(train_stats, valid_stats):
    stats = {}
    for key, value in train_stats.items():
        if key.endswith("loss"):
            stats['train_loss'] = value
        if key.endswith("metric"):
            stats['train_acc'] = value
    for key, value in valid_stats.items():
        if key.endswith("loss"):
            stats['valid_loss'] = value
        if key.endswith("metric"):
            stats['valid_acc'] = value

    return stats


def _format_stats(ref_stats, quant_stats=None):
    stats = {}

    if quant_stats is None:
        for key, value in ref_stats.items():
            if key.endswith("loss"):
                stats['loss'] = value
            if key.endswith("metric"):
                stats['acc'] = value

    else:
        for key, value in ref_stats.items():
            if key.endswith("loss"):
                stats['float_loss'] = value
            if key.endswith("metric"):
                stats['float_acc'] = value
        for key, value in quant_stats.items():
            if key.endswith("loss"):
                stats['quant_loss'] = value
            if key.endswith("metric"):
                stats['quant_acc'] = value

    return stats


def _set_dict_value(d, path, value):
    keys = path.split('.')
    d_working = d
    for key in keys[:-1]:
        d_working = d_working[key]
    d_working[keys[-1]] = value


def _just_store_anything(subject, exp_id, n_weights=None, n_activ=None):
    """ stores everything """
    # extract name of logfile
    stats_folder = os.path.join(PROBLEM, EXP_FOLDER.format(exp_id), "stats")
    log_files = [f for f in os.listdir(stats_folder) if not f.startswith('.')]
    assert(len(log_files) == 1)
    log_file = os.path.join(stats_folder, log_files[0])

    # get eventaccumulator
    ea = EventAccumulator(log_file)
    ea.Reload()

    # load data file
    if GRID_MEASUREMENT:
        name_addon = f"data_W{n_weights}_A{n_activ}_S{subject:02}"
    else:
        name_addon = f"data_S{subject:02}"
    data_file = os.path.join(PROBLEM, EXPORT_FILE.format(name_addon))
    if os.path.exists(data_file):
        with np.load(data_file) as data_loader:
            data = dict(data_loader)
    else:
        data = {'num_trials': 0}

    # update the data dictionary to keep the mean value
    num_trials = data['num_trials']
    for key in ea.Tags()['scalars']:
        new_arr = _prepare_scalar_array_from_tensorboard(ea, key)
        new_arr = np.array([new_arr])
        if num_trials == 0:
            # just add the data
            data[key] = new_arr
        else:
            assert(key in data)
            data[key] = np.concatenate((data[key], new_arr), axis=0)
    data['num_trials'] += 1

    # store data back into the same file
    np.savez(data_file, **data)


def _accumulate_logs(subject, exp_id):
    # extract name of logfile
    stats_folder = os.path.join(PROBLEM, EXP_FOLDER.format(exp_id), "stats")
    log_files = os.listdir(stats_folder)
    assert(len(log_files) == 1)
    log_file = os.path.join(stats_folder, log_files[0])

    # get eventaccumulator
    ea = EventAccumulator(log_file)
    ea.Reload()

    # load data file
    name_addon = f"data_S{subject:02}"
    data_file = os.path.join(PROBLEM, EXPORT_FILE.format(name_addon))
    if os.path.exists(data_file):
        with np.load(data_file) as data_loader:
            data = dict(data_loader)
    else:
        data = {'num_trials': 0}

    # update the data dictionary to keep the mean value
    num_trials = data['num_trials']
    for key in ea.Tags()['scalars']:
        new_arr = _prepare_scalar_array_from_tensorboard(ea, key)
        if num_trials == 0:
            # just add the data
            data[key] = new_arr
        else:
            assert(key in data)
            data[key] = (data[key] * num_trials + new_arr) / (num_trials + 1)
    data['num_trials'] += 1

    # store data back into the same file
    np.savez(data_file, **data)


def _prepare_scalar_array_from_tensorboard(ea, key, start_step=1):
    if ea.Scalars(key)[-1].step == len(ea.Scalars(key)):
        return np.array([x.value for x in ea.Scalars(key)])
    else:
        arr = np.zeros(ea.most_recent_step)
        entries = ea.Scalars(key)
        # we assume the value is zero at the beginning
        for i_entry in range(len(entries)):
            start_idx = entries[i_entry].step - start_step
            end_idx = entries[i_entry + 1].step if i_entry + 1 < len(entries) else \
                ea.most_recent_step - start_step + 1
            arr[start_idx:end_idx] = entries[i_entry].value
        return arr


class TestEnvironment():
    def __enter__(self):
        # backup config.json if it exists
        if os.path.exists(os.path.join(PROBLEM, MAIN_CONFIG)):
            os.rename(os.path.join(PROBLEM, MAIN_CONFIG),
                      os.path.join(PROBLEM, BAK_CONFIG))

        # hide progress default output
        self.devnull = open(os.devnull, 'w')
        progress.Infinite.file = self.devnull

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # remove the created config.json file
        if os.path.exists(os.path.join(PROBLEM, MAIN_CONFIG)):
            os.remove(os.path.join(PROBLEM, MAIN_CONFIG))

        # move backup back
        if os.path.exists(os.path.join(PROBLEM, BAK_CONFIG)):
            os.rename(os.path.join(PROBLEM, BAK_CONFIG),
                      os.path.join(PROBLEM, MAIN_CONFIG))

        # reenable default progress
        progress.Infinite.file = sys.stderr


if __name__ == '__main__':

    if GRID_MEASUREMENT:
        grid_measurement()
    if BENCHMARK:
        benchmark()
    elif CV:
        cross_validate()
    else:
        single_iter()
