import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_npz(filename, export=None, act_quant_line=None):
    data = dict(np.load(filename))
    if 'num_trials' in data:
        del data['num_trials']
    plot_data(data, export, act_quant_line)


def plot_tb(filename, export=None, act_quant_line=None):
    from edgeEEGnet_run import _prepare_scalar_array_from_tensorboard as prepare_tb_array
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(filename)
    ea.Reload()
    data = {key: prepare_tb_array(ea, key) for key in ea.Tags()['scalars']}
    plot_data(data, export, act_quant_line)


def plot_data(data, export=None, act_quant_line=None):
    # decide for each key to which plot it should belong
    loss_plot = {}
    acc_plot = {}

    n_epochs = None

    for name, array in data.items():
        if n_epochs is None:
            n_epochs = len(array)
        else:
            assert len(array) == n_epochs, f"{name} has length {len(array)} but should be {n_epochs}"

        l_name = name.lower()
        if 'metric' in l_name or 'acc' in l_name or 'accuracy' in l_name:
            acc_plot[name] = array
        elif 'loss' in l_name:
            loss_plot[name] = array
        elif l_name == 'learning_rate':
            pass
        else:
            # ask user to which plot it should be added
            choice = input(f"Where to put {name}? [b]oth, [l]oss, [a]ccuracy, [N]one? > ")
            choice = choice.lower() if choice else 'n'
            assert choice in ['b', 'l', 'a', 'n']
            if choice in ['b', 'l']:
                loss_plot[name] = array
            if choice in ['b', 'a']:
                acc_plot[name] = array

    generate_figure(loss_plot, acc_plot, n_epochs, export, act_quant_line)


def generate_figure(loss_plot, acc_plot, n_epochs, export=None, act_quant_line=None):

    # make sure that the environment variables are set (to hide the unnecessary output)
    if "XDG_RUNTIME_DIR" not in os.environ:
        tmp_dir = "/tmp/runtime-eegnet"
        os.environ["XDG_RUNTIME_DIR"] = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            os.chmod(tmp_dir, 700)

    # prepare data
    x = np.array(range(1, n_epochs + 1))

    # prepare the plot
    fig = plt.figure(figsize=(20, 10))

    # do loss figure
    loss_subfig = fig.add_subplot(121)
    add_subplot(loss_plot, x, loss_subfig, "Loss", "upper center", act_quant_line)

    # do accuracy figure
    acc_subfig = fig.add_subplot(122)
    add_subplot(acc_plot, x, acc_subfig, "Accuracy", "lower center", act_quant_line)

    # save the image
    if export is None:
        plt.show()
    else:
        fig.savefig(export, bbox_inches='tight')

    # close
    plt.close('all')


def add_subplot(data, x, subfig, title, legend_pos=None, act_quant_line=None):
    plt.grid()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    additional_axis = []
    lines = []

    if act_quant_line is not None:
        lines.append(plt.axvline(x=act_quant_line, label='Activation Quantization', color=colors[2]))

    for i, key in enumerate(data.keys()):
        if key.startswith('train_'):
            new_lines = subfig.plot(x, data[key], label=key, color=colors[0])
        elif key.startswith('valid_'):
            new_lines = subfig.plot(x, data[key], label=key, color=colors[1])
        else:
            tmp_axis = subfig.twinx()
            tmp_axis.set_ylabel(key)
            new_lines = tmp_axis.plot(x, data[key], label=key, color=colors[i+3])
            additional_axis.append(tmp_axis)
        lines += new_lines

    for i, axis in enumerate(additional_axis):
        axis.spines['right'].set_position(('axes', 1 + i * 0.15))
        if i > 0:
            axis.set_frame_on(True)
            axis.patch.set_visible(False)

    subfig.set_title(title)
    subfig.set_xlabel("Epoch")

    labels = [l.get_label() for l in lines]
    last_ax = additional_axis[-1] if additional_axis else subfig
    last_ax.legend(lines, labels, frameon=True, framealpha=1, facecolor='white', loc=legend_pos)

    return len(additional_axis)


def plot_tb_cv(filename, export=None, act_quant_line=None, folds=5, avg=False):
    from edgeEEGnet_run import _prepare_scalar_array_from_tensorboard as prepare_tb_array
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    #print(os.listdir(filename))
    expdirs = []
    for d in os.listdir(filename):
        if 'exp' in d:
            expdirs.append(d)
    common = os.path.commonprefix(expdirs)
    data = []
    for f in range(folds):
        datafilename = f'{filename}/{common}{f}/stats'
        datafilename = os.path.join(datafilename, os.listdir(datafilename)[0])
        ea = EventAccumulator(datafilename)
        ea.Reload()
        data.append({key: prepare_tb_array(ea, key) for key in ea.Tags()['scalars']})
    plot_data_cv(data, export, act_quant_line, folds, avg)


def plot_data_cv(data, export=None, act_quant_line=None, folds=5, avg=False):

    n_epochs = None

    loss_folds = []
    acc_folds = []

    for d in data:
        # decide for each key to which plot it should belong
        loss_plot = {}
        acc_plot = {}
        for name, array in d.items():
            if n_epochs is None:
                n_epochs = len(array)
            else:
                assert len(array) == n_epochs, f"{name} has length {len(array)} but should be {n_epochs}"

            l_name = name.lower()
            if 'metric' in l_name or 'acc' in l_name or 'accuracy' in l_name:
                acc_plot[name] = array
            elif 'loss' in l_name:
                loss_plot[name] = array
            elif l_name == 'learning_rate':
                pass
            else:
                # ask user to which plot it should be added
                choice = input(f"Where to put {name}? [b]oth, [l]oss, [a]ccuracy, [N]one? > ")
                choice = choice.lower() if choice else 'n'
                assert choice in ['b', 'l', 'a', 'n']
                if choice in ['b', 'l']:
                    loss_plot[name] = array
                if choice in ['b', 'a']:
                    acc_plot[name] = array
        loss_folds.append(loss_plot)
        acc_folds.append(acc_plot)

    generate_figure_cv(loss_folds, acc_folds, n_epochs, export, act_quant_line, folds, avg)


def generate_figure_cv(loss_plot, acc_plot, n_epochs, export=None, act_quant_line=None, folds=5, avg=False):

    # make sure that the environment variables are set (to hide the unnecessary output)
    if "XDG_RUNTIME_DIR" not in os.environ:
        tmp_dir = "/tmp/runtime-eegnet"
        os.environ["XDG_RUNTIME_DIR"] = tmp_dir
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            os.chmod(tmp_dir, 700)

    # prepare data
    x = np.array(range(1, n_epochs + 1))

    # prepare the plot
    fig = plt.figure(figsize=(20, 10))

    # do loss figure
    loss_subfig = fig.add_subplot(121)
    add_subplot_cv(loss_plot, x, loss_subfig, "Loss", "upper center", act_quant_line, folds, avg)

    # do accuracy figure
    acc_subfig = fig.add_subplot(122)
    add_subplot_cv(acc_plot, x, acc_subfig, "Accuracy", "lower center", act_quant_line, folds, avg)

    # save the image
    if export is None:
        plt.show()
    else:
        fig.savefig(export, bbox_inches='tight')

    # close
    plt.close('all')


def add_subplot_cv(data, x, subfig, title, legend_pos=None, act_quant_line=None, folds=5, avg=False):
    plt.grid()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    additional_axis = []
    lines = []

    if act_quant_line is not None:
        lines.append(plt.axvline(x=act_quant_line, label='Activation Quantization', color=colors[2]))

    if avg:

        lines_train = np.zeros((folds, x.shape[0]))
        lines_valid = np.zeros((folds, x.shape[0]))
        for f in range(folds):
            d = data[f]
            for i, key in enumerate(d.keys()):
                if key.startswith('train_'):
                    lines_train[f] = d[key]
                elif key.startswith('valid_'):
                    lines_valid[f] = d[key]
                else:
                    if not additional_axis:
                        tmp_axis = subfig.twinx()
                        tmp_axis.set_ylabel(key)
                        new_lines = tmp_axis.plot(x, d[key], label=key, color=colors[i+3])
                        lines += new_lines
                        additional_axis.append(tmp_axis)
        lines_train_avg = np.mean(lines_train, axis=0)
        lines_valid_avg = np.mean(lines_valid, axis=0)
        lines_train_std = np.std(lines_train, axis=0)
        lines_valid_std = np.std(lines_valid, axis=0)
        new_lines = subfig.plot(x, lines_train_avg, label="train", color=colors[0])
        subfig.fill_between(x, lines_train_avg-lines_train_std, lines_train_avg+lines_train_std, alpha=0.2)
        lines += new_lines
        new_lines = subfig.plot(x, lines_valid_avg, label="valid", color=colors[1])
        subfig.fill_between(x, lines_valid_avg-lines_valid_std, lines_valid_avg+lines_valid_std, alpha=0.2)
        lines += new_lines
        print(np.argmax(lines_valid_avg), lines_valid_avg[np.argmax(lines_valid_avg)])
        #print(1000, lines_valid_avg[1000-1], 1200, lines_valid_avg[1200-1], 1250, lines_valid_avg[1250-1], 1400, lines_valid_avg[1400-1], 1450, lines_valid_avg[1450-1], 1500, lines_valid_avg[1500-1])

    else:

        for f in range(folds):
            d = data[f]
            for i, key in enumerate(d.keys()):
                if key.startswith('train_'):
                    new_lines = subfig.plot(x, d[key], label=key+'_fold'+str(f), color=colors[0])
                elif key.startswith('valid_'):
                    new_lines = subfig.plot(x, d[key], label=key+'_fold'+str(f), color=colors[1])
                else:
                    tmp_axis = subfig.twinx()
                    tmp_axis.set_ylabel(key)
                    new_lines = tmp_axis.plot(x, d[key], label=key+'_fold'+str(f), color=colors[i+3])
                    additional_axis.append(tmp_axis)
                lines += new_lines

    for i, axis in enumerate(additional_axis):
        axis.spines['right'].set_position(('axes', 1 + i * 0.15))
        if i > 0:
            axis.set_frame_on(True)
            axis.patch.set_visible(False)

    subfig.set_title(title)
    subfig.set_xlabel("Epoch")

    labels = [l.get_label() for l in lines]
    last_ax = additional_axis[-1] if additional_axis else subfig
    last_ax.legend(lines, labels, frameon=True, framealpha=1, facecolor='white', loc=legend_pos)

    return len(additional_axis)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='filename of the data', nargs=1)
    parser.add_argument('-t', '--tensorboard', help='Data is of tensorboard format',
                        action='store_true')
    parser.add_argument('-n', '--numpy', help='Data is of numpy npz format',
                        action='store_true')
    parser.add_argument('-e', '--export', help='export plot to specified file', type=str)
    parser.add_argument('--act_quant_line', help='position of vertical line where activation quantization starts', type=int)
    parser.add_argument('--cv', help='plot for cross-validation', action='store_true')
    parser.add_argument('-f', '--folds', help='number of folds', type=int, default=5)
    parser.add_argument('--avg', help='compute avg of the metrics among the folds and plot the avg', action='store_true')

    args = parser.parse_args()

    # if both tensorboard and numpy are not set, infer the type by the file ending
    filename = args.file[0]
    if not args.tensorboard and not args.numpy and not args.cv:
        if 'events.out.tfevents' in filename:
            args.tensorboard = True
        elif filename.endswith('.npz'):
            args.numpy = True
        else:
            raise RuntimeError(f'Cannot automatically detect type of the file: {args.file}')

    if args.cv:
        if not os.path.isdir(filename):
            raise RuntimeError(f'Give as input the directory containing the tensorboard experiments of the cross-validation')
        for files in os.walk(filename):
            list_files = np.asarray([np.asarray(x).flatten() for x in files if x]).flatten()
            for i in list_files:
                if 'events.out.tfevents' in i:
                    tbfiles=True
                if 'stats' in i:
                    statsdir=True
        if not tbfiles and not statsdir:
            raise RuntimeError(f'Give as input the directory containing the tensorboard experiments of the cross-validation')
        plot_tb_cv(filename, args.export, args.act_quant_line, args.folds, args.avg)

    # if args.tensorboard:
    #     plot_tb(filename, args.export, args.act_quant_line)
    # elif args.numpy:
    #     plot_npz(filename, args.export, args.act_quant_line)
    # else:
    #     raise RuntimeError()
