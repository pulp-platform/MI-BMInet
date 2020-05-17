import os
import numpy as np
import matplotlib.pyplot as plt
from get_parameters import get_parameters, get_featureMapSize, get_sizeInBytes

# Plotting effects of layer freezing
def plot_freeze(NO_selected_channels, num_classes):
    train_accu_f1 = np.loadtxt(f'ss/freeze1/{NO_selected_channels}ch/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_accu_f1 = np.loadtxt(f'ss/freeze1/{NO_selected_channels}ch/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    train_loss_f1 = np.loadtxt(f'ss/freeze1/{NO_selected_channels}ch/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_loss_f1 = np.loadtxt(f'ss/freeze1/{NO_selected_channels}ch/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_avg.csv')

    train_accu_f2 = np.loadtxt(f'ss/freeze2/{NO_selected_channels}ch/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_accu_f2 = np.loadtxt(f'ss/freeze2/{NO_selected_channels}ch/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    train_loss_f2 = np.loadtxt(f'ss/freeze2/{NO_selected_channels}ch/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_loss_f2 = np.loadtxt(f'ss/freeze2/{NO_selected_channels}ch/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_avg.csv')

    train_accu_f3 = np.loadtxt(f'ss/freeze3/{NO_selected_channels}ch/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_accu_f3 = np.loadtxt(f'ss/freeze3/{NO_selected_channels}ch/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    train_loss_f3 = np.loadtxt(f'ss/freeze3/{NO_selected_channels}ch/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_loss_f3 = np.loadtxt(f'ss/freeze3/{NO_selected_channels}ch/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_avg.csv')

    train_accu = np.loadtxt(f'ss/{NO_selected_channels}ch/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_accu = np.loadtxt(f'ss/{NO_selected_channels}ch/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
    train_loss = np.loadtxt(f'ss/{NO_selected_channels}ch/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_avg.csv')
    valid_loss = np.loadtxt(f'ss/{NO_selected_channels}ch/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_avg.csv')

    # Plot Accuracy
    plt.plot(train_accu_f1, linestyle = ':', color='b',label='Train: 3 Frozen')
    plt.plot(valid_accu_f1, linestyle = '-', color='b',label='Val: 3 Frozen')
    plt.plot(train_accu_f2, linestyle = ':', color='g',label='Train: 2 Frozen')
    plt.plot(valid_accu_f2, linestyle = '-', color='g',label='Val: 2 Frozen')
    plt.plot(train_accu_f3, linestyle = ':', color='r',label='Train: 1 Frozen')
    plt.plot(valid_accu_f3, linestyle = '-', color='r',label='Val: 1 Frozen')
    plt.plot(train_accu, linestyle = ':', color='c',label='Train: 0 Frozen')
    plt.plot(valid_accu, linestyle = '-', color='c',label='Val: 0 Frozen')
    plt.title(f'SS Retraining Accuracy (C:{num_classes} CH:{NO_selected_channels})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left',fontsize='small')
    plt.grid()
    plt.savefig(f'ss/{NO_selected_channels}ch/plots/avg/accu_freeze_avg_{num_classes}_c.pdf')
    plt.clf()

    # Plot Loss
    plt.plot(train_loss_f1, linestyle = ':', color='b',label='Train: 3 Frozen')
    plt.plot(valid_loss_f1, linestyle = '-', color='b',label='Val: 3 Frozen')
    plt.plot(train_loss_f2, linestyle = ':', color='g',label='Train: 2 Frozen')
    plt.plot(valid_loss_f2, linestyle = '-', color='g',label='Val: 2 Frozen')
    plt.plot(train_loss_f3, linestyle = ':', color='r',label='Train: 1 Frozen')
    plt.plot(valid_loss_f3, linestyle = '-', color='r',label='Val: 1 Frozen')
    plt.plot(train_loss, linestyle = ':', color='c',label='Train: 0 Frozen')
    plt.plot(valid_loss, linestyle = '-', color='c',label='Val: 0 Frozen')
    plt.title(f'SS Retraining Loss (C:{num_classes} CH:{NO_selected_channels})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='lower left',fontsize='small')
    plt.grid()
    plt.savefig(f'ss/{NO_selected_channels}ch/plots/avg/loss_freeze_avg_{num_classes}_c.pdf')
    plt.clf()

# Get average for each subject for all splits and plot
def plot_subject_avg(num_classes_list,results_dir,subjects,n_epochs,lr):
    for num_classes in num_classes_list:
        os.makedirs(f'{results_dir}/stats/{num_classes}_class', exist_ok=True)
        os.makedirs(f'{results_dir}/plots/{num_classes}_class', exist_ok=True)
        for subject in subjects:
            train_accu = np.zeros(n_epochs+1)
            valid_accu = np.zeros(n_epochs+1)
            train_loss = np.zeros(n_epochs+1)
            valid_loss = np.zeros(n_epochs+1)
            sub_str = '{0:03d}'.format(subject)
            for sub_split_ctr in range(0,3):
                # Save metrics
                train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')

                train_accu += train_accu_step
                valid_accu += valid_accu_step
                train_loss += train_loss_step
                valid_loss += valid_loss_step

            train_accu = train_accu/3
            valid_accu = valid_accu/3
            train_loss = train_loss/3
            valid_loss = valid_loss/3

            np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_accu)
            np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_accu)
            np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_loss)
            np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_loss)

            # Plot Accuracy
            plt.plot(train_accu, label='Training')
            plt.plot(valid_accu, label='Validation')
            plt.title(f'S:{sub_str} C:{num_classes} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/{num_classes}_class/accu_avg_{num_classes}_c_{sub_str}.pdf')
            plt.clf()

            # Plot Loss
            plt.plot(train_loss, label='Training')
            plt.plot(valid_loss, label='Validation')
            plt.title(f'S:{sub_str} C:{num_classes} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/{num_classes}_class/loss_avg_{num_classes}_c_{sub_str}.pdf')
            plt.clf()

# Get average for everything and plot
def plot_avg(num_classes_list,results_dir,subjects,NO_selected_channels,n_epochs,lr):
    for num_classes in num_classes_list:
        train_accu = np.zeros(n_epochs+1)
        valid_accu = np.zeros(n_epochs+1)
        train_loss = np.zeros(n_epochs+1)
        valid_loss = np.zeros(n_epochs+1)
        for subject in subjects:
            sub_str = '{0:03d}'.format(subject)
            train_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            valid_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            train_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv')
            valid_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv')

            train_accu += train_accu_step
            valid_accu += valid_accu_step
            train_loss += train_loss_step
            valid_loss += valid_loss_step

        train_accu = train_accu/len(subjects)
        valid_accu = valid_accu/len(subjects)
        train_loss = train_loss/len(subjects)
        valid_loss = valid_loss/len(subjects)

        print("SS Validation Accuracy {:.4f}".format(valid_accu[-1]))

        np.savetxt(f'{results_dir}/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_avg.csv', train_accu)
        np.savetxt(f'{results_dir}/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv', valid_accu)
        np.savetxt(f'{results_dir}/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_avg.csv', train_loss)
        np.savetxt(f'{results_dir}/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_avg.csv', valid_loss)

        # Plot Accuracy
        plt.plot(train_accu, label='Training')
        plt.plot(valid_accu, label='Validation')
        plt.title(f'SS Retraining Accuracy C:{num_classes} LR: {lr} CH: {NO_selected_channels}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c.pdf')
        plt.clf()

        # Plot Loss
        plt.plot(train_loss, label='Training')
        plt.plot(valid_loss, label='Validation')
        plt.title(f'SS Retraining Loss C:{num_classes} LR: {lr} CH: {NO_selected_channels}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c.pdf')
        plt.clf()

# Get average of each model for all subjects and plot
def plot_model_avg(num_classes_list,results_dir,subjects,NO_selected_channels,n_epochs,lr):
    for num_classes in num_classes_list:
        for sub_split_ctr in range(0,3):
            train_accu = np.zeros(n_epochs+1)
            valid_accu = np.zeros(n_epochs+1)
            train_loss = np.zeros(n_epochs+1)
            valid_loss = np.zeros(n_epochs+1)
            for subject in subjects:
                sub_str = '{0:03d}'.format(subject)

                train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')

                train_accu += train_accu_step
                valid_accu += valid_accu_step
                train_loss += train_loss_step
                valid_loss += valid_loss_step

            train_accu = train_accu/len(subjects)
            valid_accu = valid_accu/len(subjects)
            train_loss = train_loss/len(subjects)
            valid_loss = valid_loss/len(subjects)

            np.savetxt(f'{results_dir}/stats/avg/train_accu_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', train_accu)
            np.savetxt(f'{results_dir}/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', valid_accu)
            np.savetxt(f'{results_dir}/stats/avg/train_loss_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', train_loss)
            np.savetxt(f'{results_dir}/stats/avg/valid_loss_v1_class_{num_classes}_ss_retrained_model_{sub_split_ctr}_avg.csv', valid_loss)

            # Plot Accuracy
            plt.plot(train_accu, label='Training')
            plt.plot(valid_accu, label='Validation')
            plt.title(f'SS Retraining Accuracy C:{num_classes} M:{sub_split_ctr} LR: {lr} CH: {NO_selected_channels}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c_model_{sub_split_ctr}.pdf')
            plt.clf()

            # Plot Loss
            plt.plot(train_loss, label='Training')
            plt.plot(valid_loss, label='Validation')
            plt.title(f'SS Retraining Loss C:{num_classes} M:{sub_split_ctr} LR: {lr} CH: {NO_selected_channels}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c_model_{sub_split_ctr}.pdf')
            plt.clf()

def plot_memory_accuracy(n_ch_vec, num_classes, kernel_length, NO_samples, pool_length, NO_subjects):
    '''
    plot memory vs accuracy scatterplot for all number of selected channels and layer freezing
    '''
    os.makedirs(f'ss/mem_acc', exist_ok=True)
    color = ['c', 'b', 'g', 'y', 'r', 'm']
    c = 0

    plt.title(f'Intra-Subject (MM) Memory vs Accuracy ({num_classes} Class, {NO_subjects} Subjects)')
    plt.xlabel('Memory (kB)')
    plt.ylabel('Accuracy')

    for n_ch in n_ch_vec:
        accuracy = np.array([])
        memory = np.array([])
        if NO_subjects != 1:
            for i in range(3):
                i = i + 1
                NO_frozen_layers = 4 - i

                data = np.loadtxt(f'ss/freeze{i}/{n_ch}ch/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
                accuracy = np.append(accuracy, data[-1])

                NO_parameters = get_parameters(kernel_length=kernel_length, NO_selected_channels=n_ch, NO_samples=NO_samples, pool_length=pool_length, NO_classes=num_classes, NO_subjects=NO_subjects, NO_frozen_layers=NO_frozen_layers)
                size = get_sizeInBytes(NO_parameters, 'kb')
                memory = np.append(memory, size)
        data = np.loadtxt(f'ss/{n_ch}ch/stats/avg/valid_accu_v1_class_{num_classes}_ss_retrained_avg.csv')
        accuracy = np.append(accuracy, data[-1])

        NO_parameters = get_parameters(kernel_length=kernel_length, NO_selected_channels=n_ch, NO_samples=NO_samples, pool_length=pool_length, NO_classes=num_classes, NO_subjects=NO_subjects, NO_frozen_layers=0)
        size = get_sizeInBytes(NO_parameters, 'kb')
        memory = np.append(memory, size)

        plt.scatter(memory, accuracy, s = 15, marker = 'x', linewidth = 1, color=color[c], label = str(n_ch) + ' channels')
        c += 1
    plt.grid()
    plt.legend(fontsize = 'small')
    plt.savefig(f'ss/mem_acc/memory_vs_accuracy_{num_classes}_class_{NO_subjects}_subjects.pdf')
    plt.clf()

''' --- global --- '''

def plot_ram_accuracy(results_dir, n_ch_vec, n_ds_vec, T_vec, num_classes):
    '''
    plot RAM vs accuracy scatterplot for all number of selected channels, downsampling and time windows
    '''
    os.makedirs(f'{results_dir}/plots/ram_acc', exist_ok=True)
    marker = ['o', 's', '^']
    s = [5, 15, 25]
    color = ['c', 'b', 'g', 'y', 'r', 'm']

    c = 0
    for n_ch in n_ch_vec:
        for T in T_vec:
            for n_ds in n_ds_vec:
                poolLength = int(np.ceil(8/n_ds)) # pool length
                n_s = int(np.ceil(T*160/n_ds)) # number of time samples

                data = np.loadtxt(f'{results_dir}/stats/valid_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv')
                accuracy = data[-1]

                feature_map_size = get_featureMapSize(NO_samples=n_s,NO_selected_channels=n_ch,pool_length=poolLength,NO_classes=num_classes)
                memory = get_sizeInBytes(feature_map_size, 'kb')

                plt.scatter(memory, accuracy, s = s[n_ds-1], marker = marker[T-1], linewidth = 1, color=color[c], label = str(n_ch) + 'ch, '+ str(n_ds) + 'ds, T=' + str(T) + 's')
        c += 1
    plt.axvline(x=128,linewidth=1,linestyle='dashed',color='black')
    plt.title(f'Intra-Subject (MM) RAM vs Accuracy ({num_classes} Class)')
    plt.xlabel('Memory (kB)')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 4, ncol = len(n_ch_vec))
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{results_dir}/plots/ram_acc/ram_vs_accuracy_class_{num_classes}.pdf')
    plt.clf()

def plot_avg_global(num_classes, n_ch, T, n_ds, results_dir):
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    train_accu = np.loadtxt(f'{results_dir}/stats/train_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv')
    valid_accu = np.loadtxt(f'{results_dir}/stats/valid_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv')
    train_loss = np.loadtxt(f'{results_dir}/stats/train_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv')
    valid_loss = np.loadtxt(f'{results_dir}/stats/valid_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv')

    # Plot Accuracy
    plt.plot(train_accu, label='Training')
    plt.plot(valid_accu, label='Validation')
    plt.title(f'Intra-Subject MM Global Accuracy (classes={num_classes}, {n_ch}ch, ds={n_ds}, T={T}s)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{results_dir}/plots/accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.pdf')
    plt.clf()

    # Plot Loss
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')
    plt.title(f'Intra-Subject MM Global Loss (classes={num_classes}, {n_ch}ch, ds={n_ds}, T={T}s)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{results_dir}/plots/loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.pdf')
    plt.clf()

# plot_memory_accuracy([8,16,19,24,38,64], 4, 128, 480, 8, 1)
# plot_ram_accuracy('global/test', [8,16,19,24,38], [1,2,3], [1,2,3], 4)
# plot_avg_global(4,64,3,1,'global')

# plot_freeze(64,2)
# plot_freeze(64,3)
# plot_freeze(64,4)
