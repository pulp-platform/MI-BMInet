import numpy as np
import torch as t
import torch.quantization as tq

from tqdm import tqdm

from edgeEEGnet import edgeEEGNet
from edgeEEGnet_quant import edgeEEGNetQuant
from utils.get_data import get_data, as_data_loader, as_tensor
from utils.kfold_cv import KFoldCV
from utils.plot_results import generate_plots
from utils.metrics import get_metrics_from_model
from utils.misc import class_decision
from utils.early_stopping import EarlyStopping

from utils.summary import print_summary


def train_subject_specific_cv(subject, n_splits=4, epochs=500, batch_size=32, lr=0.001,
                              early_stopping=True, silent=False, plot=True, **kwargs):
    """
    Trains a subject specific model for the given subject, using K-Fold Cross Validation

    Parameters:
     - subject:        Integer in the Range 1 <= subject <= 9
     - n_splits:       Number of splits for K-Fold CV
     - epochs:         Number of epochs to train
     - batch_size:     Batch Size
     - lr:             Learning Rate
     - early_stopping: bool, approximate the number of epochs to train the network for the subject.
     - silent:         bool, if True, generate no output, including progress bar
     - plot:           bool, if True, generates plots
     - kwargs:         remaining parameters are passed to the edgeEEGnet model

    Returns: (models, metrics, epoch)
     - models:  List of t.nn.Module, size = [n_splits]
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1, averaged over all splits
     - epoch:   integer, number of epochs determined by early_stopping. If early_stopping is not
                used, then this value will always be equal to the parameter epochs

    Notes:
     - Early Stopping: Uses early stopping to determine the best epoch to stop training for the
       given subject by averaging over the stopping epoch of all splits.
    """

    # load the raw data
    samples, labels = get_data(subject, training=True)
    samples, labels = as_tensor(samples, labels)

    # prepare the models
    models = [edgeEEGNet(T=samples.shape[2], **kwargs) for _ in range(n_splits)]
    metrics = t.zeros((n_splits, 4))
    best_epoch = np.zeros(n_splits)

    # prepare KFold
    kfcv = KFoldCV(n_splits)
    split = 0

    # open the progress bar
    with tqdm(desc=f"Subject {subject} CV, split 1", total=n_splits * epochs, leave=False,
              unit='epoch', ascii=True, disable=silent) as pbar:
        # loop over all splits
        for split, indices in enumerate(kfcv.split(samples, labels)):
            # set the progress bar title
            pbar.set_description(f"Subject {subject} CV, split {split + 1}")

            # generate dataset
            train_idx, val_idx = indices
            train_ds = t.utils.data.TensorDataset(samples[train_idx], labels[train_idx])
            val_ds = t.utils.data.TensorDataset(samples[val_idx], labels[val_idx])
            train_loader = t.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = t.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)

            # prepare the model
            model = models[split]
            model.initialize_params()
            if t.cuda.is_available():
                model = model.cuda()

            # prepare loss function and optimizer
            loss_function = t.nn.CrossEntropyLoss()
            optimizer = t.optim.Adam(model.parameters(), lr=lr)

            model, split_metrics, split_epoch, _ = _train_net(subject, model, train_loader,
                                                              val_loader, loss_function, optimizer,
                                                              epochs=epochs,
                                                              early_stopping=early_stopping,
                                                              plot=plot, pbar=pbar)

            metrics[split, :] = split_metrics[0, :]
            best_epoch[split] = split_epoch

    # average all metrics
    metrics = metrics.mean(axis=0).reshape(1, 4)

    # print the result
    if not silent:
        print(f"Subje`ct {subject} CV: accuracy = {metrics[0, 0]}, at epoch {best_epoch.mean()} " +
              f"+- {best_epoch.std()}")
    return models, metrics, int(best_epoch.mean().round())


def train_subject_specific(subject, epochs=500, batch_size=32, lr=0.001, silent=False, plot=True,
                           **kwargs):
    """
    Trains a subject specific model for the given subject

    Parameters:
     - subject:    Integer in the Range 1 <= subject <= 9
     - epochs:     Number of epochs to train
     - batch_size: Batch Size
     - lr:         Learning Rate
     - silent:     bool, if True, hide all output including the progress bar
     - plot:       bool, if True, generate plots
     - kwargs:     Remaining arguments passed to the edgeEEGnet model

    Returns: (model, metrics)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
    """
    # load the data
    train_samples, train_labels = get_data(subject, training=True)
    test_samples, test_labels = get_data(subject, training=False)
    train_loader = as_data_loader(train_samples, train_labels, batch_size=batch_size)
    # test_loader = as_data_loader(test_samples, test_labels, batch_size=test_labels.shape[0])
    test_loader = as_data_loader(test_samples, test_labels, batch_size=batch_size)

    # prepare the model
    model = edgeEEGNet(T=train_samples.shape[2], **kwargs)
    model.initialize_params()
    if t.cuda.is_available():
        model = model.cuda()

    # prepare loss function and optimizer
    loss_function = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    scheduler = None

    # print the training setup
    print_summary(model, optimizer, loss_function, scheduler)

    # prepare progress bar
    with tqdm(desc=f"Subject {subject}", total=epochs, leave=False, disable=silent,
              unit='epoch', ascii=True) as pbar:

        # Early stopping is not allowed in this mode, because the testing data cannot be used for
        # training!
        model, metrics, _, history = _train_net(subject, model, train_loader, test_loader,
                                                loss_function, optimizer, scheduler=scheduler,
                                                epochs=epochs, early_stopping=False, plot=plot,
                                                pbar=pbar)

    if not silent:
        print(f"Subject {subject}: accuracy = {metrics[0, 0]}")
    return model, metrics, history


def train_subject_specific_quant(subject, epochs=500, batch_size=32, lr=0.001, silent=False,
                                 plot=True, **kwargs):
    """
    Trains a subject specific model for the given subject

    Parameters:
     - subject:    Integer in the Range 1 <= subject <= 9
     - epochs:     Number of epochs to train
     - batch_size: Batch Size
     - lr:         Learning Rate
     - silent:     bool, if True, hide all output including the progress bar
     - plot:       bool, if True, generate plots
     - kwargs:     Remaining arguments passed to the edgeEEGnet model

    Returns: (model, metrics)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
    """
    # load the data
    train_samples, train_labels = get_data(subject, training=True)
    test_samples, test_labels = get_data(subject, training=False)
    train_loader = as_data_loader(train_samples, train_labels, batch_size=batch_size)
    # test_loader = as_data_loader(test_samples, test_labels, batch_size=test_labels.shape[0])
    test_loader = as_data_loader(test_samples, test_labels, batch_size=batch_size)

    # prepare quantization configuration
    qconfig = tq.QConfig(activation=tq.MinMaxObserver.with_args(dtype=t.quint8),
                         weight=tq.MinMaxObserver.with_args(dtype=t.qint8))

    # prepare the model
    model = edgeEEGNetQuant(T=train_samples.shape[2], qconfig=qconfig, **kwargs)
    model.initialize_params()
    if t.cuda.is_available():
        model = model.cuda()

    # prepare the quantization
    tq.prepare_qat(model, inplace=True)

    # prepare loss function and optimizer
    loss_function = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    scheduler = None

    # print the training setup
    print_summary(model, optimizer, loss_function, scheduler)

    # prepare progress bar
    with tqdm(desc=f"Subject {subject}", total=epochs, leave=False, disable=silent,
              unit='epoch', ascii=True) as pbar:

        # Early stopping is not allowed in this mode, because the testing data cannot be used for
        # training!
        model, metrics, _, history = _train_net(subject, model, train_loader, test_loader,
                                                loss_function, optimizer, scheduler=scheduler,
                                                epochs=epochs, early_stopping=False, plot=plot,
                                                pbar=pbar)

    # convert the model into a quantized model
    model = model.cpu()
    tq.convert(model, inplace=True)

    metrics = get_metrics_from_model(model, test_loader)

    if not silent:
        print(f"Subject {subject}: accuracy = {metrics[0, 0]}")
    return model, metrics, history


def test_model_from_keras(subject, batch_size=32,
                          folder='/scratch/sem19h24/EEGNet_reformat/EEGnet/initialModels/'):
    """
    Import the keras model and test it

    Parameters:
        subject:    Number between 1 and 9
        batch_size: Batch Size to test the model
        folder:     Folder where the model files are found.

    returns: loss, accuracy
    """

    # Just use the default values and hope all dimensions are fine...
    model = edgeEEGNet(activation='elu', constrain_w=False, permuted_flatten=True)
    model.load_model_params_from_keras(f"{folder}model{subject-1}_torch.npz")
    if t.cuda.is_available():
        model = model.cuda()

    print_summary(model, None, None, None)

    # get dataloader
    test_samples, test_labels = get_data(subject, training=False)
    test_loader = as_data_loader(test_samples, test_labels, batch_size=batch_size)

    # prepare loss function, no optimizer necessary
    loss_function = t.nn.CrossEntropyLoss()
    return _test_net(model, test_loader, loss_function, train=False)


def _train_net(subject, model, train_loader, val_loader, loss_function, optimizer, scheduler=None,
               epochs=500, early_stopping=True, plot=True, track_lr=True, pbar=None):
    """
    Main training loop

    Parameters:
     - subject:        Integer, subject ID
     - model:          t.nn.Module (is set to training mode)
     - train_loader:   t.utils.data.DataLoader: training data
     - val_loader:     t.utils.data.DataLoader: validation data
     - loss_function:  function
     - optimizer:      t.optim.Optimizer
     - scheduler:      t.optim.lr_scheduler or None
     - epochs:         Integer, number of epochs to train
     - early_stopping: boolean, if True, store models for all epochs and select the one with the
                       highest validation accuracy
     - plot:           boolean, if True, generate all plots and store on disk
     - pbar:           tqdm progress bar or None, in which case no progress will be displayed
                       (not closed afterwards)

    Returns: (model, metrics, epoch, history)
     - model:   t.nn.Module, trained model
     - metrics: t.tensor, size=[1, 4], accuracy, precision, recall, f1
     - epoch:   integer, always equal to 500 if early stopping is not used
     - history: tuple: (loss, accuracy), where both are t.tensor, size=[2, epochs]

    Notes:
     - Model and data will not be moved to gpu, do this outside of this function.
     - When early_stopping is enabled, this function will store all intermediate models
    """

    # prepare result
    loss = t.zeros((2, epochs))
    accuracy = t.zeros((2, epochs))
    lr = None
    if track_lr:
        lr = t.zeros((epochs))

    # prepare early_stopping
    if early_stopping:
        early_stopping = EarlyStopping()

    use_cuda = model.is_cuda()

    # train model for all epochs
    for epoch in range(epochs):
        # train the model
        train_loss, train_accuracy = _train_epoch(model, train_loader, loss_function, optimizer,
                                                  scheduler=scheduler, use_cuda=use_cuda)

        # collect current loss and accuracy
        validation_loss, validation_accuracy = _test_net(model, val_loader, loss_function,
                                                         train=False, use_cuda=use_cuda)
        loss[0, epoch] = train_loss
        loss[1, epoch] = validation_loss
        accuracy[0, epoch] = train_accuracy
        accuracy[1, epoch] = validation_accuracy
        if track_lr:
            lr[epoch] = optimizer.param_groups[0]['lr']

        # do early stopping
        if early_stopping:
            early_stopping.checkpoint(model, loss[1, epoch], accuracy[1, epoch], epoch)

        if pbar is not None:
            pbar.update()

    # get the best model
    if early_stopping:
        model, best_loss, best_accuracy, best_epoch = early_stopping.use_best_model(model)
    else:
        best_epoch = epoch

    # generate plots
    if plot:
        generate_plots(subject, model, val_loader, loss, accuracy, lr=lr)

    metrics = get_metrics_from_model(model, val_loader)

    return model, metrics, best_epoch + 1, (loss, accuracy)


def _train_epoch(model, loader, loss_function, optimizer, scheduler=None, use_cuda=None):
    """
    Trains a single epoch

    Parameters:
     - model:         t.nn.Module (is set to training mode)
     - loader:        t.utils.data.DataLoader
     - loss_function: function
     - optimizer:     t.optim.Optimizer
     - scheduler:     t.optim.lr_scheduler or None
     - use_cuda:      bool or None, if None, use cuda if the model is on cuda

    Returns: loss: float, accuracy: float
    """

    if use_cuda is None:
        use_cuda = model.is_cuda()

    model.train(True)
    n_samples = 0
    running_loss = 0.0
    accuracy = 0.0
    for x, y in loader:
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        # Forward step
        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # prepare loss and accuracy
        n_samples += x.shape[0]
        running_loss += loss * x.shape[0]
        decision = class_decision(output)
        accuracy += (decision == y).sum().item()

    running_loss = running_loss / n_samples

    if scheduler:
        scheduler.step(running_loss)

    accuracy = accuracy / n_samples
    return running_loss, accuracy


def _test_net(model, loader, loss_function, train=False, use_cuda=None):
    """
    Tests the model for accuracy

    Parameters:
     - model:         t.nn.Module (is set to testing mode)
     - loader:        t.utils.DataLoader
     - loss_function: function or None.
     - train:         boolean, if the model is to be set into training or testing mode
     - use_cuda:      bool or None, if None, use cuda if the model is on cuda

    Returns: loss: float, accuracy: float
    """
    if use_cuda is None:
        use_cuda = model.is_cuda()

    # set the model into testing mode
    model.train(train)
    with t.no_grad():

        running_loss = 0.0
        running_acc = 0.0
        n_total = 0

        # get the data from the loader (only one batch will be available)
        for x, y in loader:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            # compute the output
            output = model(x)

            # compute accuracy
            yhat = class_decision(output)
            prediction_correct = yhat == y
            num_correct = prediction_correct.sum().item()
            running_acc += num_correct

            # compute the loss function
            loss = loss_function(output, y)
            running_loss += loss * x.shape[0]

            # increment sample counter
            n_total += x.shape[0]

    return running_loss / n_total, running_acc / n_total
