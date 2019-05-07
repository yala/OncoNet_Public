import os
import math
import numpy as np
import sklearn.metrics
import torch
import torch.autograd as autograd
from tqdm import tqdm
import onconet.models.factory as model_factory
import onconet.learn.state_keeper as state
import onconet.utils.stats as stats
from onconet.learn.utils import cluster_results_by_exam, ignore_None_collate, init_metrics_dictionary, \
    get_train_and_dev_dataset_loaders, compute_eval_metrics, \
    get_human_preds
from onconet.learn.step import get_model_loss, model_step
import warnings
import pdb

tqdm.monitor_interval=0

def get_train_variables(args, model):
    '''
        Given args, and whether or not resuming training, return
        relevant train variales.

        returns:
        - start_epoch:  Index of initial epoch
        - epoch_stats: Dict summarizing epoch by epoch results
        - state_keeper: Object responsibile for saving and restoring training state
        - batch_size: sampling batch_size
        - models: Dict of models
        - optimizers: Dict of optimizers, one for each model
        - tuning_key: Name of epoch_stats key to control learning rate by
        - num_epoch_sans_improvement: Number of epochs since last dev improvment, as measured by tuning_key
        - num_epoch_since_reducing_lr: Number of epochs since last lr reduction
        - no_tuning_on_dev: True when training does not adapt based on dev performance
    '''
    start_epoch = 1
    if args.current_epoch is not None:
        start_epoch = args.current_epoch
    if args.lr is None:
        args.lr = args.init_lr
    if args.epoch_stats is not None:
        epoch_stats = args.epoch_stats
    else:
        epoch_stats = init_metrics_dictionary(modes=['train', 'dev'])

    state_keeper = state.StateKeeper(args)
    batch_size = args.batch_size // args.batch_splits

    # Set up models
    if isinstance(model, dict):
        models = model
    else:
        models = {'model': model }

    # Setup optimizers
    optimizers = {}
    for name in models:
        model = models[name]

        if args.cuda:
            model = model.cuda()

        optimizers[name] = model_factory.get_optimizer(model, args)

    if args.optimizer_state is not None:
        for optimizer_name in args.optimizer_state:
            state_dict = args.optimizer_state[optimizer_name]
            optimizers[optimizer_name] = state_keeper.load_optimizer(
                optimizers[optimizer_name],
                state_dict)

    num_epoch_sans_improvement = 0
    num_epoch_since_reducing_lr = 0

    no_tuning_on_dev = args.no_tuning_on_dev or args.ten_fold_cross_val

    tuning_key = "dev_{}".format(args.tuning_metric)

    return start_epoch, epoch_stats, state_keeper, batch_size, models, optimizers, tuning_key, num_epoch_sans_improvement, num_epoch_since_reducing_lr, no_tuning_on_dev


def train_model(train_data, dev_data, model, args):
    '''
        Train model and tune on dev set. If model doesn't improve dev performance within args.patience
        epochs, then halve the learning rate, restore the model to best and continue training.

        At the end of training, the function will restore the model to best dev version.

        returns epoch_stats: a dictionary of epoch level metrics for train and test
        returns models : dict of models, containing best performing model setting from this call to train
    '''

    start_epoch, epoch_stats, state_keeper, batch_size, models, optimizers, tuning_key, num_epoch_sans_improvement, num_epoch_since_reducing_lr, no_tuning_on_dev = get_train_variables(
        args, model)

    train_data_loader, dev_data_loader = get_train_and_dev_dataset_loaders(
        args,
        train_data,
        dev_data,
        batch_size)
    for epoch in range(start_epoch, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))

        for mode, data_loader in [('Train', train_data_loader), ('Dev', dev_data_loader)]:
            train_model = mode == 'Train'
            key_prefix = mode.lower()
            loss, accuracy, _, golds, preds, probs, auc, _,  reg_loss, precision, recall, f1 = run_epoch(
                data_loader,
                train_model=train_model,
                truncate_epoch=True,
                models=models,
                optimizers=optimizers,
                args=args)

            confusion_matrix = sklearn.metrics.confusion_matrix(golds, preds)
            epoch_stats['{}_loss'.format(key_prefix)].append(loss)
            epoch_stats['{}_reg_loss'.format(key_prefix)].append(reg_loss)
            epoch_stats['{}_auc'.format(key_prefix)].append(auc)
            epoch_stats['{}_accuracy'.format(key_prefix)].append(accuracy)
            epoch_stats['{}_precision'.format(key_prefix)].append(precision)
            epoch_stats['{}_recall'.format(key_prefix)].append(recall)
            epoch_stats['{}_f1'.format(key_prefix)].append(f1)
            epoch_stats['{}_confusion_matrix'.format(key_prefix)].append(confusion_matrix.tolist())

            log_statement = '{}\n{} - loss: {:.6f} reg_loss: {:.6f} accuracy: {:.2f} auc: {} precision: {} recall: {} f1: {}'.format(
                args.objective, mode, loss, reg_loss, accuracy, auc, precision, recall, f1)

            print(log_statement)

        # Save model if beats best dev, or if not tuning on dev
        best_func = min if tuning_key == 'dev_loss' else max
        improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]
        if improved or no_tuning_on_dev:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            # Subtract one because epoch is 1-indexed and arr is 0-indexed
            epoch_stats['best_epoch'] = epoch - 1
            state_keeper.save(models, optimizers, epoch, args.lr, epoch_stats)

        num_epoch_since_reducing_lr += 1
        if improved:
            num_epoch_sans_improvement = 0
        else:
            num_epoch_sans_improvement += 1

        print('---- Best Dev {} is {} at epoch {}'.format(
            args.tuning_metric,
            epoch_stats[tuning_key][epoch_stats['best_epoch']],
            epoch_stats['best_epoch'] + 1))

        if num_epoch_sans_improvement >= args.patience or \
                (no_tuning_on_dev and num_epoch_since_reducing_lr >= args.lr_reduction_interval):
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            num_epoch_since_reducing_lr = 0
            if not args.turn_off_model_reset:
                models, optimizer_states, _, _, _ = state_keeper.load()

                # Reset optimizers
                for name in optimizers:
                    optimizer = optimizers[name]
                    state_dict = optimizer_states[name]
                    optimizers[name] = state_keeper.load_optimizer(optimizer, state_dict)
            # Reduce LR
            for name in optimizers:
                optimizer = optimizers[name]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay

            # Update lr also in args for resumable usage
            args.lr *= .5

    # Restore model to best dev performance, or last epoch when not tuning on dev
    models, _, _, _, _ = state_keeper.load()

    return epoch_stats, models


def compute_threshold_and_dev_stats(dev_data, models, args):
    '''
    Compute threshold based on the Dev results
    '''
    if not isinstance(models, dict):
        models = {'model': models}
    if args.cuda:
        models['model'] = models['model'].cuda()

    dev_stats = init_metrics_dictionary(modes=['dev'])

    batch_size = args.batch_size // args.batch_splits
    data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        collate_fn = ignore_None_collate,
        pin_memory=True,
        drop_last = False)
    loss, accuracy, confusion_matrix, golds, preds, probs, auc, exams, reg_loss, precision, recall, f1 = run_epoch(
        data_loader,
        train_model=False,
        truncate_epoch=False,
        models=models,
        optimizers=None,
        args=args)

    human_preds = get_human_preds(exams, dev_data.metadata_json)


    threshold, (th_lb, th_ub) = stats.get_thresholds_interval(probs, golds, human_preds, rebalance_eval_cancers=args.rebalance_eval_cancers)
    args.threshold = threshold
    print(' Dev Threshold: {:.8f} ({:.8f} - {:.8f})'.format(threshold, th_lb, th_ub))

    log_statement, dev_stats = compute_eval_metrics(
                            args, loss, accuracy, confusion_matrix,
                            golds, preds, probs, auc, exams,
                            reg_loss, precision, recall, f1,
                            dev_stats, 'dev')
    print(log_statement)
    return dev_stats


def eval_model(test_data, models, args):
    '''
        Run model on test data, and return test stats (includes loss

        accuracy, etc)
    '''
    if not isinstance(models, dict):
        models = {'model': models}
    if args.cuda:
        models['model'] = models['model'].cuda()

    batch_size = args.batch_size // args.batch_splits
    test_stats = init_metrics_dictionary(modes=['test'])
    data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False)

    loss, accuracy, confusion_matrix, golds, preds, probs, auc, exams, meta_loss, reg_loss, precision, recall, f1 = run_epoch(
        data_loader,
        train_model=False,
        truncate_epoch=False,
        models=models,
        optimizers=None,
        args=args)

    log_statement, test_stats = compute_eval_metrics(
                            args, loss, accuracy, confusion_matrix,
                            golds, preds, probs, auc, exams, meta_loss,
                            reg_loss, precision, recall, f1,
                            test_stats, 'test')
    print(log_statement)

    return test_stats


def run_epoch(data_loader, train_model, truncate_epoch, models, optimizers, args):
    '''
        Run model for one pass of data_loader, and return epoch statistics.

        args:
        - data_loader: Pytorch dataloader over some dataset.
        - train_model: True to train the model and run the optimizers
        - models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
        - optimizer: dict of optimizers, one for each model
        - args: general runtime args defined in by argparse

        returns:
        - avg_loss: epoch loss
        - accuracy: epoch accuracy
        - confusion_matrix:
        - golds: labels for all samples in data_loader
        - preds: model predictions for all samples in data_loader
        - probs: model softmaxes for all samples in data_loader
        - auc: model auc for epoch, only well defined for binary classification
        - exams: exam ids for samples if available, used to cluster samples for evaluation.
        - avg_meta_loss: avg loss of meta setups across epoch. this includes wassertien critic, and all additional losses.
    '''
    data_iter = data_loader.__iter__()
    preds = []
    probs = []
    golds = []
    losses = []
    reg_losses = []
    exams = []

    volatile = not train_model
    for name in models:
        if train_model:
            models[name].train()
            if optimizers is not None:
                optimizers[name].zero_grad()
        else:
            models[name].eval()

    batch_loss = 0
    batch_reg_loss = 0


    if truncate_epoch:
        max_batches =  args.max_batches_per_train_epoch if train_model else args.max_batches_per_dev_epoch
        num_batches_per_epoch = min(len(data_loader), (max_batches * args.batch_splits))
    i = 0
    for batch in tqdm(data_iter, total=num_batches_per_epoch):
        if batch is None:
            warnings.warn('Empty batch')
            continue
        x, y = autograd.Variable(batch['x'], volatile=volatile), autograd.Variable(batch['y'], volatile=volatile)
        if args.cuda:
            x, y = x.cuda(async=True), y.cuda(async=True)
        if args.use_risk_factors:
            risk_factors = autograd.Variable(batch['risk_factors'], volatile=volatile)
            if args.cuda:
                risk_factors = risk_factors.cuda(async=True)
        else:
            risk_factors = None

        step_results = model_step(x, y, risk_factors,
                    batch, models, train_model, args)
        loss, reg_loss, batch_preds, batch_probs, batch_golds, batch_exams, _ = step_results

        batch_loss += loss.cpu().data[0]
        batch_reg_loss += reg_loss.cpu().data[0]
        if train_model:
            if (i + 1) % args.batch_splits == 0:
                optimizers['model'].step()
                optimizers['model'].zero_grad()

        if (i + 1) % args.batch_splits == 0:
            losses.append(batch_loss)
            reg_losses.append(batch_reg_loss)
            batch_loss = 0
            batch_reg_loss = 0

        preds.extend(batch_preds)
        probs.extend(batch_probs)
        golds.extend(batch_golds)
        if batch_exams is not None:
            exams.extend(batch_exams)

        i += 1
        if i > num_batches_per_epoch:
            data_iter.__del__()
            break
    # Recluster results by exam
    if args.cluster_exams:
        aggr = 'majority'
        if 'risk' in args.dataset or 'detection' in args.dataset:
            aggr = 'max'

        golds, preds, probs, exams = cluster_results_by_exam(golds, preds, probs, exams, aggr=aggr)

    # Calculate epoch level scores
    if args.num_classes == 2:
        precision = sklearn.metrics.precision_score(y_true=golds, y_pred=preds)
        recall = sklearn.metrics.recall_score(y_true=golds, y_pred=preds)
        f1 = sklearn.metrics.f1_score(y_true=golds, y_pred=preds)
        try:
            auc = sklearn.metrics.roc_auc_score(golds, probs, average='samples')

        except Exception as e:
            warnings.warn("Failed to calculate AUC because {}".format(e))
            auc = 'NA'
    else:
        auc = 'NA'
        precision = 'NA'
        recall = 'NA'
        f1 = 'NA'

    avg_loss = np.mean(losses)
    avg_reg_loss = np.mean(reg_losses)
    accuracy = sklearn.metrics.accuracy_score(y_true=golds, y_pred=preds)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=golds, y_pred=preds)

    return avg_loss, accuracy, confusion_matrix, golds, preds, probs, auc, exams, avg_reg_loss, precision, recall, f1
