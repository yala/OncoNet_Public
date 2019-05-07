import numpy as np
import math
import sklearn.metrics
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

def get_model_loss(logit, y, args):
    if args.objective == 'cross_entropy':
        loss = F.cross_entropy(logit, y) / args.batch_splits
    else:
        raise Exception(
            "Objective {} not supported!".format(args.objective))
    return loss

def model_step(x, y, risk_factors, batch, models, train_model,  args):
    '''
        Single step of running model on the a batch x,y and computing
        the loss. Backward pass is computed if train_model=True.
        if use_meta_reg is set to true, this preserves the derivative graph
        and also returns the meta_loss.

        Returns various stats of this single forward and backward pass.


        args:
        - x: input features
        - y: labels
        - risk_factors: additional input features corresponding to risk factors
        - target_x: target to regularize towards. Used for wgan/meta learning
        - batch: whole batch dict, can be used by various special args
        - models: dict of models. The main model, named "model" must return logit, hidden, activ
        - train_model: whether or not to compute backward on loss
        - args: various runtime args such as batch_split etc

        returns:
        - loss: scalar for loss on batch as a tensor
        - reg_loss: scalar for regularization loss on batch as a tensor
        - preds: predicted labels as numpy array
        - probs: softmax probablities as numpy array
        - golds: labels, numpy array version of arg y
        - exams: exam ids for batch if available
        - hiddens: feature rep for batch
    '''
    if args.use_risk_factors:
        logit, hidden, activ = models['model'](x, risk_factors=risk_factors)
    else:
        logit, hidden, activ = models['model'](x)
    if args.downsample_activ:
        activ = F.max_pool2d(activ, 12, stride=12)

    loss = get_model_loss(logit, y, args)
    reg_loss = autograd.Variable(torch.zeros(1))


    if train_model:
        loss.backward()

    batch_softmax = F.softmax(logit, dim=-1).cpu()

    preds = torch.max(batch_softmax, 1)[1].view(y.size()).data.numpy()
    probs = batch_softmax[:,1].data.numpy().tolist()
    golds = y.data.cpu().numpy()
    exams = batch['exam'] if 'exam' in batch else None


    if args.get_activs_instead_of_hiddens:
        hiddens = activ.data.cpu().numpy()
    else:
        hiddens = hidden.data.cpu().numpy()

    return  loss, reg_loss, preds, probs, golds, exams, hiddens
