import torch
import torch.nn as nn
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool
from onconet.models.pools.factory import get_pool
from onconet.utils.risk_factors import RiskFactorVectorizer
import torch.autograd as autograd

import pdb

MLP_HIDDEN_DIM = 100

@RegisterPool('RiskFactorPool')
class RiskFactorPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(RiskFactorPool, self).__init__(args, num_chan)
        self.args = args
        self.internal_pool = get_pool(args.pool_name)(args, num_chan)
        assert not self.internal_pool.replaces_fc()
        self.dropout = nn.Dropout(args.dropout)
        self.length_risk_factor_vector = RiskFactorVectorizer(args).vector_length
        self.fc = nn.Linear(self.length_risk_factor_vector + num_chan, args.num_classes)

        self.args.hidden_dim = self.length_risk_factor_vector + num_chan


    def replaces_fc(self):
        return True

    def forward(self, x, risk_factors):
        if self.args.replace_snapshot_pool:
            x = autograd.Variable(x.data)
            if self.args.cuda:
                x = x.cuda()

        risk_factors_hidden = risk_factors
        _, hidden = self.internal_pool(x)
        hidden = torch.cat((hidden, risk_factors_hidden), 1)
        hidden = self.dropout(hidden)
        logit = self.fc(hidden)
        return logit, hidden
