# Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
# Implementation based on PyTorch ResNet implementation: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import math
import torch
import torch.nn as nn
import pdb
import numpy as np
from onconet.models.pools.factory import get_pool


class ResNet(nn.Module):
    """
        A ResNet model. Blocks can be Basic, Non-local, bottleneck or
        anything in onconet.models.blocks and intermixed in any order.
        This is a slight generalization of orginal resnet model,
        which assumed a homogenous block type.
    """

    def __init__(self, layers, args):
        """Initializes a generalized resnet. Supports arbitrary block configurations per layer.

        Arguments:
            layers(list): A length-4 list with the list of block
            classes in each of the 4 layers. Blocks can be
            basic blocks, bottlenecks, non-locals etc.

            num_classes(int): The number of classes the network
                is predicting between, i.e. the size of the final
                layer of the network.
            args(Args): configuration of experiment. Used to determine num gpus,
            cuda mode, etc.
        """

        super(ResNet, self).__init__()

        self.args = args
        self.args.wrap_model = False
        self.args.hidden_dim = 512 * args.block_widening_factor
        self.inplanes = max(64 * args.block_widening_factor, self.args.input_dim)

        self.all_blocks = []
        downsampler = Downsampler(self.inplanes, self.args.input_dim)
        self.add_module('downsampler', downsampler)
        self.all_blocks.append('downsampler')

        layer_modules = [(self._make_layer(self.inplanes, layers[0]), 'layer1_{}')]
        current_dim = self.inplanes
        indx = 1
        for layer_i in layers[1:]:
            indx += 1
            current_dim = min(current_dim * 2, 1024)
            layer_modules.append(
                            (self._make_layer(current_dim, layer_i, stride=2),
                             'layer{}_'.format(indx)+'{}')
                            )

        args.hidden_dim = current_dim

        '''
            For all layers, register all constituent blocks to the module,
            and record block names for later access in self.all_blocks
        '''
        for layer, layer_name in layer_modules:
            for indx, block in enumerate(layer):
                block_name = layer_name.format(indx)
                self.add_module(block_name, block)
                self.all_blocks.append(block_name)

        last_block = layers[-1][-1]

        pool_name = args.pool_name
        if args.use_risk_factors:
            pool_name = 'DeepRiskFactorPool' if self.args.deep_risk_factor_pool else 'RiskFactorPool'
        self.pool = get_pool(pool_name)(args, args.hidden_dim)

        if not self.pool.replaces_fc():
            # Cannot not placed on self.all_blocks since requires intermediate op
            self.dropout = nn.Dropout(p=args.dropout)
            self.fc = nn.Linear(args.hidden_dim, args.num_classes)


        self.gpu_to_layer_assignments = self.get_gpu_to_layer()

    def get_gpu_to_layer(self):
        '''
            Given args.model_parallel, args.num_shards, will try to best
            balance layers across gpus given the number of gpus.

            returns:
            -gpu_to_layers: a list of lists of length num_shards. Each interior
            list consist of layer names to place on that index's gpu.
        '''
        if self.args.model_parallel and self.args.num_shards > 1:
            num_shards = self.args.num_shards
        else:
            num_shards = 1

        gpu_to_layers = np.array_split(self.all_blocks, num_shards)
        self._validate_gpu_assignments(gpu_to_layers)
        return gpu_to_layers

    def _validate_gpu_assignments(self, gpu_to_layers):
        """Confirms that all layers from self.all_blocks are in gpu_to_layers.

        Arguments:
            gpu_to_layers: A list of lists containing the layers assigned to each GPU.

        Raises:
            Exception if the the layers in self.all_blocks and the layers in gpu_to_layers
            don't match.
        """

        original_layers = set(self.all_blocks)
        layers_assigned = set([layer for layers in gpu_to_layers for layer in layers])

        if original_layers != layers_assigned:
            extra_layers = layers_assigned - original_layers
            missing_layers = original_layers - layers_assigned

            raise Exception(
                'GPU partitioned layers don\'t match original layers.\n\t{}\n\t{}'.format(
                    'Extra layers: {}'.format(extra_layers) if len(extra_layers) > 0 else '',
                    'Missing layers: {}'.format(missing_layers) if len(missing_layers) > 0 else ''
                )
            )

    def _make_layer(self, planes, blocks, stride=1):
        """Builds a layer of the ResNet.

        Arguments:
            planes(int): The number of filters to use in convolutions
                and therefore the depth (number of channels) of the output.
            blocks: list of block classes for this layer.
            stride(int): The stride to use in the convolutionals.

        Returns:
            A Sequential model containing all the blocks in this layer.
        """
        layers = []

        for i, block in enumerate(blocks):
            if (i == 0 and stride != 1) or self.inplanes != planes * block.expansion:

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            else:
                downsample = None

            if i != 0:
                stride = 1

            layers.append(block(self.args,
                                self.inplanes,
                                planes,
                                stride=stride,
                                downsample=downsample
                                ))

            self.inplanes = planes * block.expansion

        return layers

    def forward(self, x, risk_factors=None):
        """Computes a forward pass of the model.

        Arguments:
            x(Variable): The input to the model.

        Returns:
            The result of feeding the input through the model.
        """

        # Go through all layers up to fc
        for gpu, layers in enumerate(self.gpu_to_layer_assignments):
            if self.args.cuda and self.args.model_parallel:
                x = x.cuda(gpu)
            for name in layers:
                layer = self._modules[name]
                x = layer(x)


        logit, hidden = self.aggregate_and_classify(x, risk_factors=risk_factors)
        return logit, hidden, x


    def aggregate_and_classify(self, x, risk_factors=None):
        # Pooling layer
        if self.args.use_risk_factors:
            logit, hidden = self.pool(x, risk_factors)
        else:
            logit, hidden = self.pool(x)
        if not self.pool.replaces_fc():
            # self.fc is always on last gpu, so direct call of fc(x) is safe
            logit = self.fc(self.dropout(hidden))

        # move tensor back to gpu 0 for reuse by non-sharded models
        if self.args.cuda:
            hidden = hidden.cuda()
            logit = logit.cuda()
        return logit, hidden



    def cuda(self, device=None):
        '''
            Moves all submodules to gpu according to gpu_to_layer_assignments.
            , and returns model.
            Does not currently support start device different from 0.
            self.fc is always placed on the last GPU to reduce the amount of cross GPU communication.

            Note, must be called directly from parent module must overide it's .cuda() function to directly call this .cuda() fn trigger this
            method. Generic .cuda() call skips this function, and recurses to leaf nodes directly.
        '''
        if not self.args.model_parallel:
             return self._apply(lambda t: t.cuda(device))

        for gpu, layers in enumerate(self.gpu_to_layer_assignments):
            # fetch layers for GPU at device gpu(int)
            for name in layers:
                # move each layer (identified by module name) to corresponding gpu
                self._modules[name] = self._modules[name].cuda(gpu)

        if not self.pool.replaces_fc():
            # place fc layer at last gpu, so no need for extra gpu communication
            self.fc.cuda(len(self.gpu_to_layer_assignments) - 1)

        return self



class Downsampler(nn.Module):
    """Downsampling layers for ResNet. Downsamples input by 4x"""


    def __init__(self, inplanes, num_chan=3):

        self.inplanes = inplanes
        super(Downsampler, self).__init__()
        self.conv1 = nn.Conv2d(num_chan, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
