import math
import numpy as np
import torch
import torch.cuda.comm as comm
import torch.nn as nn
import torch.nn.functional as F

import os 
import warnings

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

 
 
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.autograd import Variable, Function
from torch.autograd.function import once_differentiable
from torch.nn import Module, Parameter
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load


# from ..encoding import lib
#also be sure that its even supposed to generate a .so file
from encoding import lib

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        
        self.cont_enx = EncHead(num_classes, num_classes, se_loss=True,
                            lateral=False, norm_layer=SyncBatchNorm,
                            up_kwargs=None)
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)  
         
        x = list(self.cont_enx(x)) 
        x = x[0] # currently just ignoring the se loss and using the context encoding module
        
        # print("\n\n\n", x, "\n\n\n")
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# Manually added everything below this pointto create munanet- also notice the 
# addition of the module to the Decoder's init and forward functions above

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Functions for Encoding Layer"""

 

class _aggregate(Function):
    @staticmethod
    def forward(ctx, A, X, C):
        # A \in(BxNxK) R \in(BxNxKxD) => E \in(BxNxD)
        ctx.save_for_backward(A, X, C)
        if A.is_cuda:
            E = lib.gpu.aggregate_forward(A, X, C)
        else:
            E = lib.cpu.aggregate_forward(A, X, C)
        return E

    @staticmethod
    def backward(ctx, gradE):
        A, X, C = ctx.saved_variables
        if A.is_cuda:
            gradA, gradX, gradC = lib.gpu.aggregate_backward(gradE, A, X, C)
        else:
            gradA, gradX, gradC = lib.cpu.aggregate_backward(gradE, A, X, C)
        return gradA, gradX, gradC

def aggregate(A, X, C):
    r""" Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
    to the codewords (:math:`C`) with assignment weights (:math:`A`).

    .. math::

        e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}`
          :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Examples:
        >>> B,N,K,D = 2,3,4,5
        >>> A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
        >>> X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> func = encoding.aggregate()
        >>> E = func(A, X, C)
    """
    return _aggregate.apply(A, X, C)

class _scaled_l2(Function):
    @staticmethod
    def forward(ctx, X, C, S):
        if X.is_cuda:
            SL = lib.gpu.scaled_l2_forward(X, C, S)
        else:
            SL = lib.cpu.scaled_l2_forward(X, C, S)
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, gradSL):
        X, C, S, SL = ctx.saved_variables
        if X.is_cuda:
            gradX, gradC, gradS = lib.gpu.scaled_l2_backward(gradSL, X, C, S, SL)
        else:
            gradX, gradC, gradS = lib.cpu.scaled_l2_backward(gradSL, X, C, S, SL)
        return gradX, gradC, gradS

def scaled_l2(X, C, S):
    r""" scaled_l2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    return _scaled_l2.apply(X, C, S)

# Experimental
def pairwise_cosine(X, C, normalize=False):
    r"""Pairwise Cosine Similarity or Dot-product Similarity
    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    if normalize:
        X = F.normalize(X, dim=2, eps=1e-8)
        C = F.normalize(C, dim=1, eps=1e-8)
    return torch.matmul(X, C.t())







class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, lateral=True,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)

class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)

class Encoding(Module):
    r"""
    Encoding Layer: a learnable residual encoder.

    .. image:: _static/img/cvpr17.svg
        :width: 50%
        :align: center

    Encoding Layer accpets 3D or 4D inputs.
    It considers an input featuremaps with the shape of :math:`C\times H\times W`
    as a set of C-dimentional input features :math:`X=\{x_1, ...x_N\}`, where N is total number
    of features given by :math:`H\times W`, which learns an inherent codebook
    :math:`D=\{d_1,...d_K\}` and a set of smoothing factor of visual centers
    :math:`S=\{s_1,...s_K\}`. Encoding Layer outputs the residuals with soft-assignment weights
    :math:`e_k=\sum_{i=1}^Ne_{ik}`, where

    .. math::

        e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}

    and the residuals are given by :math:`r_{ik} = x_i - d_k`. The output encoders are
    :math:`E=\{e_1,...e_K\}`.

    Args:
        D: dimention of the features or feature channels
        K: number of codeswords

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}` or
          :math:`\mathcal{R}^{B\times D\times H\times W}` (where :math:`B` is batch,
          :math:`N` is total number of features or :math:`H\times W`.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Attributes:
        codewords (Tensor): the learnable codewords of shape (:math:`K\times D`)
        scale (Tensor): the learnable scale factor of visual centers

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

        Hang Zhang, Jia Xue, and Kristin Dana. "Deep TEN: Texture Encoding Network."
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2017*

    Examples:
        >>> import encoding
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.autograd import Variable
        >>> B,C,H,W,K = 2,3,4,5,6
        >>> X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        >>> layer = encoding.Encoding(C,K).double().cuda()
        >>> E = layer(X)
    """
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'

class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)

class syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        assert activation == 'none'

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.gpu.expectation_forward(x)
            else:
                raise NotImplemented

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 

        # BN forward + activation
        if x.is_cuda:
            y = lib.gpu.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            y = lib.cpu.batchnorm_forward(x, _ex, _exs, gamma, beta, ctx.eps)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = \
                lib.gpu.batchnorm_backward(dz, x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            if x.is_cuda:
                dx_ = lib.gpu.expectation_backward(x, _dex, _dexs)
            else:
                raise NotImplemented
            dx = dx + dx_

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

def _act_forward(ctx, x):
    if ctx.activation.lower() == "leaky_relu":
        if x.is_cuda:
            lib.gpu.leaky_relu_forward(x, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert activation == 'none'

def _act_backward(ctx, x, dx):
    if ctx.activation.lower() == "leaky_relu":
        if x.is_cuda:
            lib.gpu.leaky_relu_backward(x, dx, ctx.slope)
        else:
            raise NotImplemented
    else:
        assert activation == 'none'

class inp_syncbatchnorm_(Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, running_mean, running_var,
                extra, sync=True, training=True, momentum=0.1, eps=1e-05,
                activation="none", slope=0.01):
        # save context
        cls._parse_extra(ctx, extra)
        ctx.sync = sync
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope

        # continous inputs
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()

        if ctx.training:
            if x.is_cuda:
                _ex, _exs = lib.gpu.expectation_forward(x)
            else:
                raise NotImplemented

            if ctx.sync:
                if ctx.is_master:
                    _ex, _exs = [_ex.unsqueeze(0)], [_exs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _ex_w, _exs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _ex.append(_ex_w.unsqueeze(0))
                        _exs.append(_exs_w.unsqueeze(0))

                    _ex = comm.gather(_ex).mean(0)
                    _exs = comm.gather(_exs).mean(0)

                    tensors = comm.broadcast_coalesced((_ex, _exs), [_ex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_ex, _exs))
                    _ex, _exs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            # Update running stats
            _var = _exs - _ex ** 2
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * _ex)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * _var)

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _var + _ex ** 2 
            ctx.mark_dirty(x)

        # BN forward + activation
        if x.is_cuda:
            lib.gpu.batchnorm_inp_forward(x, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        _act_forward(ctx, x)

        # Output
        ctx.save_for_backward(x, _ex, _exs, gamma, beta)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, _ex, _exs, gamma, beta = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        # BN backward
        if dz.is_cuda:
            dx, _dex, _dexs, dgamma, dbeta = \
                lib.gpu.batchnorm_inp_backward(dz, z, _ex, _exs, gamma, beta, ctx.eps)
        else:
            raise NotImplemented

        if ctx.training:
            if ctx.sync:
                if ctx.is_master:
                    _dex, _dexs = [_dex.unsqueeze(0)], [_dexs.unsqueeze(0)]
                    for _ in range(ctx.master_queue.maxsize):
                        _dex_w, _dexs_w = ctx.master_queue.get()
                        ctx.master_queue.task_done()
                        _dex.append(_dex_w.unsqueeze(0))
                        _dexs.append(_dexs_w.unsqueeze(0))

                    _dex = comm.gather(_dex).mean(0)
                    _dexs = comm.gather(_dexs).mean(0)

                    tensors = comm.broadcast_coalesced((_dex, _dexs), [_dex.get_device()] + ctx.worker_ids)
                    for ts, queue in zip(tensors[1:], ctx.worker_queues):
                        queue.put(ts)
                else:
                    ctx.master_queue.put((_dex, _dexs))
                    _dex, _dexs = ctx.worker_queue.get()
                    ctx.worker_queue.task_done()

            if z.is_cuda:
                lib.gpu.expectation_inp_backward(dx, z, _dex, _dexs, _ex, _exs, gamma, beta, ctx.eps)
            else:
                raise NotImplemented

        return dx, dgamma, dbeta, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]

syncbatchnorm = syncbatchnorm_.apply
inp_syncbatchnorm = inp_syncbatchnorm_.apply



class SyncBatchNorm(_BatchNorm):
    r"""Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device (GPU).
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .
    Please see the design idea in the `notes <./notes/syncbn.html>`_.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    The mean and standard-deviation are calculated per-channel over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        sync: a boolean value that when set to ``True``, synchronize across
            different gpus. Default: ``True``
        activation : str
            Name of the activation functions, one of: `leaky_relu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*

    Examples:
        >>> m = SyncBatchNorm(100)
        >>> net = torch.nn.DataParallel(m)
        >>> output = net(input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, sync=True, activation="none", slope=0.01,
                 inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        #self.inplace = inplace
        self.slope = slope
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]
        # running_exs
        #self.register_buffer('running_exs', torch.ones(num_features))

    def forward(self, x):
        # Resize the input to (B, C, -1).
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }
        if self.inplace:
            return inp_syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                     extra, self.sync, self.training, self.momentum, self.eps,
                                     self.activation, self.slope).view(input_shape)
        else:
            return syncbatchnorm(x, self.weight, self.bias, self.running_mean, self.running_var,
                                 extra, self.sync, self.training, self.momentum, self.eps,
                                 self.activation, self.slope).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(
                self.sync, self.activation, self.slope, self.inplace
            )

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)

