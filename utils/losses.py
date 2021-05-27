import torch.nn as nn
import torch

from . import base
from . import functional as F
from .metrics import Activation


class JaccardLoss(base.Loss):
    def __init__(self,
                 eps=1.,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self,
                 eps=1.,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.dice(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class NoiseRobustDiceLoss(base.Loss):
    def __init__(self,
                 eps=1.,
                 activation=None,
                 gamma=1.5,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.gamma = gamma
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.noise_robust_dice(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            gamma=self.gamma,
            ignore_channels=self.ignore_channels,
        )


class TverskyLoss(base.Loss):
    def __init__(self,
                 eps=1.,
                 activation=None,
                 alpha=0.5,
                 beta=0.5,
                 ignore_channles=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.alpha = alpha
        self.beta = beta
        self.ignore_channels = ignore_channles

    def forward(self, y_pr, y_gt):
        return 1 - F.tversky(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            alpha=self.alpha,
            beta=self.beta,
            ignore_channels=self.ignore_channels,
        )


class FLoss(base.Loss):
    def __init__(self,
                 eps=1.,
                 beta=1.,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
