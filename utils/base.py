import re
import torch
import torch.nn as nn


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        name = '{} + {}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


# # class Loss(BaseObject):
# #     def __add__(self, *other):
# #         assert len(other) > 1
# #         if isinstance(other[0], Loss):
# #             res = SumOfLosses(self, other[0])
# #         else:
# #             raise ValueError('Loss should be inherited from `Loss` class')
# #         for x in other[1:]:
# #             if isinstance(x, Loss):
# #                 res += SumOfLosses(self, x)
# #             else:
# #                 raise ValueError('Loss should be inherited from `Loss` class')
# #         return res

#     def __radd__(self, other):
#         return self.__add__(other)

#     def __mul__(self, value):
#         if isinstance(value, (int, float)):
#             return MultipliedLoss(self, value)
#         else:
#             raise ValueError('Loss should be inherited from `BaseLoss` class')

#     def __rmul__(self, other):
#         return self.__mul__(other)

# class SumOfLosses(Loss):
#     def __init__(self, *losses):
#         name = ""
#         for loss in losses:
#             name += '{} + '.format(loss.__name__)
#         name = name[:-3]
#         super().__init__(name=name)
#         self.losses = losses

#     def __call__(self, *inputs):
#         assert len(self.losses) > 1
#         res = self.losses[0].forward(*inputs)
#         for loss in self.losses[1:]:
#             res += loss.forward(*inputs)
#         return res


class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{} * ({})'.format(multiplier, loss.__name__)
        else:
            name = '{} * {}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)
