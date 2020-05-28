from torch import nn
from metann import Learner


class Reshape(nn.Module):
    def __init__(self, param):
        super(Reshape, self).__init__()
        self.param = param

    def forward(self, x):
        return x.view(x.size(0), *self.param)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_cnn(config):
    layers = []
    for item in config:
        if len(item) == 1:
            name, args, kwargs = item[0], [], {}
        elif len(item) == 2:
            name, args, kwargs = item[0], item[1], {}
        elif len(item) == 3:
            name, args, kwargs = item
        else:
            raise ValueError("items in network config should be like (name, args, kwargs)")

        if name is 'conv2d':
            layers.append(nn.Conv2d(*args, **kwargs))
        elif name is 'convt2d':
            layers.append(nn.ConvTranspose2d(*args, **kwargs))
        elif name is 'linear':
            layers.append(nn.Linear(*args, **kwargs))
        elif name is 'bn2d':
            layers.append(nn.BatchNorm2d(*args, **kwargs))
        elif name is 'flatten':
            layers.append(Flatten())
        elif name is 'reshape':
            layers.append(Reshape(*args, **kwargs))
        elif name is 'relu':
            layers.append(nn.ReLU())
        elif name is 'leakyrelu':
            layers.append(nn.LeakyReLU(*args, **kwargs))
        elif name is 'tanh':
            layers.append(nn.Tanh())
        elif name is 'sigmoid':
            layers.append(nn.Sigmoid())
        elif name is 'upsample':
            layers.append(nn.UpsamplingNearest2d(*args, **kwargs))
        elif name is 'max_pool2d':
            layers.append(nn.MaxPool2d(*args, **kwargs))
        elif name is 'avg_pool2d':
            layers.append(nn.AvgPool2d(*args, **kwargs))
        elif name is 'softmax':
            layers.append(nn.Softmax(*args, **kwargs))

        else:
            raise NotImplementedError
    return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.module = Learner(get_cnn(config))

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            return self.module(x)
        else:
            return self.module.functional(vars, bn_training, x)
