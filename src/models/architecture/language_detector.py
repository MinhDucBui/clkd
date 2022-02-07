import torch.nn as nn
import torch



class GradientReverse(torch.autograd.Function):
    scale = torch.tensor(1.0, requires_grad=False)
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = torch.tensor(scale, requires_grad=False)
    return GradientReverse.apply(x)


class LanguageDetector(nn.Module):
    """Taken from https://github.com/ccsasuke/adan/blob/master/code/models.py
    """
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm=False,
                 bias=True):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())
        
        self.net.add_module('q-linear-final', nn.Linear(hidden_size, 1, bias=bias))

    def forward(self, input):
        return self.net(input)

class LanguageDetectorGRL(LanguageDetector):
    """Taken from https://github.com/ccsasuke/adan/blob/master/code/models.py
    """
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 lambd,
                 batch_norm=False,
                 bias=True):
        super().__init__(num_layers, hidden_size, dropout, batch_norm, bias)
        self.lambd = lambd
        GradientReverse()
        
    def forward(self, input):
        input = grad_reverse(input, self.lambd)
        return self.net(input)
        
        
