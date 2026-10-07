import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)

batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]

_OriginalModel = Model
_original_get_inputs = get_inputs
STATE_NAMES = ('transition.0.weight', 'transition.0.bias', 'transition.0.running_mean', 'transition.0.running_var', 'transition.2.weight')
class Model(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.inner = _OriginalModel(*args).eval()
    def forward(self, x, bn_weight, bn_bias, running_mean, running_var, conv_weight):
        state = dict(zip(STATE_NAMES, (bn_weight, bn_bias, running_mean, running_var, conv_weight)))
        return torch.func.functional_call(self.inner, state, (x,))
def get_inputs():
    values = _original_get_inputs()
    devices = [torch.cuda.current_device()] if values[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        with torch.device('cpu'):
            original = _OriginalModel(*get_init_inputs()).eval()
        state = original.state_dict()
        values.extend(state[name].detach().to(values[0].device) for name in STATE_NAMES)
    return values
