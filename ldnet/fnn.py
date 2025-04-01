import torch
import torch.nn as nn
import torch.nn.init as init

def get_activation(name): 
    if name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']: 
        return nn.ReLU()
    elif name in ['leakyrelu', 'LeakyReLU', 'leaky_relu']:
        return nn.LeakyReLU()
    elif name in ['prelu', 'PReLU']:
        return nn.PReLU()
    elif name in ['rrelu', 'RReLU']:
        return nn.RReLU()
    elif name in ['elu', 'ELU']:
        return nn.ELU()
    elif name in ['selu', 'SELU']:
        return nn.SELU()
    elif name in ['celu', 'CELU']:
        return nn.CELU()
    elif name in ['gelu', 'GELU']:
        return nn.GELU()
    elif name in ['silu', 'SiLU']:
        return nn.SiLU()
    elif name in ['mish', 'Mish']:
        return nn.Mish()
    elif name in ['glu', 'GLU']:
        return nn.GLU()
    # elif name in ['sine', 'Sine']:
    #     return Sin()
    else:
        raise NotImplementedError

def get_initializer(name):
    if name == 'xavier_uniform' or name == "Glorot uniform":
        return init.xavier_uniform_
    elif name == 'xvaier_normal' or name == "Glorot normal":
        return init.xavier_normal_
    elif name == 'kaiming_uniform':
        return init.kaiming_uniform_
    elif name == 'kaiming_normal':
        return init.kaiming_normal_
    elif name == 'uniform':
        return init.uniform_
    elif name == 'normal':
        return init.normal_
    elif name == 'trunc_normal':
        return init.trunc_normal_
    else:
        raise NotImplementedError
    
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation=None, kernel_initializer=None, dropout=0.0):
        """
        Args:
            layer_sizes (list[int]): A list specifying the size of each layer, including input and output.
                                     For example, [10, 32, 64, 1] means:
                                     input_dim=10, two hidden layers with sizes 32 and 64, output_dim=1.
            activation (nn.Module or name): Activation function to use (e.g., nn.ReLU).
            kernel_initializer: name
            dropout_prob (float): Dropout probability, 0.0 means no dropout.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation() if callable(activation) else get_activation(activation)
        self.init = get_initializer(kernel_initializer)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        if kernel_initializer is not None:
            self.apply_kernel_initializer()

    def apply_kernel_initializer(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:  # Apply all layers except the last one
            x = layer(x)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x)  # Apply the last layer without activation
        return x
