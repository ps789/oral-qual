import torch
import torch.nn as nn
import numpy as np
import deepxde as dde
import deepxde.nn.activations as activations
import deepxde.nn.initializers as initializers

from src.model import LDNN

parallel = 40

class ActivationModule(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        
    def forward(self, x):
        return self.activation(x)
    

class ResidualBlock1d(nn.Module):
    """
    input_shape = (batch_size, input_dim)
    output_shape = (batch_size, output_dim)
    """

    def __init__(self,
                 input_dim: int, 
                 output_dim: int, 
                 activation: str, 
                 kernel_initializer: str):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation1 = activations.get(activation)
        self.activation2 = activations.get(activation)
        self.init_func = initializers.get(kernel_initializer)
        # self.init_func_str = init_func

        # self.activation1 = get_activation(self.activation_str)
        # self.activation2 = get_activation(self.activation_str)
        # self.init_func = get_initializer(self.init_func_str)

        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        # self.bn1 = nn.BatchNorm1d(self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        # self.bn2 = nn.BatchNorm1d(self.output_dim)
        self.shortcut = nn.Sequential()
        if self.input_dim != self.output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
                # nn.BatchNorm1d(self.output_dim)
            )
        self.init_params()
        

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # y = self.activation1(self.bn1(self.linear1(x)))
        # y = self.bn2(self.linear2(y))
        # y = self.activation2(y + self.shortcut(x))

        y = self.activation1(self.linear1(x))
        y = self.linear2(y)
        y = self.activation2(y + self.shortcut(x))

        return y


class ResNN(nn.Module):
    """
    input_shape = (batch_size, input_dim)
    output_shape = (batch_size, output_dim)
    """

    def __init__(self, 
                 layer_sizes: list,
                 activation: str,
                 kernel_initialier: str):

        super().__init__()
        # self.input_dim = input_dim
        # self.output_dim = output_dim  
        # self.hidden_dim = hidden_dim
        # self.hidden_depth = hidden_depth
        # self.activation_str = activation
        # self.init_func_str = init_func

        self.hidden_depth = len(layer_sizes) - 2
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.hidden_dim = layer_sizes[1] # suppose the hidden layers have the same dimension
        
        # self.activation = get_activation(activation)
        # self.init_func = get_initializer(init_func)
        self.activation = activations.get(activation)
        self.init_func = initializers.get(kernel_initialier)

        self.num_residual_blocks = (self.hidden_depth - 1)//2
        self.num_remaining_hidden_layers = (self.hidden_depth - 1)%2

        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_residual_blocks):
            self.residual_blocks.append(ResidualBlock1d(self.hidden_dim, 
                                                        self.hidden_dim, 
                                                        activation, 
                                                        kernel_initialier))

        if self.num_remaining_hidden_layers == 1:
            layers = [nn.Linear(self.hidden_dim, self.hidden_dim), 
                    #   nn.BatchNorm1d(self.hidden_dim), 
                      ActivationModule(activations.get(activation))]
            self.remaining_hidden_layers = nn.ModuleList(layers)
            # self.remaining_hidden_layers = layers
        else:
            self.remaining_hidden_layers = nn.ModuleList()

        self.input_hidden_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.init_params()

    def init_params(self):

        for block in self.residual_blocks:
            block.init_params()
        for layer in self.remaining_hidden_layers:
            if isinstance(layer, nn.Linear):
                self.init_func(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.init_func(self.input_hidden_linear.weight)
        if self.input_hidden_linear.bias is not None:
            nn.init.zeros_(self.input_hidden_linear.bias)
        self.init_func(self.hidden_output_linear.weight)
        if self.hidden_output_linear.bias is not None:
            nn.init.zeros_(self.hidden_output_linear.bias)

    def forward(self, x):

        x = self.activation(self.input_hidden_linear(x))
        for block in self.residual_blocks:
            x = block(x)
        for layer in self.remaining_hidden_layers:
            x = layer(x)
        x = self.hidden_output_linear(x)

        return x
    
class ResLDNN(LDNN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
    ):
        super().__init__(        
            layer_sizes_dyn,
            layer_sizes_rec,
            activation,
            kernel_initializer,)
        if isinstance(activation, dict):
            activation_dyn = dde.nn.activations.geat(activation["dyn"])
            self.activation_rec = dde.nn.activations.get(activation["rec"])
        else:
            activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        if callable(layer_sizes_dyn[1]):
            # User-defined network
            self.dyn = layer_sizes_dyn[1]
        else:
            # Fully connected network
            self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
            # self.dyn = ResNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        # LayerNorm
        # self.layer_norm = nn.LayerNorm(layer_sizes_dyn[-1])
        self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []
        
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        if latent_init is False:
            self.state = torch.zeros(u.shape[0],self.num_latent_states).to(device)
        else:
            self.state = data["latent"]
                       
        self.state_history = []
        for i in range(x.shape[1]):
        # dyn net to encode the input function
            if len(u.shape) == 2:
                u_ti = u
            else:
                u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
            input = torch.cat((u_ti,self.state),dim=1)
            
            dyn_output = self.dyn(input)
            # dyn_output = self.layer_norm(dyn_output)
            
            # if equilibrium:
            #     self.state = self.state + dt * (dyn_output - self.dyn(torch.zeros_like(input)))
            # else:
            self.state = self.state + dt * dyn_output
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        
        if latent_state:
            return self.state_history
        
        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
        # concatenate latent state and space
        if self.training:
            x_rec = torch.cat([x,self.state_history_],dim=-1)
            x_rec = self.rec(x_rec)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = torch.cat([x,self.state_history_],dim=-1).detach()
                for i in np.arange(0,self.ensemble_size,parallel):
                    x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)
                # print(x_rec.device)
        
        # if latent_state:
        #     return self.state_history
        # else:
        return x_rec
    