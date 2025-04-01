import torch
import torch.nn.init as init
import numpy as np
import deepxde as dde

from ncps.wirings import AutoNCP 
from ncps.torch import CfC

from ldnet.resnet import ResNN

parallel = 40
    
class LiquidLDNN(dde.nn.NN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
        resnet=False
    ):
        super().__init__()
        self.num_latent_states = layer_sizes_dyn[-1]
        self.N_inputs = layer_sizes_dyn[0]
        self.N_neu = layer_sizes_dyn[1]
        self.state = None
        self.state_history = []
        if isinstance(activation, dict):
            activation_dyn = dde.nn.activations.geat(activation["dyn"])
            self.activation_rec = dde.nn.activations.get(activation["rec"])
        else:
            activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        if callable(layer_sizes_dyn[1]):
            # User-defined network
            self.dyn = layer_sizes_dyn[1]
        else:
            # Liquid dynamic network.
            self.wiring = AutoNCP(self.N_neu, self.num_latent_states)
            self.dyn = CfC(self.N_inputs, self.wiring, batch_first = True, return_sequences = True, mixed_memory = True)
        
        # self.fourier_mapping_size = fourier_mapping_size
        self.n_coords = layer_sizes_rec[0] - layer_sizes_dyn[-1]
        
        # self.B = FourierEmbedding(self.n_coords, fourier_mapping_size)
        # self.B = torch.randn(self.n_coords, fourier_mapping_size) # scale = 1
        
        # layer_sizes_rec_f = [layer_sizes_dyn[-1] + 2 * self.fourier_mapping_size] + layer_sizes_rec[1:]
        if resnet:
            self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        else:
            self.rec = dde.nn.FNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        # if latent_init is False:
        #     self.state = torch.zeros(u.shape[0],self.num_latent_states).to(device)
        # else:
        #     self.state = data["latent"]
                       
        # self.state_history = []
        # for i in range(x.shape[1]):
        # # dyn net to encode the input function
        #     if len(u.shape) == 2:
        #         u_ti = u
        #     else:
        #         u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
        #     input = torch.cat((u_ti,self.state),dim=1)
        #     if equilibrium:
        #         self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
        #     else:
        #         self.state = self.state + dt * self.dyn(input)
        #     self.state_history.append(self.state)

        # self.state_history = torch.stack(self.state_history).transpose(0, 1)
        
        if len(u.shape) == 2:
            u = u.unsqueeze(1).expand(u.shape[0],x.shape[1],u.shape[1])
        self.state_history = self.dyn(u)[0]

        if latent_state:
            return self.state_history
        
        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
        # concatenate latent state and space
        if self.training:
            # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
            # print("points_projected shape",points_projected.shape)
            x_rec_input = torch.cat([x,self.state_history_],dim=-1)
            
            x_rec = self.rec(x_rec_input)
        else:
            x_rec = []
            with torch.no_grad():
                # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
                # print("points_projected shape",points_projected.shape)
                x_rec_input = torch.cat([x,self.state_history_],dim=-1).detach()
                for i in np.arange(0,self.ensemble_size,parallel):
                    x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)
        
        # if latent_state:
        #     return self.state_history
        # else:
        return x_rec
    
