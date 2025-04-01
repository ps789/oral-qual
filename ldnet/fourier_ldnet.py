import torch
import torch.nn.init as init
import numpy as np
import deepxde as dde

from ldnet.model import LDNN
from ldnet.resnet import ResNN
from ldnet.fnn import FNN

parallel = 2
mseloss = torch.nn.MSELoss(reduction='sum')

class FourierEmbedding(torch.nn.Module):
    """
    Cell representing a generic Fourier embedding (i.e. a 2D matrix representing an encoding).
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.encoding = torch.nn.Linear(in_feats, out_feats, bias = False)
        # self.encoding.weight.requires_grad = False
        # init.normal_(self.encoding.weight, mean=0.0, std=1/(2*torch.pi))
        # self.encoding = torch.randn(in_feats, out_feats) # scale = 1

    def forward(self, inp):
        return self.encoding(inp)
        # return inp @ self.encoding.T
        
class FourierRec(dde.nn.NN):
    def __init__(self, fourier_embedding, rec):
        super().__init__()
        self.fourier_embedding = fourier_embedding
        self.rec = rec

    def forward(self, x, state_history):
        points_projected = self.fourier_embedding(2 * torch.pi * x)
        x_rec_input = torch.cat([torch.sin(points_projected), torch.cos(points_projected), state_history], dim=-1)
        return self.rec(x_rec_input)
    
class FourierLDNN(dde.nn.NN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
        dropout=0
    ):
        super().__init__()
        # if isinstance(activation, dict):
        #     activation_dyn = dde.nn.activations.geat(activation["dyn"])
        #     self.activation_rec = dde.nn.activations.get(activation["rec"])
        # else:
        #     activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        # if callable(layer_sizes_dyn[1]):
        #     # User-defined network
        #     self.dyn = layer_sizes_dyn[1]
        # else:
        #     # Fully connected network
        #     self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        
        self.fourier_mapping_size = fourier_mapping_size
        self.n_coords = layer_sizes_rec[0] - layer_sizes_dyn[-1]
        
        self.B = FourierEmbedding(self.n_coords, fourier_mapping_size)
        # self.B = torch.randn(self.n_coords, fourier_mapping_size) # scale = 1
        
        layer_sizes_rec_f = [layer_sizes_dyn[-1] + 2 * self.fourier_mapping_size] + layer_sizes_rec[1:]
        # self.rec = dde.nn.FNN(layer_sizes_rec_f, self.activation_rec, kernel_initializer)
        # self.rec = FourierRec(self.B, dde.nn.FNN(layer_sizes_rec_f, self.activation_rec, kernel_initializer))
        if isinstance(dropout, list):
            self.dyn = FNN(layer_sizes_dyn, activation, kernel_initializer, dropout=dropout[0])
            self.rec = FourierRec(self.B, FNN(layer_sizes_rec_f, activation, kernel_initializer, dropout=dropout[1]))
        else:
            self.dyn = FNN(layer_sizes_dyn, activation, kernel_initializer, dropout=dropout)
            self.rec = FourierRec(self.B, FNN(layer_sizes_rec_f, activation, kernel_initializer, dropout=dropout))
        
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
            if equilibrium:
                self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
            else:
                self.state = self.state + dt * self.dyn(input)
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
            # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
            # print("points_projected shape",points_projected.shape)
            # x_rec_input = torch.cat([torch.sin(points_projected),torch.cos(points_projected),self.state_history_],dim=-1)
            
            # x_rec = self.rec(x_rec_input)
            x_rec = self.rec(x, self.state_history_)
        else:
            x_rec = []
            with torch.no_grad():
                # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
                # print("points_projected shape",points_projected.shape)
                # x_rec_input = torch.cat([torch.sin(points_projected),torch.cos(points_projected),self.state_history_],dim=-1).detach()
                x_rec_input = self.state_history_.detach()
                for i in np.arange(0,self.ensemble_size,parallel):
                    # x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                    x_rec_i = self.rec(x[i:i+parallel], x_rec_input[i:i+parallel]).detach()
                    
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)
        
        # if latent_state:
        #     return self.state_history
        # else:
        return x_rec
    
class ResFourierLDNN(LDNN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
    ):
        super().__init__(        
                        layer_sizes_dyn,
                        layer_sizes_rec,
                        activation,
                        kernel_initializer,
                        )
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
        
        self.fourier_mapping_size = fourier_mapping_size
        self.n_coords = layer_sizes_rec[0] - layer_sizes_dyn[-1]
        
        self.B = FourierEmbedding(self.n_coords, fourier_mapping_size)
        # self.B = torch.randn(self.n_coords, fourier_mapping_size) # scale = 1
        
        layer_sizes_rec_f = [layer_sizes_dyn[-1] + 2 * self.fourier_mapping_size] + layer_sizes_rec[1:]
        # self.rec = ResNN(layer_sizes_rec_f, self.activation_rec, kernel_initializer)
        self.rec = FourierRec(self.B, ResNN(layer_sizes_rec_f, self.activation_rec, kernel_initializer))
        
        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        # u = u + torch.randn_like(u)*0.04
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
            if equilibrium:
                self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
            else:
                self.state = self.state + dt * self.dyn(input)
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
            # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
            # print("points_projected shape",points_projected.shape)
            # x_rec_input = torch.cat([torch.sin(points_projected),torch.cos(points_projected),self.state_history_],dim=-1)
            # x_rec = self.rec(x_rec_input)
            x_rec = self.rec(x, self.state_history_)
            
        else:
            x_rec = []
            with torch.no_grad():
                # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
                # print("points_projected shape",points_projected.shape)
                # x_rec_input = torch.cat([torch.sin(points_projected),torch.cos(points_projected),self.state_history_],dim=-1).detach()
                x_rec_input = self.state_history_.detach()
                for i in np.arange(0,self.ensemble_size,parallel):
                    # x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                    x_rec_i = self.rec(x[i:i+parallel], x_rec_input[i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)
        
        # if latent_state:
        #     return self.state_history
        # else:
        return x_rec
    
    def forward_data_assimilation_lbfgs(self, data, observations, device, equilibrium=False, latent_state=False, latent_init = False):
        
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        # observations have shape Nt, Nx, dimx
        # u has shape (2)
        u = data["u"]
        u = u + torch.randn_like(u) * 2
        x = data["x"]
        dt = data["dt"]
        x_rec = []
        self.ensemble_size = u.shape[0]
        
        assert len(u.shape) == 2
    
        # observations = observations# + torch.randn_like(observations) * noise_level * observations
        # observation_vector = observations.reshape([observations.shape[0], -1])
        # encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        if latent_init is False:
            self.state = torch.zeros(u.shape[0],self.num_latent_states).to(device)
        else:
            self.state = data["latent"]
                       
        self.state_history = []
        self.state_history, u_history = [], []
        
        assimilate_time = 0
        
        observation_indices = x[:, 0, :, :].reshape(x.shape[0], 150, 150, 2)[:, ::15, ::15, :]
        observation_indices = observation_indices.reshape(x.shape[0], 100, 2)
        for i in range(x.shape[1]):
            print(i)
            input = torch.cat((u,self.state),dim=1)
            
            with torch.no_grad():
                if len(u.shape) == 2:
                    u_ti = u
                else:
                    u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
                
                input = torch.cat((u_ti,self.state),dim=1)
                
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)
            max_iter = 10
            state_var = torch.autograd.Variable(torch.Tensor(self.state).to(device), requires_grad = True)
            opt = torch.optim.LBFGS([state_var], max_iter=max_iter, lr = 0.5)
            
            def loss_closure():
                opt.zero_grad()
                state_var_ = state_var.unsqueeze(1).expand(self.ensemble_size,100,9)
                # x_rec_input = torch.cat([observation_indices,state_var_],dim=-1)
                # x_rec_i = self.rec(x_rec_input)
                x_rec_i = self.rec(observation_indices, state_var_)

                loss = mseloss(x_rec_i, observations[:, i, :, :, :].reshape(self.ensemble_size, 100, 2))
                loss.backward()
                return loss
            opt.step(loss_closure)
            for var in self.rec.parameters():
                var.grad = None
            self.state = state_var.detach()
            self.state_history.append(self.state)
            state_var_ = state_var.unsqueeze(1).expand(self.ensemble_size,22500,9)
            with torch.no_grad():
                # x_rec_input = torch.cat([x[:, 0, :, :],state_var_],dim=-1).detach()
                x_rec_i = self.rec(x[:, 0, :, :],state_var_).detach().cpu()
                x_rec.append(x_rec_i)
        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        x_rec_tensor = torch.stack(x_rec, dim=1)
        return x_rec_tensor

    def forward_data_assimilation_lbfgs_with_dynamics_multistep(self, data, observations, device, equilibrium=False, latent_state=False, latent_init = False):
        
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        # observations have shape Nt, Nx, dimx
        # u has shape (2)
        u = data["u"]
        u = u + torch.randn_like(u)*0.05
        x = data["x"]
        dt = data["dt"]
        x_rec = []
        self.ensemble_size = u.shape[0]
        
        assert len(u.shape) == 2
    
        # observations = observations# + torch.randn_like(observations) * noise_level * observations
        # observation_vector = observations.reshape([observations.shape[0], -1])
        # encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        if latent_init is False:
            self.state = torch.zeros(u.shape[0],self.num_latent_states).to(device)
        else:
            self.state = data["latent"]
        self.state_history, u_history = [], []
        
        assimilate_time = 0
        
        observation_indices = x[:, 0, :, :].reshape(x.shape[0], 150, 150, 2)[:, ::15, ::15, :]
        observation_indices = observation_indices.reshape(x.shape[0], 100, 2)
        lambda_state = 1
        lambda_u = 0.1
        for i in range(x.shape[1]):
            print(i)
            input = torch.cat((u,self.state),dim=1)
            
            with torch.no_grad():
                if len(u.shape) == 2:
                    u_ti = u
                else:
                    u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
                input = torch.cat((u_ti,self.state),dim=1)
                
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)
            #start here
            smoothing_steps = 5
            if i < x.shape[1] - smoothing_steps:
                if len(u.shape) == 2:
                    u_ti = u
                else:
                    u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
                u_ti = torch.autograd.Variable(torch.Tensor(u_ti.clone()).to(device), requires_grad = True)
                state = torch.autograd.Variable(torch.Tensor(self.state.clone()).to(device), requires_grad = True)
                max_iter = 10
                opt = torch.optim.LBFGS([u_ti, state], max_iter=max_iter, lr = 0.5)

                def loss_closure():
                    loss = 0
                    temp_states = []
                    temp_states.append(state)
                    
                    opt.zero_grad()
                    state_var_ = state.unsqueeze(1).expand(self.ensemble_size,100,9)
                    # x_rec_input = torch.cat([observation_indices,state_var_],dim=-1)
                    x_rec_i = self.rec(observation_indices,state_var_)

                    loss += mseloss(x_rec_i, observations[:, i, :, :, :].reshape(self.ensemble_size, 100, 2))
                    for j in range(smoothing_steps):
                        input = torch.cat((u_ti,temp_states[j]),dim=1)
                        dyn_output = self.dyn(input)
                        temp_states.append(temp_states[j] + dt * dyn_output)
                        state_var_ = temp_states[-1].unsqueeze(1).expand(self.ensemble_size,100,9)
                        # x_rec_input = torch.cat([observation_indices,state_var_],dim=-1)
                        x_rec_i = self.rec(observation_indices,state_var_)

                        loss += mseloss(x_rec_i, observations[:, i+j+1, :, :, :].reshape(self.ensemble_size, 100, 2))
                    loss += mseloss(state, self.state)*lambda_state
                    loss += mseloss(u, u_ti)*lambda_u
                    loss.backward()
                    return loss
                opt.step(loss_closure)
                for var in self.rec.parameters():
                    var.grad = None
                for var in self.dyn.parameters():
                    var.grad = None
                u = u_ti
                # print(data["u"] - u)
                self.state = state.detach()
                self.state_history.append(self.state)
            #end here
            state_var_ = self.state.unsqueeze(1).expand(self.ensemble_size,22500,9)
            with torch.no_grad():
                # x_rec_input = torch.cat([x[:, 0, :, :],state_var_],dim=-1).detach()
                x_rec_i = self.rec(x[:, 0, :, :],state_var_).detach().cpu()
                x_rec.append(x_rec_i)
        x_rec_tensor = torch.stack(x_rec, dim=1)
        return x_rec_tensor
        
    def forward_data_assimilation_ensf(self, data, observations, encoder, noise_level, device, equilibrium=False, latent_state=False, optimal_latent=None,
                                       scaling = 500, obs_sigma =  0.1, eps_alpha = 0.05, euler_steps = 100):
        
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        # observations have shape Nt, Nx, dimx
        # u has shape (2)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]
        
        assert len(u.shape) == 2
        
        # print(data["u"])
        observation_vector = observations.reshape([observations.shape[0], -1])
        observation_vector = observation_vector + torch.randn_like(observation_vector) * noise_level
        encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        self.state = torch.zeros(x.shape[0],self.num_latent_states).to(device)
        self.state_history, u_history = [], []
        
        assimilate_time = 0
        for i in range(x.shape[1]):
        # dyn net to encode the input function
            if i == 0 and optimal_latent is not None:
                obs_ti = torch.cat((u,optimal_latent),dim=1)
            else:
                obs_ti = encoded_obs[i, :]
            scaled_obs_ti = obs_ti * scaling
            input = torch.cat((u,self.state),dim=1)
            
            with torch.no_grad():
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)
            
            def g_tau(t):
                return 1-t
            
            def score_likelihood(xt, t):
                # obs: (d)
                # xt: (ensemble, d)
                # the line below is analytical ∇z log P(Yt+1|z)device=device
                score_x = -(xt - scaled_obs_ti)/obs_sigma**2
                tau = g_tau(t)
                return tau*score_x
            
            latent = torch.concatenate((self.state.view(self.state.shape[0], -1), u), dim = 1)
            # new_state = encoded_latent.repeat((latent.shape[0], 1))#
            # import time
            # start_time = time.time()
            post_state = self.reverse_SDE(x0=latent*scaling, score_likelihood=score_likelihood, time_steps=euler_steps, device=device, eps_alpha=eps_alpha) / scaling
            # end_time = time.time()
            # print("assimilate time at one step: ", end_time - start_time)

            # assimilate_time += end_time - start_time
            self.state = post_state[:, :self.state.shape[1]]
            u = post_state[:, self.state.shape[1]:]
            # print(u)
            self.state_history.append(self.state)
            u_history.append(u)
        # print("Assimilate time: ", assimilate_time)
        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        u_history = torch.stack(u_history).transpose(0, 1)
        if latent_state:
            return self.state_history, u_history, encoded_obs
        
        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
        # concatenate latent state and space
        # x_rec = []
        # with torch.no_grad():
        #     x_rec_input = torch.cat([x,self.state_history],dim=-1).detach()
        #     for i in np.arange(0,self.ensemble_size,parallel):
        #         x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach().cpu()
        #         x_rec.append(x_rec_i)
        #     x_rec_tensor = torch.cat(x_rec, dim=0)
        # return x_rec_tensor

        x_rec = []
        with torch.no_grad():
            # points_projected = self.B(2 * torch.pi * x) # [batch, points, times, 2 * fourier]
            # print("points_projected shape",points_projected.shape)
            # x_rec_input = torch.cat([torch.sin(points_projected),torch.cos(points_projected),self.state_history_],dim=-1).detach()
            x_rec_input = self.state_history_.detach()
            for i in np.arange(0,self.ensemble_size,parallel):
                # x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                x_rec_i = self.rec(x[i:i+parallel], x_rec_input[i:i+parallel]).detach()
                x_rec.append(x_rec_i)
            x_rec = torch.cat(x_rec, dim=0)
        
        return x_rec
    
    # generate sample with reverse SDE
    def reverse_SDE(self, x0, score_likelihood=None, time_steps=100, save_path=False, device="cpu", eps_alpha=0.05):
        ensemble_size = x0.shape[0]
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from

        # reverse SDE sampling process
        # N1 = x_T.shape[0]
        # N2 = x0.shape[0]
        # d = x_T.shape[1]
        
        # drift_fun=f_func, diffuse_fun=g_func, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq, 
        def cond_alpha(t):  
            # conditional information
            # alpha_t(0) = 1
            # alpha_t(1) = esp_alpha \approx 0
            return 1 - (1-eps_alpha)*t

        def cond_sigma_sq(t):
            # conditional sigma^2
            # sigma2_t(0) = 0
            # sigma2_t(1) = 1
            # sigma(t) = t
            return t

        # drift function of forward SDE
        def f_func(t):
            # f=d_(log_alpha)/dt
            alpha_t = cond_alpha(t)
            f_t = -(1-eps_alpha) / alpha_t
            return f_t


        def g_sq(t):
            # g = d(sigma_t^2)/dt -2f sigma_t^2
            d_sigma_sq_dt = 1
            g2 = d_sigma_sq_dt - 2*f_func(t)*cond_sigma_sq(t)
            return g2

        def g_func(t):
            return np.sqrt(g_sq(t))

        # Generate the time mesh
        dt = 1.0/time_steps

        # Initialization
        xt = torch.randn(ensemble_size,x0.shape[1], device=device)
        t = 1.0

        # define storage
        if save_path:
            path_all = [xt]
            t_vec = [t]

        # forward Euler sampling
        for i in range(time_steps):
            # prior score evaluation
            alpha_t = cond_alpha(t)
            sigma2_t = cond_sigma_sq(t)


            # Evaluate the diffusion term
            diffuse = g_func(t)

            # Evaluate the drift term
            # drift = drift_fun(t)*xt - diffuse**2 * score_eval

            # Update
            if score_likelihood is not None:
                xt += - dt*(f_func(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * score_likelihood(xt, t) ) \
                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)
            else:
                xt += - dt*(f_func(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) ) + np.sqrt(dt)*diffuse*torch.randn_like(xt)

            # Store the state in the path
            if save_path:
                path_all.append(xt)
                t_vec.append(t)

            # update time
            t = t - dt

        if save_path:
            return path_all, t_vec
        else:
            return xt