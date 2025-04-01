import torch
import numpy as np
import torch.linalg as linalg
import deepxde as dde

from .encoder import *

parallel= 10
'''
The data should be a dictionary with the following keys:
y, u, x, dt, (y_x, y_u)
x must has shape (Ni, Nt, Nx, dimx). 
'''
class LDNN(dde.nn.NN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        layer_sizes_ic,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_ic = dde.nn.activations.get(activation["ic"])
            activation_dyn = dde.nn.activations.geat(activation["dyn"])
            self.activation_rec = dde.nn.activations.get(activation["rec"])
        else:
            activation_ic = activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        if callable(layer_sizes_dyn[1]):
            # User-defined network
            self.dyn = layer_sizes_dyn[1]
        else:
            # Fully connected network
            self.ic = dde.nn.FNN(layer_sizes_ic, activation_ic, kernel_initializer)
            self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        self.rec = dde.nn.FNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        # u has shape (Ni, Nu), which should be the initial condition
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        assert len(u.shape) == 2
        assert len(x.shape) == 4
        
        if latent_init is False:
            self.state = self.ic(u)
        else:
            self.state = data["latent"]
        
        Nt = x.shape[1]
        self.state_history = []
        for i in range(Nt):
            input = self.state
            if equilibrium:
                self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
            else:
                self.state = self.state + dt * self.dyn(input)
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        # print("state_history", self.state_history)
        
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
    
    def forward_data_assimilation_ensf(self, data, observations, encoder, noise_level, device, equilibrium=False, latent_state=False, latent_init=False,
                                       scaling = 500, obs_sigma =  0.1, eps_alpha = 0.05, euler_steps = 100):
        
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        # observations have shape Nt, Nx, dimx
        # u has shape (2)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        if latent_init is False:
            self.state = self.ic(u)
        else:
            self.state = data["latent"]
            
        observation_vector = observations.reshape([observations.shape[0], -1])
        observation_vector = observation_vector + torch.randn_like(observation_vector) * noise_level
        encoder.eval()
        encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        
        self.state_history = []
        Nt = x.shape[1]
        for i in range(Nt):
        # dyn net to encode the input function
            obs_ti = encoded_obs[i, :]
            scaled_obs_ti = obs_ti*scaling
            input = self.state
            
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
                # the line below is analytical ∇z log P(Yt+1|z)
                score_x = -(xt - scaled_obs_ti)/obs_sigma**2
                tau = g_tau(t)
                return tau*score_x

            latent = self.state.view(self.state.shape[0], -1)
            # new_state = encoded_latent.repeat((latent.shape[0], 1))#
            post_state = self.reverse_SDE(x0=latent*scaling, score_likelihood=score_likelihood, time_steps=euler_steps, device=device, eps_alpha=eps_alpha)/scaling
            self.state = post_state
            # u = new_state[:, self.state.shape[1]:]
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)

        if latent_state:
            return self.state_history, encoded_obs
        
        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
      # concatenate latent state and space
        x_rec = []
        with torch.no_grad():
            x_rec_input = torch.cat([x,self.state_history],dim=-1).detach()
            for i in np.arange(0,self.ensemble_size,parallel):
                x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach().cpu()
                x_rec.append(x_rec_i)
            x_rec_tensor = torch.cat(x_rec, dim=0)
        return x_rec_tensor

    def my_EnKF(self, state, obs, N, scaling):
        """My implementation of the EnKF using PyTorch."""

        ### Init ###
        E = state
        y = obs.unsqueeze(1)  # current observation with added dimension
        Eo = E  # observed ensemble

        # Compute ensemble moments
        Y = Eo - Eo.mean(dim=0, keepdim=True)
        X = E - E.mean(dim=0, keepdim=True)
        PH = X @ Y.t() / (N - 1)
        HPH = Y @ Y.t() / (N - 1)

        # Compute Kalman Gain
        KG = linalg.solve(HPH + self.R, PH.t()).t()

        # Generate perturbations
        Perturbs = scaling * torch.matmul(self.R12, torch.randn(self.latent_dim, self.ensemble_size).to(state.device))

        # Update ensemble with KG
        return E + KG @ (y - Eo - Perturbs)
    
    def forward_data_assimilation_enkf(self, data, observations, encoder, noise_level, device,
                                        equilibrium=False, latent_state=False, latent_init=False, enkf_scaling=1, enkf_obs_sigma=0):
        
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        # observations have shape Nt, Nx, dimx
        # u has shape (2)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        
        assert len(u.shape) == 2
        assert len(x.shape) == 4
        
        if latent_init is False:
            self.state = self.ic(u)
        else:
            self.state = data["latent"]
        
        Nt = x.shape[1]
        observation_vector = observations.reshape([observations.shape[0], -1])
        observation_vector = observation_vector + torch.randn_like(observation_vector) * noise_level
        encoder.eval()
        encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        
        self.latent_dim = encoded_obs.shape[-1] 
        self.R12 = enkf_obs_sigma*torch.eye((self.latent_dim)).to(device)
        self.R = self.R12 @ self.R12.T
        self.ensemble_size = u.shape[0]
        
        self.state_history, u_history = [], []
        for i in range(x.shape[1]):
        # dyn net to encode the input function
            obs_ti = encoded_obs[i, :]
            input = self.state
            
            with torch.no_grad():
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)
            
            latent = self.state.view(self.state.shape[0], -1)
            post_state = self.my_EnKF(latent.T, obs_ti.reshape(-1) + self.R12 @ torch.randn(self.R12.shape[0]).to(device), self.ensemble_size, enkf_scaling).T
            self.state = post_state
            # self.state = obs_ti.unsqueeze(0).repeat(self.ensemble_size, 1) # Test 
            
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        # np.save("./results/encoded_latents_lstm.npy", torch.stack(obs_latent_history).detach().cpu().numpy(), allow_pickle = True)
        # print(data["u"])
        if latent_state:
            return self.state_history, encoded_obs
        
        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
        # concatenate latent state and space
        x_rec = []
        with torch.no_grad():
            x_rec_input = torch.cat([x,self.state_history],dim=-1).detach()
            for i in np.arange(0,self.ensemble_size,parallel):
                x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach().cpu()
                x_rec.append(x_rec_i)
            x_rec_tensor = torch.cat(x_rec, dim=0)
        return x_rec_tensor