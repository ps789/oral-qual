import numpy as np
import torch
class EnSF:
    def __init__(self, ensemble_size, scaling, obs_sigma, eps_alpha, euler_steps, device):
        self.ensemble_size = ensemble_size
        self.scaling = scaling
        self.eps_alpha = eps_alpha
        self.obs_sigma = obs_sigma
        self.euler_steps = euler_steps
        self.device = device

    def cond_alpha(self, t):
        # conditional information
        # alpha_t(0) = 1
        # alpha_t(1) = esp_alpha \approx 0
        return 1 - (1-self.eps_alpha)*t

    def g_tau(self, t):
        return 1-t

    def cond_sigma_sq(self, t):
        # conditional sigma^2
        # sigma2_t(0) = 0
        # sigma2_t(1) = 1
        # sigma(t) = t
        return t

    # drift function of forward SDE
    def f_func(self, t):
        # f=d_(log_alpha)/dt
        alpha_t = self.cond_alpha(t)
        f_t = -(1-self.eps_alpha) / alpha_t
        return f_t


    def g_sq(self, t):
        # g = d(sigma_t^2)/dt -2f sigma_t^2
        d_sigma_sq_dt = 1
        g2 = d_sigma_sq_dt - 2*self.f_func(t)*self.cond_sigma_sq(t)
        return g2

    def g_func(self, t):
        return np.sqrt(self.g_sq(t))


    # generate sample with reverse SDE
    def reverse_SDE(self, x0, score_likelihood=None, time_steps=100, save_path=False):
        
        # x_T: sample from standard Gaussian
        # x_0: target distribution to sample from

        # reverse SDE sampling process
        # N1 = x_T.shape[0]
        # N2 = x0.shape[0]
        # d = x_T.shape[1]

        # Generate the time mesh
        dt = 1.0/time_steps

        # Initialization
        xt = torch.randn_like(x0)
        xt.to(self.device)
        t = 1.0

        # define storage
        if save_path:
            path_all = [xt]
            t_vec = [t]
        drift_fun = self.f_func
        diffuse_fun=self.g_func
        alpha_fun=self.cond_alpha
        sigma2_fun=self.cond_sigma_sq
        # forward Euler sampling
        for i in range(time_steps):
            # prior score evaluation
            alpha_t = alpha_fun(t)
            sigma2_t = sigma2_fun(t)


            # Evaluate the diffusion term
            diffuse = diffuse_fun(t)

            # Evaluate the drift term
            # drift = drift_fun(t)*xt - diffuse**2 * score_eval

            # Update
            if score_likelihood is not None:
                xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * score_likelihood(xt, t) ) \
                    + np.sqrt(dt)*diffuse*torch.randn_like(xt)
            else:
                xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) ) + np.sqrt(dt)*diffuse*torch.randn_like(xt)

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
    def assimilate_ensf_define_likelihood(self, ensemble_states, score_likelihood):
        return self.reverse_SDE(x0=ensemble_states*self.scaling, score_likelihood=score_likelihood, time_steps=self.euler_steps)
    
    def assimilate_ensf(self, ensemble_states, observations):
        obs = observations.view(-1)*self.scaling + torch.randn_like(observations.view(-1))*self.obs_sigma
        def score_likelihood(xt, t):
            # obs: (d)
            # xt: (ensemble, d)
            # the line below is analytical ∇z log P(Yt+1|z)
            score_x = -(xt - obs)/self.obs_sigma**2
            tau = self.g_tau(t)
            return tau*score_x
        
        return self.reverse_SDE(x0=ensemble_states.view(ensemble_states.shape[0], -1)*self.scaling, score_likelihood=score_likelihood, time_steps=self.euler_steps)