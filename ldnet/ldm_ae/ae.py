import torch
import torch.nn.functional as F
from contextlib import contextmanager
from .model import *
from .quantize import VectorQuantizer2 as VectorQuantizer
from .distributions import DiagonalGaussianDistribution
class KLModel(nn.Module):
    def __init__(self,
                 in_channel,
                 num_res_blocks,
                 hidden_dim,
                 embed_dim = 4,
                 ):
        super().__init__()
        self.latent_dim = hidden_dim
        self.embed_dim = embed_dim
        self.image_key = "image"
        self.encoder = Encoder(hidden_dim, in_channel, num_res_blocks=num_res_blocks,
                 attn_resolutions=[], ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, in_channels=in_channel,
                 resolution=150, z_channels=hidden_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", downsample = True)
        self.encoder_sample = Encoder(hidden_dim, in_channel, num_res_blocks=num_res_blocks,
                 attn_resolutions=[], ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, in_channels=in_channel,
                 resolution=10, z_channels=hidden_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", downsample = False)
        self.decoder = Decoder(hidden_dim, in_channel, num_res_blocks=num_res_blocks,
                 attn_resolutions = [], ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, in_channels=in_channel,
                 resolution=150, z_channels=hidden_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla")
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=None,
        #                                 sane_index_shape=False)
        self.quant_conv = torch.nn.Conv2d(2*hidden_dim, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hidden_dim, 1)
        
        self.scheduler_config = None
        self.lr_g_factor = 1.0
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def encode_sample(self, x):
        h = self.encoder_sample(x)
        moments = self.quant_conv(h)
        return moments
    

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def split(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def forward(self, input, sample_posterior=True):
        moments = self.encode(input)
        mean, logvar = self.split(moments)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, mean, logvar
    
    def forward_sample(self, input, sample_posterior = True):
        moments = self.encode_sample(input)
        mean, logvar = self.split(moments)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, mean, logvar

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from .model import *
from .quantize import VectorQuantizer2 as VectorQuantizer
from .distributions import DiagonalGaussianDistribution
class KLModel_Dynamics(nn.Module):
    def __init__(self,
                 in_channel,
                 num_res_blocks,
                 hidden_dim,
                 embed_dim = 4,
                 ):
        super().__init__()
        self.latent_dim = hidden_dim
        self.embed_dim = embed_dim
        self.image_key = "image"
        self.dynamics = TimeSeriesLSTM(embed_dim*225, 512, embed_dim*225, num_layers=2, dropout=0)
        self.encoder = Encoder(hidden_dim, in_channel, num_res_blocks=num_res_blocks,
                 attn_resolutions=[], ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, in_channels=in_channel,
                 resolution=150, z_channels=hidden_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", downsample = True)
        self.encoder_sample = Encoder(hidden_dim, in_channel, num_res_blocks=num_res_blocks,
                 attn_resolutions=[], ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, in_channels=in_channel,
                 resolution=10, z_channels=hidden_dim, double_z=True, use_linear_attn=False, attn_type="vanilla", downsample = False)
        self.decoder = Decoder(hidden_dim, in_channel, num_res_blocks=num_res_blocks,
                 attn_resolutions = [], ch_mult=(1,2,4,8), dropout=0.0, resamp_with_conv=True, in_channels=in_channel,
                 resolution=150, z_channels=hidden_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla")
        # self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
        #                                 remap=None,
        #                                 sane_index_shape=False)
        self.quant_conv = torch.nn.Conv2d(2*hidden_dim, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hidden_dim, 1)
        
        self.scheduler_config = None
        self.lr_g_factor = 1.0
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def encode_sample(self, x):
        h = self.encoder_sample(x)
        moments = self.quant_conv(h)
        return moments
    

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def split(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def forward(self, input, sample_posterior=True):
        input_reshaped = input.view(input.shape[0]*input.shape[1], input.shape[2], input.shape[3], input.shape[4])
        moments = self.encode(input_reshaped)
        mean, logvar = self.split(moments)
        mean_reshaped = mean.view(input.shape[0], input.shape[1], mean.shape[1]*mean.shape[2]*mean.shape[3])
        mean_dynamics = self.dynamics(mean_reshaped).view(mean.shape)
        
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)

        mean_dynamics = mean_dynamics.view(input.shape[0], input.shape[1], mean.shape[1], mean.shape[2], mean.shape[3])
        mean = mean.view(input.shape[0], input.shape[1], mean.shape[1], mean.shape[2], mean.shape[3])
        logvar = logvar.view(input.shape[0], input.shape[1], logvar.shape[1], logvar.shape[2], logvar.shape[3])
        dec = dec.view(input.shape[0], input.shape[1], dec.shape[1], dec.shape[2], dec.shape[3])
        return dec, mean, logvar, mean_dynamics
    
    def forward_sample(self, input, sample_posterior = True):
        input_reshaped = input.view(input.shape[0]*input.shape[1], input.shape[2], input.shape[3], input.shape[4])
        moments = self.encode_sample(input_reshaped)
        mean, logvar = self.split(moments)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        mean = mean.view(input.shape[0], input.shape[1], mean.shape[1], mean.shape[2], mean.shape[3])
        logvar = logvar.view(input.shape[0], input.shape[1], logvar.shape[1], logvar.shape[2], logvar.shape[3])
        dec = dec.view(input.shape[0], input.shape[1], dec.shape[1], dec.shape[2], dec.shape[3])
        return dec, mean, logvar

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, proj_size=0, bias=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        #x : batch_size, num_time_steps = 51, observation_dim = 3x10x10 = 300
        output, _ = self.lstm(x)
        # batch_size, num_time_steps, hidden_size = 256
        output = self.fc(output) #256 -> 12
        # batch_size, num_time_steps, output_size = latent_dim = 12

        return output
    
    def forward_5d(self, x):
        x_reshaped = x.view(x.shape[0], x.shape[1], -1)
        #x : batch_size, num_time_steps = 51, observation_dim = 3x10x10 = 300
        output, _ = self.lstm(x_reshaped)
        # batch_size, num_time_steps, hidden_size = 256
        output = self.fc(output) #256 -> 12
        # batch_size, num_time_steps, output_size = latent_dim = 12

        return output.view(x.shape)