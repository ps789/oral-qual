import torch
import numpy as np
from ldnet.fourier_ldnet import ResFourierLDNN
from ldnet import utils

fontsize=23
fig_params = {
    'font.size': fontsize,
    "savefig.dpi": 300, 
    "figure.figsize": (8, 6),
    'lines.linewidth': 5,
    'axes.linewidth': 2.5,
    'axes.titlesize' : fontsize+3,
    #"axes.labelsize":fontsize+5,
    "xtick.labelsize":fontsize,
    "ytick.labelsize":fontsize,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'xtick.major.size': 7,
    'xtick.minor.size': 5,
    'xtick.major.width': 3,
    'xtick.minor.width': 2,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 5,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 5,
    'ytick.major.size': 7,
    'ytick.minor.size': 5,
    'ytick.major.width': 3,
    'ytick.minor.width': 2,
    'legend.frameon' : False,
    "mathtext.fontset":"cm"
}

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_time_slice_error(data, model, device, interval, normalize_y):
    """
    time step = 40
    space points = 10000
    """
    error, av_error = compute_time_slice_error(data, model, device, normalize_y)
    #plot error
    plt.rcParams.update(fig_params)
    fig = plt.figure()
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.plot(error)
    plt.xticks(np.arange(0, 40, 10), np.arange(0, 40, 10) * interval)
    plt.xlabel("time step")
    plt.ylabel("L2 relative error")
    plt.title("average error: {:.4f}".format(av_error))
    plt.savefig("./plot_dynamics.jpg")
    return error

def compute_time_slice_error(data, model, device, normalize_y=None):
    outputs = compute_outputs(data, model, device)
    y = data["y"].cpu().detach().numpy()
    if normalize_y:
        y = normalize_y.normalize_back(y)
        outputs = normalize_y.normalize_back(outputs)
    numerator = np.sum((outputs - y)**2, axis=(0,2,3))
    denominator = np.sum(y**2, axis=(0,2,3))
        
    error = np.sqrt(numerator / denominator)
    # av_error = np.sqrt(np.sum(numerator) / np.sum(denominator))
    av_error = np.mean(error)
    return error, av_error

def compute_outputs(data, model, device):
    @torch.no_grad()
    def compute_output(data, model, device):
        data_test = {}
        model.eval()
        data_test["x"] = data["x"].to(device)
        data_test["u"] = data["u"].to(device)
        data_test["dt"] = data["dt"].to(device)
        output_y = model(data_test, device=device)
        
        output_y = output_y.cpu().detach().numpy()
        return output_y

    Nx = data["x"].shape[2]
    outputs = np.zeros_like(data["y"].cpu().numpy(), dtype=np.float32)
    data_test = {}
    data_test["u"] = data["u"]
    data_test["x"] = data["x"]
    data_test["dt"] = data["dt"]
    output_y = compute_output(data_test, model, device)
    outputs = output_y
    return outputs

def plot_time_slice_error_assimilate(data, model, device, interval, normalize_y):
    """
    time step = 40
    space points = 10000
    """
    error, av_error = compute_time_slice_error_assimilate(data, model, device, normalize_y)
    #plot error
    plt.rcParams.update(fig_params)
    fig = plt.figure()
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.plot(error)
    plt.xticks(np.arange(0, 40, 10), np.arange(0, 40, 10) * interval)
    plt.xlabel("time step")
    plt.ylabel("L2 relative error")
    plt.title("average error: {:.4f}".format(av_error))
    plt.savefig("./plot_lbfgs_with_dynamics.jpg")
    return error

def compute_time_slice_error_assimilate(data, model, device, normalize_y=None):
    outputs = compute_outputs_assimilate(data, model, device)
    y = data["y"].cpu().detach().numpy()
    if normalize_y:
        y = normalize_y.normalize_back(y)
        outputs = normalize_y.normalize_back(outputs)
    numerator = np.sum((outputs - y)**2, axis=(0,2,3))
    denominator = np.sum(y**2, axis=(0,2,3))
        
    error = np.sqrt(numerator / denominator)
    # av_error = np.sqrt(np.sum(numerator) / np.sum(denominator))
    av_error = np.mean(error)
    return error, av_error

def compute_outputs_assimilate(data, model, device):
    def compute_output_assimilate(data, model, observation, device):
        data_test = {}
        model.eval()
        data_test["x"] = data["x"].to(device)
        data_test["u"] = data["u"].to(device)
        data_test["dt"] = data["dt"].to(device)
        output_y = model.forward_data_assimilation_lbfgs_with_dynamics_multistep(data_test, observation, device=device)
        # output_y = model.forward_data_assimilation_lbfgs(data_test, observation, device=device)
        
        output_y = output_y.cpu().detach().numpy()
        return output_y

    Nx = data["x"].shape[2]
    outputs = np.zeros_like(data["y"].cpu().numpy(), dtype=np.float32)
    data_test = {}
    data_test["u"] = data["u"]
    data_test["x"] = data["x"]
    data_test["dt"] = data["dt"]
    observation = data["y"][:, :, :, :]
    observation = observation.reshape(observation.shape[0], observation.shape[1], 150, 150, 2)[:, :, ::15, ::15, :].to(device)
    # observation = observation.reshape(observation.shape[0], observation.shape[1], 100, 3)
    output_y = compute_output_assimilate(data_test, model, observation, device)
    return output_y

device = torch.device("cuda:3")
model = ResFourierLDNN(
            10,
            [1+9] + 9 * [200] + [9],
            [2+9] + 14 * [500] + [2],
            activation="relu",
            kernel_initializer="Glorot normal",
)
model.to(device)
model = utils.load_model(model,"./ldnet_checkpoints/dyn_1999.ckpt",
                "./ldnet_checkpoints/retrained_rec_1999.ckpt", device)
# dataset = torch.load("/work/pengpeng/data-assimilation/kolmogorov_flow/data/data_observation_500_1500_150x150.pth", map_location="cpu")
# dataset = torch.load("/work/pengpeng/data-assimilation/kolmogorov_flow/data/data_observation_500_1500_300x300.pth", map_location="cpu")
dataset = torch.load("./data/data_observation_500_1500_150x150_seed_0_resnet_fourier_10_gamma_0.5_resd_14.pth", map_location="cpu")
data = dataset["data_test"].copy()
# from src.viz_tool import compute_time_slice_error, compute_outputs
# outputs = compute_outputs(data, model, device)
from ldnet.normalization import Normalize_gaussian
stat = np.load("./ldnet_checkpoints/cplx_Re500_1500_150x150/mean_std.npz", allow_pickle=True)
mean = stat["mean"]
std = stat["std"]
normalize_y = Normalize_gaussian(mean, std)
error = plot_time_slice_error_assimilate(data, model, device, 5, normalize_y)
# outputs = compute_outputs(data, model, device)
# np.save("/work/pengpeng/data-assimilation/kolmogorov_flow/plots/plots_data/retrained_rec_outputs.npy", outputs)
# np.save("/work/pengpeng/data-assimilation/kolmogorov_flow/plots/plots_data/rec_error_ldnet.npy", error)
# error, avg_error = compute_time_slice_error(data, model, device, normalize_y=normalize_y)
# np.savez("/work/pengpeng/data-assimilation/kolmogorov_flow/plots/plots_data/150x150_seed_0_test_retrian_reac_state_error.npz",
#          outputs=outputs,error=error, avg_error=avg_error)