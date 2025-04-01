
"""File with several visualization functions intended to use
with results from 2D shallow water model swe2D.py"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure

from matplotlib.animation import FuncAnimation

#plt.style.use("seaborn")

import matplotlib.pyplot as plt
import torch
import numpy as np

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
    outputs = np.zeros_like(data["y"], dtype=np.float32)
    for i in range(0,Nx,1000):
        data_test = {}
        data_test["u"] = data["u"]
        data_test["x"] = data["x"][:,:,i:i+1000,:]
        data_test["dt"] = data["dt"]
        output_y = compute_output(data_test, model, device)
        outputs[:,:,i:i+1000,:] = output_y
    return outputs

def compute_time_slice_error(data, model, device, normalize_y=None):
    outputs = compute_outputs(data, model, device)
    y = data["y"].cpu().detach().numpy()
    if normalize_y:
        y = normalize_y.normalize_back(y)
        outputs = normalize_y.normalize_back(outputs)
    numerator = np.sum((outputs - y)**2, axis=(0,2,3))
    denominator = np.sum(y**2, axis=(0,2,3))
        
    error = np.sqrt(numerator / denominator)
    av_error = np.sqrt(np.sum(numerator) / np.sum(denominator))
    return error, av_error

def plot_time_slice_error(data, model, device, interval):
    """
    time step = 51
    space points = 10000
    """
    error, av_error = compute_time_slice_error(data, model, device)
    #plot error
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
    plt.rcParams.update(fig_params)
    fig = plt.figure()
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    plt.plot(error)
    plt.xticks(np.arange(0, 51, 10), np.arange(0, 51, 10) * interval)
    plt.xlabel("time step")
    plt.ylabel("L2 relative error")
    plt.title("average error: {:.4f}".format(av_error))
    return fig
    
    
@torch.no_grad()
def plot_latent_state(data, interval):
    
    # data["x"] = torch.from_numpy(data["x"]).to(device)
    # data["u"] = torch.from_numpy(data["u"]).to(device)
    # data["dt"] = torch.from_numpy(data["dt"]).to(device)
    
    # latent = model(data, device=device, latent_state=True)
    latent = data["latent_states"]
    latent = latent.cpu().detach().numpy()
    fontsize=23
    fig_params = {
        'font.size': fontsize,
        "savefig.dpi": 300, 
        "figure.figsize": (8, 6),
        'lines.linewidth': 4,
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
    plt.rcParams.update(fig_params)
    fig = plt.figure()
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
    plt.plot(latent[2,:])
    plt.xticks(np.arange(0, 51, 10), np.arange(0, 51, 10) * interval)
    plt.ylabel("latent state")
    plt.xlabel("time step")
    return fig
    
@torch.no_grad()
def plot_latent_state_lstm_error(data, lstm, device, interval):
    
    # data["x"] = torch.from_numpy(data["x"]).to(device)
    # data["u"] = torch.from_numpy(data["u"]).to(device)
    # data["dt"] = torch.from_numpy(data["dt"]).to(device)
    latent = data["latent_states"]
    observation = data["observation"].to(device)
    observation = observation.reshape([observation.shape[0], observation.shape[1],-1])
    
    # observation = observation[5:, :]
    # latent = latent[5:, :]
    # slice_latent = latent[0, :].unsqueeze(0).repeat((4, 1))
    # slice_observation = observation[0, :].unsqueeze(0).repeat((4, 1))
    # observation = torch.cat((slice_observation, observation), dim = 0)
    # latent = torch.cat((slice_latent, latent), dim = 0)
    
    latent_lstm = lstm(observation).cpu().detach().numpy()[:,:,:9]
    latent = latent.cpu().detach().numpy()
    numerator = np.sum((latent - latent_lstm)**2, axis=(0,2))
    denominator = np.sum(latent**2, axis=(0,2))
    error = np.sqrt(numerator / denominator)
    av_error = np.sqrt(np.sum(numerator) / np.sum(denominator))
    plt.plot(error, label=r"Average ${:.1f}\%$".format(av_error * 100))
    plt.xticks(np.arange(0, 41, 10), np.arange(0, 41, 10) * interval)
    plt.ylabel(r"$L^2$ relative error")
    plt.xlabel("Time step")
    plt.legend()    
    # plt.show()
    
@torch.no_grad()
def plot_u_lstm_error(data, lstm, device, interval):
    
    # data["x"] = torch.from_numpy(data["x"]).to(device)
    # data["u"] = torch.from_numpy(data["u"]).to(device)
    # data["dt"] = torch.from_numpy(data["dt"]).to(device)
    u = data["u"]
    observation = data["observation"][0].to(device)
    observation = observation.reshape([observation.shape[0], -1])
    latent_lstm = lstm(observation.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    u = u.cpu().detach().numpy()
    from src import utils
    l2_error = utils.time_slice_l2_error(latent_lstm[:,10:], u[0][None,:])
    print(l2_error)
    plt.plot(u[0][None,:]-latent_lstm[:,10:])
    plt.xticks(np.arange(0, 51, 10), np.arange(0, 51, 10) * interval)
    plt.ylabel("differnce between lstm and ldnet u state")
    plt.xlabel("time step")
    plt.show()
    
    

def plot_2d_velocity(velocity, save_path=None):
    def vorticity(y):
        *batch, _, h, w = y.shape

        y = y.reshape(-1, 2, h, w)
        y = np.pad(y, pad_width=((0,0),(0,0),(1,1),(1,1)), mode='wrap')

        du = np.gradient(y[:, 0], axis=-1)
        dv = np.gradient(y[:, 1], axis=-2)

        y = du - dv
        y = y[:, 1:-1, 1:-1]
        y = y.reshape(*batch, h, w)

        return y

    vort = vorticity(velocity)
    vort = vort[0]

    def update(frame_number, data, im):
        im.set_data(data[frame_number])
        return [im]

    fig, ax = plt.subplots()
    im = ax.imshow(vort[0], cmap='viridis', interpolation='none')
    ax.axis('off')  

    ani = FuncAnimation(fig, update, frames=range(vort.shape[0]), fargs=(vort, im), blit=True)

    ani.save(save_path, writer='ffmpeg',fps=16)
    

def eta_animation(X, Y, eta_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. To get updating title one
    also need specify time step dt between each frame in the simulation, the number
    of time steps between each eta in eta_list and finally, a filename for video."""
    fig, ax = plt.subplots(1, 1)
    #plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [m]", fontname = "serif", fontsize = 12)
    pmesh = plt.pcolormesh(X, Y, eta_list[0], vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r)
    plt.colorbar(pmesh, orientation = "vertical")

    # Update function for quiver animation.
    def update_eta(num):
        ax.set_title("Surface elevation $\eta$ after t = {:.2f} hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        pmesh.set_array(eta_list[num][:-1, :-1].flatten())
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim    # Need to return anim object to see the animation


def eta_animation_overlay(X, Y, eta_list, eta_list_2, eta_list_enkf, eta_list_ensf, title_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. To get updating title one
    also need specify time step dt between each frame in the simulation, the number
    of time steps between each eta in eta_list and finally, a filename for video."""
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    fig, ax = plt.subplots(2, 2)
    ax[0][0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    ax[0][1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    ax[1][0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    ax[1][1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    #plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    pmesh = [[ax[j][i].pcolormesh(X, Y, eta_list[0], vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r) for i in range(2)] for j in range(2)]
    # plt.colorbar(pmesh[0][0], orientation = "vertical")

    # Update function for quiver animation.
    def update_eta(num):
        ax[0][0].set_title("Ground Truth $\eta$".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        ax[0][1].set_title(title_list[0], fontname = "serif", fontsize = 16)
        ax[1][0].set_title(title_list[1].format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        ax[1][1].set_title(title_list[2].format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        pmesh[0][0].set_array(eta_list[num].flatten())
        pmesh[0][1].set_array(eta_list_2[num].flatten())
        pmesh[1][0].set_array(eta_list_enkf[num].flatten())
        pmesh[1][1].set_array(eta_list_ensf[num].flatten())
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)
    pillow_writer = animation.PillowWriter(fps = 24, bitrate = 10000)
    anim.save("{}.gif".format(filename), writer = pillow_writer)
    return anim    # Need to return anim object to see the animation

def eta_animation_errors(X, Y, eta_list, eta_list_comparisons, title_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. To get updating title one
    also need specify time step dt between each frame in the simulation, the number
    of time steps between each eta in eta_list and finally, a filename for video."""
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    fig, ax = plt.subplots(2, len(eta_list_comparisons)+1)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    for x_ax in ax:
        for y_ax in x_ax:
            y_ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    #plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    pmesh = [[ax[j][i].pcolormesh(X, Y, eta_list[0]*0, vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r) for i in range(len(eta_list_comparisons)+1)] for j in range(2)]
    # plt.colorbar(pmesh[0][0], orientation = "vertical")

    # Update function for quiver animation.
    mask = np.ones((100, 100))
    mask[::10, ::10] = 0
    def update_eta(num):
        ax[0][0].set_title("Ground Truth $\eta$".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        ax[1][0].set_title("Observations $\eta$".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        pmesh[0][0].set_array(eta_list[num].flatten())
        masked_true = np.ma.array(eta_list[num], mask = mask > 0)
        pmesh[1][0].set_array(masked_true.flatten())

        for i in range(len(eta_list_comparisons)):
            ax[0][i+1].set_title(title_list[i], fontname = "serif", fontsize = 16)
            ax[1][i+1].set_title(f"{title_list[i]} Error", fontname = "serif", fontsize = 16)
            pmesh[0][i+1].set_array(eta_list_comparisons[i][num].flatten())
            pmesh[1][i+1].set_array(np.abs(eta_list_comparisons[i][num].flatten() - eta_list[num].flatten()))
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)
    pillow_writer = animation.PillowWriter(fps = 24, bitrate = 10000)
    anim.save("{}.gif".format(filename), writer = pillow_writer)
    return anim    # Need to return anim object to see the animation

def eta_animation_perturbed(X, Y, eta_list, eta_list_2, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a list of 2D arrays
    eta_list and creates an animation of all eta images. To get updating title one
    also need specify time step dt between each frame in the simulation, the number
    of time steps between each eta in eta_list and finally, a filename for video."""
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10,4)
    ax[0].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    ax[1].tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    #plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 17)
    pmesh = [ax[i].pcolormesh(X, Y, eta_list[0], vmin = -0.7*np.abs(eta_list[int(len(eta_list)/2)]).max(),
        vmax = np.abs(eta_list[int(len(eta_list)/2)]).max(), cmap = plt.cm.RdBu_r) for i in range(2)]
    # plt.colorbar(pmesh[0][0], orientation = "vertical")

    # Update function for quiver animation.
    def update_eta(num):
        ax[0].set_title("Ground Truth $\eta$".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 16)
        ax[1].set_title("Perturbed $\eta$", fontname = "serif", fontsize = 16)
        pmesh[0].set_array(eta_list[num].flatten())
        pmesh[1].set_array(eta_list_2[num].flatten())
        return pmesh,

    anim = animation.FuncAnimation(fig, update_eta,
        frames = len(eta_list), interval = 10, blit = False)
    pillow_writer = animation.PillowWriter(fps = 24, bitrate = 10000)
    anim.save("{}.gif".format(filename), writer = pillow_writer)
    return anim    # Need to return anim object to see the animation

def velocity_animation(X, Y, u_list, v_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a lists of 2D arrays
    u_list, v_list and creates an quiver animation of the velocity field (u, v). To get
    updating title one also need specify time step dt between each frame in the simulation,
    the number of time steps between each eta in eta_list and finally, a filename for video."""
    fig, ax = plt.subplots(figsize = (8, 8), facecolor = "white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname = "serif", fontsize = 19)
    plt.xlabel("x [km]", fontname = "serif", fontsize = 16)
    plt.ylabel("y [km]", fontname = "serif", fontsize = 16)
    q_int = 3
    Q = ax.quiver(X[::q_int, ::q_int]/1000.0, Y[::q_int, ::q_int]/1000.0, u_list[0][::q_int,::q_int], v_list[0][::q_int,::q_int],
        scale=0.2, scale_units='inches')
    #qk = plt.quiverkey(Q, 0.9, 0.9, 0.001, "0.1 m/s", labelpos = "E", coordinates = "figure")

    # Update function for quiver animation.
    def update_quiver(num):
        u = u_list[num]
        v = v_list[num]
        ax.set_title("Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f} hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19)
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = animation.FuncAnimation(fig, update_quiver,
        frames = len(u_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    fig.tight_layout()
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim    # Need to return anim object to see the animation

def eta_animation3D(X, Y, eta_list, frame_interval, filename):
    fig = plt.figure(figsize = (8, 8), facecolor = "white")
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, eta_list[0], cmap = plt.cm.RdBu_r)

    def update_surf(num):
        ax.clear()
        surf = ax.plot_surface(X/1000, Y/1000, eta_list[num], cmap = plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
            num*frame_interval/3600), fontname = "serif", fontsize = 19, y=1.04)
        ax.set_xlabel("x [km]", fontname = "serif", fontsize = 14)
        ax.set_ylabel("y [km]", fontname = "serif", fontsize = 14)
        ax.set_zlabel("$\eta$ [m]", fontname = "serif", fontsize = 16)
        ax.set_xlim(X.min()/1000, X.max()/1000)
        ax.set_ylim(Y.min()/1000, Y.max()/1000)
        ax.set_zlim(-0.3, 0.7)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf,
        frames = len(eta_list), interval = 10, blit = False)
    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000,
        codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer = mpeg_writer)
    return anim    # Need to return anim object to see the animation

def surface_plot3D(X, Y, eta, x_lim, y_lim, z_lim):
    """Function that takes input 1D coordinate arrays x, y and 2D array
    array psi. Then plots psi as a surface in 3D space on a meshgrid."""
    fig = plt.figure(figsize = (11, 7))
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, eta, rstride = 1, cstride = 1,
        cmap = plt.cm.jet, linewidth = 0, antialiased = True)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_title("Surface elevation $\eta$", fontname = "serif", fontsize = 17)
    ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
    ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
    ax.set_zlabel("Surface elevation [m]", fontname = "serif", fontsize = 16)
    plt.show()

def pmesh_plot(X, Y, eta, plot_title):
    """Function that generates a colored contour plot of eta in the domain X, Y"""
    plt.figure(figsize = (9, 8))
    plt.pcolormesh(X, Y, eta, cmap = plt.cm.RdBu_r)
    plt.colorbar(orientation = "vertical")
    plt.title(plot_title, fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [s]", fontname = "serif", fontsize = 12)

def quiver_plot(X, Y, U, V, plot_title):
    """Function that makes a quiver plot of (U, V) at points (X, Y)."""
    plt.figure()
    plt.title(plot_title, fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("y [m]", fontname = "serif", fontsize = 12)
    Q = plt.quiver(X[::4, ::4], Y[::4, ::4], U[::4, ::4], V[::4, ::4],
        units = "xy", scale = 0.002, scale_units = "inches")
    qk = plt.quiverkey(Q, 0.9, 0.9, 0.001, "0.1 m/s",
        labelpos = "E", coordinates = "figure")

def hovmuller_plot(x, t, eta):
    """Function that generates a Hovmuller diagram of
    eta as a function of x and t at a choosen y-coordinate"""
    X, T = np.meshgrid(x, np.array(t))
    X = np.transpose(X)         # Transpose for plotting
    T = np.transpose(T)         # Transpose for plotting
    eta_hm = np.transpose(np.array(eta))  # Transpose for plotting

    plt.figure(figsize = (5, 8))
    plt.pcolormesh(X, T, eta_hm, vmin = eta_hm.min(), vmax = eta_hm.max(), cmap = plt.cm.PiYG)
    plt.colorbar(orientation = "vertical")
    plt.title("x-t plot for middle of domain", fontname = "serif", fontsize = 17)
    plt.xlabel("x [m]", fontname = "serif", fontsize = 12)
    plt.ylabel("t [s]", fontname = "serif", fontsize = 12)
