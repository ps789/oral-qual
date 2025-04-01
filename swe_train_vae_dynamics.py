import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
from ldm_ae.ae import *
data_x = np.load("./test_data/data_coriolis_jan20.npy", allow_pickle = True)
maeloss = torch.nn.L1Loss(reduction='sum')
mseloss = torch.nn.MSELoss(reduction='sum')
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = mseloss(x_hat, x)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + 0.00001*KLD

def kld(mean1, log_var1, mean2, log_var2):
    # return torch.sum((mean1 - mean2).pow(2))
    return - 0.5 * torch.sum(1+ log_var1 - log_var2 - (mean1 - mean2).pow(2)/log_var2.exp() - log_var1.exp()/log_var2.exp())
# data_x, data_y, data_t = data_x[:5000], data_y[:5000], data_t[:5000]
X_train, X_test = train_test_split(data_x, test_size = 0.4, shuffle = True, random_state = 42)

class CustomDataset(Dataset):

    def __init__(self, x_data, device):
        self.x_data = torch.Tensor(x_data)
        self.device = device
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):

        return self.x_data[idx].to(self.device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, flush = True)
# model = VAE_Large(150, 0, 4, 4)
# model.to(device)
# model = VAE_Large(150, 0, 4, 4).to(device)
# temp_model = torch.load("/work2/phillip/shallow-water-assimilation/checkpoints/pengpeng_ver_350.ckpt")
# model.load_state_dict(temp_model.state_dict())
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train_dataset = CustomDataset(X_train, device)
# valid_dataset = CustomDataset(X_test, device)
# def train_epoch(model, optimizer, train_loader):
#     losses = []
#     model.train()
#     with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
#         for i, value in enumerate(train_loader):
model = KLModel_Dynamics_Linear(3, 1, 8, 4)
temp_model = torch.load("./checkpoints/ldm_linear_dynamics_400.ckpt")
model.load_state_dict(temp_model.state_dict())
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataset = CustomDataset(X_train, device)
valid_dataset = CustomDataset(X_test, device)
def train_epoch(model, optimizer, train_loader):
    losses = []
    model.train()
    with tqdm(total=len(train_loader), desc=f"Train {epoch}: ") as pbar:
        for i, value in enumerate(train_loader):
            x = value
            target = x
            optimizer.zero_grad()
            x_hat, mean, log_var, mean_dynamics = model(target)
            x_hat_sample, mean_sample, log_var_sample = model.forward_sample(target[:, :, :, ::15, ::15])
            loss_target = loss_function(target, x_hat, mean, log_var)
            # print(mseloss(mean[:, 1:], mean_dynamics[:, :-1])/torch.sum(mean[:, 1:]**2))
            loss_sample =  mseloss(target, x_hat_sample) + mseloss(mean_sample, mean)+ mseloss(mean[:, 1:], mean_dynamics[:, :-1])# + maeloss(log_var_sample,  log_var)
            loss = loss_target + loss_sample#kld(mean_sample, log_var_sample, mean, log_var) + kld(mean, log_var, mean_sample, log_var_sample)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)
def valid_epoch(model, valid_loader):
    losses = []
    model.eval()
    with tqdm(total=len(valid_loader), desc=f"Valid {epoch}: ") as pbar:
        for i, value in enumerate(valid_loader):
            x = value
            target = x
            optimizer.zero_grad()
            with torch.no_grad():
                x_hat, mean, log_var, mean_dynamics = model(target)
                x_hat_sample, mean_sample, log_var_sample = model.forward_sample(target[:, :, :, ::15, ::15])
                loss_target = loss_function(target, x_hat, mean, log_var)
                loss_sample =  mseloss(target, x_hat_sample) + mseloss(mean_sample, mean)+ mseloss(mean[:, 1:], mean_dynamics[:, :-1])# + maeloss(log_var_sample,  log_var)
                loss = loss_target + loss_sample#kld(mean_sample, log_var_sample, mean, log_var) + kld(mean, log_var, mean_sample, log_var_sample)
                loss_val = loss.item()
                losses.append(loss_val)
            # scheduler.step()
            pbar.update(1)
            pbar.set_postfix_str(
                f"Loss: {loss_val:.3f} ({np.mean(losses):.3f}))")
    return np.mean(losses)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)
# temp_model = torch.load("./checkpoints/large_simplified_len5_onlyvae_799.ckpt")
# model.load_state_dict(temp_model.state_dict())
# print("loaded model")
for epoch in range(1, 401):
    print('EPOCH {}:'.format(epoch + 1))
    if epoch == 200:
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)
    if epoch == 250:
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-5)
    if epoch == 300:
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-6)
    if epoch == 350:
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-7)
    train_epoch(model, optimizer, train_loader)
    valid_epoch(model, valid_loader)
    # if (epoch + 1)%50 == 0:
    #     print("done")
    if epoch % 50 == 0:
        # torch.save(model, f"./checkpoints/test_{epoch}.ckpt")
    #     print("done")
        torch.save(model, f"./checkpoints/ldm_linear_dynamics_{epoch}.ckpt")
