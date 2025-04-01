import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_x = []
data_y = []
data_t = []
for i in range(20):
    data_x.append(np.load(f"x_data_{i}.npy", allow_pickle = True))
    data_y.append(np.load(f"y_data_{i}.npy", allow_pickle = True))
    data_t.append(np.load(f"t_data_{i}.npy", allow_pickle = True))
data_x = np.concatenate(data_x, axis = 0)
data_y = np.concatenate(data_y, axis = 0)
data_t = np.concatenate(data_t, axis = 0)
np.save(f"x_data_supplemental.npy", data_x, allow_pickle=True)
np.save(f"y_data_supplemental.npy", data_y, allow_pickle=True)
np.save(f"t_data_supplemental.npy", data_t, allow_pickle=True)