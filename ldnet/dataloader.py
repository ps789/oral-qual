import torch
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

class BranchDataset(Dataset):
    def __init__(
        self,
        data,
        device
        ):
        self.data = data
        self.device = device
        
            
    def __len__(self):
        return self.data["y"].shape[0]
    
    def __getitem__(self, idx):
        new_data = {
            "u": torch.from_numpy(self.data["u"][idx]),
            "x": torch.from_numpy(self.data["x"][idx]),
            "y": torch.from_numpy(self.data["y"][idx]),
            "dt": torch.from_numpy(self.data["dt"])
        }
        return new_data
    
    def collate_fn(batch):
        # Your custom collation logic
        u_stack = torch.stack([item["u"] for item in batch], dim=0)
        x_stack = torch.stack([item["x"] for item in batch], dim=0)
        y_stack = torch.stack([item["y"] for item in batch], dim=0)
        dt_stack = batch[0]["dt"]

        return {
            "u": u_stack,
            "x": x_stack,
            "y": y_stack,
            "dt": dt_stack
        }
    
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())