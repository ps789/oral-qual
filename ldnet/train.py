import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.autograd as grad
from .dataloader import BranchDataset, DataLoaderX

"""
In this version, we use one DataLoader to load the data for branch and trunk batches. 
And backward for every trunk batch.
"""

class Trainer():
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        batch_size: int = 10,
        device = "cpu",
        lr_scheduler = None,
        equilibrium = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.criterion = criterion
        self.equilibrium = equilibrium

        self.batch_size = batch_size
        
        # self.data_train = data_train
        # self.data_valid = data_valid
        # self.num_functions = data_train["u"].shape[0]

    def _train_or_eval(self, dataset, mode="train"):
        """
        Perform training, validation or test on the given dataset.

        :param dataset: The dataset to be used in training, validation or test.
        :param mode: Specify the operation mode, 'train', 'valid' or 'test.
        :return: Result of training, validation or test.
        """
        
        assert mode in ["train", "valid", "test"], "Invalid mode."
        
        is_train = mode == "train"
        is_valid = mode == "valid"
        is_test = mode == "test"
        
        data = BranchDataset(dataset, self.device)
        dataloader = DataLoaderX(data, batch_size=self.batch_size, 
                            collate_fn=BranchDataset.collate_fn, 
                            shuffle=True, pin_memory=True, num_workers=4)
        
        if is_valid or is_test:
            self.model.eval()
            sum_squared_error = 0
            sum_squared_true = 0
        else:
            self.model.train()
        
        print_losses =  0.0
        
        for data_i in dataloader:
            
            data_i = {key: data_i[key].to(self.device, non_blocking=True) for key in data_i.keys()}
            
            if is_train:
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()
            
            outputs_y = self.model(data_i, self.device, self.equilibrium)
            
            if is_train or is_valid:
                loss = self.criterion(outputs_y, data_i["y"])

            if is_valid or is_test:
                # the l2 error is eveluated for every batch
                error = outputs_y - data_i["y"]
                sum_squared_error += (error ** 2).sum().item()
                sum_squared_true += (data_i["y"] ** 2).sum().item()

            if is_train:
                loss.backward()
                self.optimizer.step()
            
            if is_train or is_valid:
                print_losses = print_losses + loss.item()

        if is_train or is_valid:
            avg_print_loss = print_losses / len(dataloader) 

        if is_train:
            self.lr_scheduler.step()
            return avg_print_loss
        elif is_valid:
            l2_error = np.sqrt(sum_squared_error / sum_squared_true)
            return avg_print_loss, l2_error
        elif is_test:
            l2_error = np.sqrt(sum_squared_error / sum_squared_true)
            return l2_error
    
    @torch.no_grad()
    def valid(self, dataset):
        return self._train_or_eval(dataset, mode="valid")
    
    @torch.no_grad()
    def test(self, dataset):
        return self._train_or_eval(dataset, mode="test")
        
    def _train(self, dataset):
        return self._train_or_eval(dataset, mode="train")
    
    def train(self, data_train, data_valid, num_epochs=1000, eval_interval=10, save_path=None):
        for epoch in range(num_epochs):
            loss = self._train(data_train)
            
            if (epoch+1) % eval_interval == 0:
                valid_loss, valid_error = self.valid(data_valid)
            
                log_dict = {"epoch": epoch}
                log_dict["train_loss"] = loss   
                log_dict["valid_loss"] = valid_loss
                log_dict["valid_error"] = valid_error
                wandb.log(log_dict, step=epoch)
            
            
            if (epoch+1) % 500 == 0 and save_path is not None:
                    torch.save(self.model.dyn.state_dict(), save_path / f"dyn_{epoch}.ckpt")
                    torch.save(self.model.rec.state_dict(), save_path / f"rec_{epoch}.ckpt")
                    print(f"Model saved to {save_path}")
                    # wandb.save(save_path, base_path=save_path)
                
                
