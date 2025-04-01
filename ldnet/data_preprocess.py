import numpy as np

# select space points to reduce input dimension 
# not necessary when using iterator
def select_space_subset(data, num_points):
    assert len(data["x"].shape) == 4 
    assert len(data["y"].shape) == 4 

    Ni = data["y"].shape[0]
    Nt = data["y"].shape[1]
    Nx = data["y"].shape[2]
    indices = np.random.choice(Nx, (Ni, Nt, num_points))

    data["x"] = data["x"][np.arange(Ni)[:, None, None],
                        np.arange(Nt)[None, :, None],
                        indices, :]
    
    data["y"] = data["y"][np.arange(Ni)[:, None, None],
                        np.arange(Nt)[None, :, None],
                        indices, :]
    
    return data

def select_time_subset(data, num_time):
    assert len(data["x"].shape) == 4 
    assert len(data["y"].shape) == 4 
    
    # this method require to train with full dataset
    Ni = data["y"].shape[0]
    Nt = data["y"].shape[1]
    indices = np.random.choice(Nt, (Ni, num_time))
    
    data["x"] = data["x"][np.arange(Ni)[:, None],
                        indices, :, :]
    
    data["y"] = data["y"][np.arange(Ni)[:, None],
                        indices, :, :]
    
    data["time_indices"] = indices
    
    return data

class DataPreprocessor():
    """recieve the whole dataset and split it into train, valid, test
    dataset["x"]: (Ni, Nt, Nx, dim_x)
    dataset["y"]: (Ni, Nt, Nx, dim_y)
    dataset["u"]: (Ni, Nt, dim_u)
    dataset["dt"]: ()
    """
    def __init__(
        self,
        dataset: dict, # we name the whole data as dataset
        prop_train: float = 0.60,
        prop_valid: float = 0.20,
        prop_test: float = 0.20
    ):
        assert len(dataset["x"].shape) == 4 
        assert len(dataset["y"].shape) == 4 
            
        self.dataset = dataset
        self.Ni = dataset["y"].shape[0] # number of functions
        
        self.prop_train = prop_train
        self.prop_valid = prop_valid
        self.prop_test = prop_test  
        
        train_size = int(self.prop_train * self.Ni)
        valid_size = int(self.prop_valid * self.Ni)
        test_size = int(self.prop_test * self.Ni)
        
        # randomly select train, valid, test from dataset
        indices = np.random.permutation(self.Ni)
        self.train_indices = indices[:train_size]
        self.valid_indices = indices[train_size:train_size+valid_size]
        self.test_indices = indices[train_size+valid_size:test_size+train_size+valid_size]
    
    def _get_data(self, indices):
        data = {key: value for key, value in self.dataset.items()}
        data["u"] = data["u"][indices]
        data["x"] = data["x"][indices]
        data["y"] = data["y"][indices]
        return data

    def get_train_data(self):
        return self._get_data(self.train_indices)
    
    def get_valid_data(self):
        return self._get_data(self.valid_indices)

    def get_test_data(self):
        return self._get_data(self.test_indices) 
    

