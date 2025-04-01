import torch.nn as nn
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5  = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p = 0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.dropout(self.LeakyReLU(self.layer1(x)))
        h_       = self.dropout(self.LeakyReLU(self.layer2(h_)))
        h_       = self.dropout(self.LeakyReLU(self.layer3(h_)))
        h_       = self.dropout(self.LeakyReLU(self.layer4(h_)))
        h_       = self.layer5(h_)
        return h_


class Encoder_CNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim_cnn, hidden_dim, latent_dim):
        super(Encoder_CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(input_dim, hidden_dim_cnn, kernel_size = 5, padding = "same")
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_2 = nn.Conv2d(hidden_dim_cnn, hidden_dim_cnn, kernel_size = 3, padding = "same")
        self.conv2_2 = nn.Conv2d(hidden_dim_cnn, hidden_dim_cnn * 2, stride = 2, kernel_size = 3, padding = 1)
        self.conv1_3 = nn.Conv2d(hidden_dim_cnn*2, hidden_dim_cnn*2, kernel_size = 3, padding = "same")
        self.conv1_4 = nn.Conv2d(hidden_dim_cnn*2, hidden_dim_cnn*2, kernel_size = 3, padding = "same")
        self.conv2_4 = nn.Conv2d(hidden_dim_cnn*2, hidden_dim_cnn*2, kernel_size = 3, padding = "same")

        self.layer1 = nn.Linear(hidden_dim_cnn*2*5*5, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3  = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p = 0.2)
        
        self.training = True
        
    def forward(self, x):
        x = self.lrelu(self.conv1_1(x))
        x = x + self.lrelu(self.conv1_2(x))
        x = self.lrelu(self.conv2_2(x))
        x = x + self.lrelu(self.conv1_3(x))
        x = x + self.lrelu(self.conv1_4(x))
        x = self.conv2_4(x)
        x = x.reshape(x.shape[0], -1)
        h_       = self.dropout(self.LeakyReLU(self.layer1(x)))
        h_       = h_ + self.dropout(self.LeakyReLU(self.layer2(h_)))
        h_       = self.dropout(self.layer3(h_))
        return h_

class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output

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