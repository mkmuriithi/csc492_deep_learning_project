from typing import List, Callable
import os, csv, random, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from data_loaders import *

WINDOW_SIZE = 30
TRAIN_PATH = "../Data/train_stocks.csv"
VALID_PATH = "../Data/valid_data.npy"
VALID2_PATH = "../Data/valid2_data.npy"
TEST_PATH = "../Data/test_data.npy"
TEST2_PATH = "../Data/test2_data.npy"

train_loader = TrainLoader(TRAIN_PATH)
valid_loader = ValidTestLoader(VALID_PATH)
valid2_loader = ValidTestLoader(VALID2_PATH)
test_loader = ValidTestLoader(TEST_PATH)
test2_loader = ValidTestLoader(TEST2_PATH)

random.seed(time.time())

def create_batch(dataset: List[str], n: int, seq_len: int, start_date=""):
    """
    Creates n batches with size determined by size_dist() 
    by pulling from windows in the dataset that begin after start_date
    
    >>> create_batch(dataset=train, n=8, seq_len=30)
    Output is an array of size (8,30,5) and an array of size (8,30)

    """
    
    batch_x = []
    batch_t = []
    
    # Sample stocks with replacement
    stocks = random.choices(dataset, k=n)
    for stock in stocks:
        with open(os.path.join("..\\Data", stock + ".csv",)) as file:
            # Read lines and delete header line 
            # Columns: Date, Open, High, Low, Close, Adj Close, Volume, ticker
            stock_data = np.array([line.split(",") for line in file.readlines()])[1:,:]
            
            # Remove Adj Close and ticker columns
            stock_data = np.concatenate([stock_data[:,:5], stock_data[:,6:7]], axis=1)
            
            # Filter rows by date and select start index
            stock_data = stock_data[stock_data[:,0] >= start_date]
            start_index = random.randint(0, stock_data.shape[0] - seq_len - 1)
            
            # Remove date column, save time series and target value
            x, t = (stock_data[np.newaxis, start_index:start_index + seq_len,1:],
                    stock_data[np.newaxis, start_index + seq_len,4:5])
            
            batch_x.append(x.astype(float))
            batch_t.append(t.astype(float))
            
    batch_x = torch.Tensor(batch_x)[:,0]
    batch_t = torch.Tensor(batch_t)[:,0]
    
    return batch_x, batch_t
            
def batch_normalization(batch_x, batch_t):
    """
    Normalize batches as described below:
        
        Volume feature: 0-1 normalization
        Other features: relative difference from previous step, 
                        first step always == 1.0

    """
    
    # Normalize price-based features (% of previous day)
    batch_t = batch_t / batch_x[:,-1,3:4]
    #batch_x[:,:,:-1] = batch_x[:,:,:-1] / torch.concat([batch_x[:,0:1,:-1], batch_x[:,:-1,:-1]], dim=1)
    
    # Normalize volume
    #batch_t[:,-1] = batch_t[:,-1] / torch.max(batch_x[:,:,-1], axis=1)[0]
    #batch_x[:,:,-1] = batch_x[:,:,-1] / torch.unsqueeze(torch.max(batch_x[:,:,-1], axis=1)[0], dim=1)
    
    return torch.Tensor(batch_x), torch.Tensor(batch_t)

class RNNModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        super().__init__()
        self.rnn = nn.RNN(5, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size // 2, 1))
        
    def forward(self, x):
        x = self.rnn(x)[0][:,-1]
        x = self.dense(x)
        return x
    
class MLPModel(nn.Module):
    def __init__(self, hidden_sizes=[4096, 4096, 4096, 1]):
        super().__init__()
        
        hidden_sizes = [30*5] + hidden_sizes
        layers = [nn.Flatten()]
        for i in range(len(hidden_sizes) - 1):
            if i > 0:
                layers.append(nn.Sigmoid())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

def eval_loss(model, data_loader, loss_fn):
    x, t = batch_normalization(*data_loader.load_data())
    if torch.cuda.is_available():
        x = x.to('cuda')
        t = t.to('cuda')
        
    y = model(x)
    loss = loss_fn(y, t)
        
    return loss

def train_model(model, train_loader=train_loader, valid_loader=valid_loader, 
                valid2_loader=valid2_loader, seq_len=WINDOW_SIZE, 
                num_iters=20, lrate=1e-3, weight_decay=0):
    
    train_losses = []
    valid_losses = []
    valid2_losses = []
    
    loss_fn = (lambda y, t: 
               torch.sqrt(nn.MSELoss()(y, t))
               )
    optim = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
    
    for i in range(num_iters):
        
        # Training step
        optim.zero_grad()
        train_loss = eval_loss(model, train_loader, loss_fn)
        train_loss.backward()
        optim.step()
            
        # Evaluate loss
        train_loss = eval_loss(model, train_loader, loss_fn).item()
        valid_loss = eval_loss(model, valid_loader, loss_fn).item()
        valid2_loss = eval_loss(model, valid2_loader, loss_fn).item()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid2_losses.append(valid2_loss)
        print(f"Epoch {i+1} | Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f} | Valid2 Loss {valid2_loss:.4f}")
        
    return train_losses, valid_losses, valid2_losses
   
model = MLPModel().cuda()
train_losses, valid_losses, valid2_losses = train_model(model)
plt.plot(train_losses, label="Train")
plt.plot(valid_losses, label="Valid")
plt.plot(valid2_losses, label="Valid2")
plt.legend()
plt.title("MLP Model")
plt.show()
    