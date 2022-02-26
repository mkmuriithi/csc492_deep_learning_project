from typing import List, Callable
import os, csv, random, time
import numpy as np

import torch
import torch.nn as nn

from tqdm import trange

WINDOW_SIZE = 30
VALID_START = '2019-01-01'
TEST_START = '2020-06-01'

datasets = []

with open("../Data/datasets.csv") as file:
    reader = csv.reader(file)
    for row in reader:
         datasets.append(row)
        
train, valid, valid2, test, test2 = datasets

random.seed(time.time())

def create_batch(dataset: List[str], n: int, seq_len: int, start_date=""):
    """
    Creates n batches with size determined by size_dist() 
    by pulling from windows in the dataset that begin after start_date
    
    >>> create_batch(dataset=train, n=8, seq_len=30)
    Output is an array of size (8,30,5) and an array of size (8,30)

    """
    
    batch_x = np.ndarray((0,seq_len,5))
    batch_t = np.ndarray((0,1))
    
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
            
            batch_x, batch_t = (np.concatenate([batch_x, x], axis=0).astype(float),
                                np.concatenate([batch_t, t], axis=0).astype(float))
            
    batch_x = torch.Tensor(batch_x)
    batch_t = torch.Tensor(batch_t)
    
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
    batch_x[:,:,:-1] = batch_x[:,:,:-1] / torch.concat([batch_x[:,0:1,:-1], batch_x[:,:-1,:-1]], dim=1)
    
    # Normalize volume
    #batch_t[:,-1] = batch_t[:,-1] / torch.max(batch_x[:,:,-1], axis=1)[0]
    #batch_x[:,:,-1] = batch_x[:,:,-1] / torch.unsqueeze(torch.max(batch_x[:,:,-1], axis=1)[0], dim=1)
    
    return batch_x, batch_t

class RNNModel(nn.Module):
    def __init__(self, hidden_size=5, num_layers=2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, 
                          batch_first=True, nonlinearity='relu')
        
    def forward(self, x):
        return self.rnn(x)[0][:,-1,3:4]

def eval_loss(model, train, valid, seq_len, loss):
    train_x, train_t = batch_normalization(*create_batch(train, 256, seq_len))
    train_x = train_x[:,:,:-1] # removes volume parameter
    if torch.cuda.is_available():
        train_x = train_x.to('cuda')
        train_t = train_t.to('cuda')
        
    train_y = model(train_x)
    train_loss = torch.sqrt(loss(train_y, train_t))
    
    valid_x, valid_t = batch_normalization(*create_batch(valid, 256, seq_len))
    valid_x = valid_x[:,:,:-1] # removes volume parameter
    if torch.cuda.is_available():
        valid_x = valid_x.to('cuda')
        valid_t = valid_t.to('cuda')
        
    valid_y = model(valid_x)
    valid_loss = loss(valid_y, valid_t)
        
    return train_loss.item(), valid_loss.item()

def train_model(model, train=train, valid1=valid, valid2=valid, batch_size=64, 
          seq_len=30, num_iters=1024, plot_iters=16, lrate=0.0001, weight_decay=0.0):
    
    train_losses = []
    valid_losses = []
    iter_steps = [0]
    
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
    
    train_loss, valid_loss = eval_loss(model, train, valid, seq_len, loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"Iteration 0 | Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f}")
    print(list(rnn.parameters())[0])
    
    for i in trange(num_iters):
        # Create training batch
        x, t = batch_normalization(*create_batch(train, batch_size, seq_len))
        x = x[:,:,:-1] # removes volume parameter
        if torch.cuda.is_available():
            x = x.to('cuda')
            t = t.to('cuda')
        
        optim.zero_grad()
        y = model(x)
        train_loss = loss(y, t)
        train_loss.backward()
        optim.step()
        
        # Compute losses
        if (i + 1) % min(plot_iters, num_iters) == 0:
            train_loss, valid_loss = eval_loss(model, train, valid, seq_len, loss)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            iter_steps.append(i+1)
            print(f"\nIteration {i+1} | Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f}")
            print(list(rnn.parameters())[0])
            
    return train_losses, valid_losses, iter_steps
    
rnn = RNNModel(hidden_size=4).cuda() # removes volume parameter
train_losses, valid_losses, iter_steps = train_model(rnn)
    