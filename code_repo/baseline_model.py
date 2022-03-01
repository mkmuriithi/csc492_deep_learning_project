from typing import List, Callable
import os, csv, random, time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import trange, tqdm

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
    
    return batch_x, batch_t

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
    def __init__(self, hidden_sizes=[4096, 256, 16, 1]):
        super().__init__()
        
        hidden_sizes = [30*5] + hidden_sizes
        layers = [nn.Flatten()]
        for i in range(len(hidden_sizes) - 1):
            if i > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

def eval_loss(model, data, batch_size, seq_len, loss, start_date=""):
    x, t = batch_normalization(*create_batch(data, batch_size, seq_len, start_date))
    if torch.cuda.is_available():
        x = x.to('cuda')
        t = t.to('cuda')
        
    y = model(x)
    loss = torch.sqrt(loss(y, t))
        
    return loss

def train_model(model, train=train, valid=valid, valid2=valid2, batch_size=128, 
          seq_len=30, num_iters=2048, plot_iters=16, lrate=1e-6, weight_decay=0):
    
    train_losses = []
    valid_losses = []
    valid2_losses = []
    iter_steps = [0]
    
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
    
    # Evaluate iteration 0 losses
    train_loss = eval_loss(model, train, batch_size, seq_len, loss)
    valid_loss = eval_loss(model, valid, batch_size, seq_len, loss, VALID_START)
    valid2_loss = eval_loss(model, valid2, batch_size, seq_len, loss, VALID_START)
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())
    valid2_losses.append(valid2_loss.item())
    print(f"Iteration 0 | Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f} | Valid2 Loss {valid2_loss:.4f}")
    
    for i in trange(num_iters):
        optim.zero_grad()
        train_loss = torch.sqrt(eval_loss(model, train, batch_size, seq_len, loss))
        train_loss.backward()
        optim.step()
        
        # Compute losses
        if (i + 1) % min(plot_iters, num_iters) == 0:
            train_loss = eval_loss(model, train, batch_size, seq_len, loss)
            valid_loss = eval_loss(model, valid, batch_size, seq_len, loss, VALID_START)
            valid2_loss = eval_loss(model, valid2, batch_size, seq_len, loss, VALID_START)
            train_losses.append(train_loss.item())
            valid_losses.append(valid_loss.item())
            valid2_losses.append(valid2_loss.item())
            iter_steps.append(i+1)
            print(f"\nIteration {i+1} | Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f} | Valid2 Loss {valid2_loss:.4f}")
            
    return train_losses, valid_losses, valid2_losses, iter_steps
   
""" 
valid_curves = []
base_rnn = RNNModel().cuda()
for lr in [1e-2, 1e-3, 1e-4]:
    rnn = RNNModel().cuda()
    rnn.load_state_dict(base_rnn.state_dict())
    train_losses, valid_losses, valid2_losses, iter_steps = train_model(rnn, lrate=lr, num_iters=512, plot_iters=4)
    valid_curves.append(valid_losses + [f"lr={lr} Valid"])
    valid_curves.append(valid2_losses + [f"lr={lr} Valid2"])
    
for valid_curve in valid_curves:
    plt.plot(iter_steps, valid_curve[:-1], label=valid_curve[-1])
plt.legend()
plt.title("Hyperparameter Tuning")
plt.show()
"""

model = MLPModel().cuda()
train_losses, valid_losses, valid2_losses, iter_steps = train_model(model)
plt.plot(iter_steps, train_losses, label="Train")
plt.plot(iter_steps, valid_losses, label="Valid")
plt.plot(iter_steps, valid2_losses, label="Valid2")
plt.legend()
plt.title("MLP Model")
plt.show()
    