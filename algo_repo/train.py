import torch.nn as nn
import torch.optim as optim
import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
from StockDataLoader import StockDataset
from sklearn.model_selection import train_test_split
from utils import *
from data_stuff import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
import logging
import random
import os
import pickle

#setting seed for torch random number generator, for reproducability
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Linear(params.input_dim, params.model_dim)
        self.pos_encoder = PositionalEncoding(params.model_dim, params.dropout)
        encoder_layers = TransformerEncoderLayer(d_model=params.model_dim, nhead=params.num_heads,
                                                 dim_feedforward=params.forward_dim, dropout=params.dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, params.n_layers)
        self.d_model = params.model_dim
        self.decoder = nn.Linear(params.model_dim * 2, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = torch.concat([torch.max(output, dim=1)[0], torch.mean(output, dim=1)], dim=1)
        output = self.decoder(output)
        return output


class transf_params:
    n_layers = 4
    num_heads = 8
    input_dim = 10
    model_dim = 512 #embed dim
    forward_dim = 2048
    dropout = 0
    n_epochs = 10
    lr = 0.01

def train(model, data, optimizer='adam', batch_size=8, learning_rate=1e-7, num_epochs=10,
          weight_decay=0.1):
    # create training, valid and test sets of StockDataset type data
    train_custom, valid_custom, test_custom = split_data(data, window=60, minmax=True)
    # normalize data

    #preserving reproducability in dataloader with
    g = torch.Generator()
    g.manual_seed(42)
    # create loaders
    train_dataloader = DataLoader(train_custom, batch_size=16,
                                  shuffle=True,
                                  worker_init_fn=seed_worker,
                                  generator=g)  # returns the X and associated y prediction
    val_dataloader = DataLoader(valid_custom, batch_size=16, shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g)  # does same

    optimizer = optim.Adam(model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay)

    print(f'The length of the train dataloader is {len(train_dataloader)}\n'
          f'The length of the validation dataloader is {len(val_dataloader)}')
    # track learning curve
    mse = nn.L1Loss(reduction="mean")
    # criterion = lambda y, t: torch.sqrt(mse(y, t))
    criterion = torch.nn.MSELoss(reduction='mean')
    iters, train_losses, val_losses, baseline_losses =  [], [], [], []
    # train
    
    n = 0
    for epoch in range(0, num_epochs):
        print(f'Epoch {epoch} training beginning...')
        for data in train_dataloader:
            X, y, X_baseline, y_baseline = data
            mask = torch.zeros(X.shape[1], X.shape[1])
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                mask = mask.cuda()

            model.train()  # annotate for train
            out = model(X, mask)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.item() # save training loss
            
            if (n % 10 == 0):
                #annotate for evaluation
                model.eval()
                train_loss = []
                val_loss = []
                with torch.no_grad():
                    
                    for data in train_dataloader:
                        X, y, X_baseline, y_baseline = data
                        mask = torch.zeros(X.shape[1], X.shape[1])
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y = y.cuda()
                            mask = mask.cuda()
                            
                        out = model(X, mask)
                        loss = criterion(out, y)
                        train_loss.append(loss.item()) # save validation loss
                    
                    for data in val_dataloader:
                        X, y, X_baseline, y_baseline = data
                        mask = torch.zeros(X.shape[1], X.shape[1])
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y = y.cuda()
                            mask = mask.cuda()
                            
                        out = model(X, mask)
                        loss = criterion(out, y)
                        val_loss.append(loss.item()) # save validation loss
                        
                    val_loss = np.mean(val_loss) # mean reduction
                    train_loss = np.mean(train_loss)
                        
                #save current training info
                iters.append(n)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                baseline_losses.append(get_last_price_close_mse(y_baseline))
                #train_acc.append(get_accuracy(model, train_custom, train=True))
                #val_acc.append`(get_accuracy(model, valid_custom, train=False))
                #train_losses.append(loss.item())  # average loss
                print(f'Iteration: {n}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}')
            n += 1

            # predict validation

    final_loss = train_losses[1]
    print(f'Final Training Loss: {final_loss}')
    # print(f'Final Validation Loss {val_losses[-1]}')
    # graph loss
    plt.title(f"Training Curve (lr={learning_rate}, wd={weight_decay})")
    plt.plot(iters, train_losses, label='Train')
    plt.plot(iters, val_losses, label='Validation')
    plt.plot(iters, baseline_losses, label='baseline')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    fig_datetime = datetime.now().strftime("figures/fig_date_%m_%d_%Y_time_%H:%M")
    plt.savefig(fig_datetime, dpi=300, bbox_inches='tight')
    plt.show()

    final_training_loss = train_losses[-1]
    final_validation_loss = val_losses[-1]
    average_training_loss = np.mean(train_losses)
    log_datetime = datetime.now().strftime("Date: %m/%d/%Y\nTime: %H:%M")
    logging.info(f'\n{log_datetime}\nFinal Train Loss: {final_training_loss}\nFinal Validation Loss: {final_validation_loss} \n '
                 f'Average Train Loss: {average_training_loss}\n\n')

    return train_losses, val_losses, baseline_losses, iters


def get_last_price_close_mse(y):
    #note that y is a batch of ys. Need to iterate through the batches

    avg_mse = 0
    num_batches = 0

    for batch in y:
        #print(batch.reshape(-1))
        new_batch = batch.reshape(-1)
        num_batches += 1
        df_y = pd.Series(new_batch)
        df_y.name = "Close"
        df_pred = df_y.shift(1)
        df_pred.name = "Predicted"
        new_df = df_y.to_frame().join(df_pred)
        new_df.columns = ["Real Closing Price", "Predicted Closing"]
        new_df.dropna(inplace=True)
        mse = mean_squared_error(new_df["Real Closing Price"], new_df["Predicted Closing"])
        avg_mse += mse
    avg_mse = avg_mse / num_batches
    return avg_mse
    
def save_params(model):
    torch.save(model.state_dict, "model.pt")

    
def load_params():
    return torch.load("model.pt")
def seed_worker(worker_id):
    '''
    Function used to preserve reproducability, as Dataloader reseeds workers
    Inspired by https://pytorch.org/docs/stable/notes/randomness.html
    '''
    np.random.seed(42)
    random.seed(42)

def plot_predictions(model, data):
    train_custom, valid_custom, test_custom = split_data(data, window=60, minmax=True)
    train_dataloader = DataLoader(train_custom, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(valid_custom, batch_size=1, shuffle=True)
    
    criterion = torch.nn.MSELoss(reduction='mean')
    
    train_truth=[]
    train_pred=[]
    train_loss=0
    val_truth=[]
    val_pred=[]
    val_loss=0
    model.eval()
    
    with torch.no_grad():
        
        for data in val_dataloader:
            X, y, X_baseline, y_baseline = data
            mask = torch.zeros(X.shape[1], X.shape[1])
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                mask = mask.cuda()
                
            # add ground-truth and prediction
            pred = model(X, mask)
            val_loss += criterion(pred, y).item()
            val_truth.append(y.item())
            val_pred.append(pred.item())
            
        val_loss /= len(val_dataloader)
            
        for data in train_dataloader:
            X, y, X_baseline, y_baseline = data
            mask = torch.zeros(X.shape[1], X.shape[1])
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                mask = mask.cuda()
                
            # add ground-truth and prediction
            pred = model(X, mask)
            train_loss += criterion(pred, y).item()
            train_truth.append(y.item())
            train_pred.append(pred.item())
            
        train_loss /= len(train_dataloader)
            
    print(f"Training Loss: {train_loss}")
    plt.title(f"Model Prediction vs Ground Truth: Training Data")
    plt.plot(train_truth, label="Ground Truth")
    plt.plot(train_pred, label="Model Prediction")
    plt.legend()
    plt.xlabel("Time (Days)")
    plt.ylabel("Normalized Closing Price")
    plt.show()
        
    print(f"Validation Loss: {val_loss}")
    plt.title(f"Model Prediction vs Ground Truth: Validation Data")
    plt.plot(val_truth, label="Ground Truth")
    plt.plot(val_pred, label="Model Prediction")
    plt.legend()
    plt.xlabel("Time (Days)")
    plt.ylabel("Normalized Closing Price")
    plt.show()

def treat_single_stock(data):
    data.reset_index(inplace=True)
    dayoftheweek = data['Date'].dt.dayofweek + 1
    data['Date'] = pd.Series(dayoftheweek)
    data['Target'] = data["Close"].shift(-1)
    data = add_features(data)
    data = get_treated(data, to_daily_returns=True, features_to_exclude=['Date'])
    return data

def treat_multiple_stock(all_data):
    #date is a feature
    for stock_name in all_data.keys():
        all_data[stock_name] = treat_single_stock(all_data[stock_name])
    return all_data


if __name__ == '__main__':

    #configuring logger
    logging.basicConfig(filename="training.log", filemode='a', format='%(levelname)s: %(message)s', level=logging.INFO)

    try:
        if not os.path.exists('figures'):
            os.makedirs('figures')
    except Exception as e:
        print("An exception occured while trying to create path for figures to be stored in")
    try:
        if not os.path.exists('model_pickles'):
            os.makedirs('model_pickles')
    except Exception as e:
        print("An exception occured while trying to create path for model pickles to be stored in")


    #get list of stocks to train on
    #data = yf.download(tickers="AAPL", interval='1d', groupby='ticker', auto_adjust='True', start="2007-07-01")
    data = get_dataset(single=False, subset=False)
    #data = treat_single_stock(data)
    data = treat_multiple_stock(data)



    model = TransformerModel(transf_params)
    if torch.cuda.is_available():
        model = model.cuda()
    train_losses, val_losses, baseline_losses, iters = train(model, data)
    #pickle model for frontend



    model_name = "models_pickles/model_date_%m_%d_%Y_time_%H:%M.p"
    with open(model_name, 'wb') as model_save_location:
        pickle.dump(model, model_save_location)
        
