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
from data_treatment import *


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
    input_dim = 6
    model_dim = 512
    forward_dim = 2048
    dropout = 0
    n_epochs = 10
    lr = 0.01


def train(model, data, optimizer='adam', batch_size=4, learning_rate=1e-3, momentum=0.9, num_epochs=10,
          weight_decay=0.0):
    # create training, valid and test sets of StockDataset type data
    train_custom, valid_custom, test_custom = split_data(data, window=30, minmax=True)
    # normalize data

    # create loaders
    train_dataloader = DataLoader(train_custom, batch_size=16,
                                  shuffle=False)  # returns the X and associated y prediction
    val_dataloader = DataLoader(valid_custom, batch_size=16, shuffle=False)  # does same
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    # track learning curve
    mse = nn.MSELoss(reduction="mean")
    criterion = lambda y, t: torch.sqrt(mse(y, t))
    iters, train_losses, val_losses = [], [], []
    # train
    for epoch in range(0, num_epochs):
        print(f'Epoch {epoch} training beginning...')
        for n, data in enumerate(iter(train_dataloader)):
            X, y = data
            mask = torch.zeros(X.shape[1], X.shape[1])
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                mask = mask.cuda()

            model.train()  # annotate for train
            out = model(X, mask)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iters.append(n)
            train_losses.append(loss.item())  # average loss
            if (n % 20 == 0):
                print(f'Iteration: {n}, Loss: {loss.item()}')

            # predict validation

            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in iter(val_dataloader):
                    mask = torch.zeros(X_val.shape[1], X_val.shape[1])
                    if torch.cuda.is_available():
                        X_val = X_val.cuda()
                        y_val = y_val.cuda()
                        mask = mask.cuda()
                    model.eval() #annotate for test
                    val_out = model(X_val, mask)
                    val_loss += criterion(val_out, y_val).item()
            val_losses.append(val_loss)

    print(f'Final Training Loss: {train_losses[-1]}')
    print(f'Final Validation Loss {val_losses[-1]}')
    # graph loss
    plt.title("Learning Loss")
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()

    return train_losses, val_losses


if __name__ == '__main__':

    import yfinance as yf

    data = yf.download(tickers="AAPL", period='10y', interval='1d', groupby='ticker', auto_adjust='True')
    data.reset_index(inplace=True)

    data.drop(columns=['Date'], inplace=True)
    data.index = data.index.set_names(["Date"])
    data.reset_index(inplace=True)  # to keep up with order
    data['Target'] = data["Close"].shift(-1)
    # data["Date"] = data["Date"].apply(lambda x: x.value / 10 ** 9)

    data = get_treated(data)

    # dataset_train = StockDataset(X_train, y_train, 14)
    # dataset_val = StockDataset(X_val, y_val, 14)
    # dataset_test = StockDataset(X_test, y_test, 14)

    # train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=False)
    # val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=False)

    model = TransformerModel(transf_params)
    if torch.cuda.is_available():
        model = model.cuda()
    train_losses, val_losses = train(model, data)
