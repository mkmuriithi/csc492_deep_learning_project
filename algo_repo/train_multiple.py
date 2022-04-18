from train import *
import tqdm

def train_multi_batch_per_ticker(model, data_dict, optimizer='adam', batch_size=8, learning_rate=1e-7, num_epochs=10,
                 weight_decay=0.1):
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    iters, train_losses, val_losses, baseline_losses = [], [], [], []
    mse = nn.L1Loss(reduction="mean")
    criterion = torch.nn.MSELoss(reduction='mean')
    
    #note that stocks that don't have at least 3 years worth of training data will not be included
    n = 0
    for epoch in range(0, num_epochs):
        print(f'Epoch {epoch} training beginning...')
        # annotate for evaluation
        model.train()
        for iter_num, ticker in enumerate(data_dict.keys()):
            data = data_dict[ticker]
            train_custom = None
            valid_custom = None
            test_custom = None

            try:
                train_custom, valid_custom, test_custom = split_data(data, window=60, minmax=False)
            except ValueError:
                print(f'The stock that is causing issues is {ticker}')

            # preserving reproducability in dataloader with
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

            train_loss = []
            for X, y, X_baseline, y_baseline in train_dataloader:
                mask = torch.zeros(X.shape[1], X.shape[1])
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                    mask = mask.cuda()
    
                out = model(X, mask)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss.item()) # track training loss
                
            train_loss = np.mean(train_loss) # save iteration training loss
            
            if n % 100 == 0:
    
                # annotate for evaluation
                model.eval()
                with torch.no_grad():
                    val_loss = []
                    for X, y, X_baseline, y_baseline in val_dataloader:
                        mask = torch.zeros(X.shape[1], X.shape[1])
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y = y.cuda()
                            mask = mask.cuda()
                
                        out = model(X, mask)
                        loss = criterion(out, y)
                        val_loss.append(loss.item()) # track validation loss
                
                    val_loss = np.mean(val_loss) # save iteration validation loss
                
                # save current training info
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                baseline_losses.append(get_last_price_close_mse(y_baseline))
                # train_acc.append(get_accuracy(model, train_custom, train=True))
                # val_acc.append`(get_accuracy(model, valid_custom, train=False))
                print(f'Epoch {epoch} Iteration {iter_num}/{len(data_dict.keys())} (Ticker: {ticker}) | Train Loss: {round(train_losses[-1], 4)} | Val Loss: {round(val_losses[-1], 4)} | Baseline Loss: {round(baseline_losses[-1], 4)}')

            n += 1

    final_loss = train_losses[-1]
    print(f'Final Training Loss: {final_loss}')
    # print(f'Final Validation Loss {val_losses[-1]}')
    # graph loss
    plt.title(f"Training Curve (lr={learning_rate}, wd={weight_decay})")
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.plot(baseline_losses, label='baseline')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    fig_datetime = datetime.now().strftime("multi_batch_figures/fig_date_%m_%d_%Y_time_%H_%M")
    plt.savefig(fig_datetime, dpi=300, bbox_inches='tight')
    plt.show()

    final_training_loss = train_losses[-1]
    final_validation_loss = val_losses[-1]
    average_training_loss = np.mean(train_losses)
    log_datetime = datetime.now().strftime("Date: %m/%d/%Y\nTime: %H:%M")
    '''logging.info(
        f'\n{log_datetime}\nFinal Train Loss: {final_training_loss}\nFinal Validation Loss: {final_validation_loss} \n '
        f'Average Train Loss: {average_training_loss}\n\n')'''
    logging.info(
        f'\n{log_datetime}\nFinal Train Loss: {final_training_loss}\n '
        f'Average Train Loss: {average_training_loss}\n\n')
    # pickle model

    model_name = "multi_batch_model_pickles/model_date_%m_%d_%Y_time_%H_%M.pt"
    model_name = datetime.now().strftime(model_name)
    torch.save(model.state_dict(), model_name)

    val_losses = []
    return train_losses, val_losses, baseline_losses, iters


def train_multi_average_params(model, data, optimizer='adam', batch_size=8, learning_rate=1e-7, num_epochs=10,
                 weight_decay=0.1):
    # create training, valid and test sets of StockDataset type data
    train_custom, valid_custom, test_custom = split_data(data, window=60, minmax=True)
    # normalize data

    # preserving reproducability in dataloader with
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
    iters, train_losses, val_losses, baseline_losses = [], [], [], []
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
            train_loss = loss.item()  # save training loss

            if (n % 10 == 0):
                # annotate for evaluation
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
                        train_loss.append(loss.item())  # save validation loss

                    for data in val_dataloader:
                        X, y, X_baseline, y_baseline = data
                        mask = torch.zeros(X.shape[1], X.shape[1])
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y = y.cuda()
                            mask = mask.cuda()

                        out = model(X, mask)
                        loss = criterion(out, y)
                        val_loss.append(loss.item())  # save validation loss

                    val_loss = np.mean(val_loss)  # mean reduction
                    train_loss = np.mean(train_loss)

                # save current training info
                iters.append(n)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                baseline_losses.append(get_last_price_close_mse(y_baseline))
                # train_acc.append(get_accuracy(model, train_custom, train=True))
                # val_acc.append`(get_accuracy(model, valid_custom, train=False))
                # train_losses.append(loss.item())  # average loss
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
    logging.info(
        f'\n{log_datetime}\nFinal Train Loss: {final_training_loss}\nFinal Validation Loss: {final_validation_loss} \n '
        f'Average Train Loss: {average_training_loss}\n\n')

    # pickle model

    model_name = "model_pickles/model_date_%m_%d_%Y_time_%H:%M.pt"
    model_name = datetime.now().strftime(model_name)
    torch.save(model.state_dict(), model_name)

    return train_losses, val_losses, baseline_losses, iters


def treat_multiple_stock(all_data):
    # date is a feature
    # EAI was removed cause data was bad
    for stock_name in all_data.keys():
        all_data[stock_name] = treat_single_stock(all_data[stock_name])
    return all_data

if __name__ == '__main__':
    logging.basicConfig(filename="training.log", filemode='a', format='%(levelname)s: %(message)s', level=logging.INFO)

    try:
        if not os.path.exists('multi_batch_figures'):
            os.makedirs('multi_batch_figures')
    except Exception as e:
        print("An exception occurred while trying to create path for figures to be stored in")
    try:
        if not os.path.exists('multi_batch_model_pickles'):
            os.makedirs('multi_batch_model_pickles')
    except Exception as e:
        print("An exception occurred while trying to create path for model pickles to be stored in")

    data_dict = get_dataset(single=False, subset=False)
    data_dict = treat_multiple_stock(data_dict)

    model = TransformerModel(transf_params)

    if torch.cuda.is_available():
        model = model.cuda()

    train, val_losses, baseline_losses, iters = train_multi_batch_per_ticker(model, data_dict)
    #train, val_losses, baseline_losses, iters = train_multi_average_params(model, data)

