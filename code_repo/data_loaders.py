import csv, random, os
import numpy as np

WINDOW_SIZE = 30
VALID_START = '2019-01-01'

class ValidTestLoader:
    
    # VALID_PATH = "../Data/valid_data.npy"
    # TEST_PATH = "../Data/test_data.npy"
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load_data(self):
        data = np.load(self.filepath)
        return data[:,:-1,:], data[:,-1,3:4]
        
class TrainLoader:
    
    def __init__(self, filepath="../Data/train_stocks.csv"):
        with open(filepath) as f:
            reader = csv.reader(f)
            self.stocks = list(reader)
        
    def load_data(self, batch_size):
        x = []
        t = []
        random.shuffle(self.stocks)
        for i in range(0, len(self.stocks), batch_size):
            batch = np.zeros((0, WINDOW_SIZE+1, 5))
            for stock in self.stocks[i:i+batch_size]:
                with open(os.path.join("..", "Data", stock[0])) as f:
                    lines = f.readlines()[1:]
                    lines = [line.split(',') for line in lines]
                    data = np.array(lines)
                    
                    # Filter by date and remove extra features
                    data = data[data[:,0] < VALID_START]
                    data = np.concatenate([data[:,1:5], data[:,6:7]], axis=1).astype(float)
                    
                    # Add random window to batch
                    index = random.randint(0, data.shape[0] - (WINDOW_SIZE+1))
                    batch = np.concatenate([batch, np.expand_dims(data[index:index+WINDOW_SIZE+1,:], 0)], 0)
                    
            x.append(batch[:,:-1,:])
            t.append(batch[:,-1,3:4])
            
        return x, t