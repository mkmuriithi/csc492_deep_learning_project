import csv, random, os
import numpy as np

WINDOW_SIZE = 30
VALID_START = '2019-01-01'
        
class TrainLoader:
    
    # TRAIN_PATH = "../Data/train_stocks.csv"
    def __init__(self, filepath):
        with open(filepath) as f:
            reader = csv.reader(f)
            self.stocks = list(reader)
        
    def load_data(self):
        """
        x is a list of numpy arrays of size (B, 30, 5)
        t is a list of numpy arrays of size (B, 1)
        
        """
        data = np.zeros((0, WINDOW_SIZE+1, 5))
        
        for stock in self.stocks:
            with open(os.path.join("..", "Data", stock[0])) as f:
                lines = f.readlines()[1:]
                lines = [line.split(',') for line in lines]
                stock_data = np.array(lines)
                
                # Filter by date and remove extra features
                stock_data = stock_data[stock_data[:,0] < VALID_START]
                stock_data = np.concatenate([stock_data[:,1:5], stock_data[:,6:7]], axis=1).astype(float)
                
                # Add random window to batch
                index = random.randint(0, stock_data.shape[0] - (WINDOW_SIZE+1))
                data = np.concatenate([data, np.expand_dims(stock_data[index:index+WINDOW_SIZE+1,:], 0)], 0)
            
        return data[:,:-1,:], data[:,-1,3:4]

class ValidTestLoader:
    
    # VALID_PATH = "../Data/valid_data.npy"
    # VALID2_PATH = "../Data/valid2_data.npy"
    # TEST_PATH = "../Data/test_data.npy"
    # TEST2_PATH = "../Data/test2_data.npy"
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load_data(self):
        """
        x is a numpy array of size (N, 30, 5)
        t is a numpy array of size (N, 1)
        
        """
        data = np.load(self.filepath)
        return data[:,:-1,:], data[:,-1,3:4]
    
class SampleLoader:
    
    def load_data(self):
        data = np.zeros((1000, 31, 5))
        x = np.random.rand(1000, 5)
        v = np.random.rand(1000, 5)
        for i in range(data.shape[1]):
            data[:,i,:] = x
            x += v
        x = data[:,:-1,:]
        t = np.sum(data[:,-1,:], axis=1, keepdims=True)
        return x, t
            
    