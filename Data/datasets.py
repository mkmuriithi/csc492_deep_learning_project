import numpy as np
import os

WINDOW_SIZE = 30
VALID_START = '2019-01-01'
TEST_START = '2020-06-01'
paths = ['./nyse_15yr_data']

# Create data sets that include X and T
train = []
valid = []
test = []

for path in paths:
    files = os.listdir(path)
    print(f"Found {len(files)} files in {path}")
    
    for file in files:
        filepath = os.path.join(path, file)
        
        try:
            with open(filepath) as f:
                
                # Date, Open, High, Low, Close, Adj Close, Volume
                lines = f.readlines()[1:]
                lines = [line.split(',') for line in lines]
                stock = np.array(lines)
                
                # Open, High, Low, Close, Volume
                dates = stock[:,0]
                stock = np.concatenate((stock[:,1:5], stock[:,6:7]), axis=1).astype(float)
                
                for i in range(stock.shape[0] - WINDOW_SIZE):
                    # first = stock[i,:]
                    # last = stock[i+WINDOW_SIZE,:]
                    if dates[i+WINDOW_SIZE] < VALID_START:
                        train.append(stock[i:i+WINDOW_SIZE+1,:])
                    elif dates[i] >= VALID_START and dates[i+WINDOW_SIZE] < TEST_START:
                        valid.append(stock[i:i+WINDOW_SIZE+1,:])
                    elif dates[i] >= TEST_START:
                        test.append(stock[i:i+WINDOW_SIZE+1,:])
        except:
            pass
 
train = np.array(train)
valid = np.array(valid)
test = np.array(test)

print(f"T/V/T Split for {WINDOW_SIZE} day time frames:\n" + 
      f"Train: {train.shape[0]} {WINDOW_SIZE}-day time frames\n" +
      f"Valid: {valid.shape[0]} {WINDOW_SIZE}-day time frames\n" +
      f"Test: {test.shape[0]} {WINDOW_SIZE}-day time frames\n")
    