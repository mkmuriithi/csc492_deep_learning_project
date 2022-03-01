import numpy as np
import os, random, time, csv
from tqdm import tqdm

random.seed(87)

WINDOW_SIZE = 30
VALID_START = '2019-01-01'
TEST_START = '2020-06-01'
paths = ['nyse_15yr_data']

# Create data sets that include X and T
train = []
valid = []
test = []

# Counters for bad files
bad_date = 0
bad_missing = 0

# Iterate through datasets
for path in paths:
    files = os.listdir(path)
    print(f"Found {len(files)} files in {path}")
    
    # Iterate through stocks
    for file in tqdm(files):
        filepath = os.path.join(path, file)
        
        try:
            with open(filepath) as f:
                
                lines = f.readlines()[1:]
                lines = [line.split(',') for line in lines]
                stock = np.array(lines)
                
                dates = stock[:,0]
                if len(dates) < 31:
                    bad_date += 1
                    continue
                
                if any('' in line for line in lines):
                    bad_missing += 1
                    continue
                
                if dates[30] < VALID_START:
                    train.append(filepath)
                elif dates[30] < TEST_START:
                    valid.append(filepath)
                else:
                    test.append(filepath)
                    
        except:
            pass
        
print(f"\n{bad_date} files ignored due to not enough dates")
print(f"{bad_missing} files ignored due to missing value")
        
random.shuffle(train)
valid2, train = train[:len(valid)], train[len(valid):]
test2, train = train[:len(test)], train[len(test):]

for name, stock_list in [("train_stocks.csv",  train),
                         ("valid_stocks.csv",  valid),
                         ("valid2_stocks.csv", valid2),
                         ("test_stocks.csv",   test),
                         ("test2_stocks.csv",  test2),]:
    with open(name, 'w+', newline='') as file:
        w = csv.writer(file)
        w.writerows(stock_list)

valid_array = np.zeros((0,WINDOW_SIZE+1,5))
valid2_array = np.zeros((0,WINDOW_SIZE+1,5))
test_array = np.zeros((0,WINDOW_SIZE+1,5))
test2_array = np.zeros((0,WINDOW_SIZE+1,5))

for filepath in tqdm(valid):
    with open(filepath) as f:
        lines = f.readlines()[1:]
        lines = [line.split(',') for line in lines]
        stock = np.array(lines)
        
        # Remove dates outside range
        stock = stock[(VALID_START <= stock[:,0]) & (stock[:,0] < TEST_START)]
        
        # Remove additional features
        stock = np.concatenate([stock[:,1:5], stock[:,6:7]], axis=1).astype(float) 
        
        for i in range(100):
            index = random.randint(0, stock.shape[0] - (WINDOW_SIZE+1))
            valid_array = np.concatenate([valid_array, np.expand_dims(stock[index:index+WINDOW_SIZE+1,:], 0)], 0)

for filepath in tqdm(valid2):
    with open(filepath) as f:
        lines = f.readlines()[1:]
        lines = [line.split(',') for line in lines]
        stock = np.array(lines)
        
        # Remove dates outside range
        stock = stock[(VALID_START <= stock[:,0]) & (stock[:,0] < TEST_START)]
        
        # Remove additional features
        stock = np.concatenate([stock[:,1:5], stock[:,6:7]], axis=1).astype(float) 
        
        for i in range(100):
            index = random.randint(0, stock.shape[0] - (WINDOW_SIZE+1))
            valid2_array = np.concatenate([valid2_array, np.expand_dims(stock[index:index+WINDOW_SIZE+1,:], 0)], 0)

for filepath in tqdm(test):
    with open(filepath) as f:
        lines = f.readlines()[1:]
        lines = [line.split(',') for line in lines]
        stock = np.array(lines)
        
        # Remove dates outside range
        stock = stock[TEST_START <= stock[:,0]]
        
        # Remove additional features
        stock = np.concatenate([stock[:,1:5], stock[:,6:7]], axis=1).astype(float) 
        
        for i in range(100):
            index = random.randint(0, stock.shape[0] - (WINDOW_SIZE+1))
            test_array = np.concatenate([test_array, np.expand_dims(stock[index:index+WINDOW_SIZE+1,:], 0)], 0)

for filepath in tqdm(test2):
    with open(filepath) as f:
        lines = f.readlines()[1:]
        lines = [line.split(',') for line in lines]
        stock = np.array(lines)
        
        # Remove dates outside range
        stock = stock[TEST_START <= stock[:,0]]
        
        # Remove additional features
        stock = np.concatenate([stock[:,1:5], stock[:,6:7]], axis=1).astype(float) 
        
        for i in range(100):
            index = random.randint(0, stock.shape[0] - (WINDOW_SIZE+1))
            test2_array = np.concatenate([test2_array, np.expand_dims(stock[index:index+WINDOW_SIZE+1,:], 0)], 0)

np.save("./valid_data", valid)
np.save("./valid2_data", valid2)
np.save("./valid_data", test)
np.save("./valid2_data", test2)
    