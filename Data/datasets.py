import numpy as np
import os, random, time, csv

random.seed(87)

DROPOUT = 0.90
WINDOW_SIZE = 30
VALID_START = '2019-01-01'
TEST_START = '2020-06-01'
paths = ['.\\nyse_15yr_data', '.\\nasdaq_15yr_data']

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
    for file in files:
        filepath = os.path.join(path, file)
        
        try:
            with open(filepath) as f:
                
                lines = f.readlines()[1:]
                lines = [line.split(',') for line in lines]
                stock = np.array(lines)
                
                dates = stock[:,0]
                if len(dates) < 31:
                    print(f"skipping {file}: not enough dates")
                    bad_date += 1
                    continue
                
                if any('' in line for line in lines):
                    print(f"skipping {file}: found missing value")
                    bad_missing += 1
                    continue
                
                if any('NaN' in line for line in lines):
                    bad_nan += 1
                    print(f"skipping {file}: found NaN value")
                    continue
                
                if dates[30] < VALID_START:
                    train.append(filepath[2:-4])
                elif dates[30] < TEST_START:
                    valid.append(filepath[2:-4])
                else:
                    test.append(filepath[2:-4])
                    
        except:
            pass
        
print(f"{bad_date} files ignored due to not enough dates")
print(f"{bad_missing} files ignored due to missing value")
        
random.shuffle(train)
valid2, train = train[:len(valid)], train[len(valid):]
test2, train = train[:len(test)], train[len(test):]

with open('datasets.csv', 'w', newline='') as file:
    w = csv.writer(file)
    w.writerows([train, valid, valid2, test, test2])
    
print(f"len(train): {len(train)}")
print(f"len(valid): {len(valid)}")
print(f"len(test): {len(test)}")
    