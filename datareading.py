import pandas as pd
import numpy as np

def load_data():
    #reading from csv and loading into np array
    BTCcsv = pd.read_csv('data/BTC-USD.csv')
    BTCdata = np.array(BTCcsv).astype(float)

    # Creating X and y for dataset
    X = BTCdata[:,1:8]
    y = BTCdata[:,0]

    return X, y

def train_val_test():
    X, y = load_data()

    # splitting the data
    # 70-20-10 split for train-val-test
    n = X.shape[0]
    Xtrain = X[0:int(n*0.7)]
    ytrain = y[0:int(n*0.7)]
    Xval = X[int(n*0.7):int(n*0.9)]
    yval = y[int(n*0.7):int(n*0.9)]
    Xtest = X[int(n*0.9):]
    ytest = y[int(n*0.9):]
    return Xtrain, ytrain, Xval, yval, Xtest, ytest

# if __name__ == '__main__':
#     Xtrain, ytrain, Xval, yval, Xtest, ytest = train_val_test()
#     print(Xtrain.shape)
#     print(Xval.shape)
#     print(Xtest.shape)
