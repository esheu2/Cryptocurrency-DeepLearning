import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import time

def CNN_RNN(Xtrain, ytrain, Xval, yval, pickle_name, time = 1000):
    model = models.Sequential()
    
