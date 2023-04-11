#IMPORTS FOR ALL NOTEBOOKS
#Model: Pytorch and timm
from timm import *
import timm

## Data Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Image Processing
from fastai.data.external import *
from PIL import Image 

# Helper functions
from fastai import *
from fastai.vision.all import *








'''
Helper functions
'''
# take an index from the dataframe and return a tensor of shape 96,96
def get_image_tensor(training_df, index): 
    img_str = training_df['Image'][index]
    arr = np.fromstring(img_str, dtype=int, sep=' ')
    arr = ['0' if x == '' else x for x in arr]
    arr = tensor(np.array(arr).reshape(96,96))
    return arr

# take an index from the dataframe and return a 
def get_image_arr(training_df, index): 
    img_str = training_df['Image'][index]
    arr = np.fromstring(img_str, dtype=int, sep=' ')
    arr = ['0' if x == '' else x for x in arr]
    # arr = tensor(np.array(arr).reshape(96,96))
    return np.array(arr)

def get_key_points(training_df, index, num=30):
    coord_arr = []
    for i in range(0,num,2):
        coord = training_df.iloc[index][i],training_df.iloc[index][i+1]
        coord_arr.append(coord)
    return tensor(coord_arr)

def plot_image_with_key_points(training_df, index, num_key_points=30, coord_manual_input=[0]):
    image_tensor = get_image_tensor(index)
    coord_arr = get_key_points(index, num_key_points)
    
    
    plt.imshow(image_tensor,cmap='gray')
    
    if coord_manual_input == [0]:
        for coord in coord_arr:
            plt.scatter(coord[0],coord[1],c='pink', marker='s', s=50, alpha=.5)
    else:
        plt.scatter(coord_manual_input[0],coord_manual_input[1],c='r', marker='s', s=60, alpha=.5)
        
        
        

'''
Returns a training set (X, Y),
and a validation set (X,Y)

usage: 
X_train, Y_train, X_val, Y_val = create_train_test_sets(df)
'''
def create_train_test_sets(training_df, test_size=.2, normalize=False):
    
    independent_variable_images = []
    for i in range(len(training_df)):
        independent_variable_images.append(get_image_arr(training_df, i))
    image_list = np.array(independent_variable_images,dtype = 'float')
    independent_vars = image_list.reshape(-1,96,96,1) #! Optimize
    

    dependent_vars = training_df.iloc[:,0:30]
    dependent_vars = dependent_vars.to_numpy()
    
    test_size = int(test_size * len(training_df))
    train_size = len(training_df) - test_size

    # create the training and test sets
    X_train = independent_vars[:train_size]
    Y_train = dependent_vars[:train_size]
    X_val = independent_vars[train_size:]
    Y_val = dependent_vars[train_size:]
    
    return X_train, Y_train, X_val, Y_val
    if normalize: 
            return X_train/255, Y_train/255, X_val/255, Y_val/255
