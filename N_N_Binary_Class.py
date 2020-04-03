###############################################################################
#
# AUTHOR(S): Joshua Holguin
# DESCRIPTION: program that will implement stochastic gradient descent algorithm
# for a one layer neural network
# VERSION: 0.0.1v
#
###############################################################################
# import statements
import numpy as np
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    data_matrix_full = np.genfromtxt( file_name, delimiter = " " )
    return data_matrix_full

# Function: run_single_layered_NN
# INPUT ARGS:
#
#
#
#
# Return:
def run_single_layered_NN(X_mat, y_vec, hidden_layers, num_epochs):
    # set model variable to keep track on which number model is being ran
    model_number = 1

    # set color index
    color_index = 0

    # list of colors for hidden layers
    colors = ['lightblue', 'darkblue', 'black']
    # creat list of model data
    model_data_list = []

    # create a neural network with 1 hidden layer
    for hidden_layer in hidden_layers:
        # set model for single layered NN
        model = keras.Sequential([
        keras.layers.Flatten(input_shape=(np.size(X_mat, 1), )), # input layer
        keras.layers.Dense(hidden_layer, activation='sigmoid', use_bias=False), # hidden layer
        keras.layers.Dense(1, activation='sigmoid', use_bias=False) # output layer
        ])

        # compile the models
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # fit the models
        print(f"\nModel {model_number}")
        print("==============================================")
        model_data = model.fit(
                                    x=X_mat,
                                    y=y_vec,
                                    epochs=num_epochs,
                                    verbose=0,
                                    validation_split=.03)

        # update model number
        model_number += 1

        # apend model data to list
        model_data_list.append(model_data)

        ## TODO:
        ## Make graphing into its own function
        ## Use subtrain to train our model
        ## Get data from graphs to retrain the entire train set
        ## Make Predictions given our models
        plt.plot(range(0,num_epochs), model_data.history['loss'],
                        color = colors[color_index], linestyle = 'solid')

        #update color index
        color_index += 1
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.show()

    return model_data_list

# Function: main
# INPUT ARGS:
#   none
# Return: none
def main():

    # set number of epochs
    num_epochs = 60
    # denote file name for data set
    file_name = "spam.data"
    # Get data into matrix form
    X_mat_full = convert_data_to_matrix(file_name)

    X_mat_full_col_len = X_mat_full.shape[1]
    # split data into a matrix and vector of pred values
    X_mat = np.array(X_mat_full[:,:-1])
    y_vec = np.array(X_mat_full[:,X_mat_full_col_len-1])

    # Scale matrix for use
    X_sc = scale(X_mat)

    # divide data into 80% train and 20% test
    X_train, X_test = np.split( X_sc, [int(.8 * len(X_sc))])
    y_train, y_test = np.split( y_vec, [int(.8 * len(y_vec))])

    # split train data into 60% subtrain and 40% validation
    X_subtrain, X_validation = np.split( X_train, [int(.6 * len(X_train))])
    y_subtrain, y_validation = np.split( y_train, [int(.6 * len(y_train))])

    # set array of hidden layers for models
    hidden_layers = [10,100,1000]

    model_data_list = run_single_layered_NN(X_train, y_train, hidden_layers, num_epochs)



main()