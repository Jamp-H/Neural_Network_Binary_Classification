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

# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    data_matrix_full = np.genfromtxt( file_name, delimiter = " " )
    return data_matrix_full


# Function: main
# INPUT ARGS:
#   none
# Return: none
def main():
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

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


    # create a neural network with 1 hidden layer

    model_1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(np.size(X_train, 1), )), # input layer
    keras.layers.Dense(10, activation='sigmoid', use_bias=False), # hidden layer
    keras.layers.Dense(1, activation='sigmoid', use_bias=False) # output layer
    ])

    model_2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(np.size(X_train, 1), )), # input layer
    keras.layers.Dense(100, activation='sigmoid', use_bias=False), # hidden layer
    keras.layers.Dense(1, activation='sigmoid', use_bias=False) # output layer
    ])

    model_3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(np.size(X_train, 1), )), # input layer
    keras.layers.Dense(1000, activation='sigmoid', use_bias=False), # hidden layer
    keras.layers.Dense(1, activation='sigmoid', use_bias=False) # output layer
    ])

    # compile the models
    model_1.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model_2.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model_3.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # fit the models
    print("\nModel 1")
    print("==============================================")
    model_1_data = model_1.fit(
                                x=X_train,
                                y=y_train,
                                epochs=5,
                                verbose=2,
                                validation_split=.03)

    print("\nModel 2")
    print("==============================================")
    model_2.fit(
                x=X_train,
                y=y_train,
                epochs=5,
                verbose=2,
                validation_split=.03)

    print("\nModel 3")
    print("==============================================")
    model_3.fit(
                x=X_train,
                y=y_train,
                epochs=5,
                verbose=2,
                validation_split=.03)

    print([x for x in model_1_data.history])
main()