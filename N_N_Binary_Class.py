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
def run_single_layered_NN(X_mat, y_vec, hidden_layers, num_epochs, data_set):
    # set model variable to keep track on which number model is being ran
    model_number = 1

    # list of colors for hidden layers

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
        print(f"\nModel {model_number} {data_set}")
        print("==============================================")
        model_data = model.fit(
                                    x=X_mat,
                                    y=y_vec,
                                    epochs=num_epochs,
                                    verbose=0,
                                    validation_split=.05)

        # update model number
        model_number += 1

        # apend model data to list
        model_data_list.append(model_data)



    return model_data_list


# Function: graph_model_data
# INPUT ARGS:
# model_data_list: lsit of data generated by NN model
# num_epochs: total number of epoches used for the models
# line_style: style the line should be on the output graph
# model_data: a list of valuesdectionary or values from our single NN model
def graph_model_data(model_data_list, num_epochs, set):

    colors = ['lightblue', 'darkblue', 'black']
    line_style = ['solid', 'dashed']



    set_index = 0


    for model_data in model_data_list:
        color_index = 0
        model_index = 1
        for data in model_data:
            if(set[set_index] == 'Train'):
                plt.plot(range(0,num_epochs), data.history['loss'], markevery=num_epochs,
                                color = colors[color_index], linestyle = line_style[set_index],
                                label = f"Model {model_index} {set[set_index]} Data")


            if(set[set_index] == 'Validation'):
                plt.plot(range(0,num_epochs), data.history['loss'],
                                color = colors[color_index], linestyle = line_style[set_index],
                                label = f"Model {model_index} {set[set_index]} Data")

            color_index += 1

            model_index += 1
        set_index += 1
    # add grid to graphs
    plt.grid(True)

    # Add x nd y labels
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend()

    #display graph
    plt.savefig("Loss Graph")


# Function: main
# INPUT ARGS:
#   none
# Return: none
def main():

    # set number of epochs
    num_epochs = 50
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

    model_data_subtrain_list = run_single_layered_NN(X_subtrain, y_subtrain,
                                    hidden_layers, num_epochs, "Subtrain")

    model_data_valid_list = run_single_layered_NN(X_validation, y_validation,
                                    hidden_layers, num_epochs, "Validataion")

    model_data_list = [model_data_subtrain_list, model_data_valid_list]

    # plot data
    graph_model_data(model_data_list, num_epochs, ["Train" ,"Validation"])

    best_num_epochs = []
    # get best number of epochs besed off validation data
    for model in model_data_valid_list:

        loss_data = model.history["loss"]

        best_num_epochs.append(loss_data.index(min(loss_data)) + 1)

    print(best_num_epochs)
    ## Retrain whole train data based of best num of epochs
    model_data_train_list = run_single_layered_NN(X_train, y_train,
                                            hidden_layers, num_epochs, "Train")





main()