import numpy as np
from My_data import iris_data, phoneme_read, magic_read, titanic_read
from P_v3 import train_nn, test_nn
from random import *
from datetime import datetime
from NEW import rand_weith, b_p, prop
import time
from openpyxl import formatting, styles, Workbook
from openpyxl.formatting.rule import CellIsRule

def run_neuron_network(selrction, input_output_neurons, activation_funk_mas, alfa, beta, of_on):

    norm_train_selection, norm_test_selection = selrction
    np.random.shuffle(norm_train_selection)
    np.random.shuffle(norm_test_selection)


    # hiden_layer = []
    # for i in activation_funk_mas:
    #     hiden_layer.append(len(i))
    #
    # my_nn, train_loss, train_accuracy, my_B = train_nn(input_output_neurons[0], input_output_neurons[1], hiden_layer, norm_train_selection, activation_funk_mas, alfa, beta, of_on)
    # test_loss, test_accuracy = test_nn(my_nn, norm_test_selection, my_B, activation_funk_mas)

    all_layers = [input_output_neurons[0]]
    for i in activation_funk_mas:
        all_layers.append(len(i))
    all_layers.append(input_output_neurons[1])

    radom_w_b, radom_moment = rand_weith(all_layers)
    activation_funk_mas.append([0, 0, 0]) # 0 - softmax

    train_nn = b_p(radom_w_b, radom_moment, norm_train_selection, activation_funk_mas, alfa, beta)
    #print(train_nn)
    test_nn, test_acc = prop(norm_test_selection, train_nn, activation_funk_mas)

    return test_acc
