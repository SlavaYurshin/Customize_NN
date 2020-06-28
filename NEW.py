from My_data import iris_data, titanic_read, phoneme_read
import numpy as np
from random import *


class Act():
    cl = int
    def __str__(self, x):
        print("sdns")

    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    def rl(self, x):
        return 0 if x < 0 else x


def activation_neurons(x, activ):
    #print("ACTIV " + str(activ))
    if activ == 1:
        if x > 3:
            x = 3
        if x < - 3:
            x = -3
        return 1 / (1 + np.exp(-x))
    elif activ == 2:
        if x > 3:
            x = 3
        if x < - 3:
            x = -3
        return 0 if x < 0 else x
    else:
        for i in range(len(x)):
            if x[i] > 3:
                x[i] = 3
            if x[i] < - 3:
                x[i] = -3
        e_x = np.exp(x)
        return e_x / e_x.sum()

def derivative_calculation(x, activ):
    if activ == 1:
        return x * (1 - x)
    elif activ == 2:
        return 0 if x < 0 else 1


def error_output(output, accurate):
    razn = accurate - output
    return razn

def cross_entropy(output, sorce):
    l = 0
    kross_sum = 0
    for i in range(len(output)):

        if sorce.index(max(sorce)) == i:
            l = np.log(output[i])
        # else:
        #     l += 1 - math.log(soft_val[i])
        # kross_sum += (sorce[i] * (math.log(output[i]))) + ((1-sorce[i])*(1-math.log(output[i])))
    return - l

def accuracy_value(output, sorce):
    accuracy = 0
    index_max = np.argmax(output)
    if index_max == sorce.index(max(sorce)):
        accuracy = 1
    return accuracy

def rand_weith(input_output_nn):
    w_with_b = []
    moment = []
    for i in range(len(input_output_nn)):
        if i == len(input_output_nn) - 1:
            break
        w = [np.random.normal(0, 0.05, (input_output_nn[i + 1], input_output_nn[i]))]
        w.append(np.random.normal(0, 0.05, input_output_nn[i + 1]))
        moment.append([np.zeros((input_output_nn[i+1], input_output_nn[i])), np.zeros((input_output_nn[i + 1]))])
        w_with_b.append(w)

    #rand_w_b = np.array(w_with_b)
    return w_with_b, moment


def correct_moment(moment, delta_w_with_b, beta):
    for la in range(len(moment)):
        for i in range(len(moment[la][0])):
            moment[la][0][i] = moment[la][0][i] * beta + delta_w_with_b[la][0][i]
            moment[la][1][i] = moment[la][1][i] * beta + delta_w_with_b[la][1][i]
    return moment

def correct_w_with_bias(w_with_bias, moment, alfa):
    for i in range(len(w_with_bias[0])):
        for j in range(len(w_with_bias[0][i])):
            w_with_bias[0][i][j] = w_with_bias[0][i][j] + (alfa * moment[0][i][j])
    for i in range(len(w_with_bias[1])):
        w_with_bias[1][i] = w_with_bias[1][i] + (alfa * moment[1][i])
    return w_with_bias


def error_calculation(w_with_bias, error):
    layer_reeror = np.zeros((len(error), len(error[0])))
    for i in range(len(error)):
        for j in range(len(error[i])):
            layer_reeror[i][j] = error[i][j]

    er = np.dot(layer_reeror, w_with_bias[0])
    er_b = np.dot(layer_reeror, w_with_bias[1])
    return [er, er_b]


def derivative_one_layer(activ_n, form_activ_func):
    derivative_neurons = np.zeros((len(activ_n), len(activ_n[0])))
    for i in range(len(activ_n)):
        count = -1
        for j in range(len(activ_n[0])):
            count += 1
            derivative_neurons[i][j] = derivative_calculation(activ_n[i][j], form_activ_func[j])
    return derivative_neurons

def delta_calculation(w_with_bias, errors_with_b, activ_neuron):
    delta_for_w_b = []
    for i in range(len(w_with_bias)):
        transpose_error = np.zeros((len(errors_with_b[i + 1][0][0]), len(errors_with_b[i + 1][0])))
        active_for_bias = np.ones((len(errors_with_b[i + 1][0]), len(errors_with_b[i + 1][0][0])))
        for row in range(len(transpose_error)):
            for col in range(len(transpose_error[row])):
                transpose_error[row][col] = errors_with_b[i + 1][0][col][row]
        d_w = np.dot(transpose_error, activ_neuron[:, i])
        d_b = np.dot(errors_with_b[i][1], active_for_bias)
        delta_for_w_b.append([d_w, d_b])
    return delta_for_w_b



def b_p(rand_w_with_bias, moment, train, form_activ_func, my_alfa, my_beta):

    alfa = my_alfa
    beta = my_beta
    for gen in range(20):
        if gen == 0:
            w_with_bias = rand_w_with_bias
        else:
            shuffle(train)
            rand_w_with_bias = new_W
        active_neurons_and_last_error, acc = prop(train, w_with_bias, form_activ_func)
        activ_neuron = active_neurons_and_last_error[:,:-1]
        last_error = active_neurons_and_last_error[:,-1]

        count_back_prop = len(w_with_bias)
        errors_with_b = [[last_error, [0]]] # 0 ошибка для сдвига последнего слоя
        all_derivative_neurons = []
        for i in range(count_back_prop):
            count_back_prop -= 1
            errors_with_b.append(error_calculation(w_with_bias[count_back_prop], errors_with_b[i][0]))
        errors_with_b.reverse()
        for i in range(1, len(activ_neuron[0])):
            all_derivative_neurons.append(derivative_one_layer(activ_neuron[:,i], form_activ_func[i-1]))

        for i in range(1, len(activ_neuron[0])):
            errors_with_b[i][0] = errors_with_b[i][0] * all_derivative_neurons[i-1]

        delta_w_with_b = delta_calculation(w_with_bias, errors_with_b, activ_neuron)
        moment = correct_moment(moment, delta_w_with_b, beta)

        new_W = correct_w_with_bias(w_with_bias, moment, alfa)
    #print("----------------- END TRAIN -----------------")
    return w_with_bias

def prop(data, w_with_b, form_activ_func):
    active_neurons_and_last_error = []
    all_errors = []
    acc = 0
    loss = 0
    count = -1
    for d in data:
        count += 1
        layer = -1
        inp = np.array(d[0])
        one_train_activ_nuron = [inp]
        for wi in w_with_b:
            layer += 1
            if layer != len(w_with_b)-1:
                sumators = np.dot(wi[0], inp.transpose()) + wi[1]
                active_function = np.zeros((len(sumators)))
                for i in range(len(sumators)):
                    active_function[i] = activation_neurons(sumators[i], form_activ_func[layer][i])
                inp = active_function
                one_train_activ_nuron.append(active_function)
            else:
                sumators = np.dot(wi[0], inp.transpose()) + wi[1]
                active_function = activation_neurons(sumators, form_activ_func[layer])
                # добавляем ошибку последнего слоя в общий список
                one_train_activ_nuron.append(error_output(active_function, d[1]))
        active_neurons_and_last_error.append(one_train_activ_nuron)
        loss += cross_entropy(active_function, d[1])
        acc += accuracy_value(active_function, d[1])
    #print("ACC " + str(acc/len(data)))
    return np.array(active_neurons_and_last_error), acc/len(data)

