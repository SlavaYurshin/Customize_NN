import numpy as np
from math import ceil
from random import *


def sigmoid(x):
    if x > 3:
        x = 3
    if x < -3:
        x = -3
    return (1 / (1 + np.exp(-x)))

def derivative_sigmoid(x):
    return x * (1 - x)

def ReLu(x):
    if x > 10:
        x = 10
    if x < - 10:
        x = -10
    return 0 if x < 0 else x

def derivative_ReLu(x):
    return 0 if x < 0 else 1


activation_functions = {1: sigmoid, 2: ReLu}
derivative_functons = {1: derivative_sigmoid, 2: derivative_ReLu}

def softmax(x):
    for i in range(len(x)):
        if x[i] > 3:
            x[i] = 3
        if x[i] < - 3:
            x[i] = -3
    e_x = np.exp(x)
    a = e_x / e_x.sum()
    return a

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


def error_output(output, accurate):
    razn = accurate - output
    return razn

def W_random(input, output, hidden_layer):
    last_b = output
    my_nn = [input]
    for l in hidden_layer:
        my_nn.append(l)
    my_nn.append(output)
    #print(my_nn)
    all_W = []
    all_B = []
    all_Z = []
    for i in range(len(my_nn)-1):
        prev_layer = my_nn[i]
        next_layer = my_nn[i+1]
        w = np.random.normal(0, 0.05, (next_layer, prev_layer))
        z = np.zeros((next_layer, prev_layer))
        all_W.append(w)
        all_Z.append(z)
    # print(all_W)
    # for i in hidden_layer, output:
    #     all_B.append(np.random.normal(0, 0.05, i))
    # all_B.append(np.random.normal((0, 0.05, last_b)))

    return all_W, all_Z


def direct_W(data, W, B, activation_functions_mas):
    loss = 0
    error = []
    ALL_F = []
    accuracy = 0
    for d in data:
        all_f = []
        layer = -1
        inp = np.array(d[0])
        all_f.append(inp)
        for w in W:
            layer += 1
            #print(layer)
            if layer != len(W) - 1:
                sumator = np.dot(w, inp.transpose()) + B[layer]
                f_layer = np.zeros((len(sumator)))
                for n in range(len(sumator)):
                    f_layer[n] = activation_functions[activation_functions_mas[layer][n]](sumator[n])
                inp = f_layer
            else:
                sumator = np.dot(w, inp) + B[layer]

                for i in range(len(sumator)):
                    sumator[i] = round(sumator[i], 6)

                f_layer = softmax(sumator)
            all_f.append(f_layer)

        loss += cross_entropy(all_f[-1], d[1])
        accuracy += accuracy_value(all_f[-1], d[1])

        error.append(error_output(all_f[-1], d[1]))

        ALL_F.append(all_f[:-1])

    # print(loss/len(data))
    # print("Точность " + str(accuracy/len(data)))
    # print("W")
    # print(W)
    # print("error")
    # print(error)
    # print("Al F")
    # print(ALL_F)
    # print("loss")
    # print(loss)
    # print("accur")
    # print(accuracy)

    return np.array(error), np.array(ALL_F), loss/len(data), accuracy/len(data)

def error_calc(errors, w, f_layer, B, activations_func_mas):
    er = np.dot(errors, w)
    dd = np.zeros((len(f_layer), len(f_layer[0])))
    for i in range(len(f_layer)):
        count = -1
        for j in range(len(f_layer[0])):
            count += 1
            dd[i][j] = derivative_functons[activations_func_mas[count]](f_layer[i][j])
    delta = er * dd

    return delta

def correct_W(w, z, alfa):
    np.array(z)
    new_w = np.zeros((len(w), len(w[0])))
    for i in range(len(w)):
        for j in range(len(w[i])):
            new_w[i][j] = w[i][j] + (alfa*z[i][j])
    return new_w

def correct_z(z, delta, beta):
    count = -1
    new_z = []
    for i in z:
        count += 1
        new_z.append(i*beta + delta[count])
    return new_z

def parting(xs, parts):
    part_len = ceil(len(xs)/parts)
    xx = []
    for k in range(parts):
        q = xs[part_len*k:part_len*(k+1)]
        xx.append(q)
    return [xs[part_len*k:part_len*(k+1)] for k in range(parts)]

def train_nn(input_value, output, hiden_layer, norm_data, activation_funk_mas, my_alfa, my_beta, of_on):
    rand_W, Z = W_random(input_value, output, hiden_layer)  # вход, выход, скрытые слои
    B = []
    for ni in hiden_layer:
        B.append(np.random.normal(0, 0.05, ni))
    B.append(np.random.normal(0, 0.05, output))
    if of_on == 0:
        return rand_W, 0, 0, B
    # 0.009
    alfa = my_alfa
    if len(norm_data) < 100:
        iterations = 3
    else:
        iterations = 20
    bach = ceil(len(norm_data)/iterations)
    all_accuracy = []
    all_loss = []
    for gen in range(10):
        shuffle(norm_data)
        beta = my_beta
        ALL_W = []
        now_loss = []
        for it in range(iterations):

            now_accuracy = []
            first_index = bach * it
            next_index = bach * (it + 1)
            if gen == 0 and it == 0:
                if it > 0:
                    rand_W = new_W
                error, f_layers, loss, accuracy = direct_W(norm_data[first_index:next_index], rand_W, B, activation_funk_mas)
                W = rand_W
                momentum_z = Z
            else:
                alfa *= 0.99999
                W = new_W
                error, f_layers, loss, accuracy = direct_W(norm_data[first_index:next_index], W, B, activation_funk_mas)
            now_loss.append(loss)
            now_accuracy.append(accuracy)

            layer_value = len(W)
            previos_value = layer_value
            errors = []
            ALL_DELTA = []
            count = 0
            for i in range(len(W)):
                previos_value -= 1
                if previos_value + 1 == layer_value:
                    errors.append(error)
                else:
                    errors.append(error_calc(errors[count], W[previos_value+1], f_layers[:, previos_value+1], B[previos_value+1], activation_funk_mas[previos_value]))
                    count += 1

            errors.reverse()

            for i in range(len(W)):
                layer_value -= 1

                ALL_DELTA.append(np.dot(errors[layer_value].transpose(), f_layers[:, layer_value]))

            ALL_DELTA.reverse()
            ALL_DELTA = np.array(ALL_DELTA)
            # mm = errors[-1].shape
            # mm1 = errors[-1].shape[0]
            # mm2 = errors[-1].shape[1]

            desimal_B = []

            layer_B = len(B)
            for di in range(len(B)):
                layer_B -= 1
                error_B = np.dot(errors[layer_B], B[layer_B].transpose())

                ons_B = np.ones((len(B[layer_B]), len(error_B)))
                desimal_B.append(np.dot(error_B, ons_B.transpose()))
            desimal_B.reverse()
            for cb in range(len(B)):
                B[i] = B[i] + (alfa * desimal_B[i])


            for i in range(len(W)):
                momentum_z[i] = correct_z(momentum_z[i], ALL_DELTA[i], beta)
            new_W = []
            for i in range(len(W)):
                new_W.append(correct_W(W[i], momentum_z[i], alfa))

            ALL_W.append(W)
        best_loss = now_loss.index(min(now_loss))
        all_loss.append(now_loss[best_loss])
        #all_accuracy.append(now_accuracy[best_loss])
        #new_W = ALL_W[best_loss]
        #print(new_W)

    return new_W, all_loss, all_accuracy, B

def test_nn(my_W, norm_data, B, activation_functions_mas):
    # print("--------------TEST---------------------")
    np.random.shuffle(norm_data)
    loss = 0
    accuracy = 0
    for d in norm_data:
        all_f = []
        layer = -1
        inp = np.array(d[0])
        all_f.append(inp)
        for w in my_W:
            layer += 1
            if layer != len(my_W) - 1:
                sumator = np.dot(w, inp.transpose()) + B[layer]
                f_layer = np.zeros((len(sumator)))
                for n in range(len(sumator)):
                    f_layer[n] = activation_functions[activation_functions_mas[layer][n]](sumator[n])
                inp = f_layer
            else:
                sumator = np.dot(w, inp)+ B[layer]
                f_layer = softmax(sumator)
            all_f.append(f_layer)

        loss += cross_entropy(all_f[-1], d[1])
        accuracy += accuracy_value(all_f[-1], d[1])

        max_ind_softmax = all_f[-1].argmax()
        # print(max_ind_softmax)
    # print("Точность теста " + str(accuracy / len(norm_data)))
    # print("LOSS= " + str(loss/len(norm_data)))
    #print(accuracy / len(norm_data))
    return loss/len(norm_data), accuracy / len(norm_data)

