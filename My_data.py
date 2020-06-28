import re
import numpy as np
from random import *

def normalize(data):
    norm_data = []
    for i in data:
        y = []
        X = np.array(i[0])
        for x in X:
            y.append((x - np.mean(X)) / np.std(X))
        # for x in X:
        #     mi = X.min()
        #     ma = X.max()
        #     y.append(((x - X.min())*(1-0))/(X.max()-X.min()))
        q = [y, i[1]]
        norm_data.append(q)

    return norm_data

def string_to_data(my_str):
    signs = my_str[:-1]
    for d in range(len(signs)):
        signs[d] = float(signs[d])
    data = [signs, my_str[-1]]
    return data

def data_to_float(Iriss):
    for x in Iriss:
        for i in range(len(x[0])):
            x[0][i] = float(x[0][i].replace(',', '.'))
    return Iriss

def split_data(Iris_s):
    verif = []
    test = []
    for i in Iris_s:
        count = -1
        for Iris in i:
            count += 1
            if count <= 30:
                verif.append(Iris)
            else:
                test.append(Iris)

    return verif, test

def iris_data():
    with open('Iris.txt', 'r') as f:
        line_file = f.read()
    Iris_line = line_file.split('\n')
    str_data =[]
    for st in Iris_line:
        q = st.split('\t')
        if q[-1] == 'Iris-setosa':
            numb = q[0:4]
            # clas = q[4:]
            clas = [0,0,1]
            str_data.append([numb, clas])
        if q[-1] == 'Iris-versicolor':
            numb = q[0:4]
            # clas = q[4:]
            clas = [0,1,0]
            str_data.append([numb, clas])
        if q[-1] == 'Iris-virginica':
            numb = q[0:4]
            # clas = q[4:]
            clas = [1,0,0]
            str_data.append([numb, clas])
    data = data_to_float(str_data)
    norm_data = normalize(data)
    shuffle(norm_data)
    index_split = round(len(norm_data) / 100 * 60)
    train, test = norm_data[:index_split], norm_data[index_split:]

    return train, test

def phoneme_read():
    with open('phoneme.txt', 'r', encoding="utf-8") as f:
        st = f.read()
    rows = st.split('\n')
    data = []
    for row in rows:
        if not re.match('@', row):
            new_list = row.split(',')
            q = string_to_data(new_list)
            if int(q[-1]) == 0 or not str:
                q[-1] = [1, 0]
            elif int(q[-1]) == 1 or not str:
                q[-1] = [0, 1]
            data.append(q)

    index_split = round(len(data) / 100 * 60)
    norm_data = normalize(data)
    verif, test = norm_data[:index_split], norm_data[index_split:]
    return verif, test

def titanic_read():
    with open('titanic.txt', 'r', encoding="utf-8") as f:
        st = f.read()
    rows = st.split('\n')
    data = []
    for row in rows:
        if not re.match('@', row):
            new_list = row.split(',')
            q = string_to_data(new_list)
            if q[-1] == "1.0":
                q[-1] = [1, 0]
            else:
                q[-1] = [0, 1]
            data.append(q)

    index_split = round(len(data) / 100 * 60)
    norm_data = normalize(data)
    verif, test = norm_data[:index_split], norm_data[index_split:]
    return verif, test

def magic_read():
    with open('magic.txt', 'r', encoding="utf-8") as f:
        st = f.read()
    rows = st.split('\n')
    data = []
    for row in rows:
        if not re.match('@', row):
            new_list = row.split(',')
            q = string_to_data(new_list)
            if q[-1] == "g":
                q[-1] = [1, 0]
            else:
                q[-1] = [0, 1]
            data.append(q)

    index_split = round(len(data) / 100 * 60)
    norm_data = normalize(data)
    verif, test = norm_data[:index_split], norm_data[index_split:]
    return verif, test
