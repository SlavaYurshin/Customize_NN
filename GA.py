import random
import numpy as np
from My_data import iris_data, phoneme_read, magic_read, titanic_read
from run_nn import run_neuron_network
from datetime import datetime

import time

import xlwt
import win32com.client
Excel = win32com.client.Dispatch("Excel.Application")

def save_Exel_customiz(accurasy, my_nn, count_column):

    wb = Excel.Workbooks.Open(u'C:\\Users\\vyach\\PycharmProjects\\Perseptron\\Customiz.xls')
    sheet = wb.ActiveSheet
    for i in range(len(accurasy)):
        sheet.Cells(i+1, count_column).value = accurasy[i]
    wb.Save()

    sheet_1 = wb.ActiveSheet
    for i in range(len(my_nn)):
        for j in range(len(my_nn[i])):
            sheet_1.Cells(i+1, count_column+2+j).value = my_nn[i][j]

    wb.Save()
    wb.Close()
    Excel.Quit()

data_name = {1: iris_data,
             2: phoneme_read,
             3: titanic_read,
             4: magic_read}
input_output_neurons = {1: [4, 3],
                        2: [5, 2],
                        3: [3, 2],
                        4: [10, 2]}

def random_pop(pop_count):
    POP = []
    for p in range(pop_count):
        #layer_val = random.randint(1, 6)
        layer_val = 1
        hromosoma = []

        new_hromosoma = []
        for i in range(layer_val):
            gen_val = random.randint(2, 11)
            for j in range(gen_val):
                gen = random.randint(1, 2)
                #gen = 1
                hromosoma.append(gen)
            hromosoma.append(0)
        new_a = random.uniform(0.01, 0.1)
        hromosoma.insert(0, random.uniform(0.01, 0.1))
        hromosoma.insert(1, random.uniform(0.1, 0.99))
        # print(hromosoma)
        POP.append(hromosoma)
    return POP

def my_pop(hiden_layer):
    pass

def hromosoma_to_weight(hromosoma):
    new_weight = []
    now_layer = []
    for g in range(2, len(hromosoma)):
        now_layer.append(hromosoma[g])
        if hromosoma[g] == 0:
            new_weight.append(now_layer[:-1])
            now_layer = []

    # print(new_weight)
    # print("alfa " + str(hromosoma[0]))
    # print("beta " + str(hromosoma[1]))
    # hromosoma[0] == alfa
    # hromosoma[1] == beta
    return hromosoma[0], hromosoma[1], new_weight

def Crossover(Parens1, Parens2):
    ch = random.randint(0, 1)
    point1 = random.randint(3, len(Parens1))
    point2 = random.randint(3, len(Parens2))
    if ch == 0:
        child = Parens2[0:point2]
        for i in Parens1[point1:]:
            child.append(i)
    else:
        child = Parens1[0:point1]
        for i in Parens2[point2:]:
            child.append(i)
    if child[-1] != 0:
        child.append(0)
    # проверка на соседей
    for j in range(len(child)):
        count = 0
        if j == len(child)-count:
            break
        if child[j] == 0 and child[j] == child[j-1]:
            count += 1
            del child[j-1]

    #print("------------------")
    #print(child)
    return child

def TournamentSelection(Fitness, tournament_size):
    t = 0
    index_selection = []
    fit = Fitness.copy()
    while t != 2:
        index_tournament = []
        count = 0
        while count != tournament_size:
            rand_h = random.randint(0, len(fit)-1)
            if fit[rand_h] != 0:
                index_tournament.append(rand_h)
                fit[rand_h] = 0
                count += 1
        chek_mas = []
        for i in index_tournament:
            chek_mas.append(Fitness[i])
        index_selection.append(index_tournament[chek_mas.index(min(chek_mas))])
        t +=1
    return index_selection

def Mutation(Hromosoma):
    Probability_Mutation = 1.0/len(Hromosoma)

    for i in range(2, len(Hromosoma)):
        rand_val = random.uniform(0, 1)
        if rand_val < Probability_Mutation:

            if Hromosoma[i] == 1:
                Hromosoma[i] = 2
            elif Hromosoma[i] == 2:
                Hromosoma[i] = 1
    count = -1
    for i in range(2):
        if random.uniform(0, 1) < 1.0/len(Hromosoma):
            Hromosoma[0] = np.random.uniform(0.01, 0.099)
            Hromosoma[1] = np.random.uniform(0.1, 0.89)
            # n_random = 0
            # while min < n_random + ab < max:
            #     n_random = np.random.normal(0, 0.05)
            # Hromosoma[count] += n_random

    return Hromosoma

# ssdsd = [[0.01, 0.0],[0.02, 0.2],[0.04, 0.4],[0.06, 0.6],[0.08],[0.8]]
# for i in ssdsd:
#     Mutation(i)

numer_data = 2
def run_customiz_nn(norm_data):
    pop_count = 10
    pop = random_pop(pop_count)

    Fitness = []
    #Fitness = [2, 5, 10, 1, 5, 8, 7, 1]

    for i in range(len(pop)):
        now_weight = hromosoma_to_weight(pop[i])
        Fitness.append(run_neuron_network(norm_data, input_output_neurons[numer_data], now_weight[2], now_weight[0], now_weight[1], 0))
    index_best_fitness = Fitness.index(max(Fitness))
    print(pop)
    Best_individ = [pop[index_best_fitness], Fitness[index_best_fitness]]
    generation = 50
    g = 0
    ALL_Fitness = []
    while(g != generation):
        ALL_Fitness.append(Best_individ)
        g += 1
        Child = []
        Mutation_Children = []
        Fitness_Mutation_Children = []
        for i in range(len(pop)):
            index_Parens = TournamentSelection(Fitness, 2)
            Child.append(Crossover(pop[index_Parens[0]], pop[index_Parens[1]]))
        for c in range(len(Child)):
            Mutation_Children.append(Mutation(Child[c]))
            # print(Mutation(Child[c]))
            Mutation_Children_to_weight = hromosoma_to_weight(Mutation_Children[c])
            Fitness_Mutation_Children.append(run_neuron_network(norm_data, input_output_neurons[numer_data], Mutation_Children_to_weight[2], Mutation_Children_to_weight[0], Mutation_Children_to_weight[1], 1)) #!!!!!!!
        index_best_Fitness_Mutation_Children = Fitness_Mutation_Children.index(max(Fitness_Mutation_Children))
        if Fitness_Mutation_Children[index_best_Fitness_Mutation_Children] >= Best_individ[1]:
            Best_individ = [Mutation_Children[index_best_Fitness_Mutation_Children], Fitness_Mutation_Children[index_best_Fitness_Mutation_Children]]

        index_bad_Fitness_Mutation_Children = Fitness_Mutation_Children.index(min(Fitness_Mutation_Children))
        pop = Mutation_Children
        Fitness = Fitness_Mutation_Children

        pop[index_bad_Fitness_Mutation_Children] = Best_individ[0]
        Fitness[index_bad_Fitness_Mutation_Children] = Best_individ[1]
        print(Best_individ)
    return Best_individ
    #return np.array(ALL_Fitness)



Tests = 10
count_column = 1
all_accurasy_nn = []
all_best_nn = []
bests = []
start_time_train = datetime.now()
for i in range(numer_data, 3, 1):
    start_time = datetime.now()
    norm_data = data_name[i]()
    for j in range(Tests):
        best_nn, accurasy = run_customiz_nn(norm_data)
        all_accurasy_nn.append(accurasy)
        all_best_nn.append(best_nn)
        print("iteration ---- " + str(datetime.now() - start_time))
    print("TIME ---- " + str(datetime.now()-start_time))
    # save_Exel_customiz(all_accurasy_nn, all_best_nn, count_column)
    save_Exel_customiz(all_accurasy_nn, all_best_nn, count_column)
    count_column += 20
print("TIME ALL ---- " + str(datetime.now()-start_time_train) + " --------- " )


