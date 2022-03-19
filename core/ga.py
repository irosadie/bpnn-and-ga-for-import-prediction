import random
from functools import reduce


class GeneticAlgorithm:
    fitness = None
    chromosome = None
    prob_co = 0.9,
    alfa_range = [-0.25, 1.25],
    newron_n = 12
    newron_gen_n = 12
    bias_input_n = 1
    bias_input_gen_n = 12
    weight_n = 1
    weight_gen_n = 12
    bias_output_n = 1
    bias_output_gen_n = 1
    prob_m = 0.1
    r = 0.075

    def __init__(self, fitness, chromosome, alfa_range, prob_co, prob_m, r):
        self.fitness = fitness
        self.chromosome = chromosome
        self.alfa_range = alfa_range
        self.prob_co = prob_co
        self.prob_m = prob_m
        self.r = r

    def selection(self):
        total = sum(self.fitness)
        fitness_relative = []
        fitness_cumulative = []
        total_cumulative = 0
        start = 0.
        for key, item in enumerate(self.fitness):
            is_value = (item/total)
            if(key < (len(self.fitness)-1)):
                total_cumulative += is_value
            else:
                total_cumulative = 1.
            fitness_relative.append(is_value)
            fitness_cumulative.append([start, total_cumulative])
            start = total_cumulative
        position = []
        is_random = [round(random.uniform(0, 1), 2)
                     for i in range(0, len(self.fitness))]
        for key, item in enumerate(fitness_cumulative):
            for i in is_random:
                if(i >= item[0] and i <= item[1]):
                    position.append(key)
        return [self.chromosome[i] for i in position]

    def crossover(self):
        is_random = []
        for key, item in enumerate(self.fitness):
            rand_val = round(random.uniform(0, 1), 2)
            if(rand_val < self.prob_co):
                is_random.append(key)
        is_random.pop() if len(is_random) % 2 == 1 else None
        is_grouping = []
        for k, i in enumerate(is_random):
            if(k % 2 == 0):
                is_grouping.append([is_random[k], is_random[k+1]])

        for i in is_grouping:
            newron = []
            isalfa = []
            for is_key, y in enumerate(i):
                tmp_newron = (reduce(lambda a, b: a+b, self.chromosome[y]['input_hidden']['newron']) + self.chromosome[y]['input_hidden']
                              ['bias'] + self.chromosome[y]['output_hidden']['weight'] + [self.chromosome[y]['output_hidden']['bias']])
                newron.append(tmp_newron)
                alfa_helper = []
                for j in tmp_newron:
                    alfa_helper.append(random.uniform(
                        self.alfa_range[0], self.alfa_range[1]))
                isalfa.append(alfa_helper)
            for n_key, n_item in enumerate(newron):
                is_co_result_tmp = []
                if(n_key == 0):
                    for j_key, j_item in enumerate(n_item):
                        is_co_result_tmp.append(
                            (j_item+isalfa[n_key][j_key])*(newron[n_key+1][j_key]-j_item))
                else:
                    for j_key, j_item in enumerate(n_item):
                        is_co_result_tmp.append(
                            (j_item+isalfa[n_key][j_key])*(newron[n_key-1][j_key]-j_item))

                # zerro is not allowed
                if(sum(is_co_result_tmp) != 0.0):
                    replace_chromosome = {'input_hidden': {'newron': [is_co_result_tmp[:(self.newron_gen_n*1)], is_co_result_tmp[(self.newron_gen_n*1):(self.newron_gen_n*2)], is_co_result_tmp[(self.newron_gen_n*2):(self.newron_gen_n*3)], is_co_result_tmp[(self.newron_gen_n*3):(self.newron_gen_n*4)], is_co_result_tmp[(self.newron_gen_n*4):(12*5)], is_co_result_tmp[(self.newron_gen_n*5):(self.newron_gen_n*6)], is_co_result_tmp[(self.newron_gen_n*6):(self.newron_gen_n*7)], is_co_result_tmp[(
                        self.newron_gen_n*7):(self.newron_gen_n*8)], is_co_result_tmp[(self.newron_gen_n*8):(self.newron_gen_n*9)], is_co_result_tmp[(self.newron_gen_n*9):(self.newron_gen_n*10)], is_co_result_tmp[(self.newron_gen_n*10):(self.newron_gen_n*11)], is_co_result_tmp[(self.newron_gen_n*11):(self.newron_gen_n*12)]], 'bias': is_co_result_tmp[(self.bias_input_gen_n*12):(self.bias_input_gen_n*13)]}, 'output_hidden': {'weight': is_co_result_tmp[(self.weight_gen_n*13):(self.weight_gen_n*14)], 'bias': is_co_result_tmp[-2:-1][0]}}
                    for is_key, y in enumerate(i):
                        self.chromosome[y] = replace_chromosome

        return self.chromosome

    def mutaion(self):
        is_tmp_chromosome = []
        for key, item in enumerate(self.chromosome):
            is_tmp_chromosome.append(reduce(lambda a, b: a+b, self.chromosome[key]['input_hidden']['newron']) + self.chromosome[key]
                                     ['input_hidden']['bias'] + self.chromosome[key]['output_hidden']['weight'] + [self.chromosome[key]['output_hidden']['bias']])

        is_list_vertical = []
        for k, i in enumerate(is_tmp_chromosome[0]):
            list_vertical = []
            for n_k, n_i in enumerate(is_tmp_chromosome):
                list_vertical.append(is_tmp_chromosome[n_k][k])
            is_list_vertical.append(list_vertical)

        ismin = [min(k) for k in is_list_vertical]
        ismax = [max(k) for k in is_list_vertical]

        # generate random data in is_tmp_chromosome
        for key, item in enumerate(is_tmp_chromosome):
            for k, i in enumerate(item):
                rand_value = random.uniform(0, 1)
                if(rand_value < self.prob_m):
                    is_tmp_chromosome[key][k] = (i+self.r*(ismax[k]-ismin[k]))

        for key, item in enumerate(is_tmp_chromosome):
            is_replace = {'input_hidden': {'newron': [item[:(self.newron_gen_n*1)], item[(self.newron_gen_n*1):(self.newron_gen_n*2)], item[(self.newron_gen_n*2):(self.newron_gen_n*3)], item[(self.newron_gen_n*3):(self.newron_gen_n*4)], item[(self.newron_gen_n*4):(12*5)], item[(self.newron_gen_n*5):(self.newron_gen_n*6)], item[(self.newron_gen_n*6):(self.newron_gen_n*7)], item[(self.newron_gen_n*7):(
                self.newron_gen_n*8)], item[(self.newron_gen_n*8):(self.newron_gen_n*9)], item[(self.newron_gen_n*9):(self.newron_gen_n*10)], item[(self.newron_gen_n*10):(self.newron_gen_n*11)], item[(self.newron_gen_n*11):(self.newron_gen_n*12)]], 'bias': item[(self.bias_input_gen_n*12):(self.bias_input_gen_n*13)]}, 'output_hidden': {'weight': item[(self.weight_gen_n*13):(self.weight_gen_n*14)], 'bias': item[-2:-1][0]}}
            self.chromosome[key] = is_replace

        return self.chromosome

    def geneticAlgorithm(self):
        selection = self.selection()
        crossover = self.crossover()
        mutation = self.mutaion()
        return {'selection': selection, 'crossover': crossover, 'mutation': mutation}
