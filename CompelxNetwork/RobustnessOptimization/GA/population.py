import random
from copy import deepcopy

import networkx as nx

from EA.individual import Individual
from EA.utils import Make_crossover
from configuration import *


class Population:
    def __init__(self, max_size: int, init_graph: nx.Graph, init_size: int):
        self.max_size = max_size
        self.init_size = init_size
        self.pop_size = 0
        self.generation = 1
        self.individuals = []
        self.initial(init_graph, init_size)

    def initial(self, init_graph: nx.Graph, init_size: int):
        for size in range(init_size):
            G = deepcopy(init_graph)
            rewireNum = random.randint(2, MaxRewire)
            G = nx.double_edge_swap(G, nswap=rewireNum)
            self.add_individual(Individual(G))

    def add_individual(self, ind: Individual):
        self.individuals.append(ind)
        self.pop_size += 1

    def delete_individual(self, ind_index: int):
        assert self.pop_size > 0
        del self.individuals[ind_index]
        self.pop_size -= 1

    def replace_individual(self, ind: Individual, ind_index: int):
        assert self.pop_size > 0
        self.individuals[ind_index] = ind

    def crossover(self):
        if self.pop_size < self.max_size:
            for i in range(self.pop_size, self.max_size):
                c_idx = random.randint(0, self.pop_size - 1)
                G_t = deepcopy(self.individuals[c_idx].g)
                rewireNum = random.randint(2, MaxRewire)
                G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
                self.add_individual(Individual(G_t))

        for idx in range(self.pop_size):
            if random.random() < p_cross:
                c_idx = random.randint(0, self.pop_size - 1)
                while c_idx == idx:
                    c_idx = random.randint(0, self.pop_size - 1)
                G = Make_crossover(self.individuals[c_idx], self.individuals[idx])
                self.replace_individual(Individual(G), c_idx)

    def mutate(self, graph_ori):
        for i in range(self.pop_size):
            if random.random() <= p_mutate:
                G_t = deepcopy(self.individuals[i].g)
                rewireNum = random.randint(2, MaxRewire)
                # for j in range(MaxAdd):
                #     random_nodes = random.sample(G_t.nodes(), 2)
                #     new_edge = tuple(random_nodes)
                #     while new_edge in G_t.edges:
                #         random_nodes = random.sample(G_t.nodes(), 2)
                #         new_edge = tuple(random_nodes)
                #     G_t.add_edge(*new_edge)
                #     random_edge = random.choice(list(G_t.edges()))
                #     G_t.remove_edge(*random_edge)
                G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
                self.replace_individual(Individual(G_t), i)
            self.individuals[i].R = self.individuals[i].cal_R()
            # self.individuals[i].EMD = self.individuals[i].cal_EMD(graph_ori)
            self.individuals[i].fitness = self.individuals[i].R

    def find_best(self):
        index = 0
        for i in range(1, self.pop_size):
            if self.individuals[i].fitness > self.individuals[index].fitness:
                index = i
        return self.individuals[index]

    def selection(self):
        sorted_individuals = sorted(self.individuals, key=lambda obj: obj.fitness, reverse=True)
        self.individuals = sorted_individuals[:self.init_size]
        self.pop_size = self.init_size

    def display_pop(self):
        Rs = [ind.R for ind in self.individuals]
        EMDs = [ind.EMD for ind in self.individuals]
        fitness = [ind.fitness for ind in self.individuals]
        print(Rs)
        print(EMDs)
        print(fitness)
