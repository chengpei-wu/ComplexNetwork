import random
from copy import deepcopy
from typing import Union

import networkx as nx

from cnt.robustness_optimization.GA.individual import Individual
from cnt.robustness_optimization.GA.utils import make_crossover


class Population:
    def __init__(
            self, max_size: int, init_graph: Union[nx.Graph, nx.DiGraph], init_size: int,
            max_rewire: int, p_cross: float, p_mutate: float
    ):
        """
        the Population Class of evolutionary algorithm.

        Parameters
        ----------
        max_size : the max number of population size
        init_graph : the initial graph to be optimized
        init_size : the initial size of population
        max_rewire : the rewiring times while performing mutate operator
        p_cross : the possibility of performing cross operator
        p_mutate : the possibility of performing mutate operator
        """
        self.max_size = max_size
        self.init_size = init_size
        self.MaxRewire = max_rewire
        self.p_cross = p_cross
        self.p_mutate = p_mutate
        self.pop_size = 0
        self.generation = 1
        self.individuals = []
        self.initial(init_graph, init_size)

    def initial(self, init_graph: Union[nx.Graph, nx.DiGraph], init_size: int):
        """
        init population

        Parameters
        ----------
        init_graph : the initial graph to be optimized
        init_size : the initial size of population

        """
        for size in range(init_size):
            G = deepcopy(init_graph)
            rewireNum = random.randint(2, self.MaxRewire)
            G = nx.double_edge_swap(G, nswap=rewireNum)
            self.add_individual(Individual(G))

    def add_individual(self, ind: Individual):
        """
        add individual to population

        Parameters
        ----------
        ind : the individual to be added

        """
        self.individuals.append(ind)
        self.pop_size += 1

    def delete_individual(self, ind_index: int):
        """
        delete individual from population

        Parameters
        ----------
        ind_index : the individual index

        """
        assert self.pop_size > 0
        del self.individuals[ind_index]
        self.pop_size -= 1

    def replace_individual(self, ind: Individual, ind_index: int):
        """
        replace old individual with new individual

        Parameters
        ----------
        ind: the new individual
        ind_index : the old individual index

        """
        assert self.pop_size > 0
        self.individuals[ind_index] = ind

    def crossover(self):
        """
        crossover operator

        Returns
        -------

        """
        if self.pop_size < self.max_size:
            for i in range(self.pop_size, self.max_size):
                c_idx = random.randint(0, self.pop_size - 1)
                G_t = deepcopy(self.individuals[c_idx].g)
                rewireNum = random.randint(2, self.MaxRewire)
                G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
                self.add_individual(Individual(G_t))

        for idx in range(self.pop_size):
            if random.random() < self.p_cross:
                c_idx = random.randint(0, self.pop_size - 1)
                while c_idx == idx:
                    c_idx = random.randint(0, self.pop_size - 1)
                G = make_crossover(self.individuals[c_idx], self.individuals[idx], self.p_cross)
                self.replace_individual(Individual(G), c_idx)

    def mutate(self, graph_ori: Union[nx.Graph, nx.DiGraph]):

        """
        mutate operator

        Parameters
        ----------
        graph_ori : the initial graph

        """

        for i in range(self.pop_size):
            if random.random() <= self.p_mutate:
                G_t = deepcopy(self.individuals[i].g)
                rewireNum = random.randint(2, self.MaxRewire)
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

    def find_best(self) -> Individual:
        """
        find best individual from population

        Returns
        -------
        the best individual

        """
        index = 0
        for i in range(1, self.pop_size):
            if self.individuals[i].fitness > self.individuals[index].fitness:
                index = i
        return self.individuals[index]

    def selection(self):
        """
        selection operator, selecting the top-init_size best individuals

        Returns
        -------

        """
        sorted_individuals = sorted(self.individuals, key=lambda obj: obj.fitness, reverse=True)
        self.individuals = sorted_individuals[:self.init_size]
        self.pop_size = self.init_size
