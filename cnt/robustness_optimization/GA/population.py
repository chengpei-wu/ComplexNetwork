import random
from copy import deepcopy
from typing import Union

import networkx as nx

from .cross_over import make_crossover
from .individual import Individual


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
        self.add_individual(Individual(deepcopy(init_graph)))
        for size in range(init_size - 1):
            G = deepcopy(init_graph)
            rewireNum = random.randint(2, self.MaxRewire)
            G = nx.double_edge_swap(G, nswap=rewireNum)
            assert {d for n, d in init_graph.degree()} == {d for n, d in G.degree()}
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
                p1 = random.randint(0, self.pop_size - 1)
                p2 = random.randint(0, self.pop_size - 1)
                while p2 == p1:
                    p2 = random.randint(0, self.init_size - 1)
                g_get = make_crossover(self.individuals[p1], self.individuals[p2], self.p_cross)
                self.add_individual(Individual(g_get))

    def mutate(self, graph_ori: Union[nx.Graph, nx.DiGraph]):

        """
        mutate operator

        Parameters
        ----------
        graph_ori : the initial graph

        """

        for i in range(1, self.pop_size):
            if random.random() <= self.p_mutate:
                G_t = deepcopy(self.individuals[i].g)
                rewireNum = random.randint(2, self.MaxRewire)
                G_t = nx.double_edge_swap(G_t, nswap=rewireNum)
                if {d for n, d in self.individuals[i].g.degree()} == {d for n, d in G_t.degree()}:
                    self.replace_individual(Individual(G_t), i)

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
        for ind in self.individuals:
            if not ind.fitness:
                ind.fitness = ind.calculate_robustness(robustness='connectivity', attack='node', strategy='degree')
                ind.R = ind.fitness
        sorted_individuals = sorted(self.individuals, key=lambda obj: obj.fitness, reverse=True)
        self.individuals = sorted_individuals[:self.init_size]
        self.pop_size = self.init_size
