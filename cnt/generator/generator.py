import random
from typing import Union

import networkx as nx
import numpy as np

from cnt.utils.algorithm import havel_hakimi_process

__all__ = [
    'erdos_renyi_graph',
    'barabasi_albert_graph',
    'q_snapback_graph',
    'generic_scale_free_graph',
    'extremely_homogeneous_graph',
    'random_hexagon_graph',
    'random_triangle_graph',
    'network_with_degree_distribution',
    'newman_watts_samll_world_graph',
    'watts_strogatz_samll_world_graph',
    'multi_local_world_graph',
]


def erdos_renyi_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> Union[
    nx.Graph, nx.DiGraph]:
    """
    return Erdos Renyi (ER) graph.

    Parameters
    ----------
    num_nodes : number of nodes
    num_edges : number of edges
    is_directed : return directed graph
    is_weighted : return graph with random edge weight

    Returns
    -------
    nx.Graph or nx.DiGraph

    References
    ----------
    .. [1] P. Erdos and A. Renyi, "On the strength of connectedness of a random
        graph," Acta Mathematica Hungarica, vol. 12, no. 1-2, pp. 261–267, 1964.
    """
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)

    edge_list = []

    while len(edge_list) < num_edges:
        nodes = random.sample(range(num_nodes), 2)
        if not is_directed:
            edge = tuple(sorted(nodes))
        else:
            edge = tuple(nodes)
        if edge not in edge_list:
            edge_list.append(edge)

    for edge in edge_list:
        if is_weighted:
            weight = random.random()
            G.add_edge(*edge, weight=weight)
        else:
            G.add_edge(*edge)
    return G


def barabasi_albert_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    """
        return Barabasi Albert (BA) graph.

        Parameters
        ----------
        num_nodes : number of nodes
        num_edges : number of edges
        is_directed : return directed graph
        is_weighted : return graph with random edge weight

        Returns
        -------
        nx.Graph or nx.DiGraph

        References
        ----------
        .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
            random networks", Science 286, pp 509-512, 1999.
        """
    # calculate parameters
    n0 = int(np.ceil(num_edges / num_nodes) + 1)  # initial nodes
    n_add = num_nodes - n0  # rest nodes are added later on
    m_add = num_edges - np.math.comb(n0, 2)  # rest edges are added ...
    m = int(np.floor(m_add / n_add))  # edges to add per-generation
    delta = m_add % n_add
    ms = m * np.ones(num_nodes, dtype=int)
    ms[0:n0] = 0
    if delta:
        pos = np.arange(n0, n0 + delta)
        ms[pos] = ms[pos] + 1

    # create initial graph
    init_graph = nx.complete_graph(n0)
    # add rest nodes and edges by preferential attachment mechanism
    for m in ms[n0:]:
        degrees = dict(init_graph.degree())
        degree_sum = sum(degrees.values())
        new_node = len(init_graph)
        init_graph.add_node(new_node)
        # using degree as weight
        nodes = list(degrees.keys())
        weights = [degree / degree_sum for degree in degrees.values()]
        selected_nodes = np.random.choice(nodes, size=m, p=weights, replace=False)
        for selected_node in selected_nodes:
            init_graph.add_edge(new_node, selected_node)

    # check if directed
    if is_directed:
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(init_graph.nodes())
        for u, v in init_graph.edges():
            if random.random() < 0.5:
                directed_graph.add_edge(u, v)
            else:
                directed_graph.add_edge(v, u)
        init_graph = directed_graph
    # check if weighted
    if is_weighted:
        for u, v in init_graph.edges():
            weight = random.random()
            init_graph[u][v]['weight'] = weight

    return init_graph


def generic_scale_free_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    sfpara = {'theta': 0, 'mu': 0.999}
    np.random.seed(42)
    w = (1 + np.arange(num_nodes) + sfpara['theta']) ** (-sfpara['mu'])
    ransec = np.cumsum(w)

    # --- step(1) generate generic sf --- #
    adj = np.zeros((num_nodes, num_nodes))
    cnt = 0

    if dir:  # for directed networks
        while cnt < num_edges:
            r = np.random.rand() * ransec[-1]
            i = np.where(r <= ransec)[0][0]
            r = np.random.rand() * ransec[-1]
            j = np.where(r <= ransec)[0][0]
            if i != j and not adj[i, j]:
                adj[i, j] = 1
                cnt += 1
        if np.sum(adj) != num_edges:
            raise ValueError('Check edge sum ...')

    else:  # for undirected networks
        while cnt < num_edges:
            r = np.random.rand() * ransec[-1]
            i = np.where(r <= ransec)[0][0]
            r = np.random.rand() * ransec[-1]
            j = np.where(r <= ransec)[0][0]
            if i != j and not adj[i, j] and not adj[j, i]:
                adj[i, j] = 1
                cnt += 1
        tmpi = adj + adj.T
        if np.sum(tmpi) != 2 * num_edges:
            raise ValueError('Check edge sum ...')
        adj = tmpi
    if is_directed:
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(adj)

    # check if weighted
    if is_weighted:
        for u, v in graph.edges():
            weight = random.random()
            graph[u][v]['weight'] = weight

    return graph


def extremely_homogeneous_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError


def multi_local_world_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[nx.Graph, nx.DiGraph]:
    raise NotImplementedError


def q_snapback_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    def add_alink():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :idx] == 0)[0]

        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :idx] == 0)[0]

        jdx = np.random.choice(list)
        adj[idx, jdx] = 1

    def delete_alink():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :idx] == 1)[0]

        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :idx] == 1)[0]

        jdx = np.random.choice(list)
        adj[idx, jdx] = 0

    # for undirected networks
    def eadd_udr():
        idx = np.random.randint(num_nodes)
        list = np.where(~adj[idx, :])[0]
        list = np.setdiff1d(list, np.where(adj[:, idx])[0])

        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(~adj[idx, :])[0]
            list = np.setdiff1d(list, np.where(adj[:, idx])[0])

        jdx = np.random.choice(list)
        adj[idx, jdx] = 1
        adj[jdx, idx] = 1

    def edel_udr():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :])[0]

        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :])[0]

        jdx = np.random.choice(list)
        adj[idx, jdx] = 0
        adj[jdx, idx] = 0

    def q_nlink(r, n, s, x):
        # If s == 'q2e', x is the value of q in [0, 1]
        # If s == 'e2q', x is the number of edges

        if s == 'q2e':
            return q2nlink(r, n, x)
        elif s == 'e2q':
            return nlink2q(r, n, x)
        else:
            raise ValueError('Invalid input for s')

    def nlink2q(r, n, nlink):
        PRECISION = 1E5
        q_qsn = (nlink - n) / np.sum(np.arange(r + 1, n - 1) / (r - 1))
        q_qsn = round(q_qsn * PRECISION) / PRECISION
        return q_qsn

    def q2nlink(r, n, q):
        deg = np.zeros(n)
        for i in range(r):
            deg[i] = 1
        for i in range(r, n):
            deg[i] = 1 + np.floor((i - 2) / (r - 1)) * q
        total_deg = round(np.sum(deg))
        return total_deg

    r = 2
    itop = 'chain'
    q = q_nlink(r, num_nodes, 'e2q', num_edges)  # estimate 'q' according to (r, n, m)

    # --- step(1) generate a qsn --- #
    adj = np.zeros((num_nodes, num_nodes))
    for rdx in range(1, r + 1):
        if rdx == 1:
            if itop == 'chain':
                adj = np.diag(np.ones(num_nodes - 1), 1)
            elif itop == 'ring':
                adj = np.diag(np.ones(num_nodes - 1), 1)
                adj[num_nodes - 1, 0] = 1
            elif itop == 'tree':
                for i in range(1, num_nodes):
                    adj[i, i + np.random.randint(num_nodes - i)] = 1
                adj[num_nodes - 1, 0] = 1
        else:
            for i in range(rdx, num_nodes):
                for j in range(i - rdx, rdx - 1, -1):
                    if np.random.rand() <= q:
                        adj[i, j] = 1

    # --- step(2) control the exact number of edges --- #
    cnt = np.sum(adj)
    deltaE = cnt - num_edges

    if deltaE > 0:
        for i in range(int(deltaE)):
            delete_alink()
    elif deltaE < 0:
        deltaE = (abs(int(deltaE)))
        for i in range(deltaE):
            add_alink()

    if not is_directed:  # undirected
        m2 = num_edges * 2
        adj = adj + adj.T
        cnt = np.sum(adj)

        while cnt > m2:
            deltaE = abs(cnt - m2)
            for i in range(1, deltaE, 2):
                edel_udr()
            cnt = np.sum(adj)

        while cnt < m2:
            cnt = np.sum(adj)
            deltaE = abs(cnt - m2)
            for i in range(1, deltaE, 2):
                eadd_udr()
            cnt = np.sum(adj)

        if np.sum(adj) != m2:
            raise ValueError('Check edge_sum ...')

    if is_directed:
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(adj)

    # check if weighted
    if is_weighted:
        for u, v in graph.edges():
            weight = random.random()
            graph[u][v]['weight'] = weight
    return graph


def random_hexagon_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


def random_triangle_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    def add_edge():
        r = np.random.choice(num_nodes, 2, replace=False)
        r1, r2 = r[0], r[1]
        while adj[r1, r2] or adj[r2, r1]:
            r = np.random.choice(num_nodes, 2, replace=False)
            r1, r2 = r[0], r[1]
        neb1 = np.append(np.where(adj[r2] == 1)[0], np.where(adj[:, r2] == 1)[0])
        len_neb1 = len(neb1)
        if len(np.unique(neb1)) < len_neb1:
            raise Exception('.. check !')
        r3 = neb1[np.random.randint(len_neb1)]
        if adj[r3, r2]:
            adj[r2, r1] = 1
            if not adj[r1, r3] and not adj[r3, r1]:
                adj[r1, r3] = 1
        else:
            adj[r1, r2] = 1
            if not adj[r1, r3] and not adj[r3, r1]:
                adj[r3, r1] = 1

    def delete_edge():
        r1 = np.random.randint(num_nodes)
        neb1 = np.append(np.where(adj[r1] == 1)[0], np.where(adj[:, r1] == 1)[0])
        len_neb1 = len(neb1)
        if len(np.unique(neb1)) < len_neb1:
            raise Exception('.. check !')
        r2 = neb1[np.random.randint(len_neb1)]
        adj[r1, r2] = 0
        adj[r2, r1] = 0

    def add_one_edge_uda():
        r1 = np.random.randint(num_nodes)
        neb1 = np.where(uda[r1] == 0)[0]
        neb1 = neb1[neb1 != r1]
        len_neb1 = len(neb1)
        while len_neb1 == 0:
            r1 = np.random.randint(num_nodes)
            neb1 = np.where(uda[r1] == 0)[0]
            neb1 = neb1[neb1 != r1]
            len_neb1 = len(neb1)
        r2 = neb1[np.random.randint(len_neb1)]
        if not uda[r2, r1]:
            uda[r1, r2] = 1
            uda[r2, r1] = 1

    Tri = 3
    # Step(1) generate basic random triangles
    adj = np.zeros((num_nodes, num_nodes))
    for idx in range(Tri):
        adj[idx, (idx + 1) % Tri] = 1
    for idx in range(Tri, num_nodes):
        jdx = np.random.randint(idx)
        neb = np.append(np.where(adj[jdx] == 1)[0], np.where(adj[:, jdx] == 1)[0])
        len_neb = len(neb)
        if len(np.unique(neb)) < len_neb:
            raise Exception('check!')
        kdx = neb[np.random.randint(len_neb)]
        if adj[kdx, jdx]:
            tmpv = kdx
            kdx = jdx
            jdx = tmpv
        adj[idx, jdx] = 1
        adj[kdx, idx] = 1

    # step(2) control exact number of edges
    cnt = np.sum(adj)
    deltaE = cnt - num_edges
    if deltaE > 0:
        while cnt > num_edges:
            delete_edge()
            cnt = np.sum(adj)
    elif deltaE < 0:
        while cnt < num_edges:
            add_edge()
            cnt = np.sum(adj)
        while cnt > num_edges:
            delete_edge()
            cnt = np.sum(adj)

    # step(3) check output
    if is_directed:  # for directed networks
        if np.sum(adj) != num_edges:
            raise Exception('check edge sum ...')

    else:  # for undirected networks
        m2 = num_edges * 2
        uda = adj + adj.T
        deltaE = np.sum(uda) - m2
        if deltaE > 0:
            raise Exception('check edge sum ...')
        while deltaE:
            add_one_edge_uda()
            deltaE = np.sum(uda) - m2
        adj = uda

    if is_directed:
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(adj)

    # check if weighted
    if is_weighted:
        for u, v in graph.edges():
            weight = random.random()
            graph[u][v]['weight'] = weight

    return graph


def newman_watts_samll_world_graph(num_nodes: int, num_edges: int, is_directed: bool = False,
                                   is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    def eadd_dir():
        idx = np.random.randint(num_nodes)
        list = np.where(np.logical_not(adj[idx, :]))[0]
        list = list[list != idx]
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(np.logical_not(adj[idx, :]))[0]
            list = list[list != idx]
        jdx = np.random.choice(list)
        adj[idx, jdx] = 1

    def edel_dir():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :])[0]
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :])[0]
        jdx = np.random.choice(list)
        adj[idx, jdx] = 0

    def eadd_udr():
        idx = np.random.randint(num_nodes)
        list = np.where(np.logical_not(adj[idx, :]))[0]
        list = list[list != idx]
        list = np.setdiff1d(list, np.where(adj[:, idx])[0])
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(np.logical_not(adj[idx, :]))[0]
            list = list[list != idx]
            list = np.setdiff1d(list, np.where(adj[:, idx])[0])
        jdx = np.random.choice(list)
        adj[idx, jdx] = 1
        adj[jdx, idx] = 1

    def edel_udr():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :])[0]
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :])[0]
        jdx = np.random.choice(list)
        adj[idx, jdx] = 0
        adj[jdx, idx] = 0

    if is_directed:
        # --- step(1) construct a ring (k==2) --- #
        adj = np.diag(np.ones(num_nodes - 1), 1) + np.diag(np.ones(num_nodes - num_nodes + 1), -(num_nodes - 1))
        adj = adj + np.diag(np.ones(num_nodes - 2), -2).T + np.diag(np.ones(num_nodes - num_nodes + 2), num_nodes - 2).T
        # --- step(2) adds shortcut edges --- #
        edge_sum = np.sum(adj)
        while edge_sum > num_edges:
            print('   .. (too small m-val) deleting edges .. ')
            deltaE = abs(edge_sum - num_edges)
            for i in range(deltaE):
                edel_dir()
            edge_sum = np.sum(adj)
        while edge_sum < num_edges:
            deltaE = abs(edge_sum - num_edges)
            for i in range(int(deltaE)):
                eadd_dir()
            edge_sum = np.sum(adj)
        if np.sum(adj) != num_edges:
            raise ValueError('check edge_sum ...')

    else:
        m2 = num_edges * 2
        # --- step(1) construct a ring (k==2) --- #
        adj = np.diag(np.ones(num_nodes - 1), 1) + np.diag(np.ones(num_nodes - num_nodes + 1), -(num_nodes - 1))
        adj = adj + np.diag(np.ones(num_nodes - 2), -2).T + np.diag(np.ones(num_nodes - num_nodes + 2), num_nodes - 2).T
        adj = adj + adj.T
        # --- step(2) adds shortcut edges --- #
        edge_sum = np.sum(adj)
        while edge_sum > m2:
            print('   .. (too small m-val) deleting edges .. ')
            deltaE = abs(edge_sum - m2)
            for i in range(deltaE, 2):
                edel_udr()
            edge_sum = np.sum(adj)
        while edge_sum < m2:
            edge_sum = np.sum(adj)
            deltaE = abs(edge_sum - m2)
            for i in range(0, int(deltaE), 2):
                eadd_udr()
            edge_sum = np.sum(adj)
        if np.sum(adj) != m2:
            raise ValueError('check edge_sum ...')

    if is_directed:
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(adj)

    # check if weighted
    if is_weighted:
        for u, v in graph.edges():
            weight = random.random()
            graph[u][v]['weight'] = weight

    return graph


def watts_strogatz_samll_world_graph(num_nodes: int, num_edges: int, is_directed: bool = False,
                                     is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    def eadd_dir():
        idx = np.random.randint(num_nodes)
        list = np.where(np.logical_not(adj[idx, :]))[0]
        list = list[list != idx]
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(np.logical_not(adj[idx, :]))[0]
            list = list[list != idx]
        jdx = np.random.choice(list)
        adj[idx, jdx] = 1

    def edel_dir():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :])[0]
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :])[0]
        jdx = np.random.choice(list)
        adj[idx, jdx] = 0

    def eadd_udr():
        idx = np.random.randint(num_nodes)
        list = np.where(np.logical_not(adj[idx, :]))[0]
        list = list[list != idx]
        list = np.setdiff1d(list, np.where(adj[:, idx])[0])
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(np.logical_not(adj[idx, :]))[0]
            list = list[list != idx]
            list = np.setdiff1d(list, np.where(adj[:, idx])[0])
        jdx = np.random.choice(list)
        adj[idx, jdx] = 1
        adj[jdx, idx] = 1

    def edel_udr():
        idx = np.random.randint(num_nodes)
        list = np.where(adj[idx, :])[0]
        while len(list) == 0:
            idx = np.random.randint(num_nodes)
            list = np.where(adj[idx, :])[0]
        jdx = np.random.choice(list)
        adj[idx, jdx] = 0
        adj[jdx, idx] = 0

    prop = 10

    if is_directed:
        # --- step(1) construct a backbone ring (k==2)--- #
        adj = np.diag(np.ones(num_nodes - 1), 1) + np.diag(np.ones(num_nodes - num_nodes + 1), -(num_nodes - 1))
        adj = adj + np.diag(np.ones(num_nodes - 2), -2).T + np.diag(np.ones(num_nodes - num_nodes + 2), num_nodes - 2).T
        # --- step(2) adds shortcut edges --- #
        edge_sum = np.sum(adj)
        while edge_sum > num_edges:
            print('   .. (too small m-val) deleting edges .. ')
            deltaE = abs(edge_sum - num_edges)
            for i in range(deltaE):
                edel_dir()
            edge_sum = np.sum(adj)
        while edge_sum < num_edges:
            deltaE = abs(edge_sum - num_edges)
            for i in range(int(deltaE) // prop):
                edel_dir()
            edge_sum = np.sum(adj)
            deltaE = abs(edge_sum - num_edges)
            for i in range(int(deltaE)):
                eadd_dir()
            edge_sum = np.sum(adj)
        if np.sum(adj) != num_edges:
            raise ValueError('check edge_sum ...')

    else:
        m2 = num_edges * 2
        # --- step(1) construct a backbone ring (k==2)--- #
        adj = np.diag(np.ones(num_nodes - 1), 1) + np.diag(np.ones(num_nodes - num_nodes + 1), -(num_nodes - 1))
        adj = adj + np.diag(np.ones(num_nodes - 2), -2).T + np.diag(np.ones(num_nodes - num_nodes + 2), num_nodes - 2).T
        adj = adj + adj.T
        # --- step(2) adds shortcut edges --- #
        edge_sum = np.sum(adj)
        if edge_sum < m2:
            for i in range(1, int(abs(edge_sum - m2) // prop), 2):
                edel_udr()
        edge_sum = np.sum(adj)
        while edge_sum > m2:
            print('   .. (too small m-val) deleting edges .. ')
            deltaE = abs(edge_sum - m2)
            for i in range(1, deltaE, 2):
                edel_udr()
            edge_sum = np.sum(adj)
        while edge_sum < m2:
            edge_sum = np.sum(adj)
            deltaE = abs(edge_sum - m2)
            for i in range(1, int(deltaE), 2):
                eadd_udr()
            edge_sum = np.sum(adj)
        if np.sum(adj) != m2:
            raise ValueError('check edge_sum ...')

    if is_directed:
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(adj)

    # check if weighted
    if is_weighted:
        for u, v in graph.edges():
            weight = random.random()
            graph[u][v]['weight'] = weight

    return graph


def network_with_degree_distribution(num_nodes: int, avg_degree: int, degree_distribution: str):
    """
    Parameters
    ----------
    degree_distribution : the network degree distribution
    num_nodes : number of nodes
    avg_degree : mean degree

    Returns
    -------
    an undirected network with specified degree distribution

    """

    if degree_distribution == "poisson":
        degree_sequence = np.random.poisson(lam=avg_degree, size=num_nodes)
    elif degree_distribution == "uniform":
        degree_sequence = np.random.uniform(avg_degree - 2, avg_degree + 2, size=num_nodes)
    elif degree_distribution == "normal":
        degree_sequence = np.random.normal(avg_degree, 1, size=num_nodes)
    elif degree_distribution == "power-law":
        degree_sequence = np.random.zipf(2, size=num_nodes)
    else:
        raise NotImplementedError
    degree_sequence = havel_hakimi_process(list(degree_sequence))

    G = nx.havel_hakimi_graph(degree_sequence)

    return G
