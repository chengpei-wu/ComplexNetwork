import pickle
import random
from typing import Union, List

from cnt.generator.generator import erdos_renyi_graph, barabasi_albert_graph
from cnt.robustness.simulated_attack import connectivity_robustness, controllability_robustness, \
    communicability_robustness
from cnt.utils.tool_functions import print_progress


def create_network_instances(topology_type: str, is_directed: bool, is_weighted: bool, num_instance: int,
                             network_size: Union[int, tuple],
                             average_degree: Union[float, int, tuple]) -> list:
    networks = []

    for i in range(num_instance):
        print_progress(now=i, total=num_instance, length=40, prefix=f'generating {topology_type.upper()} networks:')
        if isinstance(network_size, tuple):
            num_nodes = random.randint(*network_size)
        else:
            num_nodes = network_size

        if isinstance(average_degree, tuple):
            k = random.uniform(*average_degree)
            if is_directed:
                num_edges = round(k * num_nodes)
            else:
                num_edges = round((k * num_nodes) / 2)
        else:
            k = average_degree
            if is_directed:
                num_edges = round(k * num_nodes)
            else:
                num_edges = round((k * num_nodes) / 2)

        if topology_type == 'er':
            graph = erdos_renyi_graph(
                num_nodes=num_nodes,
                num_edges=num_edges,
                is_directed=is_directed,
                is_weighted=is_weighted
            )
        if topology_type == 'ba':
            graph = barabasi_albert_graph(
                num_nodes=num_nodes,
                num_edges=num_edges,
                is_directed=is_directed,
                is_weighted=is_weighted
            )
        networks.append(graph)
    return networks


def save_simulated_network_dataset(
        topology_types: List[str], is_directed: bool, is_weighted: bool,
        num_instance: int,
        network_size: Union[int, tuple],
        average_degree: Union[float, int, tuple],
        **kwargs
):
    graghs = []
    graph_labels = []
    connectivity_curves = []
    controllability_curves = []
    communicability_curves = []

    for topology_type in topology_types:
        network_instances = create_network_instances(
            topology_type=topology_type,
            is_directed=is_directed,
            is_weighted=is_weighted,
            num_instance=num_instance,
            network_size=network_size,
            average_degree=average_degree
        )
        for i, network_instance in enumerate(network_instances):
            graghs.append(network_instance)
            graph_labels.append(topology_type)
            save_robustness = kwargs.get('save_robustness', None)
            if save_robustness:
                print_progress(now=i, total=num_instance,
                               prefix=f'calculating robustness of {topology_type.upper()} networks: ')
                for r in save_robustness:
                    if r == 'connectivity':
                        curve, _ = connectivity_robustness(
                            graph=network_instance,
                            attack=kwargs.get('attack', 'unknown'),
                            strategy=kwargs.get('strategy', 'unknown'),
                        )
                        connectivity_curves.append(curve)
                    elif r == 'controllability':
                        curve, _ = controllability_robustness(
                            graph=network_instance,
                            attack=kwargs.get('attack', 'unknown'),
                            strategy=kwargs.get('strategy', 'unknown'),
                        )
                        controllability_curves.append(curve)
                    elif r == 'communicability':
                        curve, _ = communicability_robustness(
                            graph=network_instance,
                            attack=kwargs.get('attack', 'unknown'),
                            strategy=kwargs.get('strategy', 'unknown'),
                        )
                        communicability_curves.append(curve)
                    else:
                        raise NotImplementedError(f'{r} not implemented')
    res = {
        'graphs': graghs,
        'labels': graph_labels,
    }

    if 'connectivity' in kwargs.get('save_robustness', None):
        res['connectivity_curves'] = connectivity_curves
    if 'controllability' in kwargs.get('save_robustness', None):
        res['controllability_curves'] = controllability_curves
    if 'communicability' in kwargs.get('save_robustness', None):
        res['communicability_curves'] = communicability_curves

    save_path = kwargs.get('save_path', None)
    if save_path:
        with open(f'{save_path}.pkl', 'wb') as file:
            pickle.dump(res, file)
            print(f'Simulated network dataset save at {save_path}.pkl')
    return res


def load_simulated_network_dataset(load_path: str) -> dict:
    with open(f'{load_path}.pkl', 'rb') as file:
        res = pickle.load(file)
        return res
