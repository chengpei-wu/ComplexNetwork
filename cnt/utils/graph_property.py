from cnt.spectral_measure.robustness_spectral_measure import *


def get_graph_props(graph: Union[nx.Graph, nx.DiGraph], props):
    """
    get graph-level properties of a graph

    Parameters
    ----------
    graph : input graph
    props : the properties

    Returns
    -------
    the list of graph properties

    """
    graph_props = []
    if 'betweenness' in props:
        graph_props.append(sum(nx.betweenness_centrality(graph).values()) / len(graph))
    if 'clustering' in props:
        graph_props.append(sum(nx.clustering(graph).values()) / len(graph))
    if 'closeness' in props:
        graph_props.append(sum(nx.closeness_centrality(graph).values()) / len(graph))
    if 'eigenvector' in props:
        graph_props.append(sum(nx.eigenvector_centrality(graph).values()) / len(graph))
    if 'katz' in props:
        graph_props.append(sum(nx.katz_centrality(graph).values()) / len(graph))
    if 'pageRank' in props:
        graph_props.append(sum(nx.pagerank(graph).values()) / len(graph))
    if 'efficiency' in props:
        graph_props.append(nx.global_efficiency(graph))
    if 'girth' in props:
        cycles0 = list(nx.cycle_basis(graph))
        graph_props.append(min([len(cycle) for cycle in cycles0]))
    if 'max_cycle' in props:
        cycles0 = list(nx.cycle_basis(graph))
        graph_props.append(max([len(cycle) for cycle in cycles0]))
    if 'num_cycles' in props:
        cycles0 = list(nx.cycle_basis(graph))
        graph_props.append(len(cycles0))
    if 'assortativity' in props:
        graph_props.append(assortativity(graph))
    if 'spectral_radius' in props:
        graph_props.append(spectral_radius(graph))
    if 'spectral_gap' in props:
        graph_props.append(spectral_gap(graph))
    if 'natural_connectivity' in props:
        graph_props.append(natural_connectivity(graph))
    if 'algebraic_connectivity' in props:
        graph_props.append(algebraic_connectivity(graph))
    if 'effective_resistance' in props:
        graph_props.append(effective_resistance(graph))
    if 'spanning_tree_count' in props:
        graph_props.append(spanning_tree_count(graph))

    return graph_props
