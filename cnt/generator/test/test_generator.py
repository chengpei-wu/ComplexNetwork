import pytest

import cnt


class TestValidateInput:
    @pytest.mark.parametrize(
        'num_nodes, num_edges, is_directed, is_weighted', [
            # num_edges more than a complete graph
            (100, (100 * 99) / 2 + 1, False, False),
            # num_edges can not connect all nodes
            (100, 98, False, False),
        ]
    )
    def test_valid_parameter(self, num_nodes, num_edges, is_directed, is_weighted):
        with pytest.raises(ValueError):
            cnt.erdos_renyi_graph(num_nodes, num_edges, is_directed, is_weighted)
        # with pytest.raises(ValueError):
        #     cnt.barabasi_albert_graph(num_nodes, num_edges, is_directed, is_weighted)
        # with pytest.raises(ValueError):
        #     cnt.q_snapback_graph(num_nodes, num_edges, is_directed, is_weighted)
        # with pytest.raises(ValueError):
        #     cnt.watts_strogatz_samll_world_graph(num_nodes, num_edges, is_directed, is_weighted)
        #     cnt.newman_watts_samll_world_graph(num_nodes, num_edges, is_directed, is_weighted)
        #     cnt.generic_scale_free_graph(num_nodes, num_edges, is_directed, is_weighted)
        #     cnt.random_triangle_graph(num_nodes, num_edges, is_directed, is_weighted)
