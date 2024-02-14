import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_trajectory_graph(trajectory_graph: nx.DiGraph,
                               first_node: int = None,
                               last_node: int = None,
                               whole_graph: bool = False,
                               ax: plt.Axes=None):
    nodes_list = sorted(list(trajectory_graph.nodes()))

    if whole_graph:
        G = trajectory_graph
    else:
        if first_node is None:
            first_node_index = np.random.randint(len(nodes_list) - 20)
            first_node = nodes_list[first_node_index]
        else:
            nodes_before = [n for n in nodes_list if n<=first_node]
            if len(nodes_before) > 0:
                first_node = nodes_before[-1]
            else:
                nodes_after = [n for n in nodes_list if n>first_node]
                first_node = nodes_after[0]

        if last_node is None:
            last_node = nodes_list[nodes_list.index(first_node) + 20]

        G = trajectory_graph.subgraph(range(first_node, last_node+1))

    if ax is None:
        w, h, dpi = 1800, 600, 100
        fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    else:
        fig = ax.get_figure()

    # Create a linear layout
    pos = {node: (i, 0) for i, node in enumerate(sorted(G.nodes()))}
    print(len(pos))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, linewidths=2)

    # Draw the curved edges
    for u, v, weight in G.edges.data('weight'):
        edge_color = 'r' if weight > 0 else 'k'
        # d = -0.2 * np.log(v-u)  # Control the curvature of the edges
        x = nodes_list.index(v) - nodes_list.index(u)
        d = 0.5*(1-np.exp(-0.4*(x-1)))  # Control the curvature of the edges
        d = d if u%2 == 1 else -d
        xs, ys = pos[u]
        xt, yt = pos[v]
        xc = (xs + xt) / 2
        yc = (ys + yt) / 2
        xc += d * (yt - ys)
        yc += d * (xs - xt)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=edge_color, width=1, alpha=1, connectionstyle=f'arc3,rad={d}', arrowstyle='-|>', arrowsize=10)

    # Add labels to the nodes
    nx.draw_networkx_labels(G, pos, font_color='k')

    # Set the x-axis limits to include the nodes
    ax.set_xlim(-0.5, len(G.nodes())-0.5)
    # Set the y-axis limits
    ax.set_ylim([-1, 1])

    fig.tight_layout()

    # Show the graph
    ax.set_axis_off()

    return ax
