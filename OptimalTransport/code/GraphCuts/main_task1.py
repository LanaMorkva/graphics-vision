from graph_cut import GraphCut
import numpy as np


def main():
    nodes = 3
    connections = 4
    graph_cut = GraphCut(nodes, connections)
    unaries = [[5,8], [7,7], [9,4]]
    pairwise = [[0,1,1,5], [1,2,2,3]]
    graph_cut.set_unary(unaries)
    graph_cut.set_pairwise(pairwise)

    print(f"Maximum flow: {graph_cut.minimize()}")
    labels = graph_cut.get_labeling()
    for i in range(nodes):
        group = "Sink" if labels[i] else "Source"
        print(f"Node {i} belongs to: {group}")


if __name__ == '__main__':
    main()
