import numpy as np
import networkx as nx
import sys
import time
from mpi4py import MPI
from mpi4py.util import pkl5
import math
import json

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def read_graph():
    '''
        Reads in NetworkX graph from either the Twitter and Facebook edge list file
        depending on command line arguments.

        Returns:
        - NetworkX graph represented in the Twitter or Facebook file
    '''
    if sys.argv[1] == 'facebook':
        graph = nx.read_edgelist('facebook_combined.txt.gz', create_using=nx.Graph, edgetype=int)
        graph.name = 'facebook'
    else:
        graph = nx.read_edgelist('twitter_combined.txt.gz', create_using=nx.DiGraph, edgetype=int)
        graph.name = 'twitter'
    
    return graph

def parallel_closeness(G: nx.Graph):
    '''
        Calculates parallel closeness with graph G.
    '''
    start = time.process_time()
    N = len(G.nodes)
    nodes = list(G.nodes)
    closeness = {}
    Qn = math.ceil(N / size)

    # assign graph node slices to all processors
    start = rank * Qn
    end = min(N, (rank + 1) * Qn)

    # calculate closeness centrality by running single source dijkstra on all nodes
    for node in nodes[start:end]:
        shortest_path_lengths = nx.single_source_dijkstra_path_length(G, node)
        average_path_length = sum(shortest_path_lengths.values()) / (N - 1)
        if average_path_length == 0:
            closeness[node] = 0
        else:
            closeness[node] = 1 / average_path_length
    
    # send closeness slice back to processor 0
    if rank != 0:
        comm.send(closeness, dest=0)
    else:
        for r in range(1, size):
            other_closeness = comm.recv(source=r)
            closeness.update(other_closeness)

        end = time.process_time()
        time_elapsed = end - start
        
        with open(f'{G.name}_P{size}_closeness.txt', 'w') as file:
            file.write(f'Run time: {time_elapsed} seconds\n')
            json.dump(closeness, file, indent=4)
        
        return closeness

def get_top_closeness(closeness):
    '''
        Prints the values of the top 5 nodes with the highest closeness centrality.
    '''
    sorted_closeness = {key: value for key, value in sorted(closeness.items(), key=lambda item: item[1], reverse=True)}

    print('Top 5 Nodes with Highest Closeness Centrality')
    print('-----------------------')
    
    top_5 = list(sorted_closeness.keys())[:5]
    for node in top_5:
        print(f'{node}:\t{sorted_closeness[node]}')

def get_average_closeness(closeness):
    '''
        Prints the average closeness centrality value.
    '''
    values = closeness.values()
    average_closeness = sum(values) / len(values)
    print('Average Node Centrality Value')
    print('-----------------------')
    print(average_closeness)

def main():
    G = read_graph()
    closeness = parallel_closeness(G)

    if rank == 0:
        get_top_closeness(closeness)
        print()
        get_average_closeness(closeness)

if __name__ == '__main__':
    main()