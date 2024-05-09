import numpy as np
import networkx as nx
import sys

def read_graph():
    '''
        Reads in NetworkX graph from either the Twitter and Facebook edge list file
        depending on command line arguments.

        Returns:
        - NetworkX graph represented in the Twitter or Facebook file
    '''
    if sys.argv[1] == 'facebook':
        return nx.read_edgelist('facebook_combined.txt.gz', create_using=nx.Graph)
    else:
        return nx.read_edgelist('twitter_combined.txt.gz', create_using=nx.DiGraph)

def parallel_closeness(G):
    '''
        Calculates parallel closeness with graph G.
    '''
    pass

def get_top_closeness(closeness):
    '''
        Prints the values of the top 5 nodes with the highest closeness centrality.
    '''
    pass

def get_average_closeness(closeness):
    '''
        Prints the average closeness centrality value.
    '''
    pass

def main():
    G = read_graph()