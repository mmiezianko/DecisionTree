import numpy as np
import pandas as pd
from grow_decision_tree import grow_tree

import csv
import numpy as np
import pandas as pd
from grow_decision_tree import grow_tree, plot, prune, predict

import matplotlib.pyplot as plt
import networkx as nx

def get_hash_dict(d_tree, hash_dict):
    if d_tree.branch_with_value is not None:
        hash_dict[d_tree.__hash__()] = {'value':d_tree.value,
                                        'col':d_tree.col_name,
                                        'children': [d_tree.branch_with_value.__hash__(), d_tree.branch_with_others.__hash__()],
                                        'size':d_tree.set_size
                                        }
        get_hash_dict(d_tree.branch_with_value, hash_dict)
        get_hash_dict(d_tree.branch_with_others, hash_dict)


def get_neighborhood_list(d_tree):
    # tworzymy pusty słownik, do którego bedziemy dopisywać informacje o strukturze drzewa
    hash_dict = {}
    # d_tree to jest drzewo
    get_hash_dict(d_tree, hash_dict)

    # verticles to lista kluczy d_tree.__hash__(), czyli wierzchołki
    vertices = [k for k in hash_dict.keys()]
    # do wierzchołków dopisujemy wszystkie dzieci - również liście (czyli dzieci które nie stały się parentsami)
    # robimy to ponieważ w funkcji get_hash_dict (przez warunek if d_tree.branch_with_value is not None:) nie uwzględniamy dzieci, które nie są parentsami (liście)
    for v in hash_dict.values():
        vertices.extend(v['children'])
    # używamy set, czyli struktury danych która jest listą bez powtórzeń. Skoro wzieliśmy wszystkie dzieci, to mogą
    # wsytępować powtórzenia, ponieważ niektóre dzieci są zarówno dziećmi jak i parentsami
    hashes = set(vertices)
    # tworzymy mapowanie k:i  gdzie k jest kluczem a "i" intigerem
    hash_to_int = {k:i for i,k in enumerate(hashes)}

    neighborhood_list = {}
    for k,v in hash_dict.items():
        neighborhood_list[hash_to_int[k]]={'value': v['value'],
                                           'col': v['col'],
                                           'children':[hash_to_int[v['children'][0]], hash_to_int[v['children'][1]]],
                                            'size': v['size']
                                           }

    return neighborhood_list

def draw_graph(nb_list):
    G = nx.DiGraph()
    vertices = list(set([k for k in nb_list.keys()]))
    for v in nb_list.values():
        vertices.extend(v['children'])
    G.add_nodes_from(set(vertices))
    edge_dict={}
    for k, v in nb_list.items():
        G.add_edge(k,v['children'][0], minlen=10)
        edge_dict[(k,v['children'][0])] = {'value':v['value'], 'col':v['col']}
        G.add_edge(k,v['children'][1], minlen=10)
        edge_dict[(k, v['children'][1])] = {'value': 'others', 'col': v['col']}

    for k,v in nb_list.items():
        G.nodes[k]['value'] = v['value']
    nodes_dict = {}
    for vertice in vertices:
        if vertice in nb_list.keys():
            nodes_dict[vertice] = nb_list[vertice]['size']
        else:
            nodes_dict[vertice] = 0
    return G, edge_dict, nodes_dict

def load_csv(filename):
    data = pd.read_csv(filename)
    data_header = data.columns

    return list(data.to_numpy()) , data_header

def grow_and_show_tree(filename, pruning=False, threshold=0.5, notification=True):
    training_data, data_header = load_csv(filename)
    d_tree = grow_tree(training_data, columns_map=data_header)
    plot(d_tree)
    if pruning:
      prune(d_tree, minGain=threshold, notification=notification)
    plot(d_tree)
    G, edge_dict, nodes_dict = draw_graph(get_neighborhood_list(d_tree))
    plt.figure(figsize=(25,25))
    plt.subplot(111)
    pos = nx.kamada_kawai_layout(G,scale=3)
    nx.draw_networkx_edge_labels(G,pos,edge_dict,verticalalignment='center')
    nx.draw_networkx_labels(G, pos, nodes_dict)
    nx.draw(G, pos)
    plt.show()
    return d_tree

if __name__ == "__main__":

    decission_tree = grow_and_show_tree('notebooks/dane.csv')


    