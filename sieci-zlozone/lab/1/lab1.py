import csv
import networkx as nx
import os
import random
import sys

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

graph = nx.Graph()

with open(script_directory + '\soc-sign-bitcoinotc.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        u = row['SOURCE']
        v = row['TARGET']
        raw_rating = float(row['RATING'])

        rating = raw_rating + 11
        
        graph.add_edge(u, v, rating=rating)

#Zad1
source, target = random.sample(list(graph.nodes), 2)
shortest_path = nx.shortest_path(graph, source, target)
distance = nx.shortest_path_length(graph, source, target)
print(f"Odległość od węzła {source} do węzła {target} wynosi: {distance}")
print(f"Ścieżka: {shortest_path}")

#Zad2
def is_eulerian(graph):
  return all(degree % 2 == 0 for vertex, degree in graph.degree()) and nx.is_connected(graph)

eulerian = is_eulerian(graph)
print(f"Czy graf jest eulerowski: {eulerian}")

def get_eulerian_subgraph_edges(graph):
    if nx.is_eulerian(graph):
        return list(nx.eulerian_circuit(graph))
    else:
        cycles = nx.cycle_basis(graph)
        if cycles:
            first_cycle = cycles[0]
            cycle_edges = [(first_cycle[i-1], first_cycle[i]) 
                           for i in range(len(first_cycle))]
            return cycle_edges
eulerian_path = get_eulerian_subgraph_edges(graph)
print(eulerian_path)

#Zad3
flow_value, flow_dict = nx.maximum_flow(graph, str(source), str(target), capacity='rating')
print(f"Maksymalny przepływ z węzła {source} do węzła {target} wynosi: {flow_value}")
