import csv
import networkx as nx
import os
import random
import sys

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
file_path = os.path.join(script_directory, 'soc-sign-bitcoinotc.csv')

graph = nx.DiGraph()

with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        u = row['SOURCE']
        v = row['TARGET']
        raw_rating = float(row['RATING'])

        rating = raw_rating + 11
        
        graph.add_edge(u, v, rating=rating)

reduce_graph = input("Czy chcesz zmniejszyć rozmiar grafu do 1000 węzłów? [t/n]: ").strip().lower()
if reduce_graph in {"t", "tak", "y", "yes"}:
    if graph.number_of_nodes() > 1000:
        selected_nodes = list(graph.nodes)[:1000]
        graph = graph.subgraph(selected_nodes).copy()
    print(f"Wczytano graf z {graph.number_of_nodes()} węzłami.")
else:
    print(f"Wczytano pełny graf z {graph.number_of_nodes()} węzłami.")

# Zad1
source, target = random.sample(list(graph.nodes), 2)
try:
    shortest_path = nx.shortest_path(graph, source, target)
    distance = nx.shortest_path_length(graph, source, target)
    print(f"Odległość od węzła {source} do węzła {target} wynosi: {distance}")
    print(f"Ścieżka: {shortest_path}")
except nx.NetworkXNoPath:
    print(f"Brak ścieżki skierowanej z węzła {source} do węzła {target}.")

# Zad2
eulerian = nx.is_eulerian(graph)
print(f"Czy graf jest eulerowski: {eulerian}")

def get_eulerian_subgraph_edges(graph):
    if nx.is_eulerian(graph):
        return list(nx.eulerian_circuit(graph))
    else:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
            return list(cycle)
        except nx.NetworkXNoCycle:
            return []

eulerian_path = get_eulerian_subgraph_edges(graph)
print("Krawędzie cyklu / fragmentu:", eulerian_path)

# Zad3
try:
    flow_value, flow_dict = nx.maximum_flow(graph, str(source), str(target), capacity='rating')
    print(f"Maksymalny przepływ z węzła {source} do węzła {target} wynosi: {flow_value}")
except Exception as e:
    print(f"Nie można obliczyć maksymalnego przepływu: {e}")