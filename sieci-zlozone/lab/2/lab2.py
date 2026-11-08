import urllib.request
import os
import networkx as nx

FILE_NAME = "dolphins.gml"
URL = "https://raw.githubusercontent.com/csdashes/GraphStreamCommunityDetection/master/data/dolphins.gml"

def download_and_load_graph():
    if not os.path.exists(FILE_NAME):
        print(f"Downloading dataset from {URL}...")
        req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            
        with open(FILE_NAME, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"File successfully saved as '{FILE_NAME}'.")
    else:
        print(f"File '{FILE_NAME}' already exists locally. Skipping download.")

    print(f"\nLoading graph from local file '{FILE_NAME}'...")
    G = nx.read_gml(FILE_NAME)

    print("\n--- Graph Summary ---")
    print(f"Węzły (Delfiny): {G.number_of_nodes()}")
    print(f"Krawędzie (Interakcje): {G.number_of_edges()}")
    print(f"Spójny: {nx.is_connected(G)}")
    
    return G

graph = download_and_load_graph()

# Zad 1
adjacency_matrix = nx.to_numpy_array(graph, dtype=int)

incidence_matrix = nx.incidence_matrix(graph, oriented=False).toarray()

print(f"Wymiary macierzy sąsiedztwa: {adjacency_matrix.shape}")
print(f"Wymiary macierzy incydencji: {incidence_matrix.shape}")

print("\nFragment macierzy sąsiedztwa (20x20):")
print(adjacency_matrix[:20, :20])

print("\nFragment macierzy incydencji (20x20):")
print(incidence_matrix[:20, :20])

# Zad 2

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

pos = nx.spring_layout(graph, seed=42)
nx.draw_networkx_nodes(
    graph, pos, node_size=300, node_color="skyblue", alpha=0.9
)
nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, edge_color="gray")
nx.draw_networkx_labels(graph, pos, font_size=8, font_family="sans-serif")

plt.title(
    "Graf relacji społecznych delfinów (Dolphin Social Network)", fontsize=14
)
plt.axis("off")
plt.tight_layout()
plt.show()

# Zad 3
from pyvis.network import Network

net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False)

net.from_nx(graph)

net.barnes_hut()

output_file = "dolphins.html"
net.show(output_file, notebook=False)

# Pytania pomocnicze
from networkx.algorithms import community

communities = list(community.greedy_modularity_communities(graph))
modularity = community.modularity(graph, communities)

deg_cent = nx.degree_centrality(graph)
bet_cent = nx.betweenness_centrality(graph)

top_hubs = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]
top_bridges = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:5]

isolates = list(nx.isolates(graph))
degree_one = [n for n, d in graph.degree() if d == 1]

print(f"Liczba społeczności: {len(communities)} (modularność: {modularity:.3f})")
print(f"Top huby (stopień): {[n for n, _ in top_hubs]}")
print(f"Top mosty (pośrednictwo): {[n for n, _ in top_bridges]}")
print(f"Węzły odizolowane: {isolates}")
print(f"Węzły peryferyjne (stopień 1): {degree_one}")