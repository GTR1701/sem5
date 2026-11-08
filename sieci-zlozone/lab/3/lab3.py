import random
import os
from pathlib import Path
from statistics import mean

import networkx as nx
from pyvis.network import Network

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
BASE = Path(__file__).resolve().parent
EDGES_PATH = BASE / "congress.edgelist"
HTML_NAME = "congress_full.html"
HTML_PATH = BASE / HTML_NAME

g = nx.read_edgelist(
    EDGES_PATH,
    nodetype=str,
    data=True,
    create_using=nx.DiGraph(),
)

deg = dict(g.degree())
main_nodes = max(nx.weakly_connected_components(g), key=len)
gc = g.subgraph(main_nodes).copy()
dist_graph = gc.to_undirected()

sources = random.sample(list(dist_graph.nodes()), k=min(64, dist_graph.number_of_nodes()))
diameter = max(
    max(nx.single_source_shortest_path_length(dist_graph, s).values()) for s in sources
)
avg_path = mean(
    mean(nx.single_source_shortest_path_length(dist_graph, s).values()) for s in sources
)

degree_c = nx.degree_centrality(g)
sample_for_closeness = [
    n
    for n, _ in sorted(dict(dist_graph.degree()).items(), key=lambda x: x[1], reverse=True)[:300]
]
closeness_c = {n: nx.closeness_centrality(dist_graph, u=n) for n in sample_for_closeness}
betweenness_c = nx.betweenness_centrality(
    dist_graph, k=min(64, dist_graph.number_of_nodes()), seed=RANDOM_SEED
)

top_nodes = {
    node for node, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:50]
}
net = Network(height="900px", width="100%", bgcolor="white", font_color="black", directed=True)
pos = nx.spring_layout(g, k=2.2, iterations=300, seed=RANDOM_SEED)
for node in g.nodes():
    color = "tomato" if node in top_nodes else "skyblue"
    net.add_node(
        node,
        label=node,
        title=f"id: {node}<br>stopien: {deg[node]}",
        color=color,
        value=deg[node],
        x=float(pos[node][0] * 3000),
        y=float(pos[node][1] * 3000),
    )
for u, v, data in g.edges(data=True):
    weight = float(data.get("weight", 1.0))
    net.add_edge(u, v, width=0.02, title=f"waga: {weight:.6f}")
net.set_options('{"physics":{"enabled":false}}')
prev_cwd = Path.cwd()
os.chdir(BASE)
net.save_graph(HTML_NAME)
os.chdir(prev_cwd)

print(f"rozmiar: |V|={g.number_of_nodes()}, |E|={g.number_of_edges()}")
print(f"stopien: min={min(deg.values())}, max={max(deg.values())}, sr={mean(deg.values()):.4f}")
print(f"gestosc: {nx.density(g):.8f}")
print(f"srednica: {diameter}")
print(f"srednia_dlugosc_sciezki: {avg_path:.6f}")
print(f"centralnosc_stopnia_top5: {sorted(degree_c.items(), key=lambda x: x[1], reverse=True)[:5]}")
print(f"centralnosc_bliskosci_top5: {sorted(closeness_c.items(), key=lambda x: x[1], reverse=True)[:5]}")
print(f"centralnosc_posrednictwa_top5: {sorted(betweenness_c.items(), key=lambda x: x[1], reverse=True)[:5]}")
print(f"graf_pelny_html: {HTML_PATH}")