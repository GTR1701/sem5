import math
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
HTML_NAME = "congress_islands.html"
HTML_PATH = BASE / HTML_NAME

# 1. Wczytanie grafu (nieskierowanego — dane są symetryczne)
g = nx.read_edgelist(
    EDGES_PATH,
    nodetype=str,
    data=True,
    create_using=nx.Graph(),
)

deg = dict(g.degree())
main_nodes = max(nx.connected_components(g), key=len)
gc = g.subgraph(main_nodes).copy()

# 2. Wykrycie wysp (społeczności)
communities = nx.community.louvain_communities(gc, seed=RANDOM_SEED, resolution=1.0)
communities = sorted(communities, key=len, reverse=True)
comm_of = {n: i for i, c in enumerate(communities) for n in c}

print(f"wykryto spolecznosci: {len(communities)}")
for i, c in enumerate(communities):
    print(f"  wyspa {i}: {len(c)} wezlow")

# 3. Pozycje: mniejszy okrąg, większy rozrzut wewnątrz wyspy
pos = {}
n_islands = len(communities)
RING_RADIUS = 1400.0
ISLAND_RADIUS = 1000.0

for i, comm in enumerate(communities):
    sub = gc.subgraph(comm)
    local = nx.spring_layout(sub, k=0.9, iterations=400, seed=RANDOM_SEED + i)
    angle = 2 * math.pi * i / max(n_islands, 1)
    cx, cy = RING_RADIUS * math.cos(angle), RING_RADIUS * math.sin(angle)
    xs = [p[0] for p in local.values()] or [0.0]
    ys = [p[1] for p in local.values()] or [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = (2 * ISLAND_RADIUS) / span
    mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    for n, (x, y) in local.items():
        pos[n] = (cx + (x - mx) * scale, cy + (y - my) * scale)

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]

top_nodes = {n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:50]}

net = Network(height="900px", width="100%", bgcolor="white", font_color="black", directed=False)
net.set_options("""
{
  "physics": {"enabled": false},
  "nodes": {"borderWidth": 1, "borderWidthSelected": 2},
  "edges": {"smooth": false, "color": {"opacity": 0.12}},
  "interaction": {"hover": true, "tooltipDelay": 120, "hideEdgesOnDrag": true}
}
""")

for node in gc.nodes():
    cid = comm_of.get(node, -1)
    color = PALETTE[cid % len(PALETTE)]
    is_hub = node in top_nodes
    net.add_node(
        node,
        label=node if is_hub else "",
        title=f"id: {node}<br>wyspa: {cid}<br>stopien: {deg[node]}",
        color={"background": color, "border": "black" if is_hub else color},
        value=deg[node],
        x=float(pos[node][0]),
        y=float(pos[node][1]),
        group=cid,
    )

# 4. Kolor krawędzi zależny od wagi
def weight_color(t: float, same: bool) -> str:
    """t w [0,1]: 0 = słabe podobieństwo, 1 = mocne.
    Wewnątrz wyspy: jasny szaroniebieski -> granat.
    Między wyspami: blady róż -> mocna czerwień."""
    t = max(0.0, min(1.0, t))
    if same:
        r = 0.75 - 0.55 * t
        g = 0.75 - 0.45 * t
        b = 0.80 + 0.20 * t
    else:
        r = 1.0
        g = 0.70 - 0.55 * t
        b = 0.70 - 0.55 * t
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


weights = [float(d.get("weight", 1.0)) for _, _, d in gc.edges(data=True)]
wmin, wmax = (min(weights), max(weights)) if weights else (0.0, 1.0)
wspan = (wmax - wmin) or 1.0

# 5. Diagnostyka mostu między głównymi centrystami
A, B = "367", "322"
HIGHLIGHT = {A, B}

if gc.has_edge(A, B):
    w = float(gc[A][B].get("weight", 1.0))
    t = (w - wmin) / wspan
    print(f"krawedz {A}--{B}: waga={w:.6f}")
else:
    print(f"brak bezposredniej krawedzi {A}--{B}")
    path = nx.shortest_path(gc, A, B)
    print(f"najkrotsza sciezka ({len(path) - 1} krawedzi): {' -> '.join(path)}")
    for u, v in zip(path, path[1:]):
        w = float(gc[u][v].get("weight", 1.0))
        print(f"   {u}--{v}: waga={w:.6f}")

top_edges = sorted(
    gc.edges(data=True), key=lambda e: float(e[2].get("weight", 1.0)), reverse=True
)[:10]
print("top10 najciezszych krawedzi:")
for u, v, d in top_edges:
    w = float(d.get("weight", 1.0))
    same = comm_of.get(u) == comm_of.get(v)
    kind = "wewnatrz" if same else "MIEDZY"
    print(f"   {u}--{v}: {w:.6f}  [{kind}: {comm_of.get(u)}-{comm_of.get(v)}]")

# 6. Rysowanie krawędzi (z wyróżnieniem mostu)
for u, v, data in gc.edges(data=True):
    w = float(data.get("weight", 1.0))
    t = (w - wmin) / wspan
    same = comm_of.get(u) == comm_of.get(v)

    if {u, v} == HIGHLIGHT:
        net.add_edge(
            u, v,
            width=6.0,
            color={"color": "#00bcd4", "opacity": 1.0},
            title=f"GLOWNY MOST {u}--{v}<br>waga: {w:.6f}",
        )
        continue

    if same:
        net.add_edge(
            u, v,
            width=0.1 + 2.2 * t,
            color={"color": weight_color(t, True), "opacity": 0.15 + 0.55 * t},
            title=f"waga: {w:.6f}",
        )
    else:
        net.add_edge(
            u, v,
            width=0.05 + 0.8 * t,
            color={"color": weight_color(t, False), "opacity": 0.06 + 0.30 * t},
            title=f"waga: {w:.6f}",
            dashes=True,
        )

prev_cwd = Path.cwd()
os.chdir(BASE)
net.save_graph(HTML_NAME)
os.chdir(prev_cwd)

# 8. Metryki
sources = random.sample(list(gc.nodes()), k=min(64, gc.number_of_nodes()))
diameter = max(max(nx.single_source_shortest_path_length(gc, s).values()) for s in sources)
avg_path = mean(mean(nx.single_source_shortest_path_length(gc, s).values()) for s in sources)

degree_c = nx.degree_centrality(gc)
closeness_c = nx.closeness_centrality(gc)
betweenness_c = nx.betweenness_centrality(gc, seed=RANDOM_SEED)
mod = nx.community.modularity(gc, communities)

print(f"rozmiar: |V|={gc.number_of_nodes()}, |E|={gc.number_of_edges()}")
print(f"stopien: min={min(deg.values())}, max={max(deg.values())}, sr={mean(deg.values()):.4f}")
print(f"gestosc: {nx.density(gc):.8f}")
print(f"modularnosc: {mod:.4f}")
print(f"srednica: {diameter}")
print(f"srednia_dlugosc_sciezki: {avg_path:.6f}")
print(f"centralnosc_stopnia_top5: {sorted(degree_c.items(), key=lambda x: x[1], reverse=True)[:5]}")
print(f"centralnosc_bliskosci_top5: {sorted(closeness_c.items(), key=lambda x: x[1], reverse=True)[:5]}")
print(f"centralnosc_posrednictwa_top5: {sorted(betweenness_c.items(), key=lambda x: x[1], reverse=True)[:5]}")
print(f"graf_pelny_html: {HTML_PATH}")
