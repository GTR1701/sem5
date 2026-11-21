import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "wikiElec.ElecBs3.txt")

# --- 1. Parsowanie: wyborca -> {kandydat: +1/-1} ---
votes = {}
voter = None
cand = None
with open(PATH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "U" and len(parts) > 1:
            voter = parts[1]
            votes.setdefault(voter, {})
        elif tag == "T" and len(parts) > 1:
            cand = parts[1]
        elif tag == "V" and len(parts) > 1 and voter and cand:
            try:
                val = int(parts[1])
            except ValueError:
                continue
            if val == 1:                       # tylko głosy "za"
                votes[voter][cand] = 1

votes = {v: c for v, c in votes.items() if c}
voters = sorted(votes)
cands = sorted({c for c in votes.values() for c in c})
vi = {v: i for i, v in enumerate(voters)}
ci = {c: i for i, c in enumerate(cands)}

A = np.zeros((len(voters), len(cands)), dtype=int)
for v, cs in votes.items():
    for c in cs:
        A[vi[v], ci[c]] = 1
print(f"wyborcy={A.shape[0]}, kandydaci={A.shape[1]}, krawędzie={A.sum()}")

# --- 2. Ucięcie do najaktywniejszych ---
top_v = np.argsort(-A.sum(1))[:60]
top_c = np.argsort(-A.sum(0))[:30]
A = A[np.ix_(top_v, top_c)]
A = A[A.sum(1) > 0][:, A.sum(0) > 0]
n, m = A.shape
print(f"po ucięciu: wyborcy={n}, kandydaci={m}, krawędzie={A.sum()}")

def save(fig, name):
    fig.savefig(os.path.join(BASE, name), dpi=150, bbox_inches="tight")

# --- 3a. Macierz sąsiedztwa ---
order_v = np.argsort(-A.sum(1))
order_c = np.argsort(-A.sum(0))
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(A[np.ix_(order_v, order_c)], cmap="Greys", aspect="auto")
ax.set_xlabel("kandydaci"); ax.set_ylabel("wyborcy")
ax.set_title("Macierz sąsiedztwa A (posortowana)")
save(fig, "macierz.png"); plt.show()

# --- 3b. Graf dwudzielny ---
B = nx.Graph()
B.add_nodes_from([f"v{i}" for i in range(n)], bipartite=0)
B.add_nodes_from([f"c{j}" for j in range(m)], bipartite=1)
B.add_edges_from((f"v{i}", f"c{j}") for i, j in zip(*np.nonzero(A)))

pos = nx.bipartite_layout(B, [f"v{i}" for i in range(n)])
fig, ax = plt.subplots(figsize=(10, 7))
nx.draw_networkx_nodes(B, pos, nodelist=[f"v{i}" for i in range(n)],
                       node_color="tab:blue", node_size=60, label="wyborcy", ax=ax)
nx.draw_networkx_nodes(B, pos, nodelist=[f"c{j}" for j in range(m)],
                       node_color="tab:orange", node_size=180, label="kandydaci", ax=ax)
nx.draw_networkx_edges(B, pos, alpha=0.3, ax=ax)
ax.legend(); ax.set_title("Graf dwudzielny: wyborcy – kandydaci"); ax.axis("off")
save(fig, "dwudzielny.png"); plt.show()

# --- 4. Projekcje ---
W = A @ A.T
deg = A.sum(1)
union = deg[:, None] + deg[None, :] - W
J = np.divide(W, union, out=np.zeros_like(W, dtype=float), where=union > 0)
np.fill_diagonal(W, 0); np.fill_diagonal(J, 0)

def proj_graph(M, thr):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((i, j) for i in range(n) for j in range(i + 1, n) if M[i, j] > thr)
    return G

prosta = proj_graph(W, 0)
jacc = proj_graph(J, 0.2)

for name, G in [("prosta", prosta), ("Jaccarda", jacc)]:
    print(f"{name}: węzły={G.number_of_nodes()}, krawędzie={G.number_of_edges()}, "f"gęstość={nx.density(G):.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
pos_j = nx.spring_layout(jacc, seed=42)
nx.draw_networkx_nodes(jacc, pos_j, node_size=40, node_color="tab:green", ax=ax)
nx.draw_networkx_edges(jacc, pos_j, alpha=0.4, width=[3 * J[u, v] for u, v in jacc.edges()], ax=ax)
ax.set_title("Projekcja Jaccarda (grubość = podobieństwo)"); ax.axis("off")
save(fig, "jaccard.png"); plt.show()

comms = nx.community.greedy_modularity_communities(jacc)
print("społeczności w projekcji Jaccarda:", [len(c) for c in comms])
print("modularność:", round(nx.community.modularity(jacc, comms), 3))
print("największa składowa:", nx.diameter(jacc.subgraph(max(nx.connected_components(jacc), key=len))))
