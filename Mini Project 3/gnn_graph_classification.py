# %%
import torch
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt




# %% Helper functions
def pyg_to_nx(data):
    """Convert PyG Data graph object to networkx graph."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
    # Store node "type" if x is one-hot (MUTAG: 7 types)
    if hasattr(data, 'x') and data.x is not None and data.x.dim() == 2 and data.x.size(0) == data.num_nodes:
        try:
            node_types = data.x.argmax(dim=1).detach().cpu().tolist()
            nx.set_node_attributes(G, {i: int(t) for i, t in enumerate(node_types)}, name='node_type')
        except Exception:
            pass
    return G

def wl_hash(G):
    """Weisfeiler Lehman hash of graph (for graph isomorphism checks)."""
    return nx.weisfeiler_lehman_graph_hash(G)

def collect_stats(graphs):
    """Per-node graph statistics from a list of nx graphs."""
    degrees, clusterings, centralities = [], [], []
    for G in graphs:
        degrees += [d for _, d in G.degree()]
        clusterings += list(nx.clustering(G).values())
        try:
            c = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            c = {n: 0.0 for n in G.nodes()}
        centralities += list(c.values())
    return degrees, clusterings, centralities

def load_model(device='cpu'):
    """Load saved model from config."""
    from deep_generative_model import VAE, GaussianPrior, GaussianEncoder, GraphDecoder, GNNEncoderNet
    M              = 16
    state_dim      = 64
    node_embed_dim = 32
    num_rounds     = 3

    prior   = GaussianPrior(M)
    encoder = GaussianEncoder(GNNEncoderNet(node_feature_dim, state_dim, M, num_rounds))
    decoder = GraphDecoder(M, node_embed_dim, max_nodes=28)
    model   = VAE(prior, encoder, decoder).to(device)
    checkpoint = torch.load(f"./graph_vae_mutag_{M}_{state_dim}_{num_rounds}.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# %% Load data and model
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device=device)


# %% Erdos-Renyi baseline
# Build empirical distribution
node_count_list = []
density_by_node_count = defaultdict(list)

for graph in train_dataset:
    n = graph.num_nodes
    max_edges = n * (n - 1)
    r = graph.num_edges / max_edges if max_edges > 0 else 0.0
    node_count_list.append(n)
    density_by_node_count[n].append(r)

# Precompute mean density per node count
mean_density = {n: np.mean(densities) for n, densities in density_by_node_count.items()}
all_densities = [d for densities in density_by_node_count.values() for d in densities]
print(f"Mean edge density (training): {np.mean(all_densities):.1%}, non-edge fraction: {1 - np.mean(all_densities):.1%}\n")

def sample_erdos_renyi():
    """Sample a random graph using the Erdos-Renyi model fitted to training data."""
    n = int(np.random.choice(node_count_list))
    r = mean_density[n]
    return nx.erdos_renyi_graph(n=n, p=r)


# %% Sample Erdos-Renyi baseline and deep generative model
num_samples = 1000
sampled_er = [sample_erdos_renyi() for _ in range(num_samples)]    
node_size   = np.random.choice(node_count_list, size=num_samples, replace=True)
sampled_dgm = [model.sample(remove_isolated=True, n_nodes=int(size)) for size in node_size]


# %% Compute novelty, uniqueness, novel+unique
train_hashes = set(wl_hash(pyg_to_nx(g)) for g in train_dataset)
sampled_er_hashes = [wl_hash(G) for G in sampled_er]
model_to_nx = [pyg_to_nx(g) for g in sampled_dgm]
sampled_dgm_hashes = [wl_hash(G) for G in model_to_nx]

sources = {"Erdos-Renyi baseline": sampled_er_hashes, "Deep generative model": sampled_dgm_hashes}

for name, hashes in sources.items():
    hash_counts = Counter(hashes)
    novel  = [h not in train_hashes for h in hashes]
    unique = [hash_counts[h] == 1 for h in hashes]
    novel_and_unique = [n and u for n, u in zip(novel, unique)]

    print(f'{name} (n={num_samples} samples):')
    print(f'    Novel:          {100 * np.mean(novel):.1f}%')
    print(f'    Unique:         {100 * np.mean(unique):.1f}%')
    print(f'    Novel + unique: {100 * np.mean(novel_and_unique):.1f}%\n')

# %% Plot histogram grid
train_nx = [pyg_to_nx(g) for g in train_dataset]
stats_train = collect_stats(train_nx)
stats_er = collect_stats(sampled_er)
stats_dgm = collect_stats(model_to_nx)

stat_names = ['Node degree', 'Clustering coefficient', 'Eigenvector centrality']
col_labels  = ['Training (empirical)', 'Erdos-Renyi baseline', 'Deep generative model']
all_stats   = [stats_train, stats_er, stats_dgm]

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
# fig.suptitle('Graph statistics')

for row in range(3):
    combined = np.concatenate([np.array(s[row], dtype=float) for s in all_stats])
    bins = np.linspace(combined.min(), combined.max(), 30)
    for col in range(3):
        ax = axes[row, col]
        vals = all_stats[col][row]
        ax.hist(vals, bins=bins, weights=np.ones(len(vals)) / len(vals), color=f'C{col}', edgecolor='white', linewidth=0.3)
        if row == 0:
            ax.set_title(col_labels[col])
        if col == 0:
            ax.set_ylabel(stat_names[row])
    
    y_max = max(axes[row, col].get_ylim()[1] for col in range(3))
    for col in range(3):
        axes[row, col].set_ylim(0, y_max)

plt.tight_layout()
plt.savefig('graph_statistics.png', dpi=150)


# %% Plot 5 example graphs from each source
k = 5
rng = np.random.default_rng(0)

train_idx = rng.choice(len(train_dataset), size=k, replace=False)
train_graphs = [pyg_to_nx(train_dataset[int(i)]) for i in train_idx]

er_idx = rng.choice(len(sampled_er), size=k, replace=False)
er_graphs = [sampled_er[int(i)] for i in er_idx]

dgm_idx = rng.choice(len(model_to_nx), size=k, replace=False)
dgm_graphs = [model_to_nx[int(i)] for i in dgm_idx]

col_titles = ['Training', 'Erdos-Renyi', 'Model']
fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))

for row in range(k):
    for col, G in enumerate([train_graphs[row], er_graphs[row], dgm_graphs[row]]):
        ax = axes[row, col]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor('black')
        if row == 0:
            ax.set_title(col_titles[col])
        if G.number_of_nodes() == 0:
            continue
        pos = nx.spring_layout(G, seed=1000 + 10 * row + col)
        node_types = nx.get_node_attributes(G, 'node_type')
        if node_types:
            node_colors = [node_types.get(n, 0) for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color=node_colors, cmap='tab10', linewidths=0)
        else:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color='C0', linewidths=0)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, alpha=0.6)

plt.tight_layout()
plt.savefig('graph_examples.png', dpi=200)
plt.close(fig)

# %%