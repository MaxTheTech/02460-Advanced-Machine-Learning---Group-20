# %%
import torch
import networkx as nx
import numpy as np
from collections import defaultdict
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
# %% Helper functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

def pyg_to_nx(data):
    """Convert PyG Data graph object to networkx graph."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)
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

def load_model(is_grnn=False):
    if not is_grnn:
        from deep_generative_model import VAE, GaussianPrior, GaussianEncoder, GraphDecoder, DeepGenerativeModel
        M = 16
        state_dim = 64
        num_message_passing_rounds = 4

        prior       = GaussianPrior(M)
        encoder_net = DeepGenerativeModel(node_feature_dim, state_dim, M, num_message_passing_rounds).to(device)
        encoder     = GaussianEncoder(encoder_net)
        decoder     = GraphDecoder(M, state_dim, node_feature_dim, num_message_passing_rounds).to(device)
        model       = VAE(prior, encoder, decoder).to(device)
        checkpoint  = torch.load(f"./deep_generative_model_mutag_{M}_{state_dim}_{num_message_passing_rounds}.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    else:
        from grnn  import GNNBlock, GraphEncoder, GRANDecoder, GRAN_VAE
        latent_dim = 16
        hidden_dim = 64
        num_rounds = 4
        max_nodes  = max(train_dataset[i].num_nodes for i in range(len(train_dataset))) + 5

        encoder = GraphEncoder(node_feature_dim, hidden_dim, latent_dim, num_rounds).to(device)
        decoder = GRANDecoder(node_feature_dim, hidden_dim, latent_dim, num_rounds, max_nodes).to(device)
        model   = GRAN_VAE(encoder, decoder, latent_dim).to(device)
        checkpoint = torch.load("gran_vae_mutag.pt", map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

# %% Erdos-Renyi baseline
# Build empirical distribution
node_count_list = []
density_by_node_count = defaultdict(list)

for graph in train_dataset:
    n = graph.num_nodes
    max_edges = n * (n - 1) # MUTAG is directed
    r = graph.num_edges / max_edges if max_edges > 0 else 0.0
    node_count_list.append(n)
    density_by_node_count[n].append(r)

# Precompute mean density per node count
mean_density = {n: np.mean(densities) for n, densities in density_by_node_count.items()}

def sample_erdos_renyi():
    """Sample a random graph using the Erdos-Renyi model fitted to training data."""
    # Sample N from empirical distribution of training node counts
    n = int(np.random.choice(node_count_list))

    # Compute link probability as mean graph density for graphs with N nodes
    r = mean_density[n]

    # Sample graph
    G = nx.erdos_renyi_graph(n=n, p=r)
    return G


# %% Sample Erdos-Renyi baseline and deep generative model
num_samples = 1000
is_grnn = False  
model = load_model(is_grnn=is_grnn)
sampled_er = [sample_erdos_renyi() for _ in range(num_samples)]
if is_grnn:
    sampled_dgm = [model.sample(device=device) for _ in range(num_samples)]
else:
    node_size  = np.random.choice(node_count_list, size=num_samples, replace=True)
    sampled_dgm = [model.sample(n_nodes=int(size)) for size in node_size]


# %% Compute novelty, uniqueness, novel+unique
# WL hashes (for graph isomorphism checks)
train_hashes = set(wl_hash(pyg_to_nx(g)) for g in train_dataset)
sampled_er_hashes = [wl_hash(G) for G in sampled_er]
model_to_nx = [pyg_to_nx(g) for g in sampled_dgm]
sampled_dgm_hashes = [wl_hash(G) for G in model_to_nx]

sources = {"Erods-Renyi baseline": sampled_er_hashes, "Deep generative model": sampled_dgm_hashes}

for name, hashes in sources.items():
    novel  = [h not in train_hashes for h in hashes]
    unique = [hashes.count(h) == 1 for h in hashes]
    novel_and_unique = [n and u for n, u in zip(novel, unique)]

    print(f'{name} (n={num_samples} samples):')
    print(f'    Novel:          {100 * np.mean(novel):.1f}%')
    print(f'    Unique:         {100 * np.mean(unique):.1f}%')
    print(f'    Novel + unique: {100 * np.mean(novel_and_unique):.1f}%\n')

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# %% Plot histogram grid
# Collect stats for training graphs, ER samples and DGM samples
train_nx = [pyg_to_nx(g) for g in train_dataset]
stats_train = collect_stats(train_nx)
stats_er = collect_stats(sampled_er)
stats_dgm = collect_stats(model_to_nx)

stat_names = ['Node degree', 'Clustering coefficient', 'Eigenvector centrality']
col_labels  = ['Training (empirical)', 'Erdos-Renyi baseline', 'Deep generative model']
all_stats   = [stats_train, stats_er, stats_dgm]

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
fig.suptitle('Graph statistics')

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
    # Enforce same y-axis scale across all columns for this row
    y_max = max(axes[row, col].get_ylim()[1] for col in range(3))
    for col in range(3):
        axes[row, col].set_ylim(0, y_max)

plt.tight_layout()
plt.savefig('graph_statistics.png', dpi=150)

# %%



