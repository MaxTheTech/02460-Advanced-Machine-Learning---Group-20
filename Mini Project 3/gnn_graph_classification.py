# %%
import torch
import networkx as nx
import numpy as np
from collections import defaultdict
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# %% Interactive plots
plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# %% Device
device = 'cpu'

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

# %% Define a simple GNN for graph classification
class SimpleGNN(torch.nn.Module):
    """Simple graph neural network for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        state_dim : Dimension of the node states
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # State output network
        self.output_net = torch.nn.Linear(self.state_dim, 1)

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)
        # state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Output
        out = self.output_net(graph_state).flatten()
        return out
    
# %% Set up the model, loss, and optimizer etc.
# Instantiate the model
state_dim = 16
num_message_passing_rounds = 4
model = SimpleGNN(node_feature_dim, state_dim, num_message_passing_rounds).to(device)

# Loss function
cross_entropy = torch.nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# %% Lists to store accuracy and loss
train_accuracies = []
train_losses = []
validation_accuracies = []
validation_losses = []

# %% Fit the model
# Number of epochs
epochs = 500

for epoch in range(epochs):
    # Loop over training batches
    model.train()
    train_accuracy = 0.
    train_loss = 0.
    for data in train_loader:
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = cross_entropy(out, data.y.float())

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training loss and accuracy
        train_accuracy += sum((out>0) == data.y).detach().cpu() / len(train_loader.dataset)
        train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)
    
    # Learning rate scheduler step
    scheduler.step()

    # Validation, print and plots
    with torch.no_grad():    
        model.eval()
        # Compute validation loss and accuracy
        validation_loss = 0.
        validation_accuracy = 0.
        for data in validation_loader:
            out = model(data.x, data.edge_index, data.batch)
            validation_accuracy += sum((out>0) == data.y).cpu() / len(validation_loader.dataset)
            validation_loss += cross_entropy(out, data.y.float()).cpu().item() * data.batch_size / len(validation_loader.dataset)

        # Store the training and validation accuracy and loss for plotting
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        # Print stats and update plots
        if (epoch+1)%10 == 0:
            print(f'Epoch {epoch+1}')
            print(f'- Learning rate   = {scheduler.get_last_lr()[0]:.1e}')
            print(f'- Train. accuracy = {train_accuracy:.3f}')
            print(f'         loss     = {train_loss:.3f}')
            print(f'- Valid. accuracy = {validation_accuracy:.3f}')
            print(f'         loss     = {validation_loss:.3f}')

            plt.figure('Loss').clf()
            plt.plot(train_losses, label='Train')
            plt.plot(validation_losses, label='Validation')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Cross entropy')
            plt.yscale('log')
            plt.tight_layout()
            drawnow()

            plt.figure('Accuracy').clf()
            plt.plot(train_accuracies, label='Train')
            plt.plot(validation_accuracies, label='Validation')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            drawnow()

# %% Save final predictions.
with torch.no_grad():
    data = next(iter(test_loader))
    out = model(data.x, data.edge_index, data.batch).cpu()
    torch.save(out, 'test_predictions.pt')








# %% Helper functions
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


# %% Sample Erdos-Renya baseline and deep generative model
num_samples = 1000
sampled_er = [sample_erdos_renyi() for _ in range(num_samples)]
sampled_dgm = sampled_er # REPLACE WITH ACTUAL


# %% Compute novelty, uniqueness, novel+unique
# WL hashes (for graph isomorphism checks)
train_hashes = set(wl_hash(pyg_to_nx(g)) for g in train_dataset)
sampled_er_hashes = [wl_hash(G) for G in sampled_er]
sampled_dgm_hashes = sampled_er_hashes # REPLACE WITH ACTUAL

sources = {"Erods-Renyi baseline": sampled_er_hashes, "Deep generative model": sampled_dgm_hashes}

for name, hashes in sources.items():
    novel  = [h not in train_hashes for h in hashes]
    unique = [hashes.count(h) == 1 for h in hashes]
    novel_and_unique = [n and u for n, u in zip(novel, unique)]

    print(f'{name} (n={num_samples} samples):')
    print(f'    Novel:          {100 * np.mean(novel):.1f}%')
    print(f'    Unique:         {100 * np.mean(unique):.1f}%')
    print(f'    Novel + unique: {100 * np.mean(novel_and_unique):.1f}%\n')

# %% Plot histogram grid
# Collect stats for training graphs, ER samples and DGM samples
train_nx = [pyg_to_nx(g) for g in train_dataset]
stats_train = collect_stats(train_nx)
stats_er = collect_stats(sampled_er)
stats_dgm = collect_stats(sampled_er) # REPLACE WITH ACTUAL

stat_names = ['Node degree', 'Clustering coefficient', 'Eigenvector centrality']
col_labels  = ['Training (empirical)', 'Erdos-Renyi baseline', 'Deep generative model']
all_stats   = [stats_train, stats_er, stats_dgm]

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
fig.suptitle('Graph statistics')

for col in range(3):
    combined = np.concatenate([np.array(s[col], dtype=float) for s in all_stats])
    bins = np.linspace(combined.min(), combined.max(), 30)
    for row in range(3):
        ax = axes[row, col]
        ax.hist(all_stats[row][col], bins=bins, color=f'C{row}', edgecolor='white', linewidth=0.3)
        if row == 0:
            ax.set_title(stat_names[col])
        if col == 0:
            ax.set_ylabel(col_labels[row])

plt.tight_layout()
plt.savefig('graph_statistics.png', dpi=150)
drawnow()

# %%
