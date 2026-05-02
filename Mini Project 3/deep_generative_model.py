# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(num_nodes,
           feature_dim)` and outputs a tensor of dimension `(num_nodes, 2M)`,
           where M is the dimension of the per-node latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x, edge_index, batch):
        """
        Given a batch of graphs, return a per-node Gaussian distribution.

        Parameters:
        x: [torch.Tensor]
           Node features of dimension `(num_nodes, node_feature_dim)`
        """
        mean, log_std = torch.chunk(self.encoder_net(x, edge_index, batch), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, node_feature_dim, num_message_passing_rounds):
        """
        Combined graph decoder.

        Edges:  P(A_uv = 1) = sigma(z_u^T z_v + b)  via inner product on dense latents.
        Nodes:  GNN message passing on latent embeddings → node feature logits.

        Parameters:
        latent_dim: [int]
        hidden_dim: [int]
        node_feature_dim: [int]
        num_message_passing_rounds: [int]
        """
        super(GraphDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        self.bias = nn.Parameter(torch.zeros(1))

        self.input_net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.message_net = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)])
        self.update_net = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)])
        self.output_net = nn.Linear(hidden_dim, node_feature_dim)

    def decode_edges(self, z_dense):
        """(B, n_max, M) → (B, n_max, n_max) edge logits."""
        return torch.bmm(z_dense, z_dense.transpose(1, 2)) + self.bias

    def decode_nodes(self, z, edge_index):
        """(num_nodes, M) + edge_index → (num_nodes, node_feature_dim) feature logits."""
        num_nodes = z.shape[0]
        state = self.input_net(z)
        for r in range(self.num_message_passing_rounds):
            message = self.message_net[r](state)
            aggregated = z.new_zeros((num_nodes, self.hidden_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            state = state + self.update_net[r](aggregated)
        return self.output_net(state)

    def log_prob(self, z, x):
        """
        Compute log p(A|z) + log p(X|z, A_sampled) summed per graph.

        Parameters:
        z: [torch.Tensor] Per-node latents of dimension `(N, M)`.
        x: PyG Batch object with fields x, edge_index, batch.

        Returns:
        log_p: [torch.Tensor] Per-graph log-likelihood of dimension `(B,)`.
        """
        B = int(x.batch.max().item()) + 1

        # The edge decoder needs the full pairwise (n_max x n_max) logit matrix z_u·z_v for every graph.
        # PyG stores graphs sparsely, so we pad to a dense batch. mask[b,i]=True means node i is real in graph b.
        z_dense, mask = to_dense_batch(z, x.batch)
        A_target      = to_dense_adj(x.edge_index, x.batch, max_num_nodes=mask.shape[1])
        edge_logits   = self.decode_edges(z_dense)

        # Treat graphs as undirected: score each unordered pair once (upper triangle i<j),
        # and ignore padding + self-loops.
        n_max = mask.shape[1]
        triu = torch.triu(torch.ones((n_max, n_max), dtype=torch.bool, device=z.device), diagonal=1).unsqueeze(0)
        pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)) & triu

        # Ensure the target adjacency is symmetric (robust if edge_index stores only one direction).
        A_target = torch.maximum(A_target, A_target.transpose(1, 2))
        edge_recon = (td.Bernoulli(logits=edge_logits).log_prob(A_target) * pair_mask).sum(dim=[1, 2])

        # The node decoder must see sampled edges (not the true graph edges) so its inputs at training time
        # match what it receives at generation time. The discrete sample has no gradient, but gradients
        # still reach the parameters through z → edge_logits (edge recon) and z → feat_logits (node recon).
        with torch.no_grad():
            A_upper = td.Bernoulli(logits=edge_logits).sample() * pair_mask
            A_sampled = A_upper + A_upper.transpose(1, 2)

        # A_sampled uses local node indices (0..n_max) per graph, but the GNN decoder expects a flat
        # edge_index over all N nodes in the batch. offsets[b] is the global index of graph b's first node,
        # so adding it maps each local index to its position in the global node list.
        offsets    = torch.cat([z.new_zeros(1, dtype=torch.long), torch.bincount(x.batch).cumsum(0)[:-1]])
        b, s, d    = A_sampled.nonzero(as_tuple=True)
        sampled_ei = torch.stack([offsets[b] + s, offsets[b] + d])

        # OneHotCategorical.log_prob gives one value per node; index_add sums them into per-graph totals
        # so the ELBO mean is taken over graphs, not individual nodes.
        node_recon = z.new_zeros(B).index_add(0, x.batch,
                         td.OneHotCategorical(logits=self.decode_nodes(z, sampled_ei)).log_prob(x.x))

        return edge_recon + node_recon

# %% GNN encoder for the graph VAE
class DeepGenerativeModel(torch.nn.Module):
    """GNN encoder that outputs per-node latent parameters (mean + log-std).

    Unlike a graph-level encoder, this does NOT pool to a single graph vector.
    Each node gets its own latent distribution q(z_i | G).

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the input node features (7 for MUTAG)
        state_dim        : GNN hidden state dimension (independent of latent size)
        latent_size      : Per-node latent dimension M; output is 2*M per node
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, latent_size, num_message_passing_rounds):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim        # GNN hidden dim — independent of latent dim
        self.latent_size = latent_size    # VAE latent dim M
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

        # Output 2*M per node: mean and log-std for the per-node latent distribution
        self.output_net = torch.nn.Linear(self.state_dim, 2 * self.latent_size)

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
        out : torch tensor (num_nodes x 2*latent_size)
            Per-node mean and log-std of the latent distribution.

        """
        # Extract number of nodes and graphs
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        # Output 2*M values per node — no graph-level pooling
        out = self.output_net(state)
        return out

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model with node-level latents.
    """
    def __init__(self, prior, encoder, decoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        decoder: [GraphDecoder]
              Combined edge and node feature decoder.
        """
        super(VAE, self).__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, x):
        """
        Compute the ELBO for a batch of graphs.

        Parameters:
        x: PyG Batch object with fields x, edge_index, batch.
        """
        B   = int(x.batch.max().item()) + 1
        q   = self.encoder(x.x, x.edge_index, x.batch)
        z   = q.rsample()
        kl  = z.new_zeros(B).index_add(0, x.batch, td.kl_divergence(q, self.prior()))
        return torch.mean(self.decoder.log_prob(z, x) - kl)
    
    @torch.no_grad()
    def sample(self, n_nodes):
        """
        Sample a graph with n_nodes nodes.

        Parameters:
        n_nodes: [int]
           Number of nodes to generate.

        Returns:
        torch_geometric.data.Data with x and edge_index matching the MUTAG format.
        """

        z = self.prior().sample(torch.Size([n_nodes]))        # (n_nodes, M)

        # Sample adjacency (mask diagonal to prevent self-loops)
        edge_logits = self.decoder.decode_edges(z.unsqueeze(0)).squeeze(0)  # (n_nodes, n_nodes)
        edge_logits.fill_diagonal_(float('-inf'))
        triu_mask = torch.triu(torch.ones((n_nodes, n_nodes), dtype=torch.bool, device=z.device), diagonal=1)
        adj_upper = td.Bernoulli(logits=edge_logits).sample() * triu_mask
        adj = adj_upper + adj_upper.t()

        # Convert sampled adjacency to edge_index for node decoder
        edge_index = adj.nonzero(as_tuple=False).T.contiguous()  # (2, num_edges)

        feat_logits = self.decoder.decode_nodes(z, edge_index)
        x = td.OneHotCategorical(logits=feat_logits).sample()

        return Data(x=x, edge_index=edge_index)

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

# %%
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# %% Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

# Empirical node count distribution from training data (used at sample time)
train_node_counts = torch.tensor([dataset[i].num_nodes for i in train_dataset.indices])




def train(model, optimizer, data_loader, validation_loader, epochs, device):

    model.train()

    progress_bar = tqdm(range(epochs), desc="Training")
    best_val_loss = float('inf')
    patience_left  = 200
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_data in validation_loader:
                val_data = val_data.to(device)
                val_loss += model(val_data).item()
            val_loss /= len(validation_loader)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", val_loss=f"{val_loss:.4f}")
        progress_bar.update()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left  = 200
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\nEarly stopping at epoch {epoch+1}, best val loss {best_val_loss:.4f}")
                break
        model.train()


if __name__ == "__main__":
    # %% Set up the VAE model
    M = 16                          # Per-node latent space dimension
    state_dim = 64                  # GNN hidden state dimension
    num_message_passing_rounds = 4
    
    prior = GaussianPrior(M)

    # Encoder GNN: (x, edge_index, batch) → (num_nodes, 2*M)
    encoder_net = DeepGenerativeModel(
        node_feature_dim=node_feature_dim,
        state_dim=state_dim,
        latent_size=M,
        num_message_passing_rounds=num_message_passing_rounds
    ).to(device)

    encoder = GaussianEncoder(encoder_net)

    decoder = GraphDecoder(
        latent_dim=M,
        hidden_dim=64,
        node_feature_dim=node_feature_dim,
        num_message_passing_rounds=num_message_passing_rounds
    ).to(device)

    model = VAE(prior, encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 800
    train(model, optimizer, train_loader, validation_loader, epochs, device)
    # Save the trained model
    save_path = f"./deep_generative_model_mutag_{M}_{state_dim}_{num_message_passing_rounds}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "M": M,
            "state_dim": state_dim,
            "num_message_passing_rounds": num_message_passing_rounds,
            "node_feature_dim": node_feature_dim,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    # Sample — draw n_nodes from the empirical training distribution
    n_nodes = train_node_counts[torch.randint(len(train_node_counts), (1,))].item()
    graph = model.sample(n_nodes=n_nodes)
    print(f"Sampled graph: {n_nodes} nodes, edge_index {graph.edge_index.shape}, x {graph.x.shape}")