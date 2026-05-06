import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from tqdm import tqdm
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data


class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std  = nn.Parameter(torch.ones(M),  requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)


class GNNEncoderNet(nn.Module):
    """GNN encoder with global mean pooling. Maps a graph to a single embedding."""
    def __init__(self, node_feature_dim, state_dim, latent_dim, num_rounds):
        super().__init__()
        self.input_lin = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )
        self.convs = nn.ModuleList([GraphConv(state_dim, state_dim) for _ in range(num_rounds)])
        self.output_lin = nn.Linear(state_dim, 2 * latent_dim)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_lin(x))
        for conv in self.convs:
            h = h + F.relu(conv(h, edge_index))
        h_graph = global_mean_pool(h, batch)
        return self.output_lin(h_graph)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x, edge_index, batch):
        out = self.encoder_net(x, edge_index, batch)
        mean, log_std = out.chunk(2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)


class GraphDecoder(nn.Module):
    """
    Decodes a graph-level latent z into an adjacency matrix. Each node position gets a distinct learnable base embedding (pos_embed). z_graph shifts all embeddings via a learned linear map. Small i.i.d. noise is added per node.
    """
    def __init__(self, latent_dim, node_embed_dim, max_nodes=28):
        super().__init__()
        self.node_embed_dim = node_embed_dim
        self.pos_embed  = nn.Embedding(max_nodes, node_embed_dim)
        self.z_to_delta = nn.Linear(latent_dim, node_embed_dim)
        self.log_noise  = nn.Parameter(torch.full((1,), -2.0))
        self.bias       = nn.Parameter(torch.zeros(1))

    def _node_embeds(self, z, n):
        """
        z: (B, M) or (1, M)
        Returns e: (B, n, node_embed_dim)
        """
        positions = torch.arange(n, device=z.device)
        pos   = self.pos_embed(positions)
        delta = self.z_to_delta(z)
        noise = torch.exp(self.log_noise) * torch.randn(
            z.shape[0], n, self.node_embed_dim, device=z.device
        )
        return pos.unsqueeze(0) + delta.unsqueeze(1) + noise

    def logits(self, z, node_mask):
        """z: (B, M), node_mask: (B, max_N) --> raw logits (B, max_N, max_N)"""
        _, max_N = node_mask.shape
        e = self._node_embeds(z, max_N)
        return torch.bmm(e, e.transpose(1, 2)) + self.bias

    def forward(self, z, node_mask):
        return td.Bernoulli(logits=self.logits(z, node_mask))

    @torch.no_grad()
    def sample(self, z, n_nodes, remove_isolated=True):
        """z: (M,), n_nodes: int --> PyG Data with sampled adjacency"""
        e = self._node_embeds(z.unsqueeze(0), n_nodes).squeeze(0)
        raw_logits  = e @ e.T + self.bias
        A_upper = td.Bernoulli(logits=raw_logits).sample().triu(diagonal=1)
        A = A_upper + A_upper.T
        if remove_isolated:
            connected = A.sum(dim=1) > 0
            A = A[connected][:, connected]
            n_nodes = int(connected.sum().item())
        edge_index = A.nonzero().T.contiguous()
        return Data(edge_index=edge_index, num_nodes=n_nodes)


class VAE(nn.Module):
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior   = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, data, beta=1.0, pos_weight=2.5):
        # Graph-level posterior: q(z|G), one vector per graph
        q = self.encoder(data.x, data.edge_index, data.batch)
        z = q.rsample()

        # KL is already per-graph (event_dim=1 absorbed by Independent)
        kl = td.kl_divergence(q, self.prior())

        _, node_mask = to_dense_batch(data.x, data.batch)
        adj_target   = to_dense_adj(
            data.edge_index, data.batch, max_num_nodes=node_mask.shape[1]
        )

        valid_edge_mask = (
            node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        ).triu(diagonal=1).float()

        raw_logits = self.decoder.logits(z, node_mask)
        # Weighted BCE: penalise missing a real edge pos_weight times more than a false positive.
        pw = torch.tensor(pos_weight, device=z.device)
        log_prob = -F.binary_cross_entropy_with_logits(
            raw_logits, adj_target, pos_weight=pw, reduction='none'
        )
        logp_per_graph = (log_prob * valid_edge_mask).sum(dim=[1, 2])

        return (logp_per_graph - beta * kl).mean()

    def forward(self, data, beta=1.0):
        return -self.elbo(data, beta=beta)

    @torch.no_grad()
    def sample(self, n_nodes, remove_isolated=True):
        z = self.prior().sample()
        return self.decoder.sample(z, n_nodes, remove_isolated=remove_isolated)


def train(model, optimizer, data_loader, validation_loader, epochs, device):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        beta = min(1.0, epoch / (epochs * 0.5))
        total, n = 0.0, 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item()
            n += 1

        model.eval()
        with torch.no_grad():
            val = sum(
                model(d.to(device), beta=1.0).item() for d in validation_loader
            ) / len(validation_loader)
        tqdm.write(f"Epoch {epoch+1}/{epochs}  loss={total/n:.4f}  val_loss={val:.4f}  beta={beta:.4f}")
        model.train()


if __name__ == "__main__":
    from torch.utils.data import random_split
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset          = TUDataset(root='./data/', name='MUTAG')
    node_feature_dim = 7

    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, (100, 44, 44), generator=rng)

    train_loader      = DataLoader(train_dataset,      batch_size=10, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=44)

    M              = 16
    state_dim      = 64
    node_embed_dim = 32
    num_rounds     = 3
    epochs         = 1000

    prior   = GaussianPrior(M)
    encoder = GaussianEncoder(GNNEncoderNet(node_feature_dim, state_dim, M, num_rounds))
    decoder = GraphDecoder(M, node_embed_dim, max_nodes=28)
    model   = VAE(prior, encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train(model, optimizer, train_loader, validation_loader, epochs, device)

    save_path = f"./graph_vae_mutag_{M}_{state_dim}_{num_rounds}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "M": M,
        "state_dim": state_dim,
        "node_embed_dim": node_embed_dim,
        "num_rounds": num_rounds,
        "node_feature_dim": node_feature_dim,
    }, save_path)
    print(f"Saved to {save_path}")
