import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from tqdm import tqdm
from torch_geometric.nn import GraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std  = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GNNEncoderNet(nn.Module):
    def __init__(self, node_feature_dim, state_dim, latent_dim, num_rounds):
        super().__init__()
        self.input_lin = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([GraphConv(state_dim, state_dim) for _ in range(num_rounds)])
        self.output_lin = nn.Linear(state_dim, 2 * latent_dim)  

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_lin(x))
        for conv in self.convs:
            h = h + F.relu(conv(h, edge_index))
        return self.output_lin(h)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x, edge_index, batch):
        mean, log_std = torch.chunk(self.encoder_net(x, edge_index, batch), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)


class GraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1))    

    def forward(self, z_dense):
        logits = torch.bmm(z_dense, z_dense.transpose(1, 2)) + self.bias
        return td.Bernoulli(logits=logits)


class VAE(nn.Module):
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, data, beta=1.0): 
        q = self.encoder(data.x, data.edge_index, data.batch)
        z = q.rsample()

        kl_per_node = td.kl_divergence(q, self.prior())
        num_graphs = int(data.batch.max()) + 1
        kl_per_graph = torch.zeros(num_graphs, device=z.device)
        kl_per_graph.index_add_(0, data.batch, kl_per_node)

        z_dense, node_mask = to_dense_batch(z, data.batch)
        adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=z_dense.size(1))
        valid_edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).triu(diagonal=1)

        edge_dist = self.decoder(z_dense)
        log_prob_edges = edge_dist.log_prob(adj_target)
        logp_per_graph = (log_prob_edges * valid_edge_mask).sum(dim=[1, 2])        
        return (logp_per_graph - beta * kl_per_graph).mean()

    def forward(self, data, beta=1.0):
        return -self.elbo(data, beta=beta)

    @torch.no_grad()
    def sample(self, n_nodes):
        z = self.prior().sample(torch.Size([n_nodes]))
        logits = z @ z.T + self.decoder.bias
        A_upper = td.Bernoulli(logits=logits).sample().triu(diagonal=1)
        return Data(edge_index=(A_upper + A_upper.T).nonzero().T, num_nodes=n_nodes) 


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
        avg = total / n

        model.eval()
        with torch.no_grad():
            val = sum(model(d.to(device), beta=1.0).item() for d in validation_loader) / len(validation_loader)
        tqdm.write(f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}  val_loss={val:.4f}  beta={beta:.4f}")
        model.train()


if __name__ == "__main__":
    from torch.utils.data import random_split
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = TUDataset(root='./data/', name='MUTAG')
    node_feature_dim = 7

    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, (100, 44, 44), generator=rng)

    train_loader      = DataLoader(train_dataset, batch_size=10, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=44)
    test_loader       = DataLoader(test_dataset, batch_size=44)

    train_node_counts = torch.tensor(
        [dataset[i].num_nodes for i in train_dataset.indices])

    M           = 4 
    state_dim   = 32 
    num_rounds  = 2
    epochs      = 5000

    prior   = GaussianPrior(M)
    encoder = GaussianEncoder(GNNEncoderNet(node_feature_dim, state_dim, M, num_rounds))
    decoder = GraphDecoder()
    model   = VAE(prior, encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train(model, optimizer, train_loader, validation_loader, epochs, device)

    save_path = f"./graph_vae_mutag_{M}_{state_dim}_{num_rounds}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "M": M,
        "state_dim": state_dim,
        "num_message_passing_rounds": num_rounds,
        "node_feature_dim": node_feature_dim,
    }, save_path)
    print(f"Model saved to {save_path}")

    model.eval()
    n_nodes = train_node_counts[torch.randint(len(train_node_counts), (1,))].item()
    graph = model.sample(n_nodes=n_nodes)
    print(f"Sampled graph: {n_nodes} nodes, edge_index {graph.edge_index.shape}")