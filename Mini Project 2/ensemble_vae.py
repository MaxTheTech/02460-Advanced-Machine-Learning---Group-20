# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np


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
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        data_iter = iter(data_loader)
        for step in pbar:
            try:
                x = next(data_iter)[0]
            except StopIteration:
                data_iter = iter(data_loader)
                x = next(data_iter)[0]
                epoch += 1

            x = noise(x.to(device))
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                loss = loss.detach().cpu()
                pbar.set_description(
                    f"epoch={epoch}, step={step}, loss={loss:.1f}"
                )


def pick_test_pairs(test_loader, num_pairs, seed):
    rng = random.Random(seed)
    all_labels = torch.cat([y for _, y in test_loader], dim=0)
    unique_classes = sorted(all_labels.unique().tolist())
    class_indices = {c: (all_labels == c).nonzero(as_tuple=True)[0].tolist() for c in unique_classes}

    pairs = []
    for _ in range(num_pairs):
        cls_a, cls_b = rng.sample(unique_classes, 2)
        idx_a = rng.choice(class_indices[cls_a])
        idx_b = rng.choice(class_indices[cls_b])
        pairs.append((idx_a, idx_b))
    return pairs


class GeodesicSolver(nn.Module):
    def __init__(self, model, c0, c1, T=1000):
        super().__init__()
        self.T = T
        self.model = model
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)
        t = torch.linspace(0, 1, T + 1, device=c0.device)[1:-1, None]
        interior = c0[None, :] * (1 - t) + c1[None, :] * t
        self.interior = nn.Parameter(interior)

    def get_curve(self):
        return torch.cat([self.c0[None], self.interior, self.c1[None]], dim=0)

    def pullback_metric(self, model, z):
        """Compute pullback metric G = j^T @ J"""
        B, M = z.shape

        def decode(z_):
            return model.decoder.decoder_net(z_).reshape(B, -1)

        basis = torch.eye(M, device=z.device)
        J = torch.stack([
            torch.autograd.functional.jvp(decode, (z,), (basis[j].expand(B, -1),), create_graph=True)[1]
            for j in range(M)
        ], dim=-1) # [B, D, M]

        G = J.mT @ J # [B, M, M]
        return G

    def forward(self):
        """Compute energy of piecewise curve under the pullback metric"""
        c = self.get_curve()
        dc = (c[1:] - c[:-1])
        G = self.pullback_metric(self.model, c[:-1])
        # dc[i] @ G[i] @ dc[i] for each segment, then sum
        energy = (dc.unsqueeze(1) @ G @ dc.unsqueeze(2)).squeeze().sum()
        return energy

    def geodesic_distance(self):
        """Compute geodesic arc length: sum of sqrt(dc^T G dc)"""
        c = self.get_curve()
        dc = (c[1:] - c[:-1])
        G = self.pullback_metric(self.model, c[:-1])
        seg_lengths = (dc.unsqueeze(1) @ G @ dc.unsqueeze(2)).squeeze().sqrt()
        return seg_lengths.sum()


def plot_geodesics(model, all_latents, all_labels, geodesics, num_classes, device, title, save_path):
    import matplotlib
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8, 8))

    # grid for heatmap
    z_np = all_latents.numpy()
    margin = 0.5
    x_min, x_max = z_np[:, 0].min() - margin, z_np[:, 0].max() + margin
    y_min, y_max = z_np[:, 1].min() - margin, z_np[:, 1].max() + margin
    grid_n = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n), np.linspace(y_min, y_max, grid_n))
    z_grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32).to(device)

    with torch.no_grad():
        decoded = model.decoder.decoder_net(z_grid).reshape(z_grid.shape[0], -1)
        pixel_std = decoded.std(dim=1).cpu().numpy().reshape(xx.shape)
    im = ax.pcolormesh(xx, yy, pixel_std, cmap='viridis', shading='auto')
    fig.colorbar(im, ax=ax, label='Standard deviation of pixel values')

    # plot points
    colors = ["lightcoral", "lightskyblue", "yellow"]
    for cls in range(num_classes):
        mask = all_labels == cls
        ax.scatter(all_latents[mask, 0], all_latents[mask, 1], color=colors[cls % len(colors)], alpha=0.4, s=20)

    # plot geodesics
    cmap_lines = matplotlib.colormaps.get_cmap('Set2')
    for i, curve in enumerate(geodesics):
        color = cmap_lines(i % 8)
        ax.plot([curve[0, 0], curve[-1, 0]], [curve[0, 1], curve[-1, 1]], '--', color=color, linewidth=2.2)
        ax.plot(curve[:, 0], curve[:, 1], '-', color=color, linewidth=2.2, alpha=1.0)

    handles = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2.2, label='Pullback geodesic'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2.2, label='Straight line'),
    ]
    ax.legend(handles=handles)
    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)




if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()

    ### CONFIGURATION
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        help="name of folder to save experiments in (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=1,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: %(default)s)",
    )
    ### MODEL & TRAINING
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    ### GEODESICS
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--opt-lr",
        type=float,
        default=0.001,
        help="learning rate of geodesic optimizer (default: %(default)s)",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=2000,
        metavar="N",
        help="number of iterations to train geodesic optimizer (default: %(default)s)",
    )




    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")


    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            # nn.Softmax(),
            nn.Softplus(), # original template code used the (current) Softmax implementation, I found Softplus works better
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            # nn.Softmax(),
            nn.Softplus(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            # nn.Softmax(),
            nn.Softplus(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            # nn.Softmax(),
            nn.Softplus(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # nn.Softmax(),
            nn.Softplus(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net




    # Choose mode to run
    if args.mode == "train":
        os.makedirs(args.experiment, exist_ok=True)

        for run in range(args.num_reruns):
            run_seed = args.seed + run
            torch.manual_seed(run_seed)
            random.seed(run_seed)
            np.random.seed(run_seed)
            print(f"\nTraining run {run+1}/{args.num_reruns} (seed={run_seed})")

            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, args.device)

            torch.save(model.state_dict(), os.path.join(args.experiment, f"model_run{run+1}.pt"))


    elif args.mode == "sample":
        model_paths = [os.path.join(args.experiment, f"model_run{r+1}.pt")
                       for r in range(args.num_reruns)
                       if os.path.exists(os.path.join(args.experiment, f"model_run{r+1}.pt"))]
        if not model_paths:
            raise FileNotFoundError(f"No model_run*.pt found in {args.experiment}")

        data = next(iter(mnist_test_loader))[0].to(device)

        for m, mpath in enumerate(model_paths):
            model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()), GaussianEncoder(new_encoder())).to(device)
            model.load_state_dict(torch.load(mpath, weights_only=True))
            model.eval()

            with torch.no_grad():
                samples = model.sample(64).cpu()
                save_image(samples.view(64, 1, 28, 28), os.path.join(args.experiment, f"samples_run{m+1}.png"))

                recon = model.decoder(model.encoder(data).mean).mean
                save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), os.path.join(args.experiment, f"reconstruction_means_run{m+1}.png"))

            print(f"Run {m+1}: saved samples and reconstructions")


    elif args.mode == "eval":
        model_paths = [os.path.join(args.experiment, f"model_run{r+1}.pt")
                       for r in range(args.num_reruns)
                       if os.path.exists(os.path.join(args.experiment, f"model_run{r+1}.pt"))]
        if not model_paths:
            raise FileNotFoundError(f"No model_run*.pt found in {args.experiment}")

        for m, mpath in enumerate(model_paths):
            model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()), GaussianEncoder(new_encoder())).to(device)
            model.load_state_dict(torch.load(mpath, weights_only=True))
            model.eval()

            elbos = []
            with torch.no_grad():
                for x, y in mnist_test_loader:
                    x = x.to(device)
                    elbos.append(model.elbo(x))
            mean_elbo = torch.tensor(elbos).mean().item()
            print(f"Run {m+1}: mean test ELBO = {mean_elbo:.4f}")


    elif args.mode == "geodesics":
        # fixed test-data pairs (same indices across all models for CoV)
        test_pairs = pick_test_pairs(mnist_test_loader, args.num_curves, seed=args.seed)

        # retrieve test data
        all_test_x = torch.cat([x for x, _ in mnist_test_loader], dim=0)
        all_test_y = torch.cat([y for _, y in mnist_test_loader], dim=0)

        # find all trained models
        model_paths = [os.path.join(args.experiment, f"model_run{r+1}.pt")
                       for r in range(args.num_reruns)
                       if os.path.exists(os.path.join(args.experiment, f"model_run{r+1}.pt"))]
        if not model_paths:
            raise FileNotFoundError(f"No model_run*.pt found in {args.experiment}")
        print(f"Found {len(model_paths)} trained models")

        all_model_geodesics = []
        all_model_latents = []
        for m, mpath in enumerate(model_paths):
            print(f"\nModel {m+1}/{len(model_paths)}: {mpath}")

            # load model
            model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()), GaussianEncoder(new_encoder())).to(device)
            model.load_state_dict(torch.load(mpath, weights_only=True))
            model.eval()
            # needed because jvp creates a computation graph, don't want to accidentally update VAE parameters
            for p in model.parameters():
                p.requires_grad_(False)

            # encode all test data with this model
            with torch.no_grad():
                all_latents = model.encoder(all_test_x.to(device)).mean.cpu()
            all_model_latents.append(all_latents)

            geo_dists = []
            euc_dists = []
            model_geodesics = []
            for k, (idx_a, idx_b) in enumerate(test_pairs):
                # retrieve pair endpoints
                c0 = all_latents[idx_a].to(device)
                c1 = all_latents[idx_b].to(device)

                # initialize solver
                solver = GeodesicSolver(model, c0, c1, T=args.num_t)
                optimizer = torch.optim.Adam(solver.parameters(), lr=args.opt_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-4)

                # train solver
                for it in range(args.num_iter):
                    optimizer.zero_grad()
                    energy = solver()
                    energy.backward()
                    optimizer.step()
                    scheduler.step()
                    if m == 0 and it % 500 == 0:
                        print(f"  Pair {k+1}/{args.num_curves}, iter {it+1}/{args.num_iter}: energy={energy.item():.4f}")

                # calculate geodesic and euclidean distances
                with torch.no_grad():
                    geo_dist = solver.geodesic_distance().item()
                    euc_dist = torch.norm(c1 - c0).item()
                geo_dists.append(geo_dist)
                euc_dists.append(euc_dist)

                print(f"  Pair {k+1} distances: geo={geo_dist:.4f}, euc={euc_dist:.4f}\n")

                # get geodesic curve points
                model_geodesics.append(solver.get_curve().detach().cpu())
            
            all_model_geodesics.append(model_geodesics)
        
            plot_geodesics(
                model, all_model_latents[m], all_test_y, all_model_geodesics[m], num_classes, device,
                title=f"Geodesics in latent space\n({args.num_decoders} decoder{'s' if args.num_decoders > 1 else ''}, run {m+1}/{args.num_reruns}, {args.num_curves} endpoint pairs)",
                save_path=f"{args.experiment}/geodesics_n{args.num_curves}_run{m+1}.png",
            )
