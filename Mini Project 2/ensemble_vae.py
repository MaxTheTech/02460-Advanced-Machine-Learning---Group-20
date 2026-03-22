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
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


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
        """Compute pullback metric G = j^T @ J associated with the mean of the Gaussian decoder"""
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

    def geodesic_distance(self):
        """Compute geodesic arc length: sum of sqrt(dc^T G dc) per segment"""
        c = self.get_curve()
        dc = (c[1:] - c[:-1])
        G = self.pullback_metric(self.model, c[:-1])
        seg_lengths = (dc.unsqueeze(1) @ G @ dc.unsqueeze(2)).squeeze().sqrt()
        return seg_lengths.sum()

    def forward(self):
        """Compute energy of piecewise curve under the pullback metric"""
        c = self.get_curve()
        dc = (c[1:] - c[:-1])
        G = self.pullback_metric(self.model, c[:-1])
        # dc[i] @ G[i] @ dc[i] for each segment, then sum
        energy = (dc.unsqueeze(1) @ G @ dc.unsqueeze(2)).squeeze().sum()
        return energy




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
        "--model",
        type=str,
        default='model.pt',
        help='file to save model to or load model from (default: %(default)s)'
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
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    ### MODEL & TRAINING
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    ### GEODESICS
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
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

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            # nn.Softmax(),
            nn.Softplus(),
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
            nn.Sigmoid(), # ADDED BY MAX
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        os.makedirs(args.experiment, exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )

        torch.save(
            model.state_dict(),
            os.path.join(args.experiment, args.model),
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.experiment, args.model), weights_only=True))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            samples_path = os.path.join(
                args.experiment,
                f"samples_{os.path.splitext(os.path.basename(args.model))[0]}.png"
            )
            save_image(samples.view(64, 1, 28, 28), samples_path)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            reconstruction_path = os.path.join(
                args.experiment,
                f"reconstruction_means_{os.path.splitext(os.path.basename(args.model))[0]}.png"
            )
            save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), reconstruction_path)

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.experiment, args.model), weights_only=True))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean().item()
        print(f"Print mean test elbo: {mean_elbo:.4f}")

    elif args.mode == "geodesics":
        # load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.experiment, args.model), weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # encode all test data for latent space plot
        all_latents, all_labels = [], []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                all_latents.append(model.encoder(x).mean.cpu())
                all_labels.append(y)
        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # pair points from different classes so geodesics cross between clusters        
        unique_classes = sorted(all_labels.unique().tolist())
        class_indices = {c: (all_labels == c).nonzero(as_tuple=True)[0] for c in unique_classes}

        c0_list, c1_list = [], []
        for _ in range(args.num_curves):
            cls_a, cls_b = random.sample(unique_classes, 2)
            idx_a = class_indices[cls_a][torch.randint(len(class_indices[cls_a]), (1,))].item()
            idx_b = class_indices[cls_b][torch.randint(len(class_indices[cls_b]), (1,))].item()
            c0_list.append(all_latents[idx_a])
            c1_list.append(all_latents[idx_b])

        c0s = torch.stack(c0_list).to(device)
        c1s = torch.stack(c1_list).to(device)

        geodesics = []
        print(f"\nComputing {args.num_curves} geodesics, with {args.num_t} points per ucurve")
        print(f"Solving for {args.num_iter} iterations using a learning rate of {args.opt_lr}")
        for pair_idx, (c0, c1) in enumerate(zip(c0s, c1s)):
            solver = GeodesicSolver(model, c0, c1, T=args.num_t)
            optimizer = torch.optim.Adam(solver.parameters(), lr=args.opt_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-4)

            print(f"\n  Optimizing geodesic {pair_idx+1}/{args.num_curves}:")
            for iter in range(args.num_iter):
                optimizer.zero_grad()
                energy = solver()
                energy.backward()
                optimizer.step()
                scheduler.step()
                if iter % 500 == 0:
                    print(f"    Pair {pair_idx+1}/{args.num_curves}, iter {iter+1}/{args.num_iter}: energy = {energy.item():.4f}")
                    grad = solver.interior.grad
                    # print(f"    grad norm: {grad.norm().item():.6f}, "f"grad max: {grad.abs().max().item():.6f}")

            with torch.no_grad():
                geo_dist = solver.geodesic_distance().item()
                euc_dist = torch.norm(c1 - c0).item()
                print(f"  Distances - Geodesic: {geo_dist:.4f}, Euclidean: {euc_dist:.4f}")

            # save geodesic
            geodesics.append(solver.get_curve().detach().cpu())

        # plot latent space with geodesics
        fig, ax = plt.subplots(figsize=(8, 8))

        for cls in range(num_classes):
            mask = all_labels == cls
            ax.scatter(all_latents[mask, 0], all_latents[mask, 1], label=str(cls), alpha=0.5, s=10)

        for curve in geodesics:
            ax.plot(curve[:, 0], curve[:, 1], 'k-', linewidth=0.8, alpha=0.7)

        ax.set_title(f"Latent test data with {args.num_curves} pull-back geodesics")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.legend(title='Class')
        fig.savefig(f"{args.experiment}/geodesics_n{args.num_curves}_{os.path.splitext(os.path.basename(args.model))[0]}.png", bbox_inches='tight')