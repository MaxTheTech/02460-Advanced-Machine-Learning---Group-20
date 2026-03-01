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


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians (MoG) prior distribution.

                Parameters:
        M: [int]
            Dimension of the latent space.
        K: [int]
            Number of Gaussian components.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(torch.randn(K, M))
        self.log_stds = nn.Parameter(torch.zeros(K, M))

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)


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


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


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
        # OLD: elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        elbo = torch.mean(self.decoder(z).log_prob(x) + self.prior().log_prob(z) - q.log_prob(z), dim=0)
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
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    import os

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    
    # configuration params
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')

    # model architecture
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='prior distribution (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=10, metavar='K', help='number of MoG components (default: %(default)s)')
    parser.add_argument('--decoder', type=str, default='bernoulli', choices=['bernoulli', 'gaussian'], help='decoder distribution, "bernoulli" for binarized MNIST and "gaussian" for continuous MNIST (default: %(default)s)')

    # training
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    # eval
    parser.add_argument('--save-posterior', action='store_true', help='save the posterior')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # load MNIST (binarized or continous) and create data loaders
    if args.decoder == 'bernoulli':
        # binarized MNIST
        thresshold = 0.5
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])
    else:
        # continuous MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])

    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == "gaussian":
        prior = GaussianPrior(M)
    elif args.prior == "mog":
        prior = MoGPrior(M, args.num_components)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    model_path = os.path.join("models", args.model)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu()

            samples_file = f"samples_{os.path.splitext(os.path.basename(args.model))[0]}.png"
            samples_path = os.path.join("figures", samples_file)

            os.makedirs(os.path.dirname(samples_path), exist_ok=True)
            save_image(samples.view(64, 1, 28, 28), samples_path)
            print(f"Samples saved to {samples_path}")
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
        model.eval()

        elbo_sum = 0.0
        n_samples = 0
        all_priors = []
        all_z = []
        all_z_labels = []

        with torch.no_grad():
            for x, labels in mnist_test_loader:
                x = x.to(device)
            
                # batch ELBO
                elbo_sum += model.elbo(x).sum()
                n_samples += x.shape[0]

                # prior samples
                prior = model.prior().sample(torch.Size([x.shape[0]]))
                all_priors.append(prior)

                # approximate posterior samples
                q = model.encoder(x)
                z = q.rsample()
                all_z.append(z)
                all_z_labels.append(labels)

        elbo_avg = elbo_sum / n_samples
        print(f"Avg. ELBO on test set: {elbo_avg}")

        all_priors = torch.cat(all_priors, dim=0).cpu().numpy()
        all_z = torch.cat(all_z, dim=0).cpu().numpy()
        all_labels = torch.cat(all_z_labels, dim=0).cpu().numpy()

        if M > 2:
            pca = PCA(n_components=2)
            all_z = pca.fit_transform(all_z)
            all_priors = pca.transform(all_priors)
        
        comparison_file = f"comparison_{os.path.splitext(os.path.basename(args.model))[0]}"
        comparison_path = os.path.join("figures", comparison_file)

        # plot prior vs approximate posterior samples
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(all_priors[:, 0], all_priors[:, 1], label="prior", c="blue", s=1, alpha=0.5)
        scatter = plt.scatter(all_z[:, 0], all_z[:, 1], label="posterior", c="red", s=1, alpha=0.5)
        plt.xlabel('z1' if M <= 2 else 'PC1')
        plt.ylabel('z2' if M <= 2 else 'PC2')
        plt.title('Prior VS Approximate Posterior Samples')
        plt.tight_layout()
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        plt.savefig(comparison_path, dpi=150)
        print(f"Prior VS posterior plot saved to {comparison_path}")


