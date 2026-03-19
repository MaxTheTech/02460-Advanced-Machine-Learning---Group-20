import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet
import argparse
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

# This class is from week 3
class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ### 
        t = torch.randint(0, self.T, (x.shape[0],1), device=x.device)
        epsilon = torch.randn_like(x)

        in_norm = torch.sqrt(self.alpha_cumprod[t]) * x + torch.sqrt(1 - self.alpha_cumprod[t]) * epsilon
        res = self.network(in_norm, t.float()/self.T)
        neg_elbo = F.mse_loss(res, epsilon, reduction='none').mean(dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """

        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        with torch.no_grad():
            for t in range(self.T-1, -1, -1):
                ### Implement the remaining of Algorithm 2 here ###
                z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
                t_tensor = torch.full((x_t.shape[0], 1), t / self.T, device=x_t.device, dtype=torch.float32)
                x_t = 1/torch.sqrt(self.alpha[t]) * (x_t - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_cumprod[t]) * self.network(x_t, t_tensor)) + torch.sqrt(self.beta[t]) * z
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()

    def train_mod(self, optimizer, data_loader, epochs, device):
        """
        Train a Flow model.

        Parameters:
        model: [Flow]
        The model to train.
        optimizer: [torch.optim.Optimizer]
            The optimizer to use for training.
        data_loader: [torch.utils.data.DataLoader]
                The data loader to use for training.
        epochs: [int]
            Number of epochs to train for.
        device: [torch.device]
            The device to use for training.
        """
        super().train()

        total_steps = len(data_loader)*epochs
        progress_bar = tqdm(range(total_steps), desc="Training")

        for epoch in range(epochs):
            data_iter = iter(data_loader)
            for x in data_iter:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(device)
                optimizer.zero_grad()
                loss = self.loss(x)
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
                progress_bar.update()
    def train_mod_lat(self, optimizer, data_loader, epochs, device, encoder):
        super().train()

        total_steps = len(data_loader)*epochs
        progress_bar = tqdm(range(total_steps), desc="Training")
        encoder = encoder.to(device)
        for epoch in range(epochs):
            data_iter = iter(data_loader)
            for x in data_iter:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(device)
                optimizer.zero_grad()
                z = encoder(x).rsample()
                loss = self.loss(z)
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
                progress_bar.update()

    def sample_lat(self, n_samples, latent_dim):
            z = torch.randn(n_samples, latent_dim).to(self.alpha.device)
            with torch.no_grad():
                for t in range(self.T-1, -1, -1):
                    noise = torch.randn_like(z) if t > 0 else torch.zeros_like(z)
                    t_tensor = torch.full((z.shape[0], 1), t / self.T, device=z.device, dtype=torch.float32)
                    z = 1/torch.sqrt(self.alpha[t]) * (z - (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_cumprod[t]) * self.network(z, t_tensor)) + torch.sqrt(self.beta[t]) * noise
            return z

# The following classes are from week 1, with the following changes
# VAE now is a \beta-VAE, with a beta parameter that can be set to 1 for a standard VAE.
class basic_network(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(basic_network, self).__init__()
        mid_ch = max(in_ch, out_ch)
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            nn.Linear(in_ch, mid_ch),
            nn.ReLU(),
            nn.Linear(mid_ch, mid_ch),
            nn.ReLU(),
            nn.Linear(mid_ch, out_ch),
        )

    def forward(self, x, t=None):
        if t is not None:
            t = t.to(x.device)  
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            t = t.expand(x.shape[0], -1)
            x = torch.cat([x, t], dim=-1)
        if x.dim() == 2:
            # If input is (batch_size, features), reshape appropriately
            return self.layers(x)
        elif x.dim() == 4:
            # x: (B, C, H, W) -> apply 5 linear layers per pixel
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, c)
            x = self.layers(x)
            x = x.view(b, h, w, self.out_ch).permute(0, 3, 1, 2)
        return x
class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.zeros(784), requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(self.std)), 1)

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

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, beta=1.0):
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
        self.beta = beta

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo_value = torch.mean(self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()), dim=0)
        return elbo_value

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
    def train_mod(self, optimizer, data_loader, epochs, device):
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
        super().train()

        total_steps = len(data_loader)*epochs
        progress_bar = tqdm(range(total_steps), desc="Training")

        for epoch in range(epochs):
            data_iter = iter(data_loader)
            for x in data_iter:
                x = x[0].to(device)
                optimizer.zero_grad()
                loss = self(x)
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
                progress_bar.update()

       