"""Compute and report the Fréchet Inception Distance (FID) for samples generated
from each of the three generative models (DDPM, latent DDPM, and the chosen
VAE) using the provided code (the function compute_fid in fid.py). For the latent
DDPM, you must additionally report FID scores for different values of β, including
β = 10−6"""

""" Measure and report the wall-clock sampling time (e.g., samples per second) for the
chosen VAE, DDPM, and latent DDPM"""
"""• Discuss and compare the sampling quality and FID scores across the three models
in relation to their sampling times.
"""
"""• Plot, discuss, and compare the prior from the β-VAE and the learned latent DDPM
distribution against the aggregate posterior.
"""
import time
from models_PartB import *
from mains import *
from fid import compute_fid
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

########################################## Time and FID computation ##########################################
device = "cuda" if torch.cuda.is_available() else "cpu"
M = 16
T = 2000
N_samples = 500
## get real data batch
transform = transforms . Compose ([ transforms . ToTensor () ,
    transforms . Lambda ( lambda x : x + torch . rand ( x . shape ) /255) ,
    transforms . Lambda ( lambda x : (x -0.5) *2.0) ])

train_data = datasets . MNIST ( 'Week1\data', train = True , download = True ,
transform = transform )

train_loader = torch.utils.data.DataLoader ( train_data , batch_size = N_samples , shuffle = True )
batch = next(iter(train_loader))
if isinstance(batch, (list, tuple)):
    batch = batch[0]
D = 28*28
batch_real = batch
batch_real = batch_real.to(device)

# sample time for DDPM and FID
network = Unet()
model = DDPM(network,T=T).to(device)
model.load_state_dict(torch.load("model_unet.pt", map_location=device))
model.eval()

with torch.no_grad():
    start_time = time.time()
    samples = model.sample((N_samples, 784))
    end_time = time.time()
    samples = samples.view(-1, 1, 28, 28)
    ddpm_time = end_time - start_time
print(f"DDPM sampling time for {N_samples} samples: {ddpm_time:.4f} seconds")

fid_ddpm = compute_fid(batch_real, samples, device=device, classifier_ckpt="Mini_Project_1/mnist_classifier.pth")
print(f"DDPM FID: {fid_ddpm:.4f}")

# sample time for VAE
encoder_net = basic_network(D, 2*M)  
decoder_net = basic_network(M, D)
prior_g = GaussianPrior(M)
encoder = GaussianEncoder(encoder_net)
decoder = GaussianDecoder(decoder_net)
model = VAE(prior_g, decoder, encoder, beta=0.1).to(device)
model.load_state_dict(torch.load("model_VAE.pt", map_location=device))
model.eval()

with torch.no_grad():
    start_time = time.time()
    samples = model.sample(N_samples)
    samples = samples.view(-1, 1, 28, 28)
    end_time = time.time()
    vae_time = end_time - start_time
    print(f"VAE sampling time for {N_samples} samples: {vae_time:.4f} seconds")

fid_vae = compute_fid(batch_real, samples, device=device, classifier_ckpt="Mini_Project_1/mnist_classifier.pth")
print(f"VAE FID: {fid_vae:.4f}")

# sample time for DDPM_VAE
encoder_net = basic_network(D, 2*M)  
decoder_net = basic_network(M, D)
prior_g = GaussianPrior(M)
vae = VAE(prior_g, GaussianDecoder(decoder_net), GaussianEncoder(encoder_net), beta=0.1).to(device)
vae.load_state_dict(torch.load("model_VAE.pt", map_location=device))
encoder = vae.encoder
network = basic_network(M + 1, M)
model = DDPM(network, T=T).to(device)
model.load_state_dict(torch.load("model_DDPM_VAE.pt", map_location=device))
model.eval()


with torch.no_grad():
    start_time = time.time()
    z = model.sample_lat(N_samples, M)
    samples = vae.decoder(z).mean
    samples = samples.view(-1, 1, 28, 28)
    end_time = time.time()
    ddpm_vae_time = end_time - start_time
    print(f"DDPM_VAE sampling time for {N_samples} samples: {ddpm_vae_time:.4f} seconds")

fid_ddpm_vae = compute_fid(batch_real, samples, device=device, classifier_ckpt="Mini_Project_1/mnist_classifier.pth")
print(f"DDPM_VAE FID: {fid_ddpm_vae:.4f} with beta=0.1")
############################### Plotting the prior and aggregate posterior ##########################################
def plot(ddpm, vae, beta, real_images):
    ddpm.eval()
    vae.eval()
    with torch.no_grad():
        z_posterior = vae.encoder(real_images).sample().cpu().numpy()
        z_ddpm = ddpm.sample_lat(real_images.shape[0], vae.prior.M).cpu().numpy()
        z_prior = vae.prior().sample([real_images.shape[0]]).cpu().numpy()

    pca = PCA(n_components=2)
    post_pca = pca.fit_transform(z_posterior)
    ddpm_pca = pca.transform(z_ddpm)
    prior_pca = pca.transform(z_prior)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(post_pca[:, 0], post_pca[:, 1], alpha=0.5, label="Aggregate Posterior")
    plt.scatter(prior_pca[:, 0], prior_pca[:, 1], alpha=0.5, color='green', label="Prior")
    plt.title(f"Prior vs Aggregate Posterior (beta={beta})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(post_pca[:, 0], post_pca[:, 1], alpha=0.5, label="Aggregate Posterior")
    plt.scatter(ddpm_pca[:, 0], ddpm_pca[:, 1], alpha=0.5, color='orange', label="Latent DDPM")
    plt.title(f"Latent DDPM vs Aggregate Posterior (beta={beta})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"latent_distributions_beta_{beta}.png")
    plt.close()
##########################################
betas = [1e-6, 1e-4, 1e-2, 0.1, 1.0]
fid_scores = []
lr = 1e-4
epochs = 50
batch_size = 32
transform = transforms . Compose ([ transforms . ToTensor () ,
    transforms . Lambda ( lambda x : x + torch . rand ( x . shape ) /255) ,
    transforms . Lambda ( lambda x : (x -0.5) *2.0) ,
    transforms . Lambda ( lambda x : x . flatten () ) ])

train_data = datasets . MNIST ( 'Week1\data', train = True , download = True ,
transform = transform )

train_loader = torch.utils.data.DataLoader ( train_data , batch_size = batch_size , shuffle = True )
batch = next(iter(train_loader))
if isinstance(batch, (list, tuple)):
    batch = batch[0]
    D = batch.shape[1]
M = 16
test_loader = torch.utils.data.DataLoader ( train_data , batch_size = 1000 , shuffle = False )
real_images, _ = next(iter(test_loader))

#######################################



for beta in betas:
    # train the VAE with the given beta
    encoder_net = basic_network(D, 2*M)  
    decoder_net = basic_network(M, D)
    prior_g = GaussianPrior(M)
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    vae = VAE(prior_g, decoder, encoder, beta=beta).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train_mod(optimizer, train_loader, epochs=epochs, device=device)
    for param in vae.encoder.parameters():
        param.requires_grad = False
    vae.encoder.eval()

    # use the trained VAE to train the DDPM_VAE
    network = basic_network(M + 1, M)
    model = DDPM(network, T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train_mod_lat(optimizer, train_loader, epochs=epochs, device=device, encoder=encoder)
    # compute FID
    vae.eval()
    model.eval()
    with torch.no_grad():
        z = model.sample_lat(N_samples, M)
        samples = vae.decoder(z).mean
        samples = samples.view(-1, 1, 28, 28)
    fid = compute_fid(batch_real, samples, device=device, classifier_ckpt="Mini_Project_1/mnist_classifier.pth")
    fid_scores.append(fid)
    print(f"Beta: {beta}, FID: {fid:.4f}")
    plot(model, vae, beta, real_images.to(device))


