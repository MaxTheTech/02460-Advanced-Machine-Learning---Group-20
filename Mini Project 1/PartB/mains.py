import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
from unet import Unet
from models_PartB import DDPM, VAE, GaussianEncoder, GaussianDecoder, GaussianPrior, basic_network
import argparse
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--network', type=str, default='unet', choices=['unet', "VAE", "DDPM_VAE"], help='network to use for the diffusion process (default: %(default)s)')
    parser.add_argument('--model', type=str, help='file to save model to or load model from')
    parser.add_argument('--samples', type=str, help='file to save samples in')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--beta-VAE', type=float, default=0.10, metavar='V', help='beta for VAE loss (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=16, metavar='N', help='latent dimension for VAE (default: %(default)s)')
    args = parser.parse_args()
    
    if args.model is None:
        args.model = f'model_{args.network}.pt'
    if args.samples is None:
        args.samples = f'samples_{args.network}.png'

    # Load the dataset
    transform = transforms . Compose ([ transforms . ToTensor () ,
        transforms . Lambda ( lambda x : x + torch . rand ( x . shape ) /255) ,
        transforms . Lambda ( lambda x : (x -0.5) *2.0) ,
        transforms . Lambda ( lambda x : x . flatten () ) ])

    train_data = datasets . MNIST ( 'Week1\data', train = True , download = True ,
    transform = transform )

    train_loader = torch.utils.data.DataLoader ( train_data , batch_size = args . batch_size , shuffle = True )
    test_loader = torch.utils.data.DataLoader ( train_data , batch_size = args . batch_size , shuffle = False )

    batch = next(iter(train_loader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
        D = batch.shape[1]
    # Define the network
    num_hidden = 256
    T = 2000

    if args.network == 'unet':
        network = Unet()
        model = DDPM(network,T=T).to(args.device)
    elif args.network == 'VAE':
        M = args.latent_dim
        encoder_net = basic_network(D, 2*M)  # D -> 2*M
        decoder_net = basic_network(M, D)            # M -> D
        prior_g = GaussianPrior(M)
        encoder = GaussianEncoder(encoder_net)
        decoder = GaussianDecoder(decoder_net)
        model = VAE(prior_g, decoder, encoder, beta=args.beta_VAE).to(args.device)
    elif args.network == 'DDPM_VAE':
        encoder_net = basic_network(D, 2*args.latent_dim)  
        decoder_net = basic_network(args.latent_dim, D)
        prior_g = GaussianPrior(args.latent_dim)
        vae = VAE(prior_g, GaussianDecoder(decoder_net), GaussianEncoder(encoder_net), beta=args.beta_VAE)
        vae.load_state_dict(torch.load("model_VAE.pt", map_location=args.device))
        encoder = vae.encoder
        network = basic_network(args.latent_dim + 1, args.latent_dim)
        model = DDPM(network, T=T).to(args.device)
        for param in encoder.parameters():
            param.requires_grad = False
        

    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.network == 'unet' or args.network == 'VAE':
            model.train_mod( optimizer, train_loader, args.epochs, args.device)
        else:
            model.train_mod_lat(optimizer, train_loader, args.epochs, args.device, encoder)
        torch.save(model.state_dict(), args.model)
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        model.eval()
        if args.network == 'unet':
            samples = model.sample((5, 784))
            save_image(samples.view(-1, 1, 28, 28), args.samples, normalize=True, value_range=(-1, 1))
        elif args.network == 'VAE':
            samples = model.sample(5)
            save_image(samples.view(-1, 1, 28, 28), args.samples, normalize=True, value_range=(-1, 1))
        elif args.network == 'DDPM_VAE':
            vae.to(args.device)
            z = model.sample_lat(5, args.latent_dim)
            with torch.no_grad():
                samples = vae.decoder(z).mean
            save_image(samples.view(-1, 1, 28, 28), args.samples, normalize=True, value_range=(-1, 1))
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented")

    

#python Mini_Project_1/mains.py train --network unet --epochs 40 --batch-size 64 --lr 2e-4 --device cuda
#python Mini_Project_1/mains.py train --network VAE --epochs 50 --batch-size 32  --device cuda
#python Mini_Project_1/mains.py train --network DDPM_VAE --epochs 40 --batch-size 64 --lr 2e-4 --device cuda