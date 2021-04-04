# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

def weights_init(m):
    """Apply the weights_init function to randomly initialize all weights
    to mean=0, stdev=0.2.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def main(args):
    ##################################################
    # Set random seed for reproducibility
    ##################################################
    print("Random Seed: ", args["seed"])
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    # Decide which device we want to run on
    if torch.cuda.is_available():
        print("Running on GPU")
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = torch.device("cpu")

    ##################################################
    # Create the dataset
    ##################################################
    dataset = dset.MNIST(
        root=args["data_path"],
        train=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        download=True
    )

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args["batch_size"],
        shuffle=True, 
    )

    # Initialize models
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # # Print the model
    # print("Generator architecture: \n")
    # print(netG)
    # print("Discriminator architecture: \n")
    # print(netD)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args["epochs"]):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            
            # Move data to device
            X_mb = data[0].to(device)
            
            # Forward pass real batch through D
            Y_real = netD(X_mb).view(-1)
            # Calculate loss on all-real batch
            D_real_loss = criterion(
                Y_real, 
                torch.ones_like(Y_real, dtype=torch.float)
            )

            # Sample fake data
            Z_mb = torch.randn_like(
                X_mb,
                dtype=torch.float
            )
            
            # Generate fake image batch with G
            X_hat_mb = netG(Z_mb)
            # Classify all fake batch with D
            Y_fake = netD(X_hat_mb.detach()).view(-1)
            
            # Calculate D's loss on the all-fake batch
            D_fake_loss = criterion(
                Y_fake, 
                torch.zeros_like(Y_real, dtype=torch.float)
            )

            # Add the gradients from the all-real and all-fake batches
            D_loss = D_real_loss + D_fake_loss
            # Backward pass
            D_loss.backward()
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            # Since we just updated D, perform another forward pass of all-fake batch through D
            Y_fake = netD(X_hat_mb).view(-1)
            # Calculate G's loss based on this output
            G_loss = criterion(
                Y_fake, 
                torch.ones_like(Y_fake, dtype=torch.float)
            )
            # Calculate gradients for G
            G_loss.backward()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print(f'''
                    Epoch: {epoch}
                    Loss_D: {G_loss.item()}\tLoss_G: {D_loss.item()}
                ''')
            # Save Losses for plotting later
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())

if __name__ == "__main__":
    args = {
        "data_path": "data/cifar10",
        "seed": 42
        "epochs": 5,
        "batch_size": 128,
        "image_size": 64,
        "n_channels": 3,
        "z_dim": 100,
        "h_dim": 64,
        "lr": 0.0002,
        "beta1": 0.5,
    }

    main(args)
