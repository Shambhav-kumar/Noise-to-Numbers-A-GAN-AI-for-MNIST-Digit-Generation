import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparametersa
latent_dim = 100
img_shape = (1, 28, 28)
batch_size = 64
lr = 0.0002
epochs = 200

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.reshape(img.size(0), *img_shape)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Configure data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mnist_dataset = datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    mnist_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=3  # Adjust this number if you face any issues with multiprocessing
)

# Wrap training in the "if __name__ == '__main__':" block for Windows compatibility
if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()  # Required for Windows compatibility

    # Training loop
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            
            # Configure input
            real_imgs = imgs.to(device)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(imgs.size(0), latent_dim).to(device)
            
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # Print progress
            if i % 400 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )
        
        # Save generated images every few epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_z = torch.randn(16, latent_dim).to(device)
                generated = generator(sample_z).cpu()
                
                fig, axs = plt.subplots(4, 4, figsize=(4, 4))
                cnt = 0
                for i in range(4):
                    for j in range(4):
                        axs[i,j].imshow(generated[cnt, 0, :, :], cmap='gray')
                        axs[i,j].axis('off')
                        cnt += 1
                plt.savefig(f"mnist_epoch_{epoch}.png")
                plt.close()
