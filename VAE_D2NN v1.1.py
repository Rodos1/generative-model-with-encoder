import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

sys.path.append('/kaggle/input/lskdfjb/OpticalEncoder-cluster')
!pip install minio belashovplot

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from optuna_scripts.d2nn_utilities import configure_dnn
import torch.nn.functional as F


#--------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device "{device}"')

batch_size = 100
epochs = 50
lr = 1e-3
n = 160
latent_dim = 20
m = int((n * n / 10) ** 0.5)
d2nn = True

print('Обучается D2NN') if d2nn else print('Обучается сверточный декодер')


#------------------------------------------------------------------------------------


#Преобразование MNIST в mxm, затем добавление нулевых рамок до nxn
def mnist_transform(img):
    img = transforms.Resize((m, m))(img)
    img = transforms.ToTensor()(img)
    padded = torch.zeros(1, n, n)
    pad_start = (n - m) // 2
    padded[:, pad_start:pad_start + m, pad_start:pad_start + m] = img  #Вставляем в центр
    return padded

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------------------

class VAE_D2NN(nn.Module):
    def __init__(self, latent_dim, n, m, d2nn=True):
        super(VAE_D2NN, self).__init__()

        self.d2nn = d2nn
        self.n = n
        self.m = m
        #классический полносвязный энкодер
        self.encoder = nn.Sequential(
            nn.Linear(n * n, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)

        # self.latent_to_image = nn.Sequential(
        #     nn.Linear(latent_dim, n * n),
        #     nn.ReLU()
        # )

        #декодер D2NN
        self.decoder_D2NN = configure_dnn(
            n=n, pixels=n, length=n*500E-9, wavelength=500E-9,
            masks_amount=3, distance=0.08628599497985633,
            detectors_norm='none'
        ).to(device)

        #свёрточный декодер (в качестве альтернативы D2NN)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(device)

    def crop_center(self, x):
        pad_start = (self.n - self.m) // 2
        return x[:, :, pad_start:pad_start + self.m, pad_start:pad_start + self.m]

    def encode(self, x):
        x = x.view(x.size(0), -1)  #Разворачиваем в вектор размерности (batch_size, n*n)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_D2NN(self, z):
        batch_size = z.size(0)
    
        p = int(z.size(1) ** 0.5)
        q = z.size(1) // p

        if p * q != z.size(1):
            raise ValueError("Latent dim must be a square or rectangular area.")
    
        pad_h = (self.n - p) // 2
        pad_w = (self.n - q) // 2
            
        padding = (pad_w, self.n - pad_w - q, pad_h, self.n - pad_h - p)
        z = z.view(batch_size, 1, p, q)
        padded = F.pad(z, padding, "constant", 0)

        out = self.decoder_D2NN(padded, detect=False)
    
        min_val = out.amin(dim=(1, 2, 3), keepdim=True)
        max_val = out.amax(dim=(1, 2, 3), keepdim=True)
        normalized_out = (out - min_val) / (max_val - min_val + 1e-6)
    
        return self.crop_center(normalized_out)

    def decode_conv(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # подгонка размерности под свертки
        x_recon = self.decoder_conv(z)
        return x_recon

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if self.d2nn:
            return self.decode_D2NN(z), mu, logvar
        else:
            return self.decode_conv(z), mu, logvar

#--------------------------------------------------------------------------------------------

model = VAE_D2NN(latent_dim, n, m, d2nn=d2nn).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

#-------------------------------------------------------------------------------------------

def pearson_loss(img1, img2):
    img1_mean = torch.mean(img1, dim=[1, 2, 3], keepdim=True)
    img2_mean = torch.mean(img2, dim=[1, 2, 3], keepdim=True)

    img1_centered = img1 - img1_mean
    img2_centered = img2 - img2_mean

    a = torch.sum(img1_centered * img2_centered, dim=[1, 2, 3])
    b = torch.sqrt(torch.sum(img1_centered**2, dim=[1, 2, 3]) * torch.sum(img2_centered**2, dim=[1, 2, 3]))

    correlation = a / (b + 1e-8)
    loss = 1 - correlation.mean()
    return loss

#------------------------------------------------------------------------------------------------

def loss_function(recon_x, x, mu, logvar):
    if recon_x.min() < 0. or recon_x.max() > 1.:
        print(f"Min and Max of recon_x: {recon_x.min()}, {recon_x.max()}")
    
    x_cropped = x[:, :, (n - m) // 2: (n + m) // 2, (n - m) // 2: (n + m) // 2]
    
    if x_cropped.min() < 0. or x_cropped.max() > 1.:
        print(f"Min and Max of x_cropped: {x_cropped.min()}, {x_cropped.max()}")
    
    BCE = nn.functional.binary_cross_entropy(recon_x, x_cropped, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    BCE = BCE / recon_x.size(0)
    KLD = KLD / recon_x.size(0)
    
    loss = BCE + KLD
    return loss

#---------------------------------------------------------------------------------------------------
#Обучение

model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss: {loss}")
            break
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    with torch.no_grad():
        plt.imshow(recon_batch[0].cpu().squeeze())
        plt.show()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')
    print(torch.cuda.memory_allocated())
    torch.cuda.empty_cache()

#------------------------------------------------------------------------------------------------
#Визуализация

model.eval()
with torch.no_grad():
    z = torch.randn(144, latent_dim).to(device)
    sample = model.decode_D2NN(z).cpu() if model.d2nn else model.decode_conv(z).cpu()
    fig, axes = plt.subplots(12, 12, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sample[i].squeeze(), cmap='gray')  #Теперь размер m x m
        ax.axis('off')
    plt.show()
