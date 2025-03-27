import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from optuna_scripts.d2nn_utilities import configure_dnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device "{device}"')

batch_size = 100
latent_dim = 20
epochs = 10
lr = 1e-3
n = 28  # размерность MNIST
d2nn = False

print('Обучается D2NN') if d2nn else print('Обучается сверточный декодер')

train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class VAE_D2NN(nn.Module):
    def __init__(self, latent_dim, n, d2nn=True):
        super(VAE_D2NN, self).__init__()

        self.d2nn = d2nn

        #классический полносвязный энкодер
        self.encoder = nn.Sequential(
            nn.Linear(n * n, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)

        self.latent_to_image = nn.Sequential(
            nn.Linear(latent_dim, n * n),
            nn.ReLU()
        )

        #декодер D2NN
        self.decoder_D2NN = configure_dnn(
            n=n, pixels=n, length=0.00032, wavelength=500E-9,
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

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    #Для D2NN декодера
    def decode_D2NN(self, z):
        x = self.latent_to_image(z)
        x = x.view(-1, 1, n, n)
        out = self.decoder_D2NN(x, detect=False)
        return out

    #Для свёрточного декодера
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


model = VAE_D2NN(latent_dim, n, d2nn=d2nn).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 1, n, n), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#обучение
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, n * n).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')

#Визуализация
model.eval()
with torch.no_grad():
    z = torch.randn(100, latent_dim).to(device)
    sample = model.decode_D2NN(z).cpu() if model.d2nn else model.decode_conv(z).cpu()

    fig, axes = plt.subplots(10, 10, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(sample[i].reshape(n, n), cmap='gray')
        ax.axis('off')
    plt.show()
