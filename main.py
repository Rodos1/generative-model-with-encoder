import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from optuna_scripts.d2nn_utilities import configure_dnn

latent_dim = 100
n = 64
batch_size = 1000
epochs_num = 15
lr = 0.001
num_classes = 10

class ConditionalPhaseEncoder(nn.Module):
    def __init__(self, latent_dim, n, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, n * n)
        )

    def forward(self, noise, class_label):
        input_data = torch.cat((noise, class_label), dim=1)
        phase_map = self.fc(input_data)
        phase_map = torch.clamp(phase_map, -5, 5)
        phase_map = torch.tanh(phase_map).view(-1, 1, n, n)

        return torch.clamp((phase_map + 1) * 3.14159, 0, 2 * 3.14159)

model = configure_dnn(n=n, pixels=n, length=0.001, wavelength=500E-9, masks_amount=3, distance=0.08628599497985633, detectors_norm='none')

encoder = ConditionalPhaseEncoder(latent_dim, n, num_classes)

#Обучение
transform = transforms.Compose([transforms.Resize((n, n)), transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=lr, weight_decay=1e-5)
eps = 1e-8

for epoch in range(epochs_num):
    total_loss = 0.0
    for images, labels in dataloader:
        images = images

        #Генерируем шум и классы
        noise = torch.clamp(torch.randn(images.shape[0], latent_dim), -2, 2)  # Ограничиваем шум

        class_labels = torch.nn.functional.one_hot(labels, num_classes).float()

        #Кодируем фазу с учётом класса
        phase_maps = encoder(noise, class_labels)
        phase_maps = torch.clamp(phase_maps, 0, 2 * 3.14159)  # Ограничиваем фазы

        optimizer.zero_grad()

        outputs = model.forward(phase_maps, detect=False)

        if outputs.max() == outputs.min():
            outputs = torch.zeros_like(outputs)
        else:
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + eps)

        loss = 0.8 * nn.L1Loss()(outputs, images) + 0.2 * nn.MSELoss()(outputs, images)

        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch: [{epoch+1}/{epochs_num}], Loss: {total_loss/len(dataloader):.6f}')


#Визуализация результатов
with torch.no_grad():
    batch_size = 5
    noise = torch.clamp(torch.randn(batch_size, latent_dim), -2, 2)

    target_classes = torch.tensor([5,5,5,5,5])
    class_labels = torch.nn.functional.one_hot(target_classes, num_classes).float()

    phase_maps = encoder(noise, class_labels)
    phase_maps = torch.clamp(phase_maps, 0, 2 * 3.14159)  # Ограничиваем фазы

    generated_images = model.forward(phase_maps, detect=False).cpu().squeeze()
    generated_images = generated_images.squeeze(1)

fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
for i in range(batch_size):
    axes[i].imshow(generated_images[i], cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"Class: {target_classes[i].item()}")
plt.show()
