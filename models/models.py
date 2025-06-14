import torch
import torch.optim as optim
import torch.nn as nn

from .Conv1D_Autoencoder import Conv1D_Autoencoder
from .MLP_Expanded_Autoencoder_V2 import MLP_Expanded_Autoencoder_V2
from .MLP_Expanded_Autoencoder import MLP_Expanded_Autoencoder
from .MLP_Simple_Autoencoder import MLP_Simple_Autoencoder
from .MLP_Generic_Autoencoder import MLP_Generic_Autoencoder
from .MLP_Generic_Dropout_Norm import MLP_Generic_Dropout_Norm

MODELS = {
    'conv1d_autoencoder': Conv1D_Autoencoder,
    'expanded_autoencoder': MLP_Expanded_Autoencoder,
    'expanded_autoencoder_2': MLP_Expanded_Autoencoder_V2,
    'simple_autoencoder': MLP_Simple_Autoencoder,
    'generic_autoencoder': MLP_Generic_Autoencoder,
    'generic_dropout_norm': MLP_Generic_Dropout_Norm
}


def train_autoencoder(net, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for (x,) in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = net(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch + 1}, Loss {epoch_loss}')
    return net

