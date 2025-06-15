import os
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from sklearn.metrics import mean_squared_error

from models.models import MODELS
from .CompressionMethod import CompressionMethod
from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader


def compress_and_save(input_path: str, model: CompressionMethod, output_path: Optional[str]):
    print(f"Kompresuję {input_path} modelem: {model}, {output_path}{model.extension}")
    model_path = Path(f'models/saved/{model.value}.pth')
    if not model_path.exists():
        print("Brak wytrenowanego modelu. Trening...")
        raise RuntimeError("Model nie jest wytrenowany, najpierw wywołaj trening!")

    net = MODELS[model]()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    df = load_and_preprocess_data(input_path)
    data_loader = pandas_to_loader(df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    compressed_data = []
    with torch.no_grad():
        for (x,) in data_loader:
            x = x.to(device)
            z = net.compress(x)
            compressed_data.append(z.cpu())

    output_path = Path(output_path) if output_path else Path(input_path).with_suffix(model.extension)
    torch.save(torch.cat(compressed_data), output_path)
    print(f"Zapisano skompresowane dane w {output_path}")


def decompress_and_save(input_path: str, output_path: Optional[str]):
    extension = CompressionMethod.from_extension(input_path)
    print(f"Dekompresuję {input_path} output: {output_path} {extension}")

    model_path = Path(f'models/saved/{extension.value}.pth')
    if not model_path.exists():
        raise RuntimeError(f"Model dla {extension.value} nie jest wytrenowany!")

    net = MODELS[extension]()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    compressed_data = torch.load(input_path).to(device)

    with torch.no_grad():
        decompressed = net.decompress(compressed_data)

    df = decompressed.cpu().numpy()
    output_path = Path(output_path) if output_path else Path(input_path).with_suffix('.csv')

    pd.DataFrame(df).to_csv(output_path, index=False, header=False, float_format='%.18e')

    print(f"Zapisano zdekompresowane dane w {output_path}")


def train_and_save_autoencoder(train_data_dir: str, model: CompressionMethod, epochs=10, lr=1e-3):
    net = MODELS[model]()
    df = load_and_preprocess_data(train_data_dir)
    data_loader = pandas_to_loader(df)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for (x,) in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = net(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        print(f'Epoch {epoch + 1}, Loss {epoch_loss}')

    os.makedirs('models/saved', exist_ok=True)
    path = f'models/saved/{model}.pth'
    torch.save(net.state_dict(), path)
    print(f"Model zapisany w {path}")


def test_model(input_path: str, model: CompressionMethod):
    compressed_file = input_path + model.extension
    decompressed_file = input_path + ".decompressed"

    # Kompresja
    compress_and_save(input_path, model, compressed_file)

    # Dekompresja
    decompress_and_save(compressed_file, decompressed_file)

    # Pomiar rozmiaru
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(compressed_file)
    # compression_ratio = compressed_size / original_size
    compression_ratio = original_size / compressed_size

    # Wczytywanie danych CSV
    original = load_and_preprocess_data(input_path)
    reconstructed = pd.read_csv(decompressed_file, delimiter=",", header=None).values

    print(compressed_file)
    print(decompressed_file)

    # print(original)
    # print(reconstructed)

    mse = mean_squared_error(original, reconstructed)

    print(f"Rozmiar oryginału: {original_size} bytes")
    print(f"Rozmiar skompresowany: {compressed_size} bytes")
    print(f"Współczynnik kompresji: {compression_ratio}")
    print(f"Błąd średniokwadratowy (MSE): {mse}")
