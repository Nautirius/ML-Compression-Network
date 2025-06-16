import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader
from models.models import MODELS
from tests import save_reconstruction_plot
from .CompressionMethod import CompressionMethod


def compress_and_save(
    input_path: str,
    model: CompressionMethod,
    output_path: Optional[str] = None
):
    print(f"[compress_and_save] Start – kompresuję {input_path} modelem {model}")

    model_path = Path(f"models/saved/{model.value}.pth")
    if not model_path.exists():
        msg = "[compress_and_save] Brak wytrenowanego modelu – wywołaj najpierw trening!"
        print(msg)
        raise RuntimeError(msg)

    # 1) Załaduj model
    net = MODELS[model]()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    # 2) Przygotuj urządzenie
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"[compress_and_save] Używane urządzenie: {device}")

    # 3) Wczytaj dane i loader
    df = load_and_preprocess_data(input_path)
    data_loader = pandas_to_loader(df, shuffle=False)
    print(f"[compress_and_save] DataLoader ma {len(data_loader)} batchy")

    # 4) Kompresuj każdą próbkę
    compressed_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(data_loader):
            x = x.to(device)
            z = net.compress(x)        # (B, latent_dim)
            z = z.cpu().numpy()
            compressed_rows.append(z)

    # 5) Zapis
    compressed_mat = np.vstack(compressed_rows)      # (N, latent_dim)
    df_compressed = pd.DataFrame(compressed_mat)


    if output_path is None:
        output_path = Path(input_path).with_suffix(f"{model.extension}.csv")
    else:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(f"{model.extension}.csv")

    df_compressed.to_csv(output_path, index=False, header=False, float_format="%.18e",)
    print(f"[compress] Wektor latentny zapisany w {output_path.resolve()}")



def decompress_and_save(input_path: str, output_path: Optional[str]):
    extension = CompressionMethod.from_extension(input_path)
    print(f"Dekompresuję {input_path} output: {output_path} {extension}")

    model_path = Path(f'models/saved/{extension.value}.pth')
    if not model_path.exists():
        raise RuntimeError(f"Model dla {extension.value} nie jest wytrenowany!")

    net = MODELS[extension]()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    df = pd.read_csv(
        input_path,
        header=None,
    )
    data_loader = pandas_to_loader(df.values, shuffle=False)
    decompressed_rows = []
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(data_loader):
            x = x.to(device)
            decompressed = net.decompress(x)
            decompressed_rows.append(decompressed.cpu().numpy())

    decompressed_mat = np.vstack(decompressed_rows)

    if output_path is None:
        output_path = Path(input_path).with_suffix("").with_suffix(".decompressed.csv")
    output_path = Path(output_path)

    pd.DataFrame(decompressed_mat).to_csv(
        output_path,
        index=False,
        header=False,
        float_format="%.18e",
    )
    print(f"Zapisano zdekompresowane dane w {output_path}")


def train_and_save_autoencoder(train_data_dir: str, model: CompressionMethod, epochs=10, lr=5e-4):
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
    compressed_file = str(Path(input_path).with_suffix(f"{model.extension}.csv"))
    decompressed_file = str(Path(input_path).with_suffix(f".decompressed.csv"))

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

    for i in range(min(10, original.shape[0])):
        save_reconstruction_plot(
            str(Path(input_path).with_suffix(f".{i}.png")),
            original[i],
            reconstructed[i]
        )

    print(compressed_file)
    print(decompressed_file)

    mse = mean_squared_error(original, reconstructed)

    print(f"Rozmiar oryginału: {original_size} bytes")
    print(f"Rozmiar skompresowany: {compressed_size} bytes")
    print(f"Współczynnik kompresji: {compression_ratio}")
    print(f"Błąd średniokwadratowy (MSE): {mse}")
