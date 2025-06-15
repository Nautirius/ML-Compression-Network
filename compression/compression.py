import os
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from models.models import MODELS
from .CompressionMethod import CompressionMethod
from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader


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
    data_loader = pandas_to_loader(df)
    print(f"[compress_and_save] DataLoader ma {len(data_loader)} batchy")

    # 4) Kompresuj każdą próbkę
    compressed_rows = []
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(data_loader):
            x = x.to(device)
            z = net.compress(x)            # ← 1-wymiarowy np.ndarray
            if not isinstance(z, np.ndarray):
                z = z.cpu().numpy()        # awaryjnie zamień Tensor → NumPy
            compressed_rows.append(z)
            print(z.shape)
            print(
                f"[compress_and_save] Batch {batch_idx + 1}/{len(data_loader)} "
                f"skompresowany (latent dim = {z.shape[0]})"
            )

    # 5) Zamień listę na macierz i zapisz do CSV
    compressed_array = np.vstack(compressed_rows)  # (N, latent_dim)
    df_compressed = pd.DataFrame(compressed_array)

    # 6) Ustal ścieżkę wyjściową
    if output_path is None:
        output_path = Path(input_path).with_suffix(f"{model.extension}.csv")
    else:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(f"{model.extension}.csv")

    df_compressed.to_csv(output_path, index=False, header=False)
    print(f"[compress_and_save] Zapisano skompresowane dane w {output_path.resolve()}")


def decompress_and_save(input_path: str, output_path: Optional[str]):
    extension = CompressionMethod.from_extension(input_path)
    print(f"Dekompresuję {input_path} output: {output_path} {extension}")

    model_path = Path(f'models/saved/{extension.value}.pth')
    if not model_path.exists():
        raise RuntimeError(f"Model dla {extension.value} nie jest wytrenowany!")

    net = MODELS[extension]()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    compressed_df = pd.read_csv(input_path, header=None)
    compressed_np = compressed_df.to_numpy(dtype=np.float32)  # shape: (N, latent_dim)

    # --- dekompresja (cała macierz naraz) ---
    with torch.no_grad():
        decompressed_np = net.decompress(compressed_np)       # shape: (N, input_dim)

    # --- zapis ---
    if output_path is None:
        output_path = Path(input_path).with_suffix("").with_suffix(".decompressed.csv")
    output_path = Path(output_path)

    pd.DataFrame(decompressed_np).to_csv(
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
