import json
import os
import statistics as stats
import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.image import peak_signal_noise_ratio

from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader
from models.Conv1d_Generic_Autoencoder import Conv1d_Generic_Autoencoder
from models.MLP_Generic_Autoencoder import MLP_Generic_Autoencoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_autoencoder(
        net: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        *,
        epochs: int = 10,
        lr: float = 1e-4,
        root_out: str = "tests",
        expected_train_MSE: float = 1e-7
) -> float:
    """Train *net* until MSE nie wzrośnie względem poprzedniego kroku.

    - Standardowo trenuje ``epochs`` epok.
    - **Jeśli** trenowa MSE (na train‑loaderze) wzrośnie względem poprzedniej
      epoki, pętla *kontynuuje* tak długo, aż MSE znowu spadnie poniżej ostatniego
      najlepszego poziomu.  Ten mechanizm może wydłużyć trening poza
      pierwotną liczbę epok.
    - Aby uniknąć niekończącego się treningu można ustawić
      ``max_extra_epochs``.  Jeśli ```None```, brak limitu.

    Funkcja zwraca: ``model, training_time_s, train_losses, test_losses``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(root_out, exist_ok=True)
    model_dir = os.path.join(root_out, str(net))
    os.makedirs(model_dir, exist_ok=True)

    train_losses: list[float] = []
    test_losses: list[float] = []
    pure_train_time = 0.0  # seconds

    epoch = 0

    while epoch < epochs:
        epoch += 1
        # ------------------------- train -------------------------
        net.train()
        running_train = 0.0
        t0_epoch = time.perf_counter()
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), x)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        pure_train_time += time.perf_counter() - t0_epoch
        epoch_train = running_train / len(train_loader)
        train_losses.append(epoch_train)

        # ------------------------- test -------------------------
        net.eval()
        running_test = 0.0
        with torch.no_grad():
            for (x,) in test_loader:
                x = x.to(device)
                running_test += criterion(net(x), x).item()
        test_losses.append(running_test / len(test_loader))

        print(
            f"Epoch {epoch:3d} | train MSE: {epoch_train:.6f} | test MSE: {test_losses[-1]:.6f}"
        )

        if expected_train_MSE >= epoch_train:
            break

    # -------------------- wykresy ---------------------------
    plt.figure()
    ep_range = range(1, len(train_losses) + 1)
    plt.plot(ep_range, train_losses, marker="o", label="train")
    plt.plot(ep_range, test_losses, marker="s", label="test")
    plt.title("MSE per epoch – train vs test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "loss.png"))
    plt.close()

    return pure_train_time


def evaluate_autoencoder(
        net: Union["MLP_Generic_Autoencoder", "Conv1d_Generic_Autoencoder"],
        test_loader: torch.utils.data.DataLoader,
        *,
        training_time_s: Optional[float] = None,
        root_out: str = "tests",
        save_json: bool = True,
        save_plot: bool = True,
) -> dict[str, float]:
    """
    Oblicza metryki rekonstrukcji i opcjonalnie zapisuje:
      - metrics.json
      - reconstruction.png - wykres pierwszej serii oryginalnej vs zrekonstruowanej
    """

    device = next(net.parameters()).device
    net.eval()

    # ----------------------- pętla po zbiorze testowym -----------------------
    comp_times, decomp_times = [], []
    y_true, y_pred = [], []  # do obliczeń metryk
    first_orig, first_recon = None, None

    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)

            # --------------- kompresja ----------------
            t0 = time.perf_counter()
            z = net.compress(x)
            comp_times.append(time.perf_counter() - t0)

            # --------------- dekompresja --------------
            t0 = time.perf_counter()
            recon = net.decompress(z)
            decomp_times.append(time.perf_counter() - t0)

            # --------------- akumulacja do metryk -----
            y_true.append(x.detach().cpu().numpy().reshape(-1))
            y_pred.append(recon.detach().cpu().numpy().reshape(-1))

            # zapis pierwszej serii do wykresu
            if first_orig is None:
                first_orig = y_true[-1]
                first_recon = y_pred[-1]

    # ----------------------- obliczenie metryk -----------------------
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    psnr = peak_signal_noise_ratio(
        torch.tensor(y_pred), torch.tensor(y_true), data_range=1.0
    ).item()

    # CR działa dla MLP - jeśli sieć ma inną strukturę, dostosuj
    compression_ratio = net.layer_dims[0] / net.layer_dims[-1]

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "PSNR_dB": psnr,
        "compression_ratio": compression_ratio,
        "compress_time_ms": stats.mean(comp_times) * 1e3,
        "decompress_time_ms": stats.mean(decomp_times) * 1e3,
        "MSE_per_CR": mse / compression_ratio,
        "R2_per_CR": r2 / compression_ratio,
    }
    if training_time_s is not None:
        metrics["training_time_s"] = training_time_s

    # ----------------------- zapisywanie -----------------------
    model_dir = os.path.join(root_out, str(net))
    os.makedirs(model_dir, exist_ok=True)

    if save_json:
        with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)

    if save_plot and first_orig is not None:
        save_reconstruction_plot(os.path.join(model_dir, "reconstruction.png"), first_orig, first_recon)

    # ------------------- ładny print w konsoli -----------------
    print("\n----- Evaluation metrics -----")
    for k, v in metrics.items():
        print(f"{k:18s}: {v: .6f}")

    return metrics


def save_reconstruction_plot(path, original: np.ndarray, reconstruction: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(original, label="oryginał")
    ax.plot(reconstruction, label="rekonstrukcja")
    ax.set_title("Rekonstrukcja")
    ax.legend()
    ax.set_xlabel("indeks próbki")
    ax.set_ylabel("wartość")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    models_to_train: list[Union[Conv1d_Generic_Autoencoder, MLP_Generic_Autoencoder]] = [
        Conv1d_Generic_Autoencoder(),
        # MLP_Generic_Autoencoder(layer_dims=[187, 80, 32]),
        # Conv1d_Generic_Autoencoder(latent_dim=32, conv_channels=[64, 128]),
        # MLP_Generic_Autoencoder(layer_dims=[187, 80, 16]),
        # Conv1d_Generic_Autoencoder(latent_dim=16, conv_channels=[64, 128]),
        # MLP_Generic_Autoencoder(layer_dims=[187, 80, 32, 8]),
        # Conv1d_Generic_Autoencoder(latent_dim=64, conv_channels=[128]),
        # MLP_Generic_Autoencoder(layer_dims=[187, 64]),
        # Conv1d_Generic_Autoencoder(latent_dim=32, conv_channels=[80]),
        # MLP_Generic_Autoencoder(layer_dims=[187, 64, 8]),
        # Conv1d_Generic_Autoencoder(latent_dim=8, conv_channels=[32, 80]),
    ]

    df_train = load_and_preprocess_data('./data/mitbih_train.csv')
    loader_train = pandas_to_loader(df_train)

    df_test = load_and_preprocess_data('./data/mitbih_test.csv')
    loader_test = pandas_to_loader(df_test)

    for model in models_to_train:
        training_time = train_autoencoder(model, loader_train, loader_test, epochs=6, lr=5e-4)
        evaluate_autoencoder(model, loader_test, training_time_s=training_time)


if __name__ == '__main__':
    run()
