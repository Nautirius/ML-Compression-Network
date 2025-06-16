import json
import math
import os
import statistics as stats
import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader
from models.Conv1d_Generic_Autoencoder import Conv1d_Generic_Autoencoder
from models.MLP_Generic_Autoencoder import MLP_Generic_Autoencoder


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

    # -------------------- plot curves ---------------------------
    plt.figure()
    ep_range = range(1, len(train_losses) + 1)
    plt.plot(ep_range, train_losses, marker="o", label="train")
    plt.plot(ep_range, test_losses, marker="s", label="test")
    plt.title("MSE per epoch – train vs test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "loss.png"))
    plt.close()

    return pure_train_time


###############################################################################
# Evaluation helper                                                           #
###############################################################################

def evaluate_autoencoder(
    net: Union["MLP_Generic_Autoencoder", "Conv1d_Generic_Autoencoder"],
    test_loader: torch.utils.data.DataLoader,
    *,
    training_time_s: Optional[float] = None,
    root_out: str = "tests",
    save_json: bool = True,
    save_plot: bool = True,                   # <── NOWA FLAGA
) -> dict[str, float]:
    """
    Oblicza metryki rekonstrukcji i – opcjonalnie – zapisuje:
      • metrics.json   (jak dotychczas)
      • reconstruction.png  – wykres 1-szej serii oryginalnej vs zrekonstruowanej
    """

    device = next(net.parameters()).device
    net.eval()

    # ---------- pass 1: średnia zbioru (do R²) ----------
    total_elems, mean_acc = 0, 0.0
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            total_elems += x.numel()
            mean_acc += x.sum().item()
    dataset_mean = mean_acc / total_elems

    # ---------- pass 2: metryki + chwytamy pierwszą próbkę ----------
    sq_error_sum = abs_error_sum = sq_total_sum = 0.0
    comp_times, decomp_times = [], []
    first_orig, first_recon = None, None      # ← do wykresu

    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)

            # pomiar – kompresja
            t0 = time.perf_counter()
            z = net.compress(x)
            comp_times.append(time.perf_counter() - t0)

            # pomiar – dekompresja
            t0 = time.perf_counter()
            recon = net.decompress(z)
            decomp_times.append(time.perf_counter() - t0)

            # --------------- konwersja typów do obliczeń ---------------
            recon_t = torch.to(device=device, dtype=x.dtype)

            sq_error_sum += ((recon_t - x) ** 2).sum().item()
            abs_error_sum += (recon_t - x).abs().sum().item()
            sq_total_sum += ((x - dataset_mean) ** 2).sum().item()

            # zapamiętujemy pierwszą serię do wykresu
            if first_orig is None:
                # x może mieć kształt (C, L) lub (L) – spłaszczamy do ( L )
                first_orig = x[0].detach().cpu().numpy().reshape(-1)
                first_recon = recon[0].reshape(-1)

    # ------------------------- metryki -------------------------
    mse = sq_error_sum / total_elems
    mae = abs_error_sum / total_elems
    r2  = 1.0 - sq_error_sum / sq_total_sum if sq_total_sum else float("nan")
    psnr = 20 * math.log10(1.0) - 10 * math.log10(mse) if mse > 0 else float("inf")

    # CR działa dla MLP – jeśli sieć ma inną strukturę, dostosuj
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
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(first_orig, label="oryginał")
        ax.plot(first_recon, label="rekonstrukcja")
        ax.set_title("Rekonstrukcja – pierwsza próbka")
        ax.legend()
        ax.set_xlabel("indeks próbki")
        ax.set_ylabel("wartość")
        fig.tight_layout()
        fig.savefig(os.path.join(model_dir, "reconstruction.png"), dpi=150)
        plt.close(fig)

    # ------------------- ładny print w konsoli ------------------
    print("\n----- Evaluation metrics -----")
    for k, v in metrics.items():
        print(f"{k:18s}: {v: .6f}")

    return metrics


def run():
    models_to_train = [
        MLP_Generic_Autoencoder(layer_dims=[187, 80, 32]),
        Conv1d_Generic_Autoencoder(latent_dim=32, conv_channels=[64, 128]),
        MLP_Generic_Autoencoder(layer_dims=[187, 80, 16]),
        Conv1d_Generic_Autoencoder(latent_dim=16, conv_channels=[64, 128]),
        MLP_Generic_Autoencoder(layer_dims=[187, 80, 32, 8]),
        Conv1d_Generic_Autoencoder(latent_dim=64, conv_channels=[128]),
        MLP_Generic_Autoencoder(layer_dims=[187, 64]),
        Conv1d_Generic_Autoencoder(latent_dim=32, conv_channels=[80]),
        MLP_Generic_Autoencoder(layer_dims=[187, 64, 8]),
        Conv1d_Generic_Autoencoder(latent_dim=8, conv_channels=[32, 80]),
    ]

    df_train = load_and_preprocess_data('./data/mitbih_train.csv')
    loader_train = pandas_to_loader(df_train)

    df_test = load_and_preprocess_data('./data/mitbih_test.csv')
    loader_test = pandas_to_loader(df_test)

    for model in models_to_train:
        training_time = train_autoencoder(model, loader_train, loader_test, epochs=20, lr=2e-4)
        evaluate_autoencoder(model, loader_test, training_time_s=training_time)


if __name__ == '__main__':
    run()
