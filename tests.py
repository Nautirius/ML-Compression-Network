import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import json
import math
import statistics as stats
import matplotlib.pyplot as plt
from typing import Optional

import models.Conv1d_Strided_Autoencoder_V2
from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader
from models.MLP_Generic_Dropout_Norm import MLP_Generic_Dropout_Norm
from models.MLP_Generic_Autoencoder import MLP_Generic_Autoencoder
from models.Conv1d_Generic_Autoencoder import Conv1d_Generic_Autoencoder
from models.Conv1d_Strided_Autoencoder import Conv1d_Strided_Autoencoder


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
        net: "MLP_Generic_Autoencoder",
        test_loader: torch.utils.data.DataLoader,
        *,
        training_time_s: Optional[float] = None,
        root_out: str = "tests",
        save_json: bool = True,
) -> dict[str, float]:
    """Compute a rich set of metrics on *net* using *test_loader*.

    If *training_time_s* (in seconds) is provided, it will be added to the
    output metrics.

    Metrics returned (and optionally saved to JSON):

    * **MSE**, **MAE**, **R2**, **PSNR_dB**
    * **compression_ratio**
    * **compress_time_ms**, **decompress_time_ms**
    * **MSE_per_CR**, **R2_per_CR**
    * **training_time_s** (jeśli przekazano)
    """

    device = next(net.parameters()).device
    net.eval()

    # ---------- pass 1: compute dataset mean (needed for R2) ----------
    total_elems = 0
    mean_acc = 0.0
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            total_elems += x.numel()
            mean_acc += x.sum().item()
    dataset_mean = mean_acc / total_elems

    # ---------- pass 2: metrics, timing ----------
    sq_error_sum = 0.0
    abs_error_sum = 0.0
    sq_total_sum = 0.0
    comp_times: list[float] = []
    decomp_times: list[float] = []

    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)

            # timing compression
            t0 = time.perf_counter()
            z = net.compress(x)
            comp_times.append(time.perf_counter() - t0)

            # timing decompression
            t0 = time.perf_counter()
            recon = net.decompress(z)
            decomp_times.append(time.perf_counter() - t0)

            sq_error_sum += ((recon - x) ** 2).sum().item()
            abs_error_sum += (recon - x).abs().sum().item()
            sq_total_sum += ((x - dataset_mean) ** 2).sum().item()

    mse = sq_error_sum / total_elems
    r2 = 1.0 - sq_error_sum / sq_total_sum if sq_total_sum != 0 else float("nan")
    mae = abs_error_sum / total_elems

    # Compression ratio based on architecture dims ----------------------
    compression_ratio = net.layer_dims[0] / net.layer_dims[-1]

    # Timing (ms averaged per batch) ------------------------------------
    compress_time_ms = stats.mean(comp_times) * 1000 if comp_times else 0.0
    decompress_time_ms = stats.mean(decomp_times) * 1000 if decomp_times else 0.0

    # Derived metrics ----------------------------------------------------
    mse_per_cr = mse / compression_ratio
    r2_per_cr = r2 / compression_ratio if compression_ratio != 0 else float("nan")

    # Extra metric: PSNR --------------------------------------------------
    max_val = 1.0  # assume normalized inputs; adjust if needed
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse) if mse > 0 else float("inf")

    metrics: dict[str, float] = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "PSNR_dB": psnr,
        "compression_ratio": compression_ratio,
        "compress_time_ms": compress_time_ms,
        "decompress_time_ms": decompress_time_ms,
        "MSE_per_CR": mse_per_cr,
        "R2_per_CR": r2_per_cr,
    }

    if training_time_s is not None:
        metrics["training_time_s"] = training_time_s

    # Save to JSON -------------------------------------------------------
    if save_json:
        os.makedirs(root_out, exist_ok=True)
        model_dir = os.path.join(root_out, str(net))
        os.makedirs(model_dir, exist_ok=True)
        json_path = os.path.join(model_dir, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)

    # Pretty print -------------------------------------------------------
    print("\n----- Evaluation metrics -----")
    for k, v in metrics.items():
        print(f"{k:18s}: {v: .6f}")

    return metrics


def run():
    models_to_train = [
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, code_dim=8),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=7, stride=1),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=7, stride=1, code_dim=8),
        Conv1d_Strided_Autoencoder(activation="relu", kernel_size=3, stride=1),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, conv_channels = (32, 64, 128, 256, 512), code_dim=16),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, conv_channels=(16, 32, 64, 128, 256, 512), code_dim=8),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, conv_channels=[64]),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, code_dim=8, conv_channels=[64]),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, conv_channels=[128, 64]),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, code_dim=8, conv_channels=[128, 64]),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, conv_channels=[512, 256, 128, 64, 32]),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=1, conv_channels=[512, 256, 128, 64, 32, 16], code_dim=8),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=2),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=7, stride=2),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=2, code_dim=8),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=7, stride=2, code_dim=8),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=3),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=7, stride=3),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=3, stride=3, code_dim=8),
        Conv1d_Strided_Autoencoder(activation="gelu", kernel_size=7, stride=3, code_dim=8),

        # Conv1d_Generic_Autoencoder_Pool(
        #     input_length=187,
        #     conv_channels=[32, 64, 128],
        #     kernel_size=3,
        #     pool=2,
        #     code_dim=32,  # mocna kompresja
        #     activation='leaky_relu'
        # )
        MLP_Generic_Autoencoder([187, 180, 160, 150, 140, 128, 112, 100, 96, 80, 64, 48, 32, 28, 24, 16]),
        MLP_Generic_Autoencoder([187, 64, 16]),
        MLP_Generic_Autoencoder([187, 64, 8]),
        MLP_Generic_Autoencoder([187, 8]),
        MLP_Generic_Autoencoder([187, 16]),
        MLP_Generic_Autoencoder([187, 64, 32]),
        MLP_Generic_Autoencoder([187, 180, 160, 150, 140, 128, 112, 100, 96, 80, 64, 48, 32]),

        MLP_Generic_Autoencoder([187, 128, 64, 32, 16, 8]),
        MLP_Generic_Autoencoder([187, 128, 64, 32, 16]),
        MLP_Generic_Autoencoder([187, 512, 256, 128, 64, 32, 16, 8]),
        MLP_Generic_Autoencoder([187, 512, 256, 128, 64, 32, 16]),
        MLP_Generic_Autoencoder([187, 128, 32, 8]),

        MLP_Generic_Dropout_Norm([187, 128, 64, 32, 16, 8], dropout=0.1, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 128, 64, 32, 16], dropout=0.1, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 512, 256, 128, 64, 32, 16, 8], dropout=0.1, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 512, 256, 128, 64, 32, 16], dropout=0.1, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 128, 32, 8], dropout=0.1, batchnorm=True),

        MLP_Generic_Dropout_Norm([187, 128, 64, 32, 16, 8], dropout=0.0, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 128, 64, 32, 16], dropout=0.0, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 512, 256, 128, 64, 32, 16, 8], dropout=0.0, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 512, 256, 128, 64, 32, 16], dropout=0, batchnorm=True),
        MLP_Generic_Dropout_Norm([187, 128, 32, 8], dropout=0, batchnorm=True),
    ]

    df_train = load_and_preprocess_data('./data/mitbih_train.csv')
    loader_train = pandas_to_loader(df_train)

    df_test = load_and_preprocess_data('./data/mitbih_test.csv')
    loader_test = pandas_to_loader(df_test)

    for model in models_to_train:
        training_time = train_autoencoder(model, loader_train, loader_test, epochs=15)
        evaluate_autoencoder(model, loader_test, training_time_s=training_time)


if __name__ == '__main__':
    run()
