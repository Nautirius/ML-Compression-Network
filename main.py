import os
import click
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import torch

from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader
from models.models import MODELS
from models.models import train_autoencoder
from compression.CompressionMethod import CompressionMethod
from compression.compression import compress, decompress

ALLOWED_EXTENSIONS = [m.extension for m in CompressionMethod]


@click.group()
def cli():
    pass


@cli.command()
@click.argument('filepath', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--model',
    type=click.Choice([e.value for e in CompressionMethod], case_sensitive=False),
    required=True,
    help='Metoda kompresji'
)
@click.option(
    '-o', '--output',
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help='Ścieżka pliku wyjściowego (opcjonalna)'
)
def compress_cmd(filepath: str, model: str, output: Optional[str] = None):
    compress(filepath, CompressionMethod(model.lower()), output)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '-o', '--output',
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help='Ścieżka pliku wyjściowego (opcjonalna)'
)
def decompress_cmd(filepath: str, output: Optional[str]):
    if not any(filepath.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise click.ClickException(f"Plik musi mieć jedno z rozszerzeń: {', '.join(ALLOWED_EXTENSIONS)}")

    decompress(filepath, output)


@cli.command()
@click.option('--model',
              type=click.Choice([e.value for e in CompressionMethod], case_sensitive=False),
              required=True,
              help='Metoda użyta do trenowania'
              )
@click.argument('train_data_dir',
                type=click.Path(exists=True, file_okay=True),
                )
def train(model: CompressionMethod, train_data_dir: str):
    """Trenuje dany Autoencoder na zbiorze danych."""
    net = MODELS[model]()
    df = load_and_preprocess_data(train_data_dir)
    data_loader = pandas_to_loader(df)

    train_autoencoder(net, data_loader, epochs=10)

    os.makedirs('models/saved', exist_ok=True)
    path = f'models/saved/{model}.pth'
    torch.save(net.state_dict(), path)
    print(f"Model zapisany w {path}")


@cli.command()
@click.argument('filepath', type=click.Path(exists=True, dir_okay=False))
@click.option('--model',
      type=click.Choice([e.value for e in CompressionMethod], case_sensitive=False),
      required=True,
      help='Metoda użyta do kompresji'
)
def stats_cmd(filepath: str, model: str):
    """Kompresuje plik CSV, dekompresuje i wypisuje statystyki."""
    model = CompressionMethod(model.lower())

    compressed_file = filepath + model.extension
    decompressed_file = filepath + ".decompressed"

    # Kompresja
    compress(filepath, model, compressed_file)

    # Dekompresja
    decompress(compressed_file, decompressed_file)

    # Pomiar rozmiaru
    original_size = os.path.getsize(filepath)
    compressed_size = os.path.getsize(compressed_file)
    compression_ratio = compressed_size / original_size

    # Wczytywanie danych CSV
    original = load_and_preprocess_data(filepath)
    reconstructed = pd.read_csv(decompressed_file, delimiter=",", header=None).values

    print(compressed_file)
    print(decompressed_file)

    print(original)
    print(reconstructed)

    mse = mean_squared_error(original, reconstructed)

    click.echo(f"Rozmiar oryginału: {original_size} bytes")
    click.echo(f"Rozmiar skompresowany: {compressed_size} bytes")
    click.echo(f"Stopień kompresji: {compression_ratio*100:.2f}%")
    click.echo(f"Błąd średniokwadratowy (MSE): {mse}")


cli.add_command(compress_cmd, name='compress')
cli.add_command(decompress_cmd, name='decompress')
cli.add_command(stats_cmd, name="stats")

if __name__ == '__main__':
    cli()
