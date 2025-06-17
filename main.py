import click
from pathlib import Path
from typing import Optional

from compression.CompressionMethod import CompressionMethod
from compression.compression import train_and_save_autoencoder, test_model
from compression.compression import compress_and_save, decompress_and_save

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
    """Kompresuje plik CSV przy użyciu wskazanego modelu."""
    compress_and_save(filepath, CompressionMethod(model.lower()), output)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '-o', '--output',
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help='Ścieżka pliku wyjściowego (opcjonalna)'
)
def decompress_cmd(filepath: str, output: Optional[str]):
    """Dekompresuje plik CSV."""
    if not any(filepath.endswith(f"{ext}.csv") for ext in ALLOWED_EXTENSIONS):
        raise click.ClickException(f"Plik musi mieć jedno z rozszerzeń: {', '.join(ALLOWED_EXTENSIONS)}")

    decompress_and_save(filepath, output)


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
    """Trenuje dany Autoencoder na zbiorze danych i zapisuje go do pliku."""
    if model.lower().startswith('mlp'):
        epochs = 12
    else:
        epochs = 6
    train_and_save_autoencoder(train_data_dir, model, epochs=epochs)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True, dir_okay=False))
@click.option('--model',
              type=click.Choice([e.value for e in CompressionMethod], case_sensitive=False),
              required=True,
              help='Metoda użyta do kompresji'
              )
def test_cmd(filepath: str, model: CompressionMethod):
    """Kompresuje plik CSV, dekompresuje i wypisuje statystyki dla wskazanego modelu."""
    model = CompressionMethod(model.lower())
    test_model(filepath, model)


cli.add_command(compress_cmd, name='compress')
cli.add_command(decompress_cmd, name='decompress')
cli.add_command(test_cmd, name="test")

if __name__ == '__main__':
    cli()
