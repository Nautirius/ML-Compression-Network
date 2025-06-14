from typing import Optional
from pathlib import Path
import pandas as pd

import torch

from data_utils.data_utils import load_and_preprocess_data, pandas_to_loader
from models.models import MODELS
from .CompressionMethod import CompressionMethod


def compress(input_path: str, model: CompressionMethod, output_path: Optional[str]):
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


def decompress(input_path: str, output_path: Optional[str]):
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
