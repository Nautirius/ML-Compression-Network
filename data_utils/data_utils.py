import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer
from torch.utils.data import TensorDataset, DataLoader


def load_and_preprocess_data(input_path: str, cleaning_strategy: str = "knn") -> np.ndarray:
    """Loads and preprocesses time series data from a CSV file."""
    df = pd.read_csv(input_path, header=None)
    X = df.iloc[:, :-1].copy()  # 187 punktów sygnału
    y = df.iloc[:, -1]  # signal class
    X_clean = _clean_missing_values(X, strategy=cleaning_strategy)

    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(X_clean)

    return X_clean.to_numpy()


def pandas_to_loader(x: np.ndarray, shuffle: bool=True) -> DataLoader:
    X_tensor = torch.tensor(x, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=shuffle)
    return loader


def _clean_missing_values(df: pd.DataFrame, strategy: str = "mean", k: int = 5) -> pd.DataFrame:
    """
    Czyści NaN-y w dataframe.

    strategy:
      - "drop": usuwa wiersze z NaN
      - "mean": zastępuje średnią
      - "knn": używa KNNImputer z sklearn
    """
    if strategy == "drop":
        df_cleaned = df.dropna()
    elif strategy == "mean":
        df_cleaned = df.fillna(df.mean())
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=k)
        df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        raise ValueError(f"Nieznana strategia: {strategy}")

    # Dla pewności – upewnij się, że wszystko jest OK
    assert not df_cleaned.isnull().any().any(), "Wciąż są NaNy po czyszczeniu!"
    return df_cleaned
