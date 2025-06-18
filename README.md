# ML-Compression-Network

**Kompresja danych z użyciem sieci neuronowych (Autoencoderów)**  
Projekt końcowy na kurs *Metody Inteligencji Obliczeniowej*

---

## Opis projektu

Projekt wykorzystuje sieci neuronowe typu autoenkoder do kompresji i rekonstrukcji danych EKG.  
Sygnał wejściowy jest przekształcany do przestrzeni latentnej (skompresowany), a następnie odtwarzany przez dekoder.  
Celem modelu jest zminimalizowanie błędu rekonstrukcji przy jednoczesnym zmniejszeniu rozmiaru danych.

Zastosowano modele:
- Typu `mlp` - gęsta sieć MLP (Multilayer Perceptron)
- Typu `conv1d` - konwolucyjny autoenkoder (1D)

---

## Autorzy


| Name                | GitHub Profile                               |
|---------------------|----------------------------------------------|
| Andrzej Świętek     | [GitHub Profile](https://github.com/Andrzej-Swietek)|
| Marcin Knapczyk     | [GitHub Profile](https://github.com/Nautirius)|
| Krzysztof Konieczny | [GitHub Profile](https://github.com/KrzysztofProgramming)|

---

## Instalacja i uruchomienie

Aby rozpocząć pracę z projektem:

1. **Utwórz środowisko wirtualne (venv):**

   ```bash
   python3 -m venv venv
   ```

2. **Aktywuj środowisko:**

   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - Windows (cmd):
     ```cmd
     venv\Scripts\activate
     ```

3. **Zainstaluj wymagane biblioteki:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Dekompresja plików z danymi testowymi i treningowymi**
   - Linux/macOS
       ```bash
       unzip ./data/mitbih_test.zip -d ./data && \
       unzip ./data/mitbih_train.zip -d ./data
       ```
---

## Dane wejściowe

Dane wejściowe to pliki CSV z sygnałem EKG zawierające sekwencje czasowe EKG (przykładowe pliki znajdują się w folderze `./data`).

Dane zostały pobrane z repozytorium `MIT-BIH ECG Heartbeat Categorization Dataset`: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

---

## Użycie programu (CLI)

Program dostarcza interfejs wiersza poleceń (CLI) z obsługą następujących komend:

```bash
python3 main.py --help
```

Dostępne komendy:
- `train` - trenuje model autoenkodera
- `compress` - kompresuje plik CSV
- `decompress` - dekompresuje plik CSV
- `test` - wykonuje pełen test: kompresja -> dekompresja -> statystyki

---

### Trening modelu

Trenuje autoenkoder na danych treningowych i zapisuje go do pliku `.pth` w katalogu `./models/saved`.

```bash
python3 main.py train data/mitbih_train.csv --model conv1d_32
```
#### Parametry:
- `--model` - nazwa modelu
#### Dostępne modele
Dostępne są dwa typy modeli, pierwszy opiera się na wielowarstowym perceptronie,
drugi na jednowymiarowej sieci konwolucyjnej
- mlp_8
- mlp_16
- mlp_32
- mlp_64
- conv1d_8
- conv1d_16
- conv1d_32
- conv1d_64

Liczba na końcu modelu oznacza do ilu liczb jest konwertowany sygnał
(im mniejsza liczba tym silniejsza kompresja kosztem jakości)


---

### Kompresja pliku

```bash
python3 main.py compress ./data/ekg1.csv --model conv1d_32
```

Parametry:
- `filepath` - ścieżka do pliku CSV
- `--model` - wybrany model do kompresji
- `--output` - opcjonalna ścieżka pliku wyjściowego

---

### Dekompresja pliku

```bash
python3 main.py decompress ./data/ekg1.conv1d_32.csv
```

Parametry:
- `filepath` - plik z rozszerzeniem `.model.csv` (np. `.conv1d_32.csv`)
- `--output` - opcjonalna ścieżka pliku wyjściowego

---

### Test modelu

Wykonuje kompresję, dekompresję oraz analizę jakości rekonstrukcji.

```bash
python3 main.py test ./data/ekg1.csv --model conv1d_32
```

---

## Uwagi

- Przed kompresją lub testowaniem należy najpierw wytrenować model za pomocą komendy `train`.
- Szczegóły działania każdej komendy można uzyskać poprzez `--help`, np.:

```bash
python3 main.py train --help
```

---

## Przykładowe komendy

```bash
python3 main.py train data/mitbih_train.csv --model conv1d_32
python3 main.py compress ./data/ekg1.csv --model conv1d_32
python3 main.py decompress ./data/ekg1.conv1d_32
python3 main.py test ./data/ekg1.csv --model conv1d_32
```

# Pełne testowanie modeli:
Uruchomienie skryptu `tests.py` poskutkuje wytrenowanie i wygenerowaniem statystyk jak również przykłądowych
rekonstrukcji w folderze `./tests/`. Podczas wykonywania tego skryptu wykrenowane sieci nie są
nigdzie zapisywane więc nie bedą dostępne przy uruchamianiu skryptu `main.py`
```bash
python3 tests.py
```
