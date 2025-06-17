# ML-Compression-Network
Computational Intelligence Methods course final project - Neural network compression

Kompresja danych z użyciem SSN [prowadzący: obaj]. 

Proszę wykorzystać sieć neuronową typu MLP do kompresji danych, dla przykładu przebiegu EKG z portalu https://www.kaggle.com/datasets/shayanfazeli/heartbeat Proszę stworzyć autoenkoder który będzie mógł zarówno poddać sygnał EKG kompresji jak i dekompresji. Proszę odpowiednio zająć się zagadnieniem preprocesingu danych oraz funkcją zależności pomiędzy stopniem kompresji a jakością odzyskanego przebiegu. 

Proszę wskazać w zbiory danych które Państwo wybieracie. 

___

Projekt polega na samodzielnym (w obrębie grupy projektowej) wykonaniu zadania związanego z tematyką kursu.  

Projekt wykonujemy w zespołach trzyosobowych. Przypadki, w których będzie to niemożliwe będą rozpatrywane indywidualnie.  

Projekt należy zrealizować do ostatnich zajęć projektowych. Po tym terminie ocena ostateczna będzie uwzględniała spóźnienie w realizacji zadania.  

W sprawozdaniu należy zamieścić opis zadania, uzyskane wyniki oraz plik źródłowy z rozwiązaniem (może być w formie odnośnika do repozytorium z kodem) Oceniane będą: czytelność opisu, kreatywność zastosowanego podejścia oraz jakość przedstawienia wyników.  

Pliki projektu (kod źródłowy/link do repozytorium z kodem źródłowym i plikami pozwalającymi na replikacje rozwiązania), dokument (5-10 stron) podsumowujący prace i uzyskane wyniki (przeklejaniu całości kodu mówimy stanowcze nie; wystarczy architektura plus ciekawe fragmenty) ładujemy przez platformę MS Teams (Assignement). W projekcie (sprawozdanie/wspomniany plik tekstowy/opis repozytorium) należy wyraźnie zaznaczyć kto był jego wykonawcą. 

Wybór tematu jest dowolny, każdy temat może realizować dowolna liczba grup.  

Zajęcie projektowe tj. konsultacje przeprowadzane są w tych terminach wg Unitime. Proszę pamiętać, iż konsultacje projektu odbywa się czasie zajęć przypadającym dla osób danego zespołu. W wyjątkowych przypadkach osób z różnych grup laboratoryjnych termin należy uwzględnić z prowadzącym projekt. 



# ML-Compression-Network

**Kompresja danych z użyciem sieci neuronowych (Autoencoderów)**  
Projekt końcowy na kurs *Metody Inteligencji Obliczeniowej*

---

## Opis projektu

Projekt wykorzystuje sieci neuronowe typu autoenkoder do kompresji i rekonstrukcji danych EKG.  
Sygnał wejściowy jest przekształcany do przestrzeni latentnej (skompresowany), a następnie odtwarzany przez dekoder.  
Celem modelu jest zminimalizowanie błędu rekonstrukcji przy jednoczesnym zmniejszeniu rozmiaru danych.

Zastosowano modele:
- `generic_autoencoder` - gęsta sieć MLP (Multilayer Perceptron)
- `conv1d_generic_autoencoder` - konwolucyjny autoenkoder (1D)

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
python3 main.py train data/mitbih_train.csv --model conv1d_generic_autoencoder
```

Parametry:
- `--model` - nazwa modelu (`generic_autoencoder` lub `conv1d_generic_autoencoder`)

---

### Kompresja pliku

```bash
python3 main.py compress ./data/ekg1.csv --model conv1d_generic_autoencoder
```

Parametry:
- `filepath` - ścieżka do pliku CSV
- `--model` - wybrany model do kompresji
- `--output` - opcjonalna ścieżka pliku wyjściowego

---

### Dekompresja pliku

```bash
python3 main.py decompress ./data/ekg1.conv1d_generic_autoencoder.csv
```

Parametry:
- `filepath` - plik z rozszerzeniem `.model.csv` (np. `.generic_autoencoder.csv`)
- `--output` - opcjonalna ścieżka pliku wyjściowego

---

### Test modelu

Wykonuje kompresję, dekompresję oraz analizę jakości rekonstrukcji.

```bash
python3 main.py test ./data/ekg1.csv --model conv1d_generic_autoencoder
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
python3 main.py train data/mitbih_train.csv --model generic_autoencoder
python3 main.py compress ./data/ekg1.csv --model generic_autoencoder
python3 main.py decompress ./data/ekg1.generic_autoencoder.csv
python3 main.py test ./data/ekg1.csv --model generic_autoencoder
```

---

## Autorzy


| Name                | GitHub Profile                               |
|---------------------|----------------------------------------------|
| Andrzej Świętek     | [GitHub Profile](https://github.com/Andrzej-Swietek)|
| Marcin Knapczyk     | [GitHub Profile](https://github.com/Nautirius)|
| Krzysztof Konieczny | [GitHub Profile](https://github.com/KrzysztofProgramming)|
