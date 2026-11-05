# Temporal generalisation of PV power estimation at a single plant

   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23990095.svg)](https://doi.org/10.5281/zenodo.23990095)

Code for the article *Bir Fotovoltaik Santralde Makine Öğrenmesiyle Güç Kestirimi: Eğitim
Geçmişinin Uzunluğu ve Zamansal Genelleme*.

Hourly PV output of the FMI Helsinki plant is estimated from measured weather with a physical
reference (k·POA) and six machine-learning models (MLR, KNN, SVR, Random Forest, XGBoost, LSTM).
All models are tested on 2021, a year never used in training, with two training histories:

| Case | Training | Test |
|---|---|---|
| `A_5yr` | 2016–2020 | 2021 |
| `B_1yr` | 2020 | 2021 |

## Data

Karhu, J. A., & Lindfors, A. V. (2025). PV production data with ancillary PV and meteorological
data including solar radiation measurements from FMI's outdoor solar laboratories in Helsinki,
Kuopio and Sodankylä (Finland) starting from August 2015 and ending Dec 2021 (Version 4).
https://doi.org/10.57707/fmi-b2share.c7b3ec9f19324a6497639d6d8c0cde84

Download `FMI_Helsinki_PV.csv` and place it in `data/raw/`.

The processed file `data/Helsinki_MultiYear.csv` is included in the Zenodo archive of this
repository (not on GitHub). It is derived from the dataset above, which is licensed under
CC BY: hourly averages of the Helsinki file for 2016–2021, physically cleaned and gap-filled with
`01_preprocessing.ipynb`. With this file, the notebooks can be run from `02_experiment.ipynb`.

## Usage

```bash
pip install -r requirements.txt
```

Run the notebooks in order from the repository root:

| Notebook | Output |
|---|---|
| `01_preprocessing.ipynb` | `data/Helsinki_MultiYear.csv` |
| `02_experiment.ipynb` | `results/` (metrics, predictions, models) |
| `03_results.ipynb` | tables and figures of the article |
| `04_tuning_sensitivity.ipynb` | optional: tuning of RF, XGBoost and LSTM |

Paths are set in `TemporalConfig` (`base_dir`, `data_filename`, `results_folder_name`).

## Reproducibility

All models except LSTM are deterministic. LSTM is run with five seeds. In case A, repeated runs
with the same seed can differ (R² by up to about 0.003), so a new run will not reproduce the
LSTM values of the article exactly; case B was reproduced to the third decimal. Results were produced with Python 3.13 and the versions in `requirements.txt`; other versions
(for example of scikit-learn) can change the last digits.
Training times were measured on Google Colab (LSTM on an NVIDIA T4 GPU, other models on CPU).

## License

MIT
