# Raport LSTM Autoencoder - 2026-05-05

## Cel

Porownanie dwoch wariantow modelu LSTM Autoencoder dla danych pacjenta `M1`:

- wariant `60 s`
- wariant `10 min`

Oba modele byly trenowane po aktualizacji datasetu o najnowsze sesje oraz z wykluczeniem jednej odstajacej sesji:

- `M1_normalny_22.03.2026_12.23.25`

## Datasety

### Wariant 60 s

- dataset: `Dane_przygotowane/lstm_autoencoder/M1`
- okno: `60` probek
- stride: `5`
- cechy: `12`

Ksztalty:

- `X_train`: `(19104, 60, 12)`
- `X_val`: `(2595, 60, 12)`
- `X_test_normal`: `(2947, 60, 12)`
- `X_test_anomaly`: `(535, 60, 12)`

### Wariant 10 min

- dataset: `Dane_przygotowane/lstm_autoencoder_10min/M1`
- okno: `600` probek
- stride: `5`
- cechy: `12`

Ksztalty:

- `X_train`: `(16773, 600, 12)`
- `X_val`: `(2151, 600, 12)`
- `X_test_normal`: `(2685, 600, 12)`
- `X_test_anomaly`: `(300, 600, 12)`

## Cechy wejsciowe

Model dostaje dla kazdej sekundy nastepujace cechy:

- `mean_temp_c`
- `min_temp_c`
- `max_temp_c`
- `range_temp_c`
- `delta_mean_temp`
- `delta_max_temp`
- `slope_mean_5s`
- `slope_mean_15s`
- `rolling_std_mean_15s`
- `rolling_range_mean_15s`
- `sin_time`
- `cos_time`

Czyli model widzi nie tylko poziom temperatur, ale tez ich lokalna dynamike i informacje o porze dnia.

## Jak dziala model

1. Z plikow CSV sesji budowane sa stale probkowane szeregi czasowe.
2. Dane dzielimy na nakladajace sie okna:
   - `60 s` albo `600 s`
   - przesuwane co `5 s`
3. Kazde okno jest normalizowane statystykami policzonymi tylko na `train`.
4. LSTM Autoencoder probuje zrekonstruowac cale okno.
5. Dla kazdego okna liczony jest blad rekonstrukcji.
6. Prog anomalii wyznaczany jest z `0.99` quantile na zbiorze `val`.
7. Jesli blad okna przekracza prog, okno oznaczamy jako anomalne.

## Wyniki zbiorcze

### Model 60 s

- run: `runs/lstm_autoencoder/M1/m1_60s_dynamic_20260505`
- raport: `Wykresy/lstm_reports/m1_60s_dynamic_20260505`

Najwazniejsze liczby:

- `best_epoch`: `44`
- `best_val_loss`: `0.19635169`
- `threshold`: `0.76724291`
- `ROC AUC`: `0.5493`
- `Average Precision`: `0.2212`
- `Precision`: `0.3333`
- `Recall`: `0.1009`
- `Specificity`: `0.9634`
- `F1`: `0.1549`

Podzial okien:

- `test_normal`: `3.66%` ponad prog
- `test_anomaly`: `10.09%` ponad prog

### Model 10 min

- run: `runs/lstm_autoencoder/M1/m1_10min_dynamic_20260505`
- raport: `Wykresy/lstm_reports/m1_10min_dynamic_20260505`

Najwazniejsze liczby:

- `best_epoch`: `29`
- `best_val_loss`: `0.33198287`
- `threshold`: `0.68960053`
- `ROC AUC`: `0.6924`
- `Average Precision`: `0.3766`
- `Precision`: `0.2874`
- `Recall`: `0.2367`
- `Specificity`: `0.9345`
- `F1`: `0.2596`

Podzial okien:

- `test_normal`: `6.55%` ponad prog
- `test_anomaly`: `23.67%` ponad prog

## Wniosek porownawczy

Model `10 min` jest wyraznie czulszy na anomalie niz model `60 s`, ale placi za to wieksza liczba falszywych alarmow.

W praktyce:

- `60 s` jest bardziej konserwatywny
- `10 min` lepiej lapie wolniejsze zmiany w czasie

W tym konkretnym eksperymencie wariant `10 min` daje lepsza separacje calosciowa:

- wyzsze `ROC AUC`
- wyzsze `Average Precision`
- wyzsze `Recall`
- wyzsze `F1`

## Zachowanie na sesjach testowych

### Model 60 s

- `M1_prysznic_16.04.2026_15.55.46`: `15.08%` okien ponad prog
- `M1_prysznic_28.04.2026_13.17.36`: `0.00%`
- `M1_normalny_29.04.2026_18.11.21`: `4.54%`
- `M1_normalny_04.05.2026_22.08.46`: `3.00%`

### Model 10 min

- `M1_prysznic_16.04.2026_15.55.46`: `31.28%` okien ponad prog
- `M1_prysznic_28.04.2026_13.17.36`: `0.00%`
- `M1_normalny_29.04.2026_18.11.21`: `13.77%`
- `M1_normalny_04.05.2026_22.08.46`: `0.00%`

Interpretacja:

- pierwszy prysznic daje wyrazny sygnal anomalii, szczegolnie przy oknie `10 min`
- drugi prysznic nadal nie odroznia sie od normy, czyli nie kazda sesja poprysznicowa daje wystarczajaco silny sygnal
- sesja `M1_normalny_29.04.2026_18.11.21` pozostaje trudniejsza i generuje czesc falszywych alarmow

## Sweep progu

Porownanie przy lagodniejszym progu `q = 0.90`:

### 60 s

- `precision = 0.3381`
- `recall = 0.2206`
- `specificity = 0.9216`
- `f1 = 0.2670`

### 10 min

- `precision = 0.3244`
- `recall = 0.3233`
- `specificity = 0.9248`
- `f1 = 0.3239`

To jest bardzo ciekawy wynik: przy lzejszym progu wariant `10 min` ma jednoczesnie:

- wyzszy `recall`
- lepszy `f1`
- bardzo podobna `specificity`

## Jak czytac wykresy

Kazdy raport zawiera ten sam zestaw figur:

### `01_loss_curve`

Pokazuje przebieg `train_loss` i `val_loss` w kolejnych epokach.

Na co patrzec:

- czy `val_loss` spada
- czy nie zaczyna mocno rosnac przy dalszym spadku `train_loss`
- w ktorej epoce model osiagnal najlepszy wynik

Interpretacja:

- jesli `train_loss` spada, a `val_loss` stoi lub rosnie, model zaczyna sie przeuczac
- jesli oba spadaja, model uczy sie stabilnie

### `02_error_distribution`

Rozklady bledow rekonstrukcji dla:

- `train`
- `val`
- `test_normal`
- `test_anomaly`

Na co patrzec:

- czy `test_anomaly` jest przesuniete w prawo wzgledem `test_normal`
- jak bardzo nachodza na siebie rozklady

Interpretacja:

- im mniejsze nachodzenie, tym latwiej ustawic dobry prog

### `03_boxplot_splits`

Boxplot bledow dla wszystkich splitow.

Na co patrzec:

- mediane
- szerokosc pudelka
- ogon wysokich bledow

Interpretacja:

- jesli `test_anomaly` ma wyzsza mediane i wyzszy gorony ogon niz `test_normal`, to model dobrze odroznia anomalie

### `04_roc_pr_curves`

Dwie klasyczne krzywe ewaluacyjne:

- ROC
- Precision-Recall

Na co patrzec:

- `ROC AUC`: im wyzsze, tym lepiej
- `Average Precision`: szczegolnie wazne przy niezbalansowanych danych

Interpretacja:

- w naszych danych `10 min` ma lepsze obie metryki, wiec ogolnie lepiej rozdziela okna normalne od anomalnych

### `05_confusion_matrix`

Macierz pomylek dla aktualnego progu.

Na co patrzec:

- `TP`: dobrze wykryte anomalie
- `FP`: falszywe alarmy
- `FN`: przeoczone anomalie
- `TN`: poprawnie rozpoznane okna normalne

Interpretacja:

- przy progu `0.99` wariant `60 s` jest ostrozniejszy
- wariant `10 min` wykrywa wiecej anomalii, ale daje wiecej `FP`

### `06_threshold_sweep`

Pokazuje, jak zmieniaja sie:

- `precision`
- `recall`
- `specificity`
- `f1`

gdy przesuwamy prog decyzyjny.

Na co patrzec:

- czy mozna poprawic `recall` bez dramatycznego spadku `specificity`
- przy jakim progu `f1` jest najlepsze

Interpretacja:

- ten wykres jest bardzo praktyczny, bo pozwala stroic model pod bardziej czuly albo bardziej konserwatywny tryb pracy

### `07_session_ranking`

Porownanie sesji testowych na poziomie sesji.

Na co patrzec:

- ktore sesje maja najwyzszy sredni blad
- ktore sesje maja najwyzszy `p95`
- ktore sesje maja najwyzszy udzial okien ponad prog

Interpretacja:

- to jest dobry wykres do pracy dyplomowej, bo pokazuje juz wynik blizej poziomu badania, a nie pojedynczego okna

### `08_test_timelines`

Anomaly score w czasie dla sesji testowych.

Na co patrzec:

- czy anomalia pojawia sie punktowo czy blokami
- czy prog przecinany jest chwilowo, czy przez dluzszy fragment sesji

Interpretacja:

- to jest bardzo wazne klinicznie, bo pojedynczy pik moze byc szumem
- dluzszy ciag podwyzszonych bledow jest znacznie bardziej wiarygodnym sygnalem odchylenia od normy

## Rekomendacja robocza

Na ten moment najrozsadniej traktowalbym to tak:

- `60 s` jako model bardziej zachowawczy
- `10 min` jako model bardziej czuly i obecnie bardziej obiecujacy badawczo

Jesli kolejne sesje anomalii beda krotsze lub bardziej punktowe, warto zostawic oba warianty do porownania. Jesli interesuja nas wolniejsze zmiany po prysznicu, treningu albo w stanie chorobowym, wariant `10 min` wyglada obecnie lepiej.
