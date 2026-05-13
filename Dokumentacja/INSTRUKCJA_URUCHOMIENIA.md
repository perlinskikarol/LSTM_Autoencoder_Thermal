# Instrukcja uruchomienia

## 1. Wymagania
- System: Windows + PowerShell
- Python: 3.10+ (rekomendowane)
- Kamera Hikvision dual (RGB + Thermal) dostepna po LAN
- Konto kamery z uprawnieniami do RTSP i ISAPI

## 2. Instalacja
```powershell
cd G:\Magisterka\Oprogramowanie
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Konfiguracja `.env`
Skopiuj `.env.example` do `.env` i uzupelnij dane kamery.

```powershell
Copy-Item .env.example .env
```

## 4. Najczesciej uzywane uruchomienia

### 4.1 Podstawowy pomiar (OpenCV okna)
```powershell
.\venv\Scripts\Activate.ps1
python -m src.mp_forehead_capture
```

### 4.2 UI (formularz sesji + podglad RGB/THERMAL)
```powershell
.\venv\Scripts\Activate.ps1
python -m src.mp_forehead_ui
```

### 4.3 Kalibracja homografii RGB->THERMAL
```powershell
.\venv\Scripts\Activate.ps1
python -m src.calibrate_homography
```

### 4.4 Szybki test kanalow RTSP
```powershell
.\venv\Scripts\Activate.ps1
python -m src.probe_rtsp
```

### 4.5 Podglad demo (bez ROI logiki)
```powershell
.\venv\Scripts\Activate.ps1
python -m src.demo
```

### 4.6 Przygotowanie danych pod LSTM Autoencoder
```powershell
.\venv\Scripts\Activate.ps1
python -m src.prepare_lstm_dataset
```

### 4.7 Szkic treningu LSTM Autoencoder
Najpierw doinstaluj zaleznosc ML:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
```

Potem uruchom trening:

```powershell
.\venv\Scripts\Activate.ps1
python -m src.train_lstm_autoencoder --dataset-dir Dane_przygotowane\lstm_autoencoder\M1
```

## 5. Sterowanie klawiatura (capture/UI)
- `W` / `S`: przesuniecie ROI gora / dol (`THERMAL_ROI_SHIFT_Y_RATIO`)
- `A` / `D`: przesuniecie ROI lewo / prawo (`THERMAL_ROI_SHIFT_X_RATIO`)
- `J` / `L`: skala ROI X minus / plus (`THERMAL_ROI_SCALE_X`)
- `K` / `I`: skala ROI Y minus / plus (`THERMAL_ROI_SCALE_Y`)
- `[` / `]`: zmniejsz / zwieksz krok przesuniecia
- `-` / `=`: zmniejsz / zwieksz krok skali
- `R`: reset strojenia do wartosci z `.env`
- `P`: wypisz do logu aktualne `THERMAL_ROI_*` (gotowe do wklejenia do `.env`)
- `Q` lub `ESC`: wyjscie (w UI `Q` zatrzymuje pomiar)

## 5A. Przygotowanie danych do modelu sekwencyjnego
- Skrypt: `src.prepare_lstm_dataset`
- Wejscie: wszystkie sesje CSV z folderu `Pomiary`
- Wyjscie: folder `Dane_przygotowane\lstm_autoencoder\<ID_uzytkownika>\`
- Cechy na probke:
  - `mean_temp_c`
  - `min_temp_c`
  - `max_temp_c`
  - `range_temp_c`
  - `sin_time`
  - `cos_time`
- Co robi preprocessing:
  - pomija pierwsze sekundy sesji (`--skip-initial-sec`, domyslnie `10`)
  - interpoluje pojedyncze braki temperatury
  - buduje okna sekwencji o dlugosci `--seq-len`
  - przesuwa okno co `--stride`
  - odrzuca okna z nadmiarem brakow (`--max-missing-ratio`)
  - dzieli dane po calych sesjach na `train`, `val`, `test_normal`, `test_anomaly`
  - normalizuje cechy statystykami policzonymi tylko na `train`

Przyklad z jawna konfiguracja:
```powershell
python -m src.prepare_lstm_dataset --input-dir Pomiary --output-dir Dane_przygotowane\lstm_autoencoder --seq-len 60 --stride 5 --skip-initial-sec 10 --max-missing-ratio 0.10
```

Skrypt treningowy `src.train_lstm_autoencoder`:
- wczytuje `dataset.npz`
- trenuje prosty `LSTM Autoencoder` na `X_train`
- monitoruje `val_loss`
- zapisuje `model_best.pt`, `history.csv`, `loss_curve.png`, `metrics.json`
- wyznacza prog anomalii na podstawie percentyla bledu rekonstrukcji na `val`
- zapisuje wynik dla kazdego okna do `scores_by_window.csv`

## 6. Flagi konfiguracyjne (`.env`) - pelna lista

### 6.1 Dostep do kamery i strumieni
- `HIK_IP` - IP kamery.
- `HIK_USER` - uzytkownik.
- `HIK_PASS` - haslo.
- `RTSP_PORT` - port RTSP (zwykle `554`).
- `RTSP_RGB` - URL RTSP kanalu RGB.
- `RTSP_TH` - URL RTSP kanalu thermal.
- `CHANNEL_ID_FOR_METADATA` - kanal do ISAPI metadanych (u Ciebie zwykle `2`).
- `ENABLE_METADATA` - `true/false`, czy pobierac temperatury.

### 6.2 Tryb metadanych temperatury
- `METADATA_MODE` - `legacy | auto | http_thermal | http_thermal_p2p`.
- `METADATA_LEGACY_URI` - wymuszenie legacy URI (opcjonalnie).
- `METADATA_HTTP_ENDPOINT` - wymuszenie endpointu HTTP (opcjonalnie).
- `METADATA_RETRY_SEC` - retry polaczenia metadanych.
- `METADATA_AUTH_LOCKOUT_SEC` - pauza po bledach auth.
- `METADATA_MAX_AUTH_FAILURES` - limit bledow auth przed lockout.

### 6.3 CSV i agregacja
- `FOREHEAD_CSV_PATH` - alias starej sciezki raw CSV.
- `RAW_CSV_PATH` - sciezka surowych probek.
- `AGG_CSV_PATH` - sciezka agregatow.
- `SAMPLE_PERIOD_SEC` - okres probkowania.
- `AGG_WINDOW_SEC` - okno agregacji.
- `AGG_EMIT_SEC` - jak czesto zapisywac agregaty.
- `MEDIAN_WINDOW` - okno mediany wygladzania.
- `EWMA_ALPHA` - wspolczynnik EWMA.

### 6.4 Detekcja twarzy / czola (MediaPipe)
- `MP_MIN_DETECTION_CONF` - prog pewnosci detekcji.
- `MP_MODEL_SELECTION` - model MP (`0` blisko, `1` daleko).
- `MP_DETECTION_INPUT_SCALE` - skala wejscia detektora.
- `MP_FACE_MODEL_PATH` - sciezka do modelu MP Tasks (opcjonalnie).
- `MP_FACE_TOP_EXPAND_RATIO` - rozszerzenie obszaru twarzy ku gorze.

### 6.5 Dopasowanie temperatury do ROI
- `TEMP_PROPERTY` - preferowana wlasciwosc (`average`, itp.).
- `ROI_MATCH_MIN_IOU` - minimalny IoU dla dopasowania regionu.
- `STRICT_ROI_ONLY` - tylko match w ROI (`true`) lub fallback (`false`).

### 6.6 Filtrowanie P2P (pixel matrix)
- `P2P_TEMP_MIN_C` - dolny prog temperatury pikseli.
- `P2P_TEMP_MAX_C` - gorny prog temperatury pikseli.
- `P2P_TRIM_LOW_PCT` - obciecie dolnego percentyla.
- `P2P_TRIM_HIGH_PCT` - obciecie gornego percentyla.
- `P2P_MIN_VALID_PIXELS` - minimum waznych pikseli w ROI.

### 6.7 Mapowanie RGB->THERMAL
- `RGB_TO_THERMAL_H` - homografia 3x3 (9 liczb, CSV w jednej linii).
- `HOMOGRAPHY_SWITCH_ENABLED` - przelaczanie near/far.
- `RGB_TO_THERMAL_H_NEAR` - homografia dla blisko.
- `RGB_TO_THERMAL_H_FAR` - homografia dla daleko.
- `HOMOGRAPHY_NEAR_MIN_FACE_WIDTH_RATIO` - prog wejscia w near.
- `HOMOGRAPHY_FAR_MAX_FACE_WIDTH_RATIO` - prog przejscia w far.

### 6.8 Gdy thermal ma wezsze FOV
- `THERMAL_COVERAGE_X` - pokrycie thermal wzgledem RGB w osi X.
- `THERMAL_COVERAGE_Y` - pokrycie thermal wzgledem RGB w osi Y.

### 6.9 Reczna korekta ROI na thermal
- `THERMAL_ROI_SHIFT_X_RATIO` - przesuniecie X po mapowaniu.
- `THERMAL_ROI_SHIFT_Y_RATIO` - przesuniecie Y po mapowaniu.
- `THERMAL_ROI_SCALE_X` - skala ROI X.
- `THERMAL_ROI_SCALE_Y` - skala ROI Y.

### 6.10 Dynamiczna synchronizacja regionu w kamerze (ISAPI PUT)
- `DYNAMIC_THERMOMETRY_REGION_ENABLED` - wlacz/wylacz.
- `DYNAMIC_THERMOMETRY_SCENE_ID` - ID sceny termometrii.
- `DYNAMIC_THERMOMETRY_REGION_ID` - ID regionu w scenie.
- `DYNAMIC_THERMOMETRY_UPDATE_SEC` - minimalny interwal aktualizacji.
- `DYNAMIC_THERMOMETRY_MIN_MOVE_NORM` - minimalna zmiana ROI do wysylki.
- `DYNAMIC_THERMOMETRY_MAX_FAILURES` - limit bledow przed auto-disable.
- `DYNAMIC_THERMOMETRY_FAILURE_BACKOFF_SEC` - backoff po bledzie.

## 7. Typowy workflow (zalecany)
1. Ustaw `.env` (`HIK_*`, RTSP, `ENABLE_METADATA=true`, `METADATA_MODE=http_thermal_p2p`).
2. Sprawdz RTSP: `python -m src.probe_rtsp`.
3. Skalibruj mapowanie: `python -m src.calibrate_homography`, wklej `RGB_TO_THERMAL_H` do `.env`.
4. Uruchom pomiar UI: `python -m src.mp_forehead_ui`.
5. Dostroj ROI klawiszami, wcisnij `P`, przepisz `THERMAL_ROI_*` do `.env`.
6. Zbieraj dane do `logs/forehead_raw.csv` i `logs/forehead_aggregated.csv`.

## 8. Najczestsze problemy
- `403` na endpointach thermometry: konto/firmware/tryb endpointu; sprawdz `METADATA_MODE=http_thermal_p2p`.
- `valid=0`: ROI nie trafia w czolo lub filtry P2P sa za ostre (`P2P_*`, `THERMAL_ROI_*`).
- Za wysokie temperatury: sprawdz ustawienia kompensacji/blackbody w GUI kamery.
- Brak startu: `ENABLE_METADATA=false` (w `mp_forehead_capture` to blad krytyczny).
