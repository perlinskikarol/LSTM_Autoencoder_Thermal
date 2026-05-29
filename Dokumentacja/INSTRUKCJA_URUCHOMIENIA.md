# Instrukcja uruchomienia

## 1. Wymagania
- Windows + PowerShell.
- Python 3.10 lub nowszy.
- Kamera Hikvision dual RGB + Thermal dostepna w sieci LAN.
- Konto kamery z dostepem do RTSP i ISAPI.

## 2. Instalacja
```powershell
cd G:\Magisterka\Oprogramowanie
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Zaleznosci do treningu modelu sa w osobnym pliku:

```powershell
pip install -r requirements-ml.txt
```

## 3. Konfiguracja
Utworz lokalny plik `.env` na podstawie przykladu:

```powershell
Copy-Item .env.example .env
```

Uzupelnij co najmniej:
- `HIK_IP`
- `HIK_USER`
- `HIK_PASS`
- `RTSP_RGB`
- `RTSP_TH`
- `ENABLE_METADATA`
- `METADATA_MODE`

Plik `.env` zawiera dane dostepowe i nie powinien byc commitowany.

## 4. Podstawowe komendy

### Test kanalow RTSP
```powershell
python -m src.probe_rtsp
```

### Podglad demo
```powershell
python -m src.demo
```

### Kalibracja RGB -> thermal
```powershell
python -m src.calibrate_homography
```

Po kalibracji przepisz wyliczona wartosc `RGB_TO_THERMAL_H` do `.env`.

### Pomiar w oknach OpenCV
```powershell
python -m src.mp_forehead_capture
```

### Pomiar w interfejsie Tkinter
```powershell
python -m src.mp_forehead_ui
```

## 5. Sterowanie podczas pomiaru
- `W` / `S` - przesuniecie ROI gora/dol.
- `A` / `D` - przesuniecie ROI lewo/prawo.
- `J` / `L` - zmiana skali ROI w osi X.
- `K` / `I` - zmiana skali ROI w osi Y.
- `[` / `]` - zmiana kroku przesuniecia.
- `-` / `=` - zmiana kroku skalowania.
- `R` - reset korekty ROI.
- `P` - wypisanie aktualnych parametrow `THERMAL_ROI_*`.
- `Q` lub `ESC` - wyjscie.

## 6. Przygotowanie danych dla LSTM Autoencoder
Przyklad:

```powershell
python -m src.prepare_lstm_dataset --input-dir Pomiary --output-dir Dane_przygotowane\lstm_autoencoder --seq-len 60 --stride 5 --skip-initial-sec 10 --max-missing-ratio 0.10
```

Skrypt:
- wczytuje sesje CSV z `Pomiary/`,
- interpoluje pojedyncze braki temperatury,
- buduje okna sekwencyjne,
- dzieli dane po sesjach na zbiory treningowe/testowe,
- zapisuje wynik do `Dane_przygotowane/`.

## 7. Trening modelu
Przyklad:

```powershell
python -m src.train_lstm_autoencoder --dataset-dir Dane_przygotowane\lstm_autoencoder\M1
```

Skrypt zapisuje lokalnie:
- najlepszy model `model_best.pt`,
- historie treningu `history.csv`,
- wykres straty `loss_curve.png`,
- metryki `metrics.json`,
- wyniki okien `scores_by_window.csv`.

Wyniki trafiaja do `runs/`, ktory jest ignorowany przez Git.

## 8. Eksport modelu do Netron
Do podgladu struktury sieci w Netron mozna wyeksportowac model po treningu:

```powershell
python -m src.export_lstm_autoencoder_netron --run-dir runs\lstm_autoencoder\M1\m1_60s_dynamic_20260505
```

Netron otwiera plik `.torchscript.pt` albo `.onnx`. Plik `.metadata.json` zawiera opis warstw, ksztaltow wejscia/wyjscia i cech.

## 9. Generowanie raportu i wykresow
Raporty i wykresy sa artefaktami lokalnymi. Przykladowo:

```powershell
python -m src.generate_lstm_visual_report
python -m src.plot_roc_auc_comparison
python -m src.plot_three_session_timeline_comparison
```

Wygenerowane pliki powinny zostac w `Wykresy/` albo innym katalogu lokalnym.

## 10. Typowy workflow
1. Skonfiguruj `.env`.
2. Sprawdz RTSP: `python -m src.probe_rtsp`.
3. Skalibruj mapowanie: `python -m src.calibrate_homography`.
4. Uruchom pomiar: `python -m src.mp_forehead_ui`.
5. Zapisz sesje pomiarowe lokalnie.
6. Przygotuj dane: `python -m src.prepare_lstm_dataset`.
7. Wytrenuj model: `python -m src.train_lstm_autoencoder`.
8. Wygeneruj wykresy/raporty lokalne.

## 10. Najczestsze problemy
- Blad autoryzacji ISAPI: sprawdz `HIK_USER`, `HIK_PASS` i uprawnienia konta.
- Brak RTSP: sprawdz adresy `RTSP_RGB`, `RTSP_TH`, port oraz dostep sieciowy.
- `valid=0` w pomiarach: ROI nie trafia w czolo albo filtry P2P sa zbyt ostre.
- Niestabilne temperatury: sprawdz kalibracje, odleglosc od kamery i ustawienia
  termometrii w panelu Hikvision.
- Brak danych dla modelu: upewnij sie, ze w `Pomiary/` sa pliki CSV w formacie
  oczekiwanym przez `prepare_lstm_dataset`.
