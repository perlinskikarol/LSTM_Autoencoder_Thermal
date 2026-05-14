# Opis projektu

## Cel
Projekt sluzy do zbierania pomiarow temperatury czola z kamery Hikvision dual
(RGB + thermal), przygotowania danych sekwencyjnych oraz trenowania modelu
LSTM Autoencoder do analizy odchylen/anomalii w przebiegach temperatury.

Repozytorium zawiera kod zrodlowy i dokumentacje. Dane pomiarowe, przygotowane
zbiory, logi, wykresy, modele oraz wyniki eksperymentow sa generowane lokalnie i
nie sa przechowywane w repozytorium.

## Glowne elementy systemu

### Pomiar temperatury
- `src/mp_forehead_capture.py` - glowny pipeline pomiarowy z oknami OpenCV.
- `src/mp_forehead_ui.py` - interfejs Tkinter do prowadzenia sesji pomiarowej.
- `src/rtsp_stream.py` - klient RTSP dla strumieni RGB i thermal.
- `src/roi_provider.py` - detekcja twarzy oraz wyznaczanie ROI czola.
- `src/hik_isapi_metadata.py` - pobieranie metadanych temperatury z ISAPI.
- `src/calibrate_homography.py` - kalibracja mapowania RGB -> thermal.
- `src/probe_rtsp.py` - szybki test kanalow RTSP kamery.
- `src/demo.py` - prosty podglad strumieni i metadanych.

### Przygotowanie danych i model
- `src/prepare_lstm_dataset.py` - budowa okien sekwencyjnych z plikow CSV.
- `src/train_lstm_autoencoder.py` - trening LSTM Autoencoder i zapis metryk.
- `src/generate_lstm_visual_report.py` - generowanie lokalnego raportu z wynikow.

### Wizualizacje pomocnicze
- `src/plot_temperature_csv.py`
- `src/plot_reconstruction_comparison.py`
- `src/plot_reconstruction_session_views.py`
- `src/plot_roc_auc_comparison.py`
- `src/plot_mean_temp_session_comparison.py`
- `src/plot_three_session_timeline_comparison.py`

## Przeplyw pracy
1. Konfiguracja dostepu do kamery w lokalnym pliku `.env`.
2. Test kanalow RTSP i ISAPI.
3. Kalibracja mapowania obrazu RGB na obraz thermal.
4. Pomiar sesji i zapis probek temperatury do CSV.
5. Przygotowanie danych dla LSTM Autoencoder.
6. Trening modelu na sekwencjach normalnych.
7. Ocena bledu rekonstrukcji i analiza anomalii.
8. Generowanie wykresow oraz raportow lokalnych.

## Dane i artefakty lokalne
Nastepujace katalogi sa ignorowane przez Git:
- `Pomiary/` - surowe sesje CSV.
- `Dane_przygotowane/` - zbiory danych dla modelu.
- `runs/` - wyniki treningow, checkpointy i metryki.
- `Wykresy/` - wygenerowane wykresy oraz raporty.
- `logs/` - logi aplikacji i pomiarow.
- `venv/` - lokalne srodowisko Python.

Do repozytorium powinny trafiac tylko pliki z kodem, dokumentacja, przykladowa
konfiguracja `.env.example` oraz pliki zaleznosci.

## Ograniczenia
- Jakosc pomiaru zalezy od kalibracji RGB -> thermal, pozycji twarzy i ustawien
  termometrii kamery.
- Model LSTM Autoencoder jest wrazliwy na sposob podzialu sesji i jakosc danych
  treningowych.
- Pliki wynikowe moga byc duze, dlatego powinny pozostac poza repozytorium albo
  trafic do osobnego archiwum/dysku, jesli sa potrzebne do odtworzenia badan.
