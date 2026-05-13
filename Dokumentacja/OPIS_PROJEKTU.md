# Opis projektu

## Cel
Projekt realizuje pomiar temperatury czola z kamery Hikvision dual (RGB + thermal), z:
- detekcja twarzy i ROI czola po stronie Python,
- mapowaniem ROI RGB -> thermal,
- pobieraniem temperatury z ISAPI (w tym P2P pixel matrix),
- zapisem surowych i agregowanych danych do CSV.

## Architektura

### `src/config.py`
- Wczytuje `.env` przez `python-dotenv`.
- Buduje obiekt `Settings` (dane kamery, RTSP, tryb metadanych).
- Waliduje wymagane pola (`HIK_IP`, `HIK_USER`, `HIK_PASS`).

### `src/rtsp_stream.py`
- Watkowy klient RTSP dla RGB i thermal.
- Utrzymuje ostatnia ramke + reconnect.
- Interfejs: `start()`, `get_last_frame()`, `stop()`.

### `src/roi_provider.py`
- Detekcja twarzy i wyznaczanie ROI czola (MediaPipe).
- Zwraca `RoiBox` dla czola.
- Przechowuje tez box twarzy i confidence.

### `src/hik_isapi_metadata.py`
- Klient metadanych temperatury Hikvision.
- Obsluguje kilka trybow (`legacy`, `auto`, `http_thermal`, `http_thermal_p2p`).
- W trybie P2P pobiera i dekoduje macierz temperatur pikselowych.

### `src/mp_forehead_capture.py` (glowny pipeline)
- Laczy RTSP RGB, RTSP thermal i metadane.
- Pobiera ROI czola z RGB, mapuje na thermal.
- Wybiera temperature z ROI:
  - region rule-based (legacy/http_thermal),
  - pixel-by-pixel (http_thermal_p2p).
- Liczy agregacje w czasie (mean/median/std/min/max/ewma).
- Zapisuje:
  - `logs/forehead_raw.csv`,
  - `logs/forehead_aggregated.csv`.
- Udostepnia reczne strojenie ROI klawiszami.

### `src/mp_forehead_ui.py` (GUI Tkinter)
- Interfejs pomiaru z polami:
  - data sesji,
  - osoba / ID.
- Start/Stop pomiaru.
- Podglad RGB i thermal obok siebie.
- Zapis CSV rozszerzony o `session_date` i `subject_id`.

### `src/calibrate_homography.py`
- Narzedzie kalibracji odwzorowania RGB->thermal.
- Uzytkownik klika punkty korespondencyjne w obu oknach.
- Wyznacza homografie 3x3 (`RGB_TO_THERMAL_H`), liczy blad reprojekcji.

### `src/probe_rtsp.py`
- Test dostepnosci kanalow RTSP `101/102/201/202`.

### `src/demo.py`
- Prosty podglad strumieni i opcjonalny podglad metadanych.

## Przeplyw danych (runtime)
1. Start konfiguracji z `.env`.
2. Start RTSP RGB + thermal.
3. Start klienta ISAPI temperatur.
4. Detekcja twarzy i ROI czola na RGB.
5. Mapowanie ROI do thermal (homografia lub model coverage+shift+scale).
6. Odczyt temperatury w ROI (rule lub P2P).
7. Zapis probki raw.
8. Aktualizacja agregatora i okresowy zapis aggregated.
9. Podglad overlay + korekta reczna ROI.

## Logi i wyniki
- `logs/forehead_raw.csv` - kazda probka, status, boxy, metadata mode, temp.
- `logs/forehead_aggregated.csv` - statystyki z okna czasowego.
- `logs/rgb_to_thermal_homography.json` - wynik kalibracji homografii.
- `logs/raw_http_thermo_empty.log` - pomocniczy log debug HTTP.

## Zalozenia i ograniczenia
- Dokladnosc zalezy od:
  - jakosci mapowania RGB->thermal,
  - ustawien termometrii i kompensacji w kamerze,
  - dystansu i kata twarzy.
- Przy slabym mapowaniu temperatura moze pochodzic czesciowo z wlosow/tla.
- `http_thermal_p2p` daje najwieksza kontrole, ale wymaga dobrego ROI.
