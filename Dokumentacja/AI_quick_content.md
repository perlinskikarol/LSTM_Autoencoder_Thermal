# AI Quick Content

## 1. Co to jest
Pipeline Python do pomiaru temperatury czola z kamery Hikvision dual:
- detekcja czola na RGB (MediaPipe),
- mapowanie ROI na thermal,
- temperatura z ISAPI (preferowany `http_thermal_p2p`),
- zapis raw + aggregated CSV.

## 2. Najwazniejsze entry-pointy
- `python -m src.mp_forehead_capture` - glowny tryb OpenCV.
- `python -m src.mp_forehead_ui` - GUI Tkinter (data + osoba + start/stop).
- `python -m src.calibrate_homography` - kalibracja `RGB_TO_THERMAL_H`.
- `python -m src.probe_rtsp` - test kanalow RTSP.

## 3. Krytyczne pliki
- `src/config.py` - ladowanie i walidacja `.env`.
- `src/hik_isapi_metadata.py` - ISAPI + parsing temperatur (w tym P2P matrix).
- `src/roi_provider.py` - twarz + ROI czola.
- `src/mp_forehead_capture.py` - pelna logika pomiaru, mapowanie, CSV, klawisze.
- `src/mp_forehead_ui.py` - UI, sesje badanych (`session_date`, `subject_id`).

## 4. Najwazniejsze zmienne `.env`
- Polaczenie: `HIK_IP`, `HIK_USER`, `HIK_PASS`, `RTSP_RGB`, `RTSP_TH`.
- Metadane: `ENABLE_METADATA=true`, `METADATA_MODE=http_thermal_p2p`.
- ROI mapping: `RGB_TO_THERMAL_H` lub `THERMAL_COVERAGE_*`.
- Reczne dostrojenie: `THERMAL_ROI_SHIFT_*`, `THERMAL_ROI_SCALE_*`.
- Filtr P2P: `P2P_TEMP_MIN_C`, `P2P_TEMP_MAX_C`, `P2P_TRIM_*`, `P2P_MIN_VALID_PIXELS`.

## 5. Szybkie debug-checki
1. RTSP dziala? `python -m src.probe_rtsp`.
2. Metadane wchodza? log powinien pokazac aktywny endpoint P2P i `valid>0`.
3. ROI trafia w czolo? podglad zielonego boxa thermal.
4. Jesli zaniza temp: box wchodzi we wlosy/tlo -> stroic `WASD JIKL [] -=`.
5. Po dostrojeniu: `P` i wkleic `THERMAL_ROI_*` do `.env`.

## 6. Klawisze runtime (capture/UI)
- `WASD`: shift ROI.
- `J/L`: scale X -/+.
- `K/I`: scale Y -/+.
- `[` `]`: krok shift.
- `-` `=`: krok scale.
- `R`: reset do `.env`.
- `P`: wypisz aktualne `THERMAL_ROI_*`.
- `Q`/`ESC`: stop.

## 7. Pliki wynikowe
- `logs/forehead_raw.csv`
- `logs/forehead_aggregated.csv`
- `logs/rgb_to_thermal_homography.json`

## 8. Typowe pulapki
- `403` ISAPI: konto/uprawnienia/endpoint.
- `valid=0`: zle ROI lub zbyt ostre filtry P2P.
- wysokie odczyty: aktywna kompensacja/blackbody po stronie kamery.
