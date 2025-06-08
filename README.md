# Kolorowanka - Projekt z Geometrii Obliczeniowej

Projekt z przedmiotu Geometria obliczeniowa - interaktywna kolorowanka wykorzystująca triangulację Delaunaya.

## Opis

Aplikacja umożliwia:
1. Wybór obrazu z dostępnych plików
2. Automatyczne tworzenie siatki triangulacji na obrazie
3. Dostosowanie gęstości siatki triangulacji
4. Wybór kolorów z palety
5. Kolorowanie trójkątów poprzez kliknięcie

## Jak uruchomić

```
python main.py
```

Uruchomienie bez argumentów spowoduje włączenie trybu kolorowanki.

## Jak używać

1. **Wybór obrazu**:
   - Po uruchomieniu aplikacji pojawi się okno z miniaturami dostępnych obrazów
   - Kliknij na wybrany obraz, aby go załadować

2. **Wybór koloru**:
   - W oknie "Wybór koloru" kliknij na wybrany kolor z palety
   - Aktualnie wybrany kolor jest zaznaczony czarną ramką

3. **Dostosowanie gęstości siatki**:
   - W oknie "Gęstość siatki" możesz dostosować gęstość triangulacji
   - Przeciągnij górny suwak, aby zmienić liczbę punktów na konturze
   - Przeciągnij dolny suwak, aby zmienić gęstość punktów wewnętrznych
   - Po zwolnieniu suwaka, siatka zostanie automatycznie przerysowana
   - Możesz również użyć klawiszy `+` i `-` do zwiększania i zmniejszania gęstości siatki

4. **Kolorowanie**:
   - Kliknij na dowolny trójkąt w głównym oknie, aby pokolorować go wybranym kolorem
   - Możesz wielokrotnie zmieniać kolory i kolorować różne trójkąty

5. **Skróty klawiszowe**:
   - `r` - resetuje kolorowanie (przywraca oryginalny obraz z triangulacją)
   - `s` - zapisuje aktualny stan kolorowanki do pliku "kolorowanka_wynik.jpg"
   - `d` - pokazuje/ukrywa okno kontroli gęstości siatki
   - `+` - zwiększa gęstość siatki triangulacji
   - `-` - zmniejsza gęstość siatki triangulacji
   - `ESC` - zamyka aplikację

## Tryb wiersza poleceń

Aplikacja obsługuje również tryb wiersza poleceń do przetwarzania pojedynczych obrazów:

```
python main.py ścieżka_do_obrazu [opcje]
```

Opcje:
- `--points N` - liczba punktów na konturze (domyślnie 25)
- `--density N` - gęstość punktów wewnętrznych (domyślnie 8)
- `--output PLIK` - ścieżka do zapisu wyniku
- `--no-display` - nie wyświetlaj wyników
