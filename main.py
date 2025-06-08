#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Główny plik aplikacji kolorowanki z triangulacją Delaunaya.

Aplikacja umożliwia:
1. Wybór obrazu z dostępnych plików
2. Automatyczne tworzenie siatki triangulacji na obrazie
3. Dostosowanie gęstości siatki triangulacji
4. Wybór kolorów z rozszerzonej palety
5. Kolorowanie trójkątów poprzez kliknięcie

Autor: Projekt z geometrii obliczeniowej
Data: 2024
"""

import sys
import os
import argparse
import cv2

# Importy z własnych modułów
from triangulation import ImageTriangulation
from coloring_game import ColoringGame
from utils import (display_results, save_result, print_help,
                   validate_image_path, validate_parameters,
                   check_opencv_version, setup_matplotlib_polish,
                   log_action, estimate_processing_time)


def main():
    """
    Główna funkcja aplikacji. Obsługuje zarówno tryb kolorowanki 
    jak i tryb wiersza poleceń.
    """
    # Konfiguracja kodowania dla polskich znaków
    if sys.stdout.encoding != 'utf-8':
        try:
            # Próba ustawienia kodowania UTF-8 dla terminala
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            # Jeśli nie udało się, kontynuuj z domyślnym kodowaniem
            pass

    # Sprawdź wersję OpenCV
    check_opencv_version()

    # Konfiguruj matplotlib dla polskich znaków
    setup_matplotlib_polish()

    print("=== KOLOROWANKA Z TRIANGULACJĄ DELAUNAYA ===")
    print("Wersja: 2.0 (rozszerzona)")
    print()

    # Sprawdź czy podano argumenty wiersza poleceń
    if len(sys.argv) > 1:
        log_action("Uruchomienie", "Tryb wiersza poleceń")
        run_command_line_mode()
    else:
        log_action("Uruchomienie", "Tryb kolorowanki")
        run_coloring_game_mode()


def run_command_line_mode():
    """
    Uruchamia aplikację w trybie wiersza poleceń do przetwarzania pojedynczych obrazów.
    """
    # Konfiguracja parsera argumentów
    parser = argparse.ArgumentParser(
        description="Triangulacja konturów obrazów z użyciem algorytmu Delaunaya",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py obraz.jpg
  python main.py obraz.jpg --points 50 --density 10
  python main.py obraz.jpg --output wynik.jpg --no-display
  python main.py obraz.jpg --points 30 --density 5 --output triangulacja.png
        """
    )

    # Argumenty pozycyjne
    parser.add_argument("image", type=str,
                        help="Ścieżka do obrazu wejściowego")

    # Argumenty opcjonalne
    parser.add_argument("--points", type=int, default=25,
                        help="Liczba punktów na konturze (domyślnie: 25, zakres: 3-200)")
    parser.add_argument("--density", type=int, default=8,
                        help="Gęstość punktów wewnętrznych (domyślnie: 8, zakres: 1-50)")
    parser.add_argument("--output", type=str,
                        help="Ścieżka zapisu wyniku (opcjonalnie)")
    parser.add_argument("--no-display", action="store_true",
                        help="Nie wyświetlaj wyników w oknie matplotlib")
    parser.add_argument("--help-extended", action="store_true",
                        help="Wyświetl rozszerzoną pomoc")

    try:
        # Parsuj argumenty
        args = parser.parse_args()

        # Wyświetl rozszerzoną pomoc jeśli żądana
        if args.help_extended:
            print_help()
            return

        # Walidacja ścieżki obrazu
        if not validate_image_path(args.image):
            sys.exit(1)

        # Walidacja parametrów
        is_valid, error_message = validate_parameters(args.points, args.density)
        if not is_valid:
            print(f"Błąd parametrów: {error_message}")
            sys.exit(1)

        log_action("Walidacja", "Parametry są prawidłowe")

        # Utwórz instancję triangulacji z podanymi parametrami
        triangulator = ImageTriangulation()
        triangulator.n_contour_points = args.points
        triangulator.interior_density = args.density

        print(f"Przetwarzanie obrazu: {args.image}")
        print(f"Parametry - Punkty konturu: {args.points}, Gęstość: {args.density}")

        # Szacuj czas przetwarzania
        temp_img = cv2.imread(args.image)
        if temp_img is not None:
            estimated_time = estimate_processing_time(temp_img.shape, args.points, args.density)
            print(f"Szacowany czas przetwarzania: {estimated_time:.1f} sekund")

        # Przetwórz obraz
        log_action("Przetwarzanie", f"Rozpoczęcie triangulacji obrazu {args.image}")
        original, result = triangulator.process_from_file(args.image)

        if original is not None and result is not None:
            log_action("Przetwarzanie", "Triangulacja zakończona pomyślnie")

            # Wyświetl wyniki jeśli nie wyłączono
            if not args.no_display:
                log_action("Wyświetlanie", "Pokazanie wyników w matplotlib")
                display_results(original, result, triangulator)

            # Zapisz wynik jeśli podano ścieżkę
            if args.output:
                log_action("Zapisywanie", f"Zapis wyniku do {args.output}")
                if save_result(result, args.output):
                    print(f"✓ Wynik zapisany pomyślnie: {args.output}")
                else:
                    print(f"✗ Błąd podczas zapisywania: {args.output}")
                    sys.exit(1)
            else:
                # Automatyczny zapis z domyślną nazwą
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                auto_output = f"{base_name}_triangulacja.jpg"
                log_action("Zapisywanie", f"Automatyczny zapis do {auto_output}")
                save_result(result, auto_output)

            print("✓ Przetwarzanie zakończone pomyślnie")

        else:
            log_action("Błąd", "Nie udało się przetworzyć obrazu")
            print("✗ Błąd podczas przetwarzania obrazu")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠ Przerwano przez użytkownika")
        sys.exit(0)
    except Exception as e:
        log_action("Błąd", f"Nieoczekiwany błąd: {str(e)}")
        print(f"✗ Nieoczekiwany błąd: {e}")
        sys.exit(1)


def run_coloring_game_mode():
    """
    Uruchamia aplikację w trybie interaktywnej gry kolorowanki.
    """
    try:
        print("Uruchamianie interaktywnej kolorowanki...")
        print("Naciśnij 'H' w aplikacji aby wyświetlić pomoc")
        print()

        # Sprawdź czy są dostępne pliki obrazów
        import glob
        image_files = (glob.glob("*.jpg") + glob.glob("*.jpeg") +
                       glob.glob("*.png") + glob.glob("*.bmp") +
                       glob.glob("*.tiff"))

        if not image_files:
            print("⚠ Uwaga: Nie znaleziono plików obrazów w bieżącym katalogu")
            print("Umieść pliki obrazów (.jpg, .png, .bmp, .tiff) w tym katalogu")
            print("i uruchom aplikację ponownie")
            input("Naciśnij Enter aby zakończyć...")
            return

        print(f"✓ Znaleziono {len(image_files)} plików obrazów")

        # Utwórz i uruchom grę kolorowanki
        game = ColoringGame()
        log_action("Gra", "Inicjalizacja kolorowanki")

        # Uruchom główną pętlę gry
        game.run()

        log_action("Gra", "Kolorowanka została zamknięta")
        print("Dziękujemy za korzystanie z kolorowanki!")

    except KeyboardInterrupt:
        print("\n⚠ Przerwano przez użytkownika")
        log_action("Gra", "Przerwano przez użytkownika")
    except Exception as e:
        log_action("Błąd", f"Błąd w trybie gry: {str(e)}")
        print(f"✗ Wystąpił błąd w trybie kolorowanki: {e}")
        print("Sprawdź czy wszystkie wymagane biblioteki są zainstalowane:")
        print("- opencv-python")
        print("- numpy")
        print("- scipy")
        print("- matplotlib")


def check_dependencies():
    """
    Sprawdza czy wszystkie wymagane biblioteki są zainstalowane.

    Returns:
        bool: True jeśli wszystkie zależności są dostępne
    """
    required_modules = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }

    missing_modules = []

    for module, package in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - BRAK")
            missing_modules.append(package)

    if missing_modules:
        print(f"\nBrakujące pakiety: {', '.join(missing_modules)}")
        print("Zainstaluj je używając:")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    return True


if __name__ == "__main__":
    """
    Punkt wejścia aplikacji. Sprawdza zależności i uruchamia główną funkcję.
    """
    print("Sprawdzanie zależności...")

    # Sprawdź czy wszystkie wymagane moduły są dostępne
    if not check_dependencies():
        print("\n✗ Niektóre wymagane biblioteki nie są zainstalowane")
        print("Zainstaluj brakujące pakiety i uruchom aplikację ponownie")
        sys.exit(1)

    print("✓ Wszystkie zależności są dostępne\n")

    # Uruchom główną funkcję
    main()