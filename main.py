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
import cv2

# Importy z własnych modułów
from triangulation import ImageTriangulation
from coloring_game import ColoringGame
from utils import (display_results, save_result, print_help,
                   check_opencv_version, setup_matplotlib_polish,
                   log_action)


def main():
    """
    Główna funkcja aplikacji. Uruchamia tryb kolorowanki.
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

    log_action("Uruchomienie", "Tryb kolorowanki")
    run_coloring_game_mode()


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
