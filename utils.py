"""
Moduł zawierający funkcje pomocnicze dla aplikacji kolorowanki.
"""

import matplotlib.pyplot as plt
import cv2


def display_results(original_image, processed_image, triangulator):
    """
    Wyświetla porównanie oryginalnego obrazu z obrazem po triangulacji
    używając matplotlib.

    Args:
        original_image (numpy.ndarray): Oryginalny obraz w formacie BGR
        processed_image (numpy.ndarray): Obraz z triangulacją w formacie BGR
        triangulator (ImageTriangulation): Instancja triangulacji z parametrami
    """
    # Utwórz nowe okno matplotlib z określonym rozmiarem
    plt.figure(figsize=(15, 10))

    # Pierwszy subplot - oryginalny obraz
    plt.subplot(1, 2, 1)  # 1 rząd, 2 kolumny, pierwszy obraz
    # Konwertuj z BGR (OpenCV) na RGB (matplotlib) do poprawnego wyświetlania kolorów
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Oryginalny obraz", fontsize=14, fontweight='bold')
    plt.axis('off')  # Ukryj osie współrzędnych

    # Drugi subplot - obraz z triangulacją
    plt.subplot(1, 2, 2)  # 1 rząd, 2 kolumny, drugi obraz
    # Konwertuj z BGR na RGB
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    # Tytuł z informacjami o parametrach triangulacji
    plt.title(f"Triangulacja Delaunaya\n"
              f"Punkty konturu: {triangulator.n_contour_points}, "
              f"Gęstość: {triangulator.interior_density}",
              fontsize=14, fontweight='bold')
    plt.axis('off')  # Ukryj osie współrzędnych

    # Dopasuj layout aby uniknąć nakładania się elementów
    plt.tight_layout()

    # Wyświetl okno z wynikami
    plt.show()


def save_result(processed_image, output_path):
    """
    Zapisuje przetworzony obraz do pliku.

    Args:
        processed_image (numpy.ndarray): Obraz do zapisania
        output_path (str): Ścieżka do pliku wynikowego

    Returns:
        bool: True jeśli zapis się powiódł, False w przeciwnym razie
    """
    try:
        # Zapisz obraz używając OpenCV
        success = cv2.imwrite(output_path, processed_image)

        if success:
            print(f"Wynik zapisany pomyślnie do: {output_path}")
            return True
        else:
            print(f"Błąd podczas zapisywania obrazu do: {output_path}")
            return False

    except Exception as e:
        print(f"Wystąpił błąd podczas zapisywania: {e}")
        return False


def print_help():
    """
    Wyświetla pomoc dotyczącą obsługi aplikacji.
    """
    help_text = """
    === KOLOROWANKA Z TRIANGULACJĄ DELAUNAYA ===

    STEROWANIE:

    Mysz:
    - Kliknij lewym przyciskiem na trójkąt aby go pokolorować
    - Kliknij na kolor w palecie aby go wybrać
    - Przeciągnij suwaki aby zmienić gęstość siatki

    Klawiatura:
    - ESC          : Zamknij aplikację
    - R            : Resetuj kolorowanie (przywróć oryginalny obraz)
    - S            : Zapisz aktualny stan kolorowanki do pliku
    - D            : Pokaż/ukryj okno kontroli gęstości siatki
    - + (lub =)    : Zwiększ gęstość siatki triangulacji
    - - (lub _)    : Zmniejsz gęstość siatki triangulacji

    OKNA:
    - Główne okno    : Kolorowanka z triangulacją
    - Wybór koloru   : Paleta kolorów do wyboru
    - Wybór obrazu   : Miniatury dostępnych obrazów
    - Gęstość siatki : Kontrola parametrów triangulacji

    PLIKI OBRAZÓW:
    Aplikacja automatycznie wyszukuje pliki obrazów w bieżącym katalogu:
    - .jpg, .jpeg, .png, .bmp, .tiff

    PARAMETRY TRIANGULACJI:
    - Punkty konturu   : Liczba punktów rozmieszczonych na konturze (10-100)
    - Gęstość wewnętrzna : Gęstość punktów wewnątrz obszaru (2-20)

    Im większe wartości, tym gęstsza siatka triangulacji.
    """
    print(help_text)


def validate_image_path(image_path):
    """
    Sprawdza czy ścieżka do obrazu jest prawidłowa i czy plik istnieje.

    Args:
        image_path (str): Ścieżka do pliku obrazu

    Returns:
        bool: True jeśli ścieżka jest prawidłowa, False w przeciwnym razie
    """
    import os

    # Sprawdź czy ścieżka nie jest pusta
    if not image_path:
        print("Błąd: Nie podano ścieżki do obrazu")
        return False

    # Sprawdź czy plik istnieje
    if not os.path.exists(image_path):
        print(f"Błąd: Plik {image_path} nie istnieje")
        return False

    # Sprawdź czy to jest plik (a nie katalog)
    if not os.path.isfile(image_path):
        print(f"Błąd: {image_path} nie jest plikiem")
        return False

    # Sprawdź rozszerzenie pliku
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    file_extension = os.path.splitext(image_path)[1].lower()

    if file_extension not in valid_extensions:
        print(f"Błąd: Nieobsługiwane rozszerzenie pliku {file_extension}")
        print(f"Obsługiwane formaty: {', '.join(valid_extensions)}")
        return False

    return True


def validate_parameters(n_contour_points, interior_density):
    """
    Sprawdza czy parametry triangulacji są w prawidłowych zakresach.

    Args:
        n_contour_points (int): Liczba punktów na konturze
        interior_density (int): Gęstość punktów wewnętrznych

    Returns:
        tuple: (bool, str) - (czy_prawidłowe, komunikat_błędu)
    """
    # Sprawdź punkty konturu
    if not isinstance(n_contour_points, int):
        return False, "Liczba punktów konturu musi być liczbą całkowitą"

    if n_contour_points < 3:
        return False, "Liczba punktów konturu musi być co najmniej 3"

    if n_contour_points > 200:
        return False, "Liczba punktów konturu nie może przekraczać 200"

    # Sprawdź gęstość wewnętrzną
    if not isinstance(interior_density, int):
        return False, "Gęstość wewnętrzna musi być liczbą całkowitą"

    if interior_density < 1:
        return False, "Gęstość wewnętrzna musi być co najmniej 1"

    if interior_density > 50:
        return False, "Gęstość wewnętrzna nie może przekraczać 50"

    return True, "Parametry są prawidłowe"


def get_color_name(color_bgr):
    """
    Zwraca nazwę koloru na podstawie wartości BGR.

    Args:
        color_bgr (tuple): Kolor w formacie BGR (B, G, R)

    Returns:
        str: Nazwa koloru lub opis RGB
    """
    # Słownik podstawowych kolorów (BGR -> nazwa)
    color_names = {
        (255, 0, 0): "Czerwony",
        (0, 255, 0): "Zielony",
        (0, 0, 255): "Niebieski",
        (255, 255, 0): "Cyan",
        (255, 0, 255): "Magenta",
        (0, 255, 255): "Żółty",
        (0, 165, 255): "Pomarańczowy",
        (128, 0, 128): "Fioletowy",
        (0, 0, 0): "Czarny",
        (255, 255, 255): "Biały",
        (128, 128, 128): "Szary",
        (0, 69, 255): "Czerwono-pomarańczowy",
        (75, 0, 130): "Indygo",
        (165, 42, 42): "Brązowy",
        (192, 192, 192): "Srebrny",
        (0, 215, 255): "Złoty"
    }

    # Sprawdź czy kolor jest w słowniku
    if color_bgr in color_names:
        return color_names[color_bgr]

    # Jeśli nie ma w słowniku, zwróć opis RGB
    b, g, r = color_bgr
    return f"RGB({r}, {g}, {b})"


def setup_matplotlib_polish():
    """
    Konfiguruje matplotlib do wyświetlania polskich znaków.
    """
    try:
        import matplotlib
        # Ustaw czcionkę obsługującą polskie znaki
        matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # Jeśli dostępna jest czcionka z polskimi znakami
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
        except:
            pass

    except ImportError:
        print("Uwaga: Nie można skonfigurować matplotlib dla polskich znaków")


def create_color_palette_image(colors, selected_index=0):
    """
    Tworzy obraz palety kolorów do zapisania lub wyświetlenia.

    Args:
        colors (list): Lista kolorów w formacie BGR
        selected_index (int): Indeks aktualnie wybranego koloru

    Returns:
        numpy.ndarray: Obraz palety kolorów
    """
    import numpy as np

    # Parametry palety
    color_width = 60
    color_height = 60
    colors_per_row = 8
    border_width = 2

    # Oblicz wymiary
    rows = (len(colors) + colors_per_row - 1) // colors_per_row
    palette_width = colors_per_row * color_width
    palette_height = rows * color_height

    # Utwórz białe tło
    palette_image = np.ones((palette_height, palette_width, 3), dtype=np.uint8) * 255

    # Narysuj kolory
    for i, color in enumerate(colors):
        row = i // colors_per_row
        col = i % colors_per_row

        x_start = col * color_width
        y_start = row * color_height
        x_end = x_start + color_width
        y_end = y_start + color_height

        # Narysuj kolor
        cv2.rectangle(palette_image, (x_start, y_start), (x_end, y_end), color, -1)

        # Dodaj ramkę
        border_color = (0, 0, 0) if i == selected_index else (128, 128, 128)
        border_thickness = 3 if i == selected_index else 1
        cv2.rectangle(palette_image, (x_start, y_start), (x_end, y_end),
                      border_color, border_thickness)

    return palette_image


def log_action(action, details=""):
    """
    Loguje akcje użytkownika do celów debugowania.

    Args:
        action (str): Nazwa akcji
        details (str): Dodatkowe szczegóły
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_message = f"[{timestamp}] {action}"

    if details:
        log_message += f" - {details}"

    print(log_message)


def check_opencv_version():
    """
    Sprawdza wersję OpenCV i wyświetla informacje o kompatybilności.

    Returns:
        bool: True jeśli wersja jest kompatybilna
    """
    try:
        opencv_version = cv2.__version__
        print(f"Wersja OpenCV: {opencv_version}")

        # Sprawdź czy wersja jest wystarczająca (minimum 4.0.0)
        version_parts = opencv_version.split('.')
        major_version = int(version_parts[0])

        if major_version >= 4:
            print("✓ Wersja OpenCV jest kompatybilna")
            return True
        else:
            print("⚠ Uwaga: Stara wersja OpenCV, niektóre funkcje mogą nie działać poprawnie")
            print("Zalecana wersja: OpenCV 4.0 lub nowsza")
            return False

    except Exception as e:
        print(f"Błąd podczas sprawdzania wersji OpenCV: {e}")
        return False


def estimate_processing_time(image_shape, n_contour_points, interior_density):
    """
    Szacuje czas przetwarzania obrazu na podstawie jego rozmiaru i parametrów.

    Args:
        image_shape (tuple): Kształt obrazu (wysokość, szerokość)
        n_contour_points (int): Liczba punktów na konturze
        interior_density (int): Gęstość punktów wewnętrznych

    Returns:
        float: Szacowany czas w sekundach
    """
    # Podstawowe szacowanie na podstawie rozmiaru obrazu i liczby punktów
    height, width = image_shape[:2]
    total_pixels = height * width

    # Szacowana liczba punktów do triangulacji
    estimated_points = n_contour_points + (interior_density ** 2) * 10

    # Bardzo przybliżone szacowanie (może wymagać kalibracji)
    base_time = (total_pixels / 1000000) * 0.5  # 0.5 sekundy na megapiksel
    triangulation_time = (estimated_points / 100) * 0.1  # 0.1 sekundy na 100 punktów

    total_time = base_time + triangulation_time

    return max(0.1, total_time)  # Minimum 0.1 sekundy