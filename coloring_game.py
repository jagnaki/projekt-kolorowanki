"""
Moduł zawierający klasę ColoringGame - główną grę kolorowanki z triangulacją.
"""

import numpy as np
import cv2
import os
import glob
from triangulation import ImageTriangulation


class ColoringGame:
    """
    Klasa implementująca interaktywną grę kolorowanki z triangulacją Delaunaya.
    Umożliwia wybór obrazu, kolorów i interaktywne kolorowanie trójkątów.
    """

    def __init__(self):
        """
        Inicjalizuje grę kolorowanki z domyślnymi ustawieniami.
        """
        # Utwórz instancję triangulacji z odpowiednimi parametrami
        self.triangulator = ImageTriangulation()
        self.triangulator.n_contour_points = 30  # Liczba punktów na konturze
        self.triangulator.interior_density = 8  # Gęstość punktów wewnętrznych

        # Standardowy rozmiar okna dla wszystkich obrazów
        self.window_width = 800
        self.window_height = 600

        # Zakresy parametrów triangulacji dla suwaków kontrolnych
        self.min_contour_points = 10  # Minimalna liczba punktów na konturze
        self.max_contour_points = 100  # Maksymalna liczba punktów na konturze
        self.min_interior_density = 2  # Minimalna gęstość punktów wewnętrznych
        self.max_interior_density = 20  # Maksymalna gęstość punktów wewnętrznych

        # Znacznie rozszerzona paleta kolorów do wyboru (format BGR dla OpenCV)
        self.colors = [
            # Kolory podstawowe
            (255, 0, 0),  # Czerwony
            (0, 255, 0),  # Zielony (jasny)
            (0, 0, 255),  # Niebieski
            (255, 255, 0),  # Żółty (cyan w BGR)
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Żółty (w BGR)

            # Kolory ciepłe
            (0, 165, 255),  # Pomarańczowy
            (0, 69, 255),  # Czerwono-pomarańczowy
            (0, 140, 255),  # Ciemny pomarańczowy
            (0, 0, 255),  # Czerwony
            (85, 107, 47),  # Ciemny oliwkowy zielony

            # Kolory zimne
            (128, 0, 128),  # Fioletowy
            (205, 92, 92),  # Indiancz czerwony
            (139, 69, 19),  # Brązowy siodłowy
            (160, 82, 45),  # Brązowy
            (75, 0, 130),  # Indygo

            # Kolory pastelowe
            (255, 182, 193),  # Jasny różowy
            (221, 160, 221),  # Śliwkowy
            (173, 216, 230),  # Jasny niebieski
            (144, 238, 144),  # Jasny zielony
            (255, 255, 224),  # Jasny żółty

            # Kolory ziemiste
            (42, 42, 165),  # Brązowy
            (19, 69, 139),  # Ciemny brązowy
            (35, 142, 107),  # Morski zielony
            (87, 139, 46),  # Oliwkowy
            (45, 82, 160),  # Sienna

            # Kolory metaliczne/specjalne
            (192, 192, 192),  # Srebrny
            (0, 215, 255),  # Złoty
            (130, 130, 130),  # Szary
            (105, 105, 105),  # Ciemny szary
            (0, 0, 0),  # Czarny
            (255, 255, 255),  # Biały

            # Kolory neonowe/jasne
            (0, 255, 127),  # Zielony wiosenny
            (255, 20, 147),  # Głęboki różowy
            (0, 191, 255),  # Głęboki niebo niebieski
            (50, 205, 50),  # Limonka zielona
            (255, 0, 255),  # Fuksja

            # Dodatkowe odcienie
            (238, 130, 238),  # Fioletowy
            (218, 112, 214),  # Orchidea
            (186, 85, 211),  # Medium orchidea
            (147, 112, 219),  # Medium fioletowy niebieski
            (123, 104, 238),  # Medium łupkowy niebieski
            (106, 90, 205),  # Łupkowy niebieski
            (72, 61, 139),  # Ciemny łupkowy niebieski
            (25, 25, 112),  # Nocny niebieski
        ]

        # Indeks aktualnie wybranego koloru
        self.current_color_index = 0
        # Aktualnie wybrany kolor
        self.current_color = self.colors[self.current_color_index]

        # Dane obrazu
        self.original_image = None  # Oryginalny obraz
        self.display_image = None  # Obraz wyświetlany z triangulacją
        self.triangles = []  # Lista wszystkich trójkątów
        self.triangle_colors = {}  # Słownik przechowujący kolory trójkątów (indeks -> kolor)

        # Nazwy okien (z polskimi znakami)
        self.main_window = "Kolorowanka"
        self.color_window = "Wybór koloru"
        self.image_select_window = "Wybór obrazu"
        self.density_window = "Gęstość siatki"

        # Flagi stanu aplikacji
        self.running = True  # Czy aplikacja jest uruchomiona
        self.image_selected = False  # Czy obraz został wybrany

        # Ustawienie kodowania dla polskich znaków w OpenCV
        self._setup_polish_encoding()

    def _setup_polish_encoding(self):
        """
        Konfiguruje obsługę polskich znaków w OpenCV.
        OpenCV ma ograniczoną obsługę znaków UTF-8, więc używamy podstawowych ustawień.
        """
        # OpenCV nie obsługuje bezpośrednio polskich znaków w tytułach okien
        # Możemy użyć transliteracji dla lepszej kompatybilności
        self.main_window = "Kolorowanka"
        self.color_window = "Wybor koloru"
        self.image_select_window = "Wybor obrazu"
        self.density_window = "Gestosc siatki"

    def find_image_files(self):
        """
        Znajduje wszystkie pliki obrazów w bieżącym katalogu.

        Returns:
            list: Lista nazw plików obrazów
        """
        # Wyszukaj pliki z rozszerzeniami obrazów
        image_files = (glob.glob("*.jpg") +
                       glob.glob("*.jpeg") +
                       glob.glob("*.png") +
                       glob.glob("*.bmp") +
                       glob.glob("*.tiff"))
        return image_files

    def create_image_selection_window(self, image_files):
        """
        Tworzy okno wyboru obrazu z miniaturami dostępnych plików.

        Args:
            image_files (list): Lista ścieżek do plików obrazów

        Returns:
            bool: True jeśli obraz został wybrany, False w przeciwnym razie
        """
        # Sprawdź czy są dostępne pliki obrazów
        if not image_files:
            print("Nie znaleziono plików obrazów!")
            return False

        # Oblicz wymiary okna wyboru (wysokość stała, szerokość zależna od liczby obrazów)
        window_height = 120
        window_width = min(150 * len(image_files), 1200)  # Ogranicz maksymalną szerokość

        # Utwórz jasnoszare tło dla okna
        selection_image = np.ones((window_height, window_width, 3), dtype=np.uint8) * 240

        # Wczytaj i umieść miniatury obrazów
        thumbnails = []
        max_images = min(len(image_files), 8)  # Ogranicz liczbę wyświetlanych miniatur

        for i in range(max_images):
            file = image_files[i]
            img = cv2.imread(file)
            if img is not None:
                # Zmień rozmiar obrazu do miniatury (120x80 pikseli)
                thumbnail = cv2.resize(img, (120, 80))
                thumbnails.append(thumbnail)

                # Oblicz pozycję miniatury w oknie
                x_offset = i * 150 + 15

                # Sprawdź czy miniatura mieści się w oknie
                if x_offset + 120 <= window_width:
                    # Umieść miniaturę w oknie wyboru
                    selection_image[10:10 + 80, x_offset:x_offset + 120] = thumbnail

                    # Dodaj nazwę pliku pod miniaturą (obetnij długie nazwy)
                    filename = os.path.basename(file)
                    if len(filename) > 15:
                        filename = filename[:12] + "..."

                    # Napisz nazwę pliku używając standardowej czcionki
                    cv2.putText(selection_image, filename,
                                (x_offset, window_height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Wyświetl okno wyboru
        cv2.imshow(self.image_select_window, selection_image)

        # Funkcja obsługująca kliknięcia myszy w oknie wyboru
        def select_image(event, x, y, flags, param):
            """
            Obsługuje kliknięcie myszy w oknie wyboru obrazu.
            """
            if event == cv2.EVENT_LBUTTONDOWN:  # Lewy przycisk myszy został naciśnięty
                # Sprawdź na którą miniaturę kliknięto
                for i in range(min(len(image_files), 8)):
                    x_start = i * 150 + 15  # Początek miniatury w osi X
                    x_end = x_start + 120  # Koniec miniatury w osi X

                    # Sprawdź czy kliknięcie było w obszarze miniatury
                    if x_start <= x <= x_end and 10 <= y <= 90:
                        # Załaduj wybrany obraz
                        self.load_image(image_files[i])
                        # Zamknij okno wyboru
                        cv2.destroyWindow(self.image_select_window)
                        # Ustaw flagę, że obraz został wybrany
                        self.image_selected = True
                        break

        # Podłącz funkcję obsługi myszy do okna
        cv2.setMouseCallback(self.image_select_window, select_image)

        # Czekaj na wybór obrazu lub zamknięcie okna
        while (not self.image_selected and
               cv2.getWindowProperty(self.image_select_window, cv2.WND_PROP_VISIBLE) >= 1):
            cv2.waitKey(100)  # Sprawdzaj co 100ms

        return self.image_selected

    def display_main_window(self):
        """
        Wyświetla obraz w głównym oknie z zachowaniem stałego rozmiaru okna.
        """
        # Upewnij się, że okno ma odpowiednie właściwości
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        # Ustaw stały rozmiar okna
        cv2.resizeWindow(self.main_window, self.window_width, self.window_height)
        # Wyświetl obraz
        cv2.imshow(self.main_window, self.display_image)

    def load_image(self, image_path):
        """
        Wczytuje obraz z pliku i tworzy triangulację.

        Args:
            image_path (str): Ścieżka do pliku obrazu

        Returns:
            bool: True jeśli obraz został pomyślnie wczytany
        """
        # Wczytaj obraz z dysku
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Nie można wczytać obrazu: {image_path}")
            return False

        # Zmień rozmiar obrazu do standardowego rozmiaru okna
        self.original_image = cv2.resize(self.original_image, (self.window_width, self.window_height))

        # Przetwórz obraz i utwórz triangulację
        self.process_image()
        return True

    def process_image(self):
        """
        Przetwarza obraz i tworzy triangulację Delaunaya.

        Returns:
            bool: True jeśli przetwarzanie się powiodło
        """
        if self.original_image is None:
            return False

        # Znajdź kontury w obrazie używając triangulacji
        contours = self.triangulator.get_contours_from_image(self.original_image)

        # Utwórz kopię obrazu do wyświetlania z triangulacją
        self.display_image = self.original_image.copy()
        self.triangles = []  # Wyczyść listę trójkątów

        # Przetwórz każdy znaleziony kontur
        for contour in contours:
            # Rozmieść punkty równomiernie na konturze
            contour_points = self.triangulator.place_points_on_contour(
                contour, self.triangulator.n_contour_points)

            # Pomiń kontury z niewystarczającą liczbą punktów
            if len(contour_points) < 3:
                continue

            # Generuj punkty wewnętrzne w obszarze konturu
            interior_points = self.triangulator.generate_interior_points(
                contour_points, self.original_image.shape)

            # Utwórz triangulację Delaunaya
            triangles = self.triangulator.create_triangulation(
                contour_points, interior_points, self.original_image.shape)

            # Dodaj nowe trójkąty do głównej listy
            self.triangles.extend(triangles)

            # Narysuj linie triangulacji na obrazie wyświetlanym
            self.triangulator.draw_delaunay_triangles(
                self.display_image, triangles, self.triangulator.triangle_color)

        # Zresetuj słownik kolorów trójkątów (usuń poprzednie kolorowanie)
        self.triangle_colors = {}

        return True

    def create_color_selection_window(self):
        """
        Tworzy okno wyboru koloru z paletą dostępnych kolorów.
        Każdy kolor jest reprezentowany jako kolorowy prostokąt.
        """
        # Oblicz wymiary okna (wysokość stała, szerokość zależy od liczby kolorów)
        color_height = 60
        color_width = 40
        colors_per_row = 12  # Liczba kolorów w jednym rzędzie

        # Oblicz liczbę rzędów potrzebnych do wyświetlenia wszystkich kolorów
        rows = (len(self.colors) + colors_per_row - 1) // colors_per_row

        # Oblicz wymiary okna
        window_width = colors_per_row * color_width
        window_height = rows * color_height + 30  # Dodatkowe miejsce na tekst

        # Utwórz jasnoszare tło
        color_image = np.ones((window_height, window_width, 3), dtype=np.uint8) * 240

        # Narysuj próbki kolorów w siatce
        for i, color in enumerate(self.colors):
            # Oblicz pozycję koloru w siatce
            row = i // colors_per_row
            col = i % colors_per_row

            # Oblicz współrzędne prostokąta
            x_start = col * color_width
            y_start = row * color_height
            x_end = x_start + color_width
            y_end = y_start + color_height

            # Narysuj kolorowy prostokąt
            cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), color, -1)

            # Zaznacz aktualnie wybrany kolor czarną ramką
            if i == self.current_color_index:
                cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 0, 0), 3)
            else:
                # Narysuj cienką szarą ramkę wokół innych kolorów
                cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (100, 100, 100), 1)

        # Dodaj informację o aktualnie wybranym kolorze
        info_text = f"Wybrany kolor: {self.current_color_index + 1}/{len(self.colors)}"
        cv2.putText(color_image, info_text,
                    (10, window_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Wyświetl okno wyboru koloru
        cv2.imshow(self.color_window, color_image)

        # Funkcja obsługująca kliknięcia myszy w oknie wyboru koloru
        def select_color(event, x, y, flags, param):
            """
            Obsługuje kliknięcie myszy w oknie wyboru koloru.
            """
            if event == cv2.EVENT_LBUTTONDOWN:  # Lewy przycisk myszy
                # Oblicz na który kolor kliknięto
                col = x // color_width
                row = y // color_height
                color_index = row * colors_per_row + col

                # Sprawdź czy indeks koloru jest prawidłowy
                if 0 <= color_index < len(self.colors):
                    # Ustaw nowy wybrany kolor
                    self.current_color_index = color_index
                    self.current_color = self.colors[self.current_color_index]
                    # Odśwież okno wyboru koloru aby pokazać nowy wybór
                    self.create_color_selection_window()

        # Podłącz funkcję obsługi myszy do okna
        cv2.setMouseCallback(self.color_window, select_color)

    def create_density_control_window(self):
        """
        Tworzy okno kontroli gęstości siatki triangulacji z suwakami.
        Umożliwia interaktywne dostosowanie parametrów triangulacji.
        """
        # Wymiary okna kontrolnego
        window_width = 450
        window_height = 180

        # Utwórz jasnoszare tło
        density_image = np.ones((window_height, window_width, 3), dtype=np.uint8) * 240

        # Dodaj tytuły dla suwaków
        cv2.putText(density_image, "Punkty konturu:", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(density_image, "Gestosc wewnetrzna:", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Dodaj aktualne wartości parametrów
        cv2.putText(density_image, str(self.triangulator.n_contour_points), (380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(density_image, str(self.triangulator.interior_density), (380, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Parametry suwaków
        slider_width = 280  # Szerokość obszaru suwaka
        slider_x = 20  # Pozycja X początku suwaka
        slider_y_contour = 40  # Pozycja Y suwaka punktów konturu
        slider_y_interior = 100  # Pozycja Y suwaka gęstości wewnętrznej

        # Narysuj tło suwaka dla punktów kontoru
        cv2.rectangle(density_image, (slider_x, slider_y_contour),
                      (slider_x + slider_width, slider_y_contour + 12), (200, 200, 200), -1)

        # Oblicz pozycję uchwytu suwaka dla punktów konturu
        contour_ratio = ((self.triangulator.n_contour_points - self.min_contour_points) /
                         (self.max_contour_points - self.min_contour_points))
        contour_pos = int(slider_x + contour_ratio * slider_width)

        # Narysuj uchwyt suwaka dla punktów konturu
        cv2.rectangle(density_image, (contour_pos - 6, slider_y_contour - 3),
                      (contour_pos + 6, slider_y_contour + 15), (0, 0, 200), -1)

        # Narysuj tło suwaka dla gęstości wewnętrznej
        cv2.rectangle(density_image, (slider_x, slider_y_interior),
                      (slider_x + slider_width, slider_y_interior + 12), (200, 200, 200), -1)

        # Oblicz pozycję uchwytu suwaka dla gęstości wewnętrznej
        interior_ratio = ((self.triangulator.interior_density - self.min_interior_density) /
                          (self.max_interior_density - self.min_interior_density))
        interior_pos = int(slider_x + interior_ratio * slider_width)

        # Narysuj uchwyt suwaka dla gęstości wewnętrznej
        cv2.rectangle(density_image, (interior_pos - 6, slider_y_interior - 3),
                      (interior_pos + 6, slider_y_interior + 15), (0, 0, 200), -1)

        # Dodaj instrukcje użytkowania
        cv2.putText(density_image, "Kliknij i przeciagnij suwaki aby zmienic gestosc siatki",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(density_image, "Uzyj klawiszy + i - lub przeciagnij suwaki",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # Wyświetl okno kontroli gęstości
        cv2.imshow(self.density_window, density_image)

        # Zmienne do śledzenia stanu przeciągania suwaków
        self.dragging_contour = False  # Czy przeciągamy suwak punktów konturu
        self.dragging_interior = False  # Czy przeciągamy suwak gęstości wewnętrznej

        # Funkcja obsługująca myszy w oknie kontroli gęstości
        def handle_density_control(event, x, y, flags, param):
            """
            Obsługuje interakcję z suwakami w oknie kontroli gęstości.
            """
            if event == cv2.EVENT_LBUTTONDOWN:  # Naciśnięcie lewego przycisku myszy
                # Sprawdź czy kliknięto na suwak punktów konturu
                if (slider_y_contour - 5 <= y <= slider_y_contour + 17 and
                        slider_x - 10 <= x <= slider_x + slider_width + 10):
                    self.dragging_contour = True

                # Sprawdź czy kliknięto na suwak gęstości wewnętrznej
                elif (slider_y_interior - 5 <= y <= slider_y_interior + 17 and
                      slider_x - 10 <= x <= slider_x + slider_width + 10):
                    self.dragging_interior = True

            elif event == cv2.EVENT_LBUTTONUP:  # Zwolnienie lewego przycisku myszy
                # Jeśli zakończono przeciąganie, zastosuj zmiany
                if self.dragging_contour or self.dragging_interior:
                    self.dragging_contour = False
                    self.dragging_interior = False
                    # Przetwórz obraz ponownie z nowymi parametrami
                    self.process_image()
                    # Wyświetl zaktualizowany obraz
                    self.display_main_window()

            elif event == cv2.EVENT_MOUSEMOVE:  # Ruch myszy
                # Aktualizuj pozycję suwaka podczas przeciągania
                if self.dragging_contour:
                    # Ogranicz pozycję X do granic suwaka
                    slider_x_pos = max(slider_x, min(x, slider_x + slider_width))
                    # Oblicz nową wartość punktów konturu na podstawie pozycji
                    ratio = (slider_x_pos - slider_x) / slider_width
                    new_contour_points = int(self.min_contour_points +
                                             ratio * (self.max_contour_points - self.min_contour_points))
                    # Zaktualizuj parametr triangulacji
                    self.triangulator.n_contour_points = new_contour_points
                    # Odśwież okno kontroli
                    self.create_density_control_window()

                elif self.dragging_interior:
                    # Ogranicz pozycję X do granic suwaka
                    slider_x_pos = max(slider_x, min(x, slider_x + slider_width))
                    # Oblicz nową wartość gęstości wewnętrznej
                    ratio = (slider_x_pos - slider_x) / slider_width
                    new_interior_density = int(self.min_interior_density +
                                               ratio * (self.max_interior_density - self.min_interior_density))
                    # Zaktualizuj parametr triangulacji
                    self.triangulator.interior_density = new_interior_density
                    # Odśwież okno kontroli
                    self.create_density_control_window()

        # Podłącz funkcję obsługi myszy do okna
        cv2.setMouseCallback(self.density_window, handle_density_control)

    def point_in_triangle(self, point, triangle):
        """
        Sprawdza czy punkt znajduje się wewnątrz trójkąta używając algorytmu barycentrycznego.

        Args:
            point (tuple): Punkt do sprawdzenia (x, y)
            triangle (tuple): Trójkąt jako krotka trzech punktów

        Returns:
            bool: True jeśli punkt jest wewnątrz trójkąta
        """

        def sign(p1, p2, p3):
            """Oblicza znak dla algorytmu punkt-w-trójkącie."""
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        # Rozpakuj wierzchołki trójkąta
        pt1, pt2, pt3 = triangle

        # Oblicz znaki dla trzech par punktów
        d1 = sign(point, pt1, pt2)
        d2 = sign(point, pt2, pt3)
        d3 = sign(point, pt3, pt1)

        # Sprawdź czy wszystkie znaki mają tę samą orientację
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        # Punkt jest wewnątrz jeśli nie ma zarówno dodatnich jak i ujemnych znaków
        return not (has_neg and has_pos)

    def fill_triangle(self, img, triangle, color):
        """
        Wypełnia trójkąt zadanym kolorem i rysuje jego krawędzie.

        Args:
            img (numpy.ndarray): Obraz na którym rysujemy
            triangle (tuple): Trójkąt jako krotka trzech punktów
            color (tuple): Kolor wypełnienia w formacie BGR
        """
        # Konwertuj punkty trójkąta do formatu wymaganego przez OpenCV
        pts = np.array([triangle[0], triangle[1], triangle[2]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Wypełnij trójkąt kolorem
        cv2.fillPoly(img, [pts], color)

        # Narysuj ponownie krawędzie trójkąta aby były widoczne
        pt1, pt2, pt3 = triangle
        cv2.line(img, pt1, pt2, self.triangulator.triangle_color, 1, cv2.LINE_AA)
        cv2.line(img, pt2, pt3, self.triangulator.triangle_color, 1, cv2.LINE_AA)
        cv2.line(img, pt3, pt1, self.triangulator.triangle_color, 1, cv2.LINE_AA)

    def handle_click(self, event, x, y, flags, param):
        """
        Obsługuje kliknięcie myszy w głównym oknie kolorowanki.

        Args:
            event: Typ zdarzenia myszy
            x, y: Współrzędne kliknięcia
            flags: Dodatkowe flagi zdarzenia
            param: Dodatkowe parametry (nieużywane)
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # Kliknięcie lewym przyciskiem myszy
            # Sprawdź wszystkie trójkąty aby znaleźć ten, w który kliknięto
            for i, triangle in enumerate(self.triangles):
                if self.point_in_triangle((x, y), triangle):
                    # Zapamiętaj kolor trójkąta w słowniku
                    self.triangle_colors[i] = self.current_color

                    # Wypełnij trójkąt wybranym kolorem
                    self.fill_triangle(self.display_image, triangle, self.current_color)

                    # Zaktualizuj wyświetlany obraz
                    self.display_main_window()
                    break  # Przerwij po znalezieniu pierwszego pasującego trójkąta

    def run(self):
        """
        Główna funkcja uruchamiająca grę kolorowanki.
        Obsługuje całą logikę interfejsu użytkownika i interakcji.
        """
        # Znajdź dostępne pliki obrazów w bieżącym katalogu
        image_files = self.find_image_files()

        # Pokaż okno wyboru obrazu i czekaj na wybór użytkownika
        if not self.create_image_selection_window(image_files):
            print("Nie wybrano obrazu!")
            return

        # Utwórz okno wyboru koloru
        self.create_color_selection_window()

        # Utwórz okno kontroli gęstości siatki
        self.create_density_control_window()

        # Wyświetl obraz z triangulacją w głównym oknie
        self.display_main_window()
        # Podłącz obsługę kliknięć myszy do głównego okna
        cv2.setMouseCallback(self.main_window, self.handle_click)

        # Główna pętla aplikacji
        density_window_visible = True  # Czy okno kontroli gęstości jest widoczne

        while self.running:
            # Czekaj na naciśnięcie klawisza (100ms timeout)
            key = cv2.waitKey(100) & 0xFF  # Maska dla zgodności z różnymi systemami

            # Sprawdź czy główne okno zostało zamknięte
            if cv2.getWindowProperty(self.main_window, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False

            # Obsługa skrótów klawiszowych
            if key == 27:  # Klawisz ESC - zakończ aplikację
                self.running = False

            elif key == ord('r') or key == ord('R'):  # Reset kolorowania
                # Przywróć oryginalny obraz z triangulacją
                self.display_image = self.original_image.copy()
                self.triangulator.draw_delaunay_triangles(
                    self.display_image, self.triangles, self.triangulator.triangle_color)
                # Wyczyść słownik kolorów
                self.triangle_colors = {}
                self.display_main_window()
                print("Kolorowanie zostało zresetowane")

            elif key == ord('s') or key == ord('S'):  # Zapisz wynik
                output_filename = "kolorowanka_wynik.jpg"
                cv2.imwrite(output_filename, self.display_image)
                print(f"Zapisano wynik do: {output_filename}")

            elif key == ord('d') or key == ord('D'):  # Pokaż/ukryj okno gęstości
                if density_window_visible:
                    cv2.destroyWindow(self.density_window)
                    density_window_visible = False
                    print("Ukryto okno kontroli gęstości")
                else:
                    self.create_density_control_window()
                    density_window_visible = True
                    print("Pokazano okno kontroli gęstości")

            elif key == ord('+') or key == ord('='):  # Zwiększ gęstość siatki
                # Zwiększ oba parametry triangulacji
                self.triangulator.n_contour_points = min(
                    self.triangulator.n_contour_points + 5, self.max_contour_points)
                self.triangulator.interior_density = min(
                    self.triangulator.interior_density + 1, self.max_interior_density)

                # Przetwórz obraz ponownie
                self.process_image()
                self.display_main_window()

                # Zaktualizuj okno kontroli jeśli jest widoczne
                if density_window_visible:
                    self.create_density_control_window()
                print("Zwiększono gęstość siatki")

            elif key == ord('-') or key == ord('_'):  # Zmniejsz gęstość siatki
                # Zmniejsz oba parametry triangulacji
                self.triangulator.n_contour_points = max(
                    self.triangulator.n_contour_points - 5, self.min_contour_points)
                self.triangulator.interior_density = max(
                    self.triangulator.interior_density - 1, self.min_interior_density)

                # Przetwórz obraz ponownie
                self.process_image()
                self.display_main_window()

                # Zaktualizuj okno kontroli jeśli jest widoczne
                if density_window_visible:
                    self.create_density_control_window()
                print("Zmniejszono gęstość siatki")

        # Zamknij wszystkie okna po zakończeniu aplikacji
        cv2.destroyAllWindows()
        print("Aplikacja została zamknięta")
