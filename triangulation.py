"""
Moduł zawierający klasę ImageTriangulation do tworzenia triangulacji Delaunaya na obrazach.
"""

import numpy as np
import cv2
from scipy.spatial import Delaunay


class ImageTriangulation:
    """
    Klasa odpowiedzialna za tworzenie triangulacji Delaunaya na obrazach.
    Umożliwia automatyczne wykrywanie konturów i generowanie siatki trójkątów.
    """

    def __init__(self):
        """
        Inicjalizuje obiekt triangulacji z domyślnymi parametrami.
        """
        # Parametry triangulacji - kontrolują gęstość siatki
        self.n_contour_points = 30  # Liczba punktów rozmieszczonych na konturze
        self.interior_density = 8  # Gęstość punktów wewnętrznych (im większa, tym więcej punktów)
        self.min_contour_area = 1000  # Minimalny obszar konturu do przetworzenia

        # Kolory używane do rysowania elementów triangulacji
        self.triangle_color = (0, 0, 0)  # Czarne linie triangulacji (format BGR)
        self.contour_color = (0, 255, 0)  # Zielony kolor konturów (nieużywany aktualnie)
        self.interior_color = (255, 0, 0)  # Czerwony kolor punktów wewnętrznych

    def rect_contains(self, rect, point):
        """
        Sprawdza czy punkt znajduje się wewnątrz prostokąta.

        Args:
            rect (tuple): Prostokąt w formacie (x_min, y_min, x_max, y_max)
            point (tuple): Punkt w formacie (x, y)

        Returns:
            bool: True jeśli punkt jest wewnątrz prostokąta
        """
        # Sprawdź czy punkt jest na lewo lub powyżej lewego górnego rogu
        if point[0] < rect[0] or point[1] < rect[1]:
            return False
        # Sprawdź czy punkt jest na prawo lub poniżej prawego dolnego rogu
        if point[0] > rect[2] or point[1] > rect[3]:
            return False
        return True

    def draw_point(self, img, point, color, radius=2):
        """
        Rysuje punkt na obrazie jako wypełnione kółko.

        Args:
            img (numpy.ndarray): Obraz na którym rysujemy
            point (tuple): Współrzędne punktu (x, y)
            color (tuple): Kolor w formacie BGR
            radius (int): Promień kółka
        """
        # Konwertuj współrzędne na liczby całkowite i narysuj kółko
        cv2.circle(img, tuple(map(int, point)), radius, color, -1, cv2.LINE_AA)

    def draw_delaunay_triangles(self, img, triangles, color):
        """
        Rysuje krawędzie trójkątów triangulacji Delaunaya na obrazie.

        Args:
            img (numpy.ndarray): Obraz na którym rysujemy
            triangles (list): Lista trójkątów, każdy jako tuple trzech punktów
            color (tuple): Kolor linii w formacie BGR
        """
        # Iteruj przez wszystkie trójkąty
        for triangle in triangles:
            pt1, pt2, pt3 = triangle  # Rozpakuj współrzędne wierzchołków
            # Narysuj każdą krawędź trójkąta z wygładzaniem
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA)
            cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA)

    def point_in_polygon(self, point, polygon_points):
        """
        Sprawdza czy punkt znajduje się wewnątrz wielokąta używając algorytmu ray casting.

        Args:
            point (tuple): Punkt do sprawdzenia (x, y)
            polygon_points (list): Lista punktów definiujących wielokąt

        Returns:
            bool: True jeśli punkt jest wewnątrz wielokąta
        """
        x, y = point  # Rozpakuj współrzędne punktu
        n = len(polygon_points)  # Liczba wierzchołków wielokąta
        inside = False  # Flaga określająca czy punkt jest wewnątrz

        # Pobierz pierwszy punkt wielokąta
        p1x, p1y = polygon_points[0]

        # Iteruj przez wszystkie krawędzie wielokąta
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]  # Następny punkt (z wraparound)

            # Sprawdź czy promień z punktu przecina krawędź
            if y > min(p1y, p2y):  # Punkt jest powyżej dolnego końca krawędzi
                if y <= max(p1y, p2y):  # Punkt jest poniżej górnego końca krawędzi
                    if x <= max(p1x, p2x):  # Punkt jest na lewo od prawego końca krawędzi
                        if p1y != p2y:  # Krawędź nie jest pozioma
                            # Oblicz punkt przecięcia promienia z krawędzią
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        # Jeśli krawędź jest pionowa lub punkt jest na lewo od przecięcia
                        if p1x == p2x or x <= xinters:
                            inside = not inside  # Przełącz stan (wewnątrz/na zewnątrz)

            # Przejdź do następnej krawędzi
            p1x, p1y = p2x, p2y

        return inside

    def get_contours_from_image(self, image):
        """
        Znajduje kontury w obrazie używając detekcji krawędzi Canny.

        Args:
            image (numpy.ndarray): Obraz wejściowy

        Returns:
            list: Lista konturów spełniających kryteria powierzchni
        """
        # Konwersja do skali szarości jeśli obraz jest kolorowy
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Rozmycie gaussowskie w celu redukcji szumu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detekcja krawędzi algorytmem Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Znajdź kontury na obrazie krawędzi
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtruj kontury według minimalnej powierzchni
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)  # Oblicz powierzchnię konturu
            if area > self.min_contour_area:  # Zachowaj tylko duże kontury
                filtered_contours.append(contour)

        return filtered_contours

    def place_points_on_contour(self, contour, n_points):
        """
        Rozmieszcza równomiernie punkty wzdłuż konturu.

        Args:
            contour (numpy.ndarray): Kontur jako tablica punktów
            n_points (int): Liczba punktów do rozmieszczenia

        Returns:
            list: Lista punktów równomiernie rozmieszczonych na konturze
        """
        # Przekształć kontur do tablicy punktów 2D
        contour = contour.reshape(-1, 2).astype(np.float32)

        # Oblicz długości segmentów między kolejnymi punktami
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))

        # Oblicz skumulowane długości
        cumulative = np.cumsum(distances)
        total_length = cumulative[-1]  # Całkowita długość konturu

        # Jeśli kontur ma zerową długość, zwróć pierwsze n punktów
        if total_length == 0:
            return [(int(p[0]), int(p[1])) for p in contour[:n_points]]

        # Oblicz odstęp między punktami
        spacing = total_length / n_points
        points = [contour[0]]  # Dodaj pierwszy punkt

        # Rozmieść pozostałe punkty równomiernie
        for i in range(1, n_points):
            target_distance = i * spacing  # Docelowa odległość od początku

            # Znajdź segment, w którym znajduje się punkt docelowy
            idx = np.searchsorted(cumulative, target_distance)

            # Zabezpieczenie przed przekroczeniem granic tablicy
            if idx >= len(distances):
                idx = len(distances) - 1
            if idx == 0:
                idx = 1

            # Interpoluj pozycję punktu wewnątrz segmentu
            ratio = (target_distance - cumulative[idx - 1]) / distances[idx] if distances[idx] != 0 else 0
            point = contour[idx - 1] + ratio * (contour[idx] - contour[idx - 1])
            points.append(point)

        # Konwertuj do liczb całkowitych i zwróć jako listę krotek
        return [(int(p[0]), int(p[1])) for p in points]

    def generate_interior_points(self, contour_points, img_shape):
        """
        Generuje punkty wewnątrz obszaru ograniczonego konturem.

        Args:
            contour_points (list): Lista punktów konturu
            img_shape (tuple): Kształt obrazu (wysokość, szerokość)

        Returns:
            list: Lista punktów wewnętrznych
        """
        # Sprawdź czy kontur ma wystarczającą liczbę punktów
        if len(contour_points) < 3:
            return []

        # Znajdź prostokąt obejmujący kontur
        contour_array = np.array(contour_points)
        min_x, min_y = np.min(contour_array, axis=0)
        max_x, max_y = np.max(contour_array, axis=0)

        # Oblicz wymiary prostokąta
        width = max_x - min_x
        height = max_y - min_y

        # Oblicz krok siatki na podstawie gęstości
        step_x = max(width // self.interior_density, 5)
        step_y = max(height // self.interior_density, 5)

        interior_points = []

        # Generuj punkty w regularnej siatce
        for y in range(int(min_y + step_y), int(max_y), int(step_y)):
            for x in range(int(min_x + step_x), int(max_x), int(step_x)):
                # Sprawdź czy punkt jest w granicach obrazu i wewnątrz konturu
                if (0 <= x < img_shape[1] and 0 <= y < img_shape[0] and
                        self.point_in_polygon((x, y), contour_points)):
                    interior_points.append((x, y))

        # Dodaj punkt centralny konturu
        center_x = int((min_x + max_x) / 2)
        center_y = int((min_y + max_y) / 2)
        if self.point_in_polygon((center_x, center_y), contour_points):
            interior_points.append((center_x, center_y))

        return interior_points

    def create_triangulation(self, contour_points, interior_points, img_shape):
        """
        Tworzy triangulację Delaunaya z punktów konturu i wewnętrznych.

        Args:
            contour_points (list): Punkty na konturze
            interior_points (list): Punkty wewnętrzne
            img_shape (tuple): Kształt obrazu

        Returns:
            list: Lista trójkątów jako krotek trzech punktów
        """
        # Sprawdź czy mamy wystarczającą liczbę punktów
        if len(contour_points) < 3:
            return []

        # Połącz wszystkie punkty
        all_points = contour_points + interior_points

        if len(all_points) < 3:
            return []

        try:
            # Konwertuj punkty do tablicy numpy
            points_array = np.array(all_points, dtype=np.float32)

            # Utwórz triangulację Delaunaya
            tri = Delaunay(points_array)

            triangles = []

            # Przetwórz każdy simplex (trójkąt) z triangulacji
            for simplex in tri.simplices:
                # Pobierz współrzędne wierzchołków trójkąta
                pt1 = tuple(map(int, points_array[simplex[0]]))
                pt2 = tuple(map(int, points_array[simplex[1]]))
                pt3 = tuple(map(int, points_array[simplex[2]]))

                # Sprawdź czy trójkąt jest w granicach obrazu
                rect = (0, 0, img_shape[1], img_shape[0])
                if (self.rect_contains(rect, pt1) and
                        self.rect_contains(rect, pt2) and
                        self.rect_contains(rect, pt3)):

                    # Sprawdź czy środek trójkąta jest wewnątrz konturu
                    center = ((pt1[0] + pt2[0] + pt3[0]) // 3,
                              (pt1[1] + pt2[1] + pt3[1]) // 3)
                    if self.point_in_polygon(center, contour_points):
                        triangles.append((pt1, pt2, pt3))

            return triangles

        except Exception as e:
            print(f"Błąd triangulacji: {e}")
            return []

    def process_image(self, image):
        """
        Przetwarza obraz - znajduje kontury i tworzy triangulację.

        Args:
            image (numpy.ndarray): Obraz wejściowy

        Returns:
            numpy.ndarray: Obraz z nałożoną triangulacją
        """
        # Znajdź kontury w obrazie
        contours = self.get_contours_from_image(image)

        # Utwórz kopię obrazu do rysowania wyników
        image_result = image.copy()

        # Przetwórz każdy znaleziony kontur
        for contour in contours:
            # Rozmieść punkty na konturze
            contour_points = self.place_points_on_contour(contour, self.n_contour_points)

            # Pomiń kontury z niewystarczającą liczbą punktów
            if len(contour_points) < 3:
                continue

            # Generuj punkty wewnętrzne
            interior_points = self.generate_interior_points(contour_points, image.shape)

            # Utwórz triangulację
            triangles = self.create_triangulation(contour_points, interior_points, image.shape)

            # Rysuj triangulację na obrazie
            self.draw_delaunay_triangles(image_result, triangles, self.triangle_color)

            # Rysuj punkty wewnętrzne (dwa razy - prawdopodobnie błąd w oryginalnym kodzie)
            for point in interior_points:
                self.draw_point(image_result, point, self.interior_color, 2)

            # To jest duplikacja powyższej pętli - prawdopodobnie błąd
            for point in interior_points:
                self.draw_point(image_result, point, self.interior_color, 2)

        return image_result

    def process_from_file(self, image_path):
        """
        Przetwarza obraz z pliku.

        Args:
            image_path (str): Ścieżka do pliku obrazu

        Returns:
            tuple: (oryginalny_obraz, przetworzony_obraz) lub (None, None) w przypadku błędu
        """
        import os

        # Sprawdź czy plik istnieje
        if not os.path.exists(image_path):
            print(f"Plik {image_path} nie istnieje!")
            return None, None

        # Wczytaj obraz z pliku
        image = cv2.imread(image_path)
        if image is None:
            print("Nie można wczytać obrazu!")
            return None, None

        # Przetwórz obraz
        result = self.process_image(image)

        return image, result