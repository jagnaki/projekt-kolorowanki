import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import os
import glob


class ImageTriangulation:
    def __init__(self):
        # Parametry triangulacji
        self.n_contour_points = 30
        self.interior_density = 8
        self.min_contour_area = 1000

        # Kolory
        self.triangle_color = (0, 0, 0)  # Czarne linie triangulacji
        self.contour_color = (0, 255, 0)
        self.interior_color = (255, 0, 0)

    def rect_contains(self, rect, point):
        """Sprawdza czy punkt jest wewnątrz prostokąta"""
        if point[0] < rect[0] or point[1] < rect[1]:
            return False
        if point[0] > rect[2] or point[1] > rect[3]:
            return False
        return True

    def draw_point(self, img, point, color, radius=2):
        """Rysuje punkt na obrazie"""
        cv2.circle(img, tuple(map(int, point)), radius, color, -1, cv2.LINE_AA)

    def draw_delaunay_triangles(self, img, triangles, color):
        """Rysuje trójkąty triangulacji Delaunaya"""
        for triangle in triangles:
            pt1, pt2, pt3 = triangle
            cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA)
            cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA)

    def point_in_polygon(self, point, polygon_points):
        """Sprawdza czy punkt jest wewnątrz wielokąta"""
        x, y = point
        n = len(polygon_points)
        inside = False

        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_contours_from_image(self, image):
        """Znajduje kontury w obrazie"""
        # Konwersja do skali szarości
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detekcja krawędzi
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Znajdź kontury
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtruj kontury po wielkości
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                filtered_contours.append(contour)

        return filtered_contours

    def place_points_on_contour(self, contour, n_points):
        """Rozmieszcza równomiernie punkty na konturze"""
        contour = contour.reshape(-1, 2).astype(np.float32)

        # Oblicz długości segmentów
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        cumulative = np.cumsum(distances)
        total_length = cumulative[-1]

        if total_length == 0:
            return [(int(p[0]), int(p[1])) for p in contour[:n_points]]

        # Rozmieść punkty równomiernie
        spacing = total_length / n_points
        points = [contour[0]]

        for i in range(1, n_points):
            target_distance = i * spacing
            idx = np.searchsorted(cumulative, target_distance)

            if idx >= len(distances):
                idx = len(distances) - 1
            if idx == 0:
                idx = 1

            ratio = (target_distance - cumulative[idx - 1]) / distances[idx] if distances[idx] != 0 else 0
            point = contour[idx - 1] + ratio * (contour[idx] - contour[idx - 1])
            points.append(point)

        return [(int(p[0]), int(p[1])) for p in points]

    def generate_interior_points(self, contour_points, img_shape):
        """Generuje punkty wewnątrz konturu"""
        if len(contour_points) < 3:
            return []

        contour_array = np.array(contour_points)
        min_x, min_y = np.min(contour_array, axis=0)
        max_x, max_y = np.max(contour_array, axis=0)

        width = max_x - min_x
        height = max_y - min_y
        step_x = max(width // self.interior_density, 5)
        step_y = max(height // self.interior_density, 5)

        interior_points = []

        for y in range(int(min_y + step_y), int(max_y), int(step_y)):
            for x in range(int(min_x + step_x), int(max_x), int(step_x)):
                if (0 <= x < img_shape[1] and 0 <= y < img_shape[0] and
                        self.point_in_polygon((x, y), contour_points)):
                    interior_points.append((x, y))

        # Dodaj punkt centralny
        center_x = int((min_x + max_x) / 2)
        center_y = int((min_y + max_y) / 2)
        if self.point_in_polygon((center_x, center_y), contour_points):
            interior_points.append((center_x, center_y))

        return interior_points

    def create_triangulation(self, contour_points, interior_points, img_shape):
        """Tworzy triangulację Delaunaya"""
        if len(contour_points) < 3:
            return []

        all_points = contour_points + interior_points

        if len(all_points) < 3:
            return []

        try:
            points_array = np.array(all_points, dtype=np.float32)
            tri = Delaunay(points_array)

            triangles = []
            for simplex in tri.simplices:
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
        """Przetwarza obraz"""
        # Znajdź kontury
        contours = self.get_contours_from_image(image)

        image_result = image.copy()

        for contour in contours:
            # Rozmieść punkty na konturze
            contour_points = self.place_points_on_contour(contour, self.n_contour_points)

            if len(contour_points) < 3:
                continue

            # Generuj punkty wewnętrzne
            interior_points = self.generate_interior_points(contour_points, image.shape)

            # Utwórz triangulację
            triangles = self.create_triangulation(contour_points, interior_points, image.shape)

            # Rysuj triangulację
            self.draw_delaunay_triangles(image_result, triangles, self.triangle_color)

            # Rysuj punkty wewnętrzne
            for point in interior_points:
                self.draw_point(image_result, point, self.interior_color, 2)

            # Rysuj punkty wewnętrzne
            for point in interior_points:
                self.draw_point(image_result, point, self.interior_color, 2)

        return image_result

    def process_from_file(self, image_path):
        """Przetwarza obraz z pliku"""
        if not os.path.exists(image_path):
            print(f"Plik {image_path} nie istnieje!")
            return None, None

        # Wczytaj obraz
        image = cv2.imread(image_path)
        if image is None:
            print("Nie można wczytać obrazu!")
            return None, None

        # Przetwórz obraz
        result = self.process_image(image)

        return image, result

    def display_results(self, original_image, processed_image):
        """Wyświetla wyniki porównania"""
        plt.figure(figsize=(15, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Oryginalny obraz")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Triangulacja Delaunaya\nPunkty konturu: {self.n_contour_points}, Gęstość: {self.interior_density}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def save_result(self, processed_image, output_path):
        """Zapisuje przetworzony obraz"""
        cv2.imwrite(output_path, processed_image)
        print(f"Wynik zapisany do: {output_path}")


class ColoringGame:
    def __init__(self):
        self.triangulator = ImageTriangulation()
        self.triangulator.n_contour_points = 30
        self.triangulator.interior_density = 8

        # Zakresy parametrów triangulacji
        self.min_contour_points = 10
        self.max_contour_points = 100
        self.min_interior_density = 2
        self.max_interior_density = 20

        # Kolory do wyboru
        self.colors = [
            (255, 0, 0),    # Czerwony
            (0, 255, 0),    # Zielony
            (0, 0, 255),    # Niebieski
            (255, 255, 0),  # Żółty
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Pomarańczowy
            (128, 0, 128),  # Fioletowy
            (165, 42, 42),  # Brązowy
            (0, 0, 0),      # Czarny
            (255, 255, 255) # Biały
        ]
        self.current_color_index = 0
        self.current_color = self.colors[self.current_color_index]

        # Dane obrazu
        self.original_image = None
        self.display_image = None
        self.triangles = []
        self.triangle_colors = {}  # Słownik do przechowywania kolorów trójkątów

        # Nazwy okien
        self.main_window = "Kolorowanka"
        self.color_window = "Wybór koloru"
        self.image_select_window = "Wybór obrazu"
        self.density_window = "Gęstość siatki"

        # Flagi
        self.running = True
        self.image_selected = False

    def find_image_files(self):
        """Znajduje pliki obrazów w katalogu"""
        image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
        return image_files

    def create_image_selection_window(self, image_files):
        """Tworzy okno wyboru obrazu"""
        if not image_files:
            print("Nie znaleziono plików obrazów!")
            return False

        # Utwórz okno wyboru obrazu
        window_height = 100
        window_width = 150 * len(image_files)
        selection_image = np.ones((window_height, window_width, 3), dtype=np.uint8) * 240

        # Wczytaj miniatury obrazów
        thumbnails = []
        for i, file in enumerate(image_files):
            img = cv2.imread(file)
            if img is not None:
                # Zmień rozmiar do miniatury
                thumbnail = cv2.resize(img, (120, 80))
                thumbnails.append(thumbnail)

                # Umieść miniaturę w oknie wyboru
                x_offset = i * 150 + 15
                selection_image[10:10+80, x_offset:x_offset+120] = thumbnail

                # Dodaj nazwę pliku
                cv2.putText(selection_image, os.path.basename(file), 
                           (x_offset, window_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.imshow(self.image_select_window, selection_image)

        # Funkcja obsługi kliknięcia myszy
        def select_image(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for i in range(len(image_files)):
                    x_start = i * 150 + 15
                    x_end = x_start + 120
                    if x_start <= x <= x_end and 10 <= y <= 90:
                        self.load_image(image_files[i])
                        cv2.destroyWindow(self.image_select_window)
                        self.image_selected = True
                        break

        cv2.setMouseCallback(self.image_select_window, select_image)

        # Czekaj na wybór obrazu
        while not self.image_selected and cv2.getWindowProperty(self.image_select_window, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(100)

        return self.image_selected

    def load_image(self, image_path):
        """Wczytuje obraz i tworzy triangulację"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Nie można wczytać obrazu: {image_path}")
            return False

        # Przetwórz obraz
        self.process_image()
        return True

    def process_image(self):
        """Przetwarza obraz i tworzy triangulację"""
        if self.original_image is None:
            return False

        # Znajdź kontury
        contours = self.triangulator.get_contours_from_image(self.original_image)

        # Utwórz kopię obrazu do wyświetlania
        self.display_image = self.original_image.copy()
        self.triangles = []

        for contour in contours:
            # Rozmieść punkty na konturze
            contour_points = self.triangulator.place_points_on_contour(contour, self.triangulator.n_contour_points)

            if len(contour_points) < 3:
                continue

            # Generuj punkty wewnętrzne
            interior_points = self.triangulator.generate_interior_points(contour_points, self.original_image.shape)

            # Utwórz triangulację
            triangles = self.triangulator.create_triangulation(contour_points, interior_points, self.original_image.shape)

            # Dodaj trójkąty do listy
            self.triangles.extend(triangles)

            # Rysuj triangulację
            self.triangulator.draw_delaunay_triangles(self.display_image, triangles, self.triangulator.triangle_color)

        # Inicjalizuj słownik kolorów trójkątów
        self.triangle_colors = {}

        return True

    def create_color_selection_window(self):
        """Tworzy okno wyboru koloru"""
        # Utwórz okno wyboru koloru
        color_image = np.ones((50, 50 * len(self.colors), 3), dtype=np.uint8) * 240

        # Narysuj próbki kolorów
        for i, color in enumerate(self.colors):
            # Konwersja z BGR na RGB dla wyświetlania
            cv2.rectangle(color_image, (i * 50, 0), ((i + 1) * 50, 50), color, -1)

            # Zaznacz aktualnie wybrany kolor
            if i == self.current_color_index:
                cv2.rectangle(color_image, (i * 50, 0), ((i + 1) * 50, 50), (0, 0, 0), 2)

        cv2.imshow(self.color_window, color_image)

        # Funkcja obsługi kliknięcia myszy
        def select_color(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                color_index = x // 50
                if 0 <= color_index < len(self.colors):
                    self.current_color_index = color_index
                    self.current_color = self.colors[self.current_color_index]
                    self.create_color_selection_window()  # Odśwież okno wyboru koloru

        cv2.setMouseCallback(self.color_window, select_color)

    def create_density_control_window(self):
        """Tworzy okno kontroli gęstości siatki"""
        # Utwórz okno kontroli gęstości
        window_width = 400
        window_height = 150
        density_image = np.ones((window_height, window_width, 3), dtype=np.uint8) * 240

        # Dodaj tytuły
        cv2.putText(density_image, "Punkty konturu:", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(density_image, "Gęstość wewnętrzna:", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Dodaj wartości
        cv2.putText(density_image, str(self.triangulator.n_contour_points), (350, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(density_image, str(self.triangulator.interior_density), (350, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Narysuj suwaki
        # Suwak dla punktów konturu
        slider_width = 250
        slider_x = 20
        slider_y_contour = 40
        slider_y_interior = 90

        cv2.rectangle(density_image, (slider_x, slider_y_contour), 
                     (slider_x + slider_width, slider_y_contour + 10), (200, 200, 200), -1)

        # Pozycja suwaka dla punktów konturu
        contour_pos = int(slider_x + (self.triangulator.n_contour_points - self.min_contour_points) * 
                         slider_width / (self.max_contour_points - self.min_contour_points))
        cv2.rectangle(density_image, (contour_pos - 5, slider_y_contour - 5), 
                     (contour_pos + 5, slider_y_contour + 15), (0, 0, 255), -1)

        # Suwak dla gęstości wewnętrznej
        cv2.rectangle(density_image, (slider_x, slider_y_interior), 
                     (slider_x + slider_width, slider_y_interior + 10), (200, 200, 200), -1)

        # Pozycja suwaka dla gęstości wewnętrznej
        interior_pos = int(slider_x + (self.triangulator.interior_density - self.min_interior_density) * 
                          slider_width / (self.max_interior_density - self.min_interior_density))
        cv2.rectangle(density_image, (interior_pos - 5, slider_y_interior - 5), 
                     (interior_pos + 5, slider_y_interior + 15), (0, 0, 255), -1)

        # Dodaj instrukcje
        cv2.putText(density_image, "Kliknij i przeciągnij suwaki, aby zmienić gęstość siatki", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow(self.density_window, density_image)

        # Zmienne do śledzenia stanu suwaków
        self.dragging_contour = False
        self.dragging_interior = False

        # Funkcja obsługi myszy
        def handle_density_control(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Sprawdź czy kliknięto suwak punktów konturu
                if slider_y_contour - 5 <= y <= slider_y_contour + 15 and slider_x - 5 <= x <= slider_x + slider_width + 5:
                    self.dragging_contour = True
                # Sprawdź czy kliknięto suwak gęstości wewnętrznej
                elif slider_y_interior - 5 <= y <= slider_y_interior + 15 and slider_x - 5 <= x <= slider_x + slider_width + 5:
                    self.dragging_interior = True

            elif event == cv2.EVENT_LBUTTONUP:
                # Zakończ przeciąganie i zastosuj zmiany
                if self.dragging_contour or self.dragging_interior:
                    self.dragging_contour = False
                    self.dragging_interior = False
                    # Przetworz obraz z nowymi parametrami
                    self.process_image()
                    cv2.imshow(self.main_window, self.display_image)

            elif event == cv2.EVENT_MOUSEMOVE:
                # Aktualizuj pozycję suwaka podczas przeciągania
                if self.dragging_contour:
                    # Ogranicz x do zakresu suwaka
                    slider_x_pos = max(slider_x, min(x, slider_x + slider_width))
                    # Oblicz nową wartość punktów konturu
                    ratio = (slider_x_pos - slider_x) / slider_width
                    new_contour_points = int(self.min_contour_points + ratio * (self.max_contour_points - self.min_contour_points))
                    # Aktualizuj wartość
                    self.triangulator.n_contour_points = new_contour_points
                    # Odśwież okno
                    self.create_density_control_window()

                elif self.dragging_interior:
                    # Ogranicz x do zakresu suwaka
                    slider_x_pos = max(slider_x, min(x, slider_x + slider_width))
                    # Oblicz nową wartość gęstości wewnętrznej
                    ratio = (slider_x_pos - slider_x) / slider_width
                    new_interior_density = int(self.min_interior_density + ratio * (self.max_interior_density - self.min_interior_density))
                    # Aktualizuj wartość
                    self.triangulator.interior_density = new_interior_density
                    # Odśwież okno
                    self.create_density_control_window()

        cv2.setMouseCallback(self.density_window, handle_density_control)

    def point_in_triangle(self, point, triangle):
        """Sprawdza czy punkt jest wewnątrz trójkąta"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        pt1, pt2, pt3 = triangle
        d1 = sign(point, pt1, pt2)
        d2 = sign(point, pt2, pt3)
        d3 = sign(point, pt3, pt1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def fill_triangle(self, img, triangle, color):
        """Wypełnia trójkąt kolorem"""
        pts = np.array([triangle[0], triangle[1], triangle[2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color)

        # Narysuj ponownie krawędzie trójkąta
        pt1, pt2, pt3 = triangle
        cv2.line(img, pt1, pt2, self.triangulator.triangle_color, 1, cv2.LINE_AA)
        cv2.line(img, pt2, pt3, self.triangulator.triangle_color, 1, cv2.LINE_AA)
        cv2.line(img, pt3, pt1, self.triangulator.triangle_color, 1, cv2.LINE_AA)

    def handle_click(self, event, x, y, flags, param):
        """Obsługuje kliknięcie myszy w oknie głównym"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Sprawdź, który trójkąt został kliknięty
            for i, triangle in enumerate(self.triangles):
                if self.point_in_triangle((x, y), triangle):
                    # Zapamiętaj kolor trójkąta
                    self.triangle_colors[i] = self.current_color

                    # Wypełnij trójkąt kolorem
                    self.fill_triangle(self.display_image, triangle, self.current_color)
                    cv2.imshow(self.main_window, self.display_image)
                    break

    def run(self):
        """Uruchamia grę kolorowanki"""
        # Znajdź pliki obrazów
        image_files = self.find_image_files()

        # Utwórz okno wyboru obrazu
        if not self.create_image_selection_window(image_files):
            print("Nie wybrano obrazu!")
            return

        # Utwórz okno wyboru koloru
        self.create_color_selection_window()

        # Utwórz okno kontroli gęstości siatki
        self.create_density_control_window()

        # Utwórz główne okno
        cv2.imshow(self.main_window, self.display_image)
        cv2.setMouseCallback(self.main_window, self.handle_click)

        # Główna pętla
        density_window_visible = True
        while self.running:
            key = cv2.waitKey(100)

            # Sprawdź czy okna są otwarte
            if cv2.getWindowProperty(self.main_window, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False

            # Obsługa klawiszy
            if key == 27:  # ESC
                self.running = False
            elif key == ord('r'):  # Reset
                self.display_image = self.original_image.copy()
                self.triangulator.draw_delaunay_triangles(self.display_image, self.triangles, self.triangulator.triangle_color)
                self.triangle_colors = {}
                cv2.imshow(self.main_window, self.display_image)
            elif key == ord('s'):  # Zapisz
                cv2.imwrite("kolorowanka_wynik.jpg", self.display_image)
                print("Zapisano wynik do: kolorowanka_wynik.jpg")
            elif key == ord('d'):  # Pokaż/ukryj okno gęstości
                if density_window_visible:
                    cv2.destroyWindow(self.density_window)
                    density_window_visible = False
                else:
                    self.create_density_control_window()
                    density_window_visible = True
            elif key == ord('+') or key == ord('='):  # Zwiększ gęstość siatki
                self.triangulator.n_contour_points = min(self.triangulator.n_contour_points + 5, self.max_contour_points)
                self.triangulator.interior_density = min(self.triangulator.interior_density + 1, self.max_interior_density)
                self.process_image()
                cv2.imshow(self.main_window, self.display_image)
                if density_window_visible:
                    self.create_density_control_window()
            elif key == ord('-') or key == ord('_'):  # Zmniejsz gęstość siatki
                self.triangulator.n_contour_points = max(self.triangulator.n_contour_points - 5, self.min_contour_points)
                self.triangulator.interior_density = max(self.triangulator.interior_density - 1, self.min_interior_density)
                self.process_image()
                cv2.imshow(self.main_window, self.display_image)
                if density_window_visible:
                    self.create_density_control_window()

        # Zamknij wszystkie okna
        cv2.destroyAllWindows()


def main():
    import argparse
    import sys

    # Sprawdź czy podano argumenty wiersza poleceń
    if len(sys.argv) > 1:
        # Tryb wiersza poleceń (kompatybilność wsteczna)
        parser = argparse.ArgumentParser(description="Triangulacja konturów obrazów")
        parser.add_argument("image", type=str, help="Ścieżka do obrazu")
        parser.add_argument("--points", type=int, default=25, help="Liczba punktów na konturze")
        parser.add_argument("--density", type=int, default=8, help="Gęstość punktów wewnętrznych")
        parser.add_argument("--output", type=str, help="Ścieżka zapisu wyniku")
        parser.add_argument("--no-display", action="store_true", help="Nie wyświetlaj wyników")

        args = parser.parse_args()

        # Utwórz triangulator
        triangulator = ImageTriangulation()
        triangulator.n_contour_points = args.points
        triangulator.interior_density = args.density

        # Przetwórz obraz
        original, result = triangulator.process_from_file(args.image)

        if original is not None and result is not None:
            # Wyświetl wyniki
            if not args.no_display:
                triangulator.display_results(original, result)

            # Zapisz wynik jeśli podano ścieżkę
            if args.output:
                triangulator.save_result(result, args.output)
        else:
            print("Błąd podczas przetwarzania obrazu")
    else:
        # Tryb gry kolorowanki
        print("Uruchamianie gry kolorowanki...")
        game = ColoringGame()
        game.run()


if __name__ == "__main__":
    main()
