import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
from PIL import Image, ImageFilter
from skimage import feature, measure, color, filters


def detect_image_type(image_path):
    filename = os.path.basename(image_path).lower()

    if "duch" in filename:
        return "clockwise"
    else:
        return "standard"


def create_structural_grid(points, image_type):
    if len(points) < 2:
        return []

    use_clockwise = image_type == "clockwise"
    grid_points = []

    for i in range(len(points) - 1):
        a = np.array([points[i][0], points[i][1]])
        b = np.array([points[i + 1][0], points[i + 1][1]])

        ab_vector = b - a
        ab_length = np.linalg.norm(ab_vector)

        if use_clockwise:
            perpendicular = np.array([ab_vector[1], -ab_vector[0]]) / ab_length
        else:
            perpendicular = np.array([-ab_vector[1], ab_vector[0]]) / ab_length

        height = (math.sqrt(3) / 2) * ab_length

        c = ((a + b) / 2) + perpendicular * height
        grid_points.append((int(c[0]), int(c[1])))

    return grid_points


def get_contour(image_path, use_skimage=True):
    img = Image.open(image_path)
    img_array = np.array(img)
    image_type = detect_image_type(image_path)

    if use_skimage:
        if len(img_array.shape) == 3:
            gray = color.rgb2gray(img_array)
        else:
            gray = img_array / 255.0

        blurred = filters.gaussian(gray, sigma=1)
        edges = feature.canny(blurred, sigma=2)
        contours = measure.find_contours(edges, 0.5)
    else:

        gray_img = img.convert('L')

        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.SMOOTH)

        edges_array = np.array(edges)
        threshold = filters.threshold_otsu(edges_array)
        binary = edges_array > threshold
        contours = measure.find_contours(binary, 0.5)

    return img_array, contours, image_type


def filter_contours(contours, img_shape, image_type):
    border_margin = 5
    img_height, img_width = img_shape[:2]
    filtered_contours = []

    for contour in contours:
        is_border_contour = False
        for point in contour:
            y, x = point
            if (y < border_margin or y >= img_height - border_margin or
                    x < border_margin or x >= img_width - border_margin):
                is_border_contour = True
                break

        if not is_border_contour:
            filtered_contours.append(contour)

    return filtered_contours if filtered_contours else contours


def place_points_on_contour(contour, n_points):
    perimeter = 0
    for i in range(len(contour) - 1):
        p1 = contour[i]
        p2 = contour[i + 1]
        perimeter += math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    distance = perimeter / n_points

    points = []
    accumulated_distance = 0
    for i in range(len(contour) - 1):
        p1 = contour[i]
        p2 = contour[i + 1]
        segment_distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        while accumulated_distance + segment_distance >= distance:
            ratio = (distance - accumulated_distance) / segment_distance
            y = int(p1[0] + ratio * (p2[0] - p1[0]))
            x = int(p1[1] + ratio * (p2[1] - p1[1]))
            points.append((x, y))

            accumulated_distance = 0
            segment_distance = segment_distance - (distance - accumulated_distance)

        accumulated_distance += segment_distance

    return points, perimeter


def draw_points(img_array, points, color):
    result_img = img_array.copy()

    for point in points:
        x, y = point
        if 0 <= y < result_img.shape[0] and 0 <= x < result_img.shape[1]:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if (dy ** 2 + dx ** 2) <= 5:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < result_img.shape[0] and 0 <= nx < result_img.shape[1]:
                            if len(result_img.shape) == 3:
                                result_img[ny, nx] = color
                            else:
                                result_img[ny, nx] = 255 if sum(color) > 0 else 0

    return result_img


def is_point_on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if val == 0:
        return 0
    return 1 if val > 0 else 2


def do_segments_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and is_point_on_segment(p1, q1, p2): return True
    if o2 == 0 and is_point_on_segment(p1, q2, p2): return True
    if o3 == 0 and is_point_on_segment(q1, p1, q2): return True
    if o4 == 0 and is_point_on_segment(q1, p2, q2): return True

    return False


def line_intersects_contour(p1, p2, contour):
    p1_yx = (p1[1], p1[0])
    p2_yx = (p2[1], p2[0])

    for i in range(len(contour) - 1):
        q1 = contour[i]
        q2 = contour[i + 1]

        if do_segments_intersect(p1_yx, p2_yx, q1, q2):
            return True

    return False


def triangle_intersects_contour(p1, p2, p3, contour):
    return (line_intersects_contour(p1, p2, contour) or
            line_intersects_contour(p2, p3, contour) or
            line_intersects_contour(p3, p1, contour))


def find_neighbors(point, grid_points, radius):
    neighbors = []
    for p in grid_points:
        dist = math.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)
        if dist <= radius and dist > 0:
            neighbors.append(p)
    return neighbors


def process_image(image_path, n_points=20, use_skimage=False, show_result=True):
    """
    Przetwarza pojedynczy obraz i tworzy siatkę strukturalną
    """
    if not os.path.exists(image_path):
        print(f"Plik {image_path} nie istnieje!")
        return None, [], []

    print(f"Przetwarzanie obrazu: {image_path}")

    img_array, contours, image_type = get_contour(image_path, use_skimage)

    if not contours:
        print("Nie znaleziono konturów w obrazie!")
        return img_array, [], []

    filtered_contours = filter_contours(contours, img_array.shape, image_type)
    main_contour = max(filtered_contours, key=len)
    points, perimeter = place_points_on_contour(main_contour, n_points)
    grid_points = create_structural_grid(points, image_type)

    # Przygotowanie obrazu wynikowego
    result_img = img_array.copy()
    for i in range(len(main_contour)):
        y, x = main_contour[i]
        if 0 <= int(y) < result_img.shape[0] and 0 <= int(x) < result_img.shape[1]:
            if len(result_img.shape) == 3:
                result_img[int(y), int(x)] = [0, 0, 0]
            else:
                result_img[int(y), int(x)] = 0

    result_img = draw_points(result_img, points, [255, 0, 0])
    result_img = draw_points(result_img, grid_points, [0, 0, 255])

    # Wizualizacja
    if show_result and len(points) >= 2:
        plt.figure(figsize=(12, 10))
        plt.imshow(result_img)

        # Rysowanie linii między punktami konturu
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1)

        # Rysowanie siatki strukturalnej
        for i in range(len(grid_points)):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            x3, y3 = grid_points[i]


            plt.plot([x1, x3], [y1, y3], 'g-', linewidth=1)
            plt.plot([x2, x3], [y2, y3], 'g-', linewidth=1)

        method = "scikit-image" if use_skimage else "PIL"
        plt.title(
            f"{os.path.basename(image_path)} - Siatka strukturalna ({method})\nDługość konturu: {perimeter:.2f} - {len(points)} punktów")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return result_img, points, grid_points


def print_usage():
    """
    Wyświetla instrukcję użycia
    """
    print("Użycie:")
    print("  python script.py <nazwa_pliku> [opcje]")
    print()
    print("Argumenty:")
    print("  <nazwa_pliku>     - ścieżka do pliku obrazu (JPG, PNG, itp.)")
    print()
    print("Opcje:")
    print("  --points N        - liczba punktów do umieszczenia (domyślnie: 20)")
    print("  --skimage         - użyj scikit-image zamiast PIL do detekcji krawędzi")
    print("  --no-show         - nie wyświetlaj wizualizacji")
    print()
    print("Przykłady:")
    print("  python script.py moj_obraz.jpg")
    print("  python script.py obraz.png --points 50 --skimage")
    print("  python script.py zdjecie.jpg --points 30 --no-show")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Parsowanie argumentów
    args = sys.argv[1:]
    image_path = args[0]
    n_points = 20
    use_skimage = False
    show_result = True

    # Parsowanie opcji
    i = 1
    while i < len(args):
        if args[i] == '--points' and i + 1 < len(args):
            try:
                n_points = int(args[i + 1])
                if n_points <= 0:
                    print("Liczba punktów musi być większa od 0!")
                    sys.exit(1)
                i += 2
            except ValueError:
                print("Nieprawidłowa liczba punktów!")
                sys.exit(1)
        elif args[i] == '--skimage':
            use_skimage = True
            i += 1
        elif args[i] == '--no-show':
            show_result = False
            i += 1
        else:
            print(f"Nieznana opcja: {args[i]}")
            print_usage()
            sys.exit(1)

    # Sprawdzenie czy plik istnieje
    if not os.path.exists(image_path):
        print(f"Błąd: Plik '{image_path}' nie istnieje!")
        sys.exit(1)

    # Przetwarzanie obrazu
    try:
        result_img, points, grid_points = process_image(image_path, n_points, use_skimage, show_result)

        if result_img is not None:
            print("Przetwarzanie zakończone pomyślnie!")
        else:
            print("Wystąpił błąd podczas przetwarzania obrazu.")

    except Exception as e:
        print(f"Błąd: {str(e)}")
        sys.exit(1)