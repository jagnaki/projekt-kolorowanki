import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter
from skimage import feature, measure, color, filters

def detect_image_type(image_path):
    return "clockwise" if "duch" in os.path.basename(image_path).lower() else "standard"

def create_structural_grid(points, image_type):
    if len(points) < 2:
        return []

    sign = -1 if image_type == "standard" else 1
    grid_points = []

    for i in range(len(points) - 1):
        a, b = np.array(points[i]), np.array(points[i + 1])
        ab = b - a
        ab_len = np.linalg.norm(ab)
        perp = sign * np.array([ab[1], -ab[0]]) / ab_len
        c = (a + b) / 2 + perp * (np.sqrt(3) / 2 * ab_len)
        grid_points.append(tuple(map(int, c)))

    return grid_points

def get_contour(image_path, use_skimage=True):
    img = Image.open(image_path)
    img_array = np.array(img)

    if use_skimage:
        gray = color.rgb2gray(img_array) if img_array.ndim == 3 else img_array / 255.0
        edges = feature.canny(filters.gaussian(gray, sigma=1), sigma=2)
    else:
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.SMOOTH)
        edges = np.array(edges)
        edges = edges > filters.threshold_otsu(edges)

    contours = measure.find_contours(edges, 0.5)
    return img_array, contours, detect_image_type(image_path)

def filter_contours(contours, shape):
    h, w = shape[:2]
    margin = 5
    return [c for c in contours if not any((y < margin or y >= h - margin or x < margin or x >= w - margin) for y, x in c)] or contours

def place_points_on_contour(contour, n_points):
    contour = np.array(contour)
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative = np.cumsum(distances)
    total = cumulative[-1]
    spacing = total / n_points
    points = [contour[0]]

    for i in range(1, n_points):
        target = i * spacing
        idx = np.searchsorted(cumulative, target)
        ratio = (target - cumulative[idx - 1]) / distances[idx]
        point = contour[idx - 1] + ratio * (contour[idx] - contour[idx - 1])
        points.append(point)

    return [(int(x[1]), int(x[0])) for x in points], total

def draw_points(img_array, points, color):
    result = img_array.copy()
    for x, y in points:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx**2 + dy**2 <= 5:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < result.shape[0] and 0 <= nx < result.shape[1]:
                        result[ny, nx] = color if result.ndim == 3 else 255
    return result

def process_image(image_path, n_points=20, use_skimage=False, show_result=True):
    if not os.path.exists(image_path):
        print(f"Plik {image_path} nie istnieje!")
        return None, [], []

    img_array, contours, image_type = get_contour(image_path, use_skimage)
    if not contours:
        print("Brak konturów.")
        return img_array, [], []

    contour = max(filter_contours(contours, img_array.shape), key=len)
    points, perimeter = place_points_on_contour(contour, n_points)
    grid_points = create_structural_grid(points, image_type)

    result_img = draw_points(img_array, points, [255, 0, 0])
    result_img = draw_points(result_img, grid_points, [0, 0, 255])

    if show_result:
        plt.imshow(result_img)
        for i in range(len(points) - 1):
            plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'r-')
            c = grid_points[i]
            plt.plot([points[i][0], c[0]], [points[i][1], c[1]], 'g-')
            plt.plot([points[i+1][0], c[0]], [points[i+1][1], c[1]], 'g-')
        plt.title(f"{os.path.basename(image_path)} - Siatka strukturalna\nDługość konturu: {perimeter:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return result_img, points, grid_points

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Analiza obrazu i tworzenie siatki strukturalnej")
    parser.add_argument("image_path", help="Ścieżka do pliku graficznego")
    parser.add_argument("--points", type=int, default=20, help="Liczba punktów na konturze")
    parser.add_argument("--skimage", action="store_true", help="Użyj scikit-image")
    parser.add_argument("--no-show", action="store_true", help="Nie pokazuj wyników")
    args = parser.parse_args()

    process_image(args.image_path, args.points, args.skimage, not args.no_show)
