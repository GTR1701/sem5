import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from PIL import Image, ImageTk

# ---------- pomocnicze ----------

def to_gray(image):
    # luminancja wg Rec.601
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float64)

def as_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)

def convolve(image, kernel):
    """Konwolucja 2D z odbiciem brzegów."""
    kernel = np.asarray(kernel, dtype=np.float64)
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    result = np.zeros_like(image, dtype=np.float64)
    for kernel_row in range(kernel_height):
        for kernel_col in range(kernel_width):
            result += kernel[kernel_row, kernel_col] * padded[
                kernel_row:kernel_row + image.shape[0],
                kernel_col:kernel_col + image.shape[1]
            ]
    return result

def gauss_kernel(sigma):
    radius = max(1, int(round(3 * sigma)))
    positions = np.arange(-radius, radius + 1)
    weights = np.exp(-(positions ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()
    return np.outer(weights, weights)

def box_kernel(size):
    return np.ones((size, size)) / (size * size)

def median_filter(image, size):
    radius = size // 2
    padded = np.pad(image, radius, mode='reflect')
    height, width = image.shape
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-1, -2))[:height, :width]

def threshold(image, threshold_value):
    return np.where(image >= threshold_value, 255.0, 0.0)

def otsu(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = histogram.sum()
    probabilities = histogram / total_pixels
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))
    global_mean = cumulative_mean[-1]
    between_class_variance_denominator = cumulative_probability * (1 - cumulative_probability)
    with np.errstate(divide='ignore', invalid='ignore'):
        between_class_variance = (
            global_mean * cumulative_probability - cumulative_mean
        ) ** 2 / between_class_variance_denominator
    between_class_variance[~np.isfinite(between_class_variance)] = 0
    return int(np.argmax(between_class_variance))

def sobel(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = kernel_x.T
    gradient_x = convolve(image, kernel_x)
    gradient_y = convolve(image, kernel_y)
    return np.hypot(gradient_x, gradient_y)

def canny(image, low_threshold, high_threshold):
    smoothed = convolve(image, gauss_kernel(1.0))
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_x, gradient_y = convolve(smoothed, kernel_x), convolve(smoothed, kernel_x.T)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = (np.rad2deg(np.arctan2(gradient_y, gradient_x)) + 180) % 180

    height, width = image.shape
    suppressed = np.zeros_like(magnitude)
    # kierunek skwantowany do 0°, 45°, 90°, 135°
    direction = np.round(angle / 45).astype(int) % 4
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            current_direction = direction[row, col]
            if current_direction == 0:
                neighbour_a, neighbour_b = magnitude[row, col - 1], magnitude[row, col + 1]
            elif current_direction == 1:
                neighbour_a, neighbour_b = magnitude[row - 1, col + 1], magnitude[row + 1, col - 1]
            elif current_direction == 2:
                neighbour_a, neighbour_b = magnitude[row - 1, col], magnitude[row + 1, col]
            else:
                neighbour_a, neighbour_b = magnitude[row - 1, col - 1], magnitude[row + 1, col + 1]
            if magnitude[row, col] >= neighbour_a and magnitude[row, col] >= neighbour_b:
                suppressed[row, col] = magnitude[row, col]

    strong_edges = suppressed >= high_threshold
    weak_edges = suppressed >= low_threshold
    # histereza: rozrost silnych krawędzi na słabe sąsiadów
    edges = strong_edges.copy()
    changed = True
    while changed:
        grown = edges.copy()
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                grown |= np.roll(np.roll(edges, row_shift, 0), col_shift, 1)
        new_edges = grown & weak_edges
        changed = not np.array_equal(new_edges, edges)
        edges = new_edges
    return edges.astype(np.float64) * 255

def add_gauss_noise(image, sigma):
    return image + np.random.normal(0, sigma, image.shape)

def add_salt_pepper(image, amount):
    noisy = image.copy()
    random_values = np.random.rand(*image.shape)
    noisy[random_values < amount / 2] = 0
    noisy[random_values > 1 - amount / 2] = 255
    return noisy

def add_poisson(image, scale):
    lam = np.clip(image, 0, 255) * scale
    return np.random.poisson(lam) / scale

def mse(image_a, image_b):
    return float(np.mean((image_a.astype(np.float64) - image_b.astype(np.float64)) ** 2))

def psnr(image_a, image_b):
    error = mse(image_a, image_b)
    return float('inf') if error == 0 else 10 * np.log10(255 ** 2 / error)

# ---------- GUI ----------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Przetwarzanie obrazów")
        self.original = None      # obraz źródłowy (float RGB lub gray)
        self.history = []         # lista (nazwa, obraz)
        self.gray_mode = tk.BooleanVar(value=False)

        toolbar = ttk.Frame(root); toolbar.pack(fill='x', padx=6, pady=4)
        ttk.Button(toolbar, text="Wczytaj", command=self.load).pack(side='left')
        ttk.Button(toolbar, text="Cofnij", command=self.undo).pack(side='left', padx=4)
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side='left')
        ttk.Checkbutton(toolbar, text="Skala szarości", variable=self.gray_mode).pack(side='left', padx=8)

        self.threshold_value = tk.IntVar(value=1)
        self.low_or_sigma = tk.IntVar(value=1)
        self.high_or_window = tk.IntVar(value=1)
        self.sliders = ttk.Frame(root); self.sliders.pack(fill='x', padx=6)
        self.slider("Próg", self.threshold_value, 0, 255)
        self.slider("Sigma / niskie", self.low_or_sigma, 1, 100)
        self.slider("Wysokie / okno", self.high_or_window, 1, 255)

        operations_bar = ttk.Frame(root); operations_bar.pack(fill='x', padx=6, pady=4)
        for button_text, operation in [
            ("Szarość", self.op_gray), ("Progowa", self.op_thresh), ("Otsu", self.op_otsu),
            ("Sobel", self.op_sobel), ("Canny", self.op_canny),
            ("Szum Gauss", self.op_gauss_noise), ("Szum P&S", self.op_sp_noise),
            ("Szum Poisson", self.op_poisson_noise),
            ("Filtr uśred.", self.op_mean), ("Filtr Gauss", self.op_gauss_filter),
            ("Filtr med.", self.op_median),
        ]:
            ttk.Button(operations_bar, text=button_text, command=operation).pack(side='left', padx=2, pady=2)

        self.metrics_text = tk.StringVar(value="MSE: -   PSNR: -")
        ttk.Label(root, textvariable=self.metrics_text).pack(anchor='w', padx=6)

        view = ttk.Frame(root); view.pack(fill='both', expand=True)
        self.before_label = ttk.Label(view); self.before_label.pack(side='left', expand=True)
        self.after_label = ttk.Label(view); self.after_label.pack(side='left', expand=True)

    def slider(self, name, variable, minimum, maximum):
        slider_frame = ttk.Frame(self.sliders); slider_frame.pack(side='left', padx=6)
        ttk.Label(slider_frame, text=name).pack(side='left')
        ttk.Scale(slider_frame, from_=minimum, to=maximum, variable=variable,
                  orient='horizontal', length=120).pack(side='left')

    # --- stan ---
    def current(self):
        return self.history[-1][1] if self.history else self.original

    def show(self, image):
        array = as_uint8(image)
        if array.ndim == 2:
            pil_image = Image.fromarray(array, mode='L')
        else:
            pil_image = Image.fromarray(array, mode='RGB')
        pil_image.thumbnail((480, 480))
        return ImageTk.PhotoImage(pil_image)

    def refresh(self):
        if self.original is None:
            return
        previous_image = self.history[-2][1] if len(self.history) > 1 else self.original
        current_image = self.current()
        self.before_photo = self.show(previous_image)
        self.after_photo = self.show(current_image)
        self.before_label.config(image=self.before_photo)
        self.after_label.config(image=self.after_photo)
        if len(self.history) > 0:
            self.metrics_text.set(
                f"MSE: {mse(self.original, current_image):.2f}   "
                f"PSNR: {psnr(self.original, current_image):.2f} dB"
            )
        else:
            self.metrics_text.set("MSE: -   PSNR: -")

    def push(self, image):
        self.history.append((None, as_uint8(image)))
        self.refresh()

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.png *.jpg *.jpeg *.bmp *.ppm")])
        if not path:
            return
        self.original = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)
        self.history = []
        self.refresh()

    def undo(self):
        if self.history:
            self.history.pop()
            self.refresh()

    def reset(self):
        self.history = []
        self.refresh()

    # --- operacje ---
    def target(self):
        """Zwraca obraz do przetworzenia (kanał po kanale)."""
        image = self.current()
        if self.gray_mode.get() or image.ndim == 2:
            return to_gray(image) if image.ndim == 3 else image
        return image

    def apply(self, operation):
        image = self.target()
        if image.ndim == 3:
            result = np.stack([operation(image[..., channel]) for channel in range(3)], axis=-1)
        else:
            result = operation(image)
        self.push(result)

    def op_gray(self):
        current_image = self.current()
        self.push(to_gray(current_image) if current_image.ndim == 3 else current_image)

    def op_thresh(self):
        self.apply(lambda channel: threshold(channel, self.threshold_value.get()))

    def op_otsu(self):
        image = self.target()
        gray_image = to_gray(image) if image.ndim == 3 else image
        otsu_threshold = otsu(gray_image)
        self.push(threshold(gray_image, otsu_threshold))
        self.show_histogram(gray_image, otsu_threshold)

    def op_sobel(self):
        self.apply(sobel)

    def op_canny(self):
        low_threshold = self.low_or_sigma.get()
        high_threshold = max(self.high_or_window.get(), low_threshold + 1)
        self.apply(lambda channel: canny(channel, low_threshold, high_threshold))

    def op_gauss_noise(self):
        sigma = self.low_or_sigma.get()
        self.apply(lambda channel: add_gauss_noise(channel, sigma))

    def op_sp_noise(self):
        amount = self.low_or_sigma.get() / 100
        self.apply(lambda channel: add_salt_pepper(channel, amount))

    def op_poisson_noise(self):
        scale = self.low_or_sigma.get() / 10
        self.apply(lambda channel: add_poisson(channel, scale))

    def op_mean(self):
        window_size = max(1, self.high_or_window.get() // 10 * 2 + 1)
        self.apply(lambda channel: convolve(channel, box_kernel(window_size)))

    def op_gauss_filter(self):
        sigma = self.low_or_sigma.get() / 10
        self.apply(lambda channel: convolve(channel, gauss_kernel(sigma)))

    def op_median(self):
        window_size = max(1, self.high_or_window.get() // 10 * 2 + 1)
        self.apply(lambda channel: median_filter(channel, window_size))

    def show_histogram(self, gray_image, otsu_threshold):
        window = tk.Toplevel(self.root)
        window.title(f"Histogram, próg Otsu = {otsu_threshold}")
        canvas = tk.Canvas(window, width=512, height=200, bg='white')
        canvas.pack()
        histogram, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        histogram = histogram / histogram.max() * 190
        for index, bar_height in enumerate(histogram):
            canvas.create_line(index * 2, 200, index * 2, 200 - bar_height, fill='black')
        canvas.create_line(otsu_threshold * 2, 0, otsu_threshold * 2, 200, fill='red')

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
