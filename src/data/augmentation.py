"""
Training Augmentations for OCR.

RandAugment-style augmentation pipeline with OCR-specific transforms.
All transforms preserve text readability while adding visual diversity.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Callable


def jpeg_compress(img: Image.Image, quality_range: tuple = (30, 90)) -> Image.Image:
    """Apply JPEG compression artifacts."""
    import io
    quality = random.randint(*quality_range)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(img: Image.Image, sigma_range: tuple = (0.5, 2.0)) -> Image.Image:
    """Apply Gaussian blur."""
    sigma = random.uniform(*sigma_range)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def salt_pepper_noise(img: Image.Image, amount: float = 0.02) -> Image.Image:
    """Add salt and pepper noise."""
    arr = np.array(img)
    # Salt
    num_salt = int(amount * arr.size / 2)
    coords = tuple(np.random.randint(0, d, num_salt) for d in arr.shape[:2])
    arr[coords[0], coords[1]] = 255
    # Pepper
    coords = tuple(np.random.randint(0, d, num_salt) for d in arr.shape[:2])
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)


def brightness_jitter(img: Image.Image, factor_range: tuple = (0.7, 1.3)) -> Image.Image:
    """Adjust brightness."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast_jitter(img: Image.Image, factor_range: tuple = (0.7, 1.3)) -> Image.Image:
    """Adjust contrast."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def rotation(img: Image.Image, max_angle: float = 3.0) -> Image.Image:
    """Slight rotation."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))


def perspective_warp(img: Image.Image, strength: float = 0.05) -> Image.Image:
    """Simulate camera angle perspective distortion."""
    w, h = img.size
    s = strength

    # Random perspective corners
    tl = (random.uniform(0, s * w), random.uniform(0, s * h))
    tr = (w - random.uniform(0, s * w), random.uniform(0, s * h))
    br = (w - random.uniform(0, s * w), h - random.uniform(0, s * h))
    bl = (random.uniform(0, s * w), h - random.uniform(0, s * h))

    coeffs = _find_perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [tl, tr, br, bl],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def _find_perspective_coeffs(src, dst):
    """Compute perspective transform coefficients."""
    import numpy as np
    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([s for pair in src for s in pair], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


def shadow_gradient(img: Image.Image, direction: str = "random") -> Image.Image:
    """Apply uneven lighting / shadow gradient."""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    if direction == "random":
        direction = random.choice(["left", "right", "top", "bottom"])

    gradient = np.ones((h, w), dtype=np.float32)
    strength = random.uniform(0.3, 0.7)

    if direction == "left":
        gradient *= np.linspace(strength, 1.0, w)[np.newaxis, :]
    elif direction == "right":
        gradient *= np.linspace(1.0, strength, w)[np.newaxis, :]
    elif direction == "top":
        gradient *= np.linspace(strength, 1.0, h)[:, np.newaxis]
    elif direction == "bottom":
        gradient *= np.linspace(1.0, strength, h)[:, np.newaxis]

    arr = arr * gradient[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def downsample_upsample(img: Image.Image, scale_range: tuple = (0.5, 0.8)) -> Image.Image:
    """Simulate low resolution by downsampling then upsampling."""
    w, h = img.size
    scale = random.uniform(*scale_range)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def invert(img: Image.Image) -> Image.Image:
    """Invert colors (simulate white-on-black or negative)."""
    return ImageOps.invert(img)


def motion_blur(img: Image.Image) -> Image.Image:
    """Horizontal motion blur — simulates camera shake or scanner movement."""
    size = random.choice([3, 5])
    kernel = [0] * (size * size)
    mid = size // 2
    for i in range(size):
        kernel[mid * size + i] = 1
    return img.filter(ImageFilter.Kernel(size=(size, size), kernel=kernel, scale=size, offset=0))


def elastic_distortion(img: Image.Image) -> Image.Image:
    """Simulate ink bleed, paper warping, character deformation."""
    from scipy.ndimage import gaussian_filter as gf, map_coordinates as mc

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    strength = random.uniform(1.0, 3.0)
    dx = gf(np.random.randn(h, w) * strength, sigma=3)
    dy = gf(np.random.randn(h, w) * strength, sigma=3)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
    y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)

    result = np.zeros_like(arr)
    for c in range(3):
        result[:, :, c] = mc(arr[:, :, c], [y_new, x_new], order=1, mode='reflect')

    return Image.fromarray(result.astype(np.uint8))


def random_erasing(img: Image.Image) -> Image.Image:
    """Random rectangular cutout — simulates occlusion, stains, tape."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Erase 1-3 small rectangles
    for _ in range(random.randint(1, 3)):
        rh = random.randint(2, max(3, h // 3))
        rw = random.randint(2, max(3, w // 6))
        ry = random.randint(0, h - rh)
        rx = random.randint(0, w - rw)
        # Fill with random color (paper-like)
        fill = random.randint(180, 255)
        arr[ry:ry+rh, rx:rx+rw] = fill

    return Image.fromarray(arr)


def color_jitter(img: Image.Image) -> Image.Image:
    """Shift hue/saturation — simulates yellowed paper, colored ink, scanner color drift."""
    # Convert to HSV, jitter, convert back
    arr = np.array(img, dtype=np.float32)

    # Tint towards yellow/brown (old paper) or blue (photocopy)
    tint = random.choice(['yellow', 'blue', 'none'])
    if tint == 'yellow':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * random.uniform(1.0, 1.1), 0, 255)  # boost red
        arr[:, :, 1] = np.clip(arr[:, :, 1] * random.uniform(0.95, 1.05), 0, 255)  # slight green
        arr[:, :, 2] = np.clip(arr[:, :, 2] * random.uniform(0.85, 0.95), 0, 255)  # reduce blue
    elif tint == 'blue':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * random.uniform(0.9, 0.95), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * random.uniform(1.0, 1.1), 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def erosion_dilation(img: Image.Image) -> Image.Image:
    """Make text thinner or thicker — simulates ink weight variation."""
    if random.random() < 0.5:
        # Erosion (thinner text)
        return img.filter(ImageFilter.MinFilter(size=3))
    else:
        # Dilation (thicker text)
        return img.filter(ImageFilter.MaxFilter(size=3))


def paper_texture(img: Image.Image) -> Image.Image:
    """Add paper grain noise — simulates scanned paper texture."""
    arr = np.array(img, dtype=np.float32)
    intensity = random.uniform(3, 15)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# All available augmentation transforms
AUGMENT_OPS: list[Callable] = [
    # Basic image quality
    jpeg_compress,
    gaussian_blur,
    salt_pepper_noise,
    brightness_jitter,
    contrast_jitter,
    downsample_upsample,
    # Geometric
    rotation,
    perspective_warp,
    # Real-world degradation
    shadow_gradient,
    motion_blur,
    color_jitter,
    paper_texture,
    erosion_dilation,
    random_erasing,
]

def paper_warp(img: Image.Image) -> Image.Image:
    """Simulate paper curling/warping — no scipy needed.

    Uses a sinusoidal displacement to create a wavy distortion,
    like a page that's not flat on the scanner.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Horizontal wave (paper curling left-right)
    amplitude = random.uniform(1.0, 3.0)
    frequency = random.uniform(1.0, 3.0)
    phase = random.uniform(0, 2 * np.pi)

    result = np.zeros_like(arr)
    for y in range(h):
        shift = int(amplitude * np.sin(2 * np.pi * frequency * y / h + phase))
        for x in range(w):
            src_x = min(max(x + shift, 0), w - 1)
            result[y, x] = arr[y, src_x]

    return Image.fromarray(result)


def spot_light(img: Image.Image) -> Image.Image:
    """Simulate a spotlight or desk lamp on part of the image.

    Creates a radial brightness falloff from a random point,
    like a phone flashlight or desk lamp illuminating unevenly.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random light source position
    cx = random.uniform(0.1, 0.9) * w
    cy = random.uniform(0.1, 0.9) * h
    radius = random.uniform(0.3, 0.8) * max(w, h)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    # Bright at center, dim at edges
    light = 1.0 - 0.5 * np.clip(dist / radius, 0, 1)
    arr = arr * light[:, :, np.newaxis]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def flash_glare(img: Image.Image) -> Image.Image:
    """Simulate phone camera flash washing out part of the text.

    Creates a bright white spot that fades outward — common artifact
    when photographing documents with flash.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random glare position (usually off-center)
    cx = random.uniform(0.2, 0.8) * w
    cy = random.uniform(0.2, 0.8) * h
    radius = random.uniform(0.15, 0.4) * max(w, h)
    intensity = random.uniform(0.3, 0.7)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    # Gaussian glare
    glare = intensity * np.exp(-0.5 * (dist / radius) ** 2)
    arr = arr + glare[:, :, np.newaxis] * 255

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert to grayscale and back to RGB.

    Forces the model to recognize text by shape alone, not color.
    The standard luminance formula (0.299R + 0.587G + 0.114B) works
    for 99% of cases. Red-on-green edge cases are rare in real OCR.
    """
    return img.convert("L").convert("RGB")


AUGMENT_OPS.extend([paper_warp, spot_light, flash_glare, to_grayscale])

# elastic_distortion needs scipy — add only if available
try:
    import scipy.ndimage  # noqa: F401
    AUGMENT_OPS.append(elastic_distortion)
except ImportError:
    pass


class RandAugmentOCR:
    """RandAugment-style augmentation for OCR training.

    Randomly applies N transforms from the pool, each with
    random intensity within its configured range.
    """

    def __init__(self, n_ops: int = 2, p: float = 0.5):
        """
        Args:
            n_ops: Number of augmentation ops to apply per image.
            p: Probability of applying augmentation at all.
        """
        self.n_ops = n_ops
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations.

        Args:
            img: PIL Image (RGB).

        Returns:
            Augmented PIL Image.
        """
        if random.random() > self.p:
            return img

        ops = random.sample(AUGMENT_OPS, min(self.n_ops, len(AUGMENT_OPS)))
        for op in ops:
            img = op(img)

        return img
