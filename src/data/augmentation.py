"""
Training Augmentations for OCR.

Simulates real-world conditions for text in the wild:
  - Documents: scans, photocopies, aged paper, folds
  - Phone captures: perspective, blur, shadows, fingers
  - Signs & banners: outdoor lighting, weather
  - Handwriting: ink variation, smudges

RandAugment-style: randomly applies N transforms per image.
Each op is designed to degrade but never destroy readability.
"""

import io
import math
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from typing import Callable


# =========================================================================
# Image quality
# =========================================================================

def jpeg_compress(img: Image.Image) -> Image.Image:
    """JPEG artifacts — scanned/shared documents."""
    quality = random.randint(20, 80)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def blur(img: Image.Image) -> Image.Image:
    """Gaussian or motion blur — out of focus, camera shake."""
    if random.random() < 0.6:
        # Gaussian
        sigma = random.uniform(0.3, 1.8)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    else:
        # Motion blur (horizontal or vertical)
        size = random.choice([3, 5])
        kernel = [0] * (size * size)
        mid = size // 2
        horizontal = random.random() < 0.7
        for i in range(size):
            if horizontal:
                kernel[mid * size + i] = 1
            else:
                kernel[i * size + mid] = 1
        return img.filter(ImageFilter.Kernel(size=(size, size), kernel=kernel, scale=size, offset=0))


def low_resolution(img: Image.Image) -> Image.Image:
    """Low res — distant photo, thumbnail, cheap camera."""
    w, h = img.size
    scale = random.uniform(0.4, 0.75)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def photocopy(img: Image.Image) -> Image.Image:
    """Photocopy degradation — contrast boost + speckle noise."""
    arr = np.array(img, dtype=np.float32)
    # Boost contrast
    mean = arr.mean()
    arr = mean + (arr - mean) * random.uniform(1.1, 1.3)
    # Speckle
    noise = np.random.normal(0, random.uniform(3, 8), arr.shape)
    arr = arr + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Lighting & exposure
# =========================================================================

def exposure_jitter(img: Image.Image) -> Image.Image:
    """Brightness + contrast variation — auto-exposure, different lighting."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
    return img


def uneven_lighting(img: Image.Image) -> Image.Image:
    """Shadow gradient or spotlight — desk lamp, window, overhead light."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    strength = random.uniform(0.4, 0.75)

    style = random.choice(["gradient", "radial"])
    if style == "gradient":
        direction = random.choice(["left", "right", "top", "bottom"])
        if direction in ("left", "right"):
            g = np.linspace(strength, 1.0, w) if direction == "left" else np.linspace(1.0, strength, w)
            mask = g[np.newaxis, :]
        else:
            g = np.linspace(strength, 1.0, h) if direction == "top" else np.linspace(1.0, strength, h)
            mask = g[:, np.newaxis]
        arr = arr * mask[:, :, np.newaxis] if mask.ndim == 2 else arr * np.expand_dims(mask, -1)
    else:
        # Radial spotlight
        cx = random.uniform(0.15, 0.85) * w
        cy = random.uniform(0.15, 0.85) * h
        radius = random.uniform(0.4, 0.9) * max(w, h)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        light = 1.0 - (1.0 - strength) * np.clip(dist / radius, 0, 1)
        arr = arr * light[:, :, np.newaxis]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def glare(img: Image.Image) -> Image.Image:
    """Flash or surface reflection — phone flash, laminated surface.

    Kept mild — brightens an area but text remains readable.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx = random.uniform(0.2, 0.8) * w
    cy = random.uniform(0.2, 0.8) * h
    radius = random.uniform(0.15, 0.4) * max(w, h)
    intensity = random.uniform(0.15, 0.35)  # mild — never wash out text
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    glow = intensity * np.exp(-0.5 * (dist / radius) ** 2)
    arr = arr + glow[:, :, np.newaxis] * 255
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def striped_shadow(img: Image.Image) -> Image.Image:
    """Shadows from blinds or fingers — parallel dark bands."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    num_stripes = random.randint(2, 5)
    stripe_width = random.uniform(0.04, 0.1) * h
    darkness = random.uniform(0.5, 0.75)

    dim = h
    for _ in range(num_stripes):
        pos = random.uniform(0, dim)
        coords = np.arange(dim)
        stripe = np.exp(-0.5 * ((coords - pos) / stripe_width) ** 2)
        mask = 1.0 - (1.0 - darkness) * stripe
        arr = arr * mask[:, np.newaxis, np.newaxis]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Geometric distortion
# =========================================================================

def rotation(img: Image.Image) -> Image.Image:
    """Slight tilt — not-quite-straight scan or photo."""
    angle = random.uniform(-4, 4)
    bg = tuple(random.randint(220, 255) for _ in range(3))
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg)


def perspective_warp(img: Image.Image) -> Image.Image:
    """Camera angle — phone held at angle to surface."""
    w, h = img.size
    s = random.uniform(0.03, 0.07)
    tl = (random.uniform(0, s * w), random.uniform(0, s * h))
    tr = (w - random.uniform(0, s * w), random.uniform(0, s * h))
    br = (w - random.uniform(0, s * w), h - random.uniform(0, s * h))
    bl = (random.uniform(0, s * w), h - random.uniform(0, s * h))
    coeffs = _find_perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)], [tl, tr, br, bl])
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def _find_perspective_coeffs(src, dst):
    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([s for pair in src for s in pair], dtype=np.float64)
    return tuple(np.linalg.lstsq(A, B, rcond=None)[0].tolist())


def wave_distortion(img: Image.Image) -> Image.Image:
    """Paper curl, book spine, or baseline wobble — mild wave."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    amplitude = random.uniform(0.5, 2.0)
    frequency = random.uniform(0.5, 2.5)
    phase = random.uniform(0, 2 * math.pi)
    vertical = random.random() < 0.5

    result = np.zeros_like(arr)
    if vertical:
        for x in range(w):
            shift = int(amplitude * math.sin(2 * math.pi * frequency * x / w + phase))
            for y in range(h):
                src_y = min(max(y + shift, 0), h - 1)
                result[y, x] = arr[src_y, x]
    else:
        for y in range(h):
            shift = int(amplitude * math.sin(2 * math.pi * frequency * y / h + phase))
            for x in range(w):
                src_x = min(max(x + shift, 0), w - 1)
                result[y, x] = arr[y, src_x]

    return Image.fromarray(result)


# =========================================================================
# Document degradation
# =========================================================================

def bleed_through(img: Image.Image) -> Image.Image:
    """Reverse-side text showing through thin paper."""
    arr = np.array(img, dtype=np.float32)
    ghost = np.flip(arr, axis=1).copy()
    alpha = random.uniform(0.05, 0.15)
    arr = arr * (1 - alpha) + ghost * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fold_crease(img: Image.Image) -> Image.Image:
    """Fold line — dark line across the paper."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    horizontal = random.random() < 0.6
    if horizontal:
        pos = random.uniform(0.2, 0.8) * h
        width = random.uniform(1, 2.5)
        darkness = random.uniform(0.4, 0.7)
        coords = np.arange(h, dtype=np.float32)
        mask = 1.0 - (1.0 - darkness) * np.exp(-0.5 * ((coords - pos) / width) ** 2)
        arr = arr * mask[:, np.newaxis, np.newaxis]
    else:
        pos = random.uniform(0.2, 0.8) * w
        width = random.uniform(1, 2.5)
        darkness = random.uniform(0.4, 0.7)
        coords = np.arange(w, dtype=np.float32)
        mask = 1.0 - (1.0 - darkness) * np.exp(-0.5 * ((coords - pos) / width) ** 2)
        arr = arr * mask[np.newaxis, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def aged_document(img: Image.Image) -> Image.Image:
    """Aged paper with yellowing, fading, and grain.

    Combines color tint + edge fading + paper texture.
    Text stays readable — just looks old.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Yellow tint
    tint = random.uniform(0.05, 0.15)
    arr[:, :, 0] *= 1 + tint * 0.3
    arr[:, :, 1] *= 1 + tint * 0.15
    arr[:, :, 2] *= 1 - tint * 0.3

    # Slight edge fade
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    edge_dist = np.minimum(
        np.minimum(x_coords, w - 1 - x_coords) / w,
        np.minimum(y_coords, h - 1 - y_coords) / h,
    )
    fade = 1.0 - random.uniform(0.03, 0.1) * (1.0 - np.clip(edge_dist * 4, 0, 1))
    arr = arr * fade[:, :, np.newaxis]

    # Paper grain
    grain = np.random.normal(0, random.uniform(2, 6), arr.shape)
    arr = arr + grain

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def scanner_edge(img: Image.Image) -> Image.Image:
    """Dark edge from flatbed scanner — page not flush with glass."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    edge = random.choice(["left", "right", "top", "bottom"])
    width = random.uniform(0.05, 0.12)
    darkness = random.uniform(0.3, 0.55)

    if edge in ("left", "right"):
        coords = np.arange(w, dtype=np.float32) / w
        if edge == "right":
            coords = 1 - coords
        mask = darkness + (1 - darkness) * np.clip(coords / width, 0, 1)
        arr = arr * mask[np.newaxis, :, np.newaxis]
    else:
        coords = np.arange(h, dtype=np.float32) / h
        if edge == "bottom":
            coords = 1 - coords
        mask = darkness + (1 - darkness) * np.clip(coords / width, 0, 1)
        arr = arr * mask[:, np.newaxis, np.newaxis]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def water_stain(img: Image.Image) -> Image.Image:
    """Water or coffee ring stain — semi-transparent."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx = random.uniform(0.1, 0.9) * w
    cy = random.uniform(0.1, 0.9) * h
    radius = random.uniform(0.1, 0.3) * max(w, h)
    ring_width = radius * random.uniform(0.2, 0.4)

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    ring = np.exp(-0.5 * ((dist - radius) / ring_width) ** 2)

    stain_color = random.choice([
        np.array([200, 180, 120]),  # coffee
        np.array([180, 170, 150]),  # water
    ])
    intensity = random.uniform(0.08, 0.2)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - ring * intensity) + stain_color[c] * ring * intensity

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Ink & stroke variation
# =========================================================================

def stroke_variation(img: Image.Image) -> Image.Image:
    """Thinner or thicker strokes — pen pressure, ink amount, bleed."""
    if random.random() < 0.4:
        return img.filter(ImageFilter.MinFilter(size=3))  # thinner
    else:
        # Thicken via dilation, optionally blended for ink bleed effect
        dilated = img.filter(ImageFilter.MaxFilter(size=3))
        if random.random() < 0.5:
            return dilated
        # Blend for subtle ink bleed
        alpha = random.uniform(0.3, 0.6)
        arr = np.array(img, dtype=np.float32) * (1 - alpha) + np.array(dilated, dtype=np.float32) * alpha
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def smudge(img: Image.Image) -> Image.Image:
    """Ink smudge — localized blur from finger or wet ink."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cx = random.uniform(0.15, 0.85) * w
    cy = random.uniform(0.15, 0.85) * h
    radius = random.uniform(0.08, 0.2) * max(w, h)

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    mask = np.clip(1.0 - dist / radius, 0, 1)

    blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
    arr = arr * (1 - mask[:, :, np.newaxis]) + blurred * mask[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Noise & color
# =========================================================================

def noise(img: Image.Image) -> Image.Image:
    """Gaussian or salt-pepper noise — sensor noise, paper grain."""
    arr = np.array(img, dtype=np.float32)
    if random.random() < 0.6:
        # Gaussian (paper grain)
        intensity = random.uniform(3, 12)
        arr = arr + np.random.normal(0, intensity, arr.shape)
    else:
        # Salt & pepper (sensor)
        arr_int = arr.astype(np.uint8)
        amount = random.uniform(0.005, 0.02)
        num = int(amount * arr_int.size / 2)
        coords = tuple(np.random.randint(0, d, num) for d in arr_int.shape[:2])
        arr_int[coords[0], coords[1]] = 255
        coords = tuple(np.random.randint(0, d, num) for d in arr_int.shape[:2])
        arr_int[coords[0], coords[1]] = 0
        return Image.fromarray(arr_int)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def color_jitter(img: Image.Image) -> Image.Image:
    """Color cast — scanner drift, aged paper, monitor color."""
    arr = np.array(img, dtype=np.float32)
    tint = random.choice(['yellow', 'blue', 'warm', 'cool'])
    s = random.uniform(0.03, 0.1)
    if tint == 'yellow':
        arr[:, :, 0] *= 1 + s
        arr[:, :, 2] *= 1 - s
    elif tint == 'blue':
        arr[:, :, 2] *= 1 + s
        arr[:, :, 0] *= 1 - s * 0.5
    elif tint == 'warm':
        arr[:, :, 0] *= 1 + s
        arr[:, :, 1] *= 1 + s * 0.3
    elif tint == 'cool':
        arr[:, :, 1] *= 1 + s * 0.3
        arr[:, :, 2] *= 1 + s
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def to_grayscale(img: Image.Image) -> Image.Image:
    """Grayscale — B&W scanner, photocopy."""
    return img.convert("L").convert("RGB")


# =========================================================================
# Occlusion
# =========================================================================

def occlusion(img: Image.Image) -> Image.Image:
    """Partial occlusion — finger, tape, sticker, stamp.

    Rectangular or elliptical, semi-transparent.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 6 or w < 6:
        return img

    shape = random.choice(["rect", "ellipse"])
    color = random.choice([
        np.array([200, 180, 160]),  # finger
        np.array([200, 50, 50]),    # red stamp
        np.array([240, 240, 240]),  # white label
        np.array([50, 50, 50]),     # dark smudge
    ])
    opacity = random.uniform(0.15, 0.45)

    if shape == "rect":
        rh = random.randint(2, max(3, h // 3))
        rw = random.randint(2, max(3, w // 5))
        ry = random.randint(0, max(1, h - rh))
        rx = random.randint(0, max(1, w - rw))
        for c in range(3):
            arr[ry:ry+rh, rx:rx+rw, c] = arr[ry:ry+rh, rx:rx+rw, c] * (1 - opacity) + color[c] * opacity
    else:
        cx = random.uniform(0.1, 0.9) * w
        cy = random.uniform(0.1, 0.9) * h
        rx = random.uniform(0.05, 0.15) * w
        ry = random.uniform(0.05, 0.25) * h
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = ((x_coords - cx) / max(rx, 1)) ** 2 + ((y_coords - cy) / max(ry, 1)) ** 2
        mask = np.clip(1.0 - dist, 0, 1)
        for c in range(3):
            arr[:, :, c] = arr[:, :, c] * (1 - mask * opacity) + color[c] * mask * opacity

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Outdoor / weather
# =========================================================================

def weather_damage(img: Image.Image) -> Image.Image:
    """Sun fading on outdoor signs — mild bleaching from one direction."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    fade_strength = random.uniform(0.05, 0.15)
    fade_dir = random.choice(["top", "left"])
    if fade_dir == "top":
        gradient = np.linspace(1 - fade_strength, 1, h)[:, np.newaxis, np.newaxis]
    else:
        gradient = np.linspace(1 - fade_strength, 1, w)[np.newaxis, :, np.newaxis]
    arr = arr * gradient + (255 - arr) * (1 - gradient) * 0.3
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Elastic distortion (requires scipy)
# =========================================================================

def elastic_distortion(img: Image.Image) -> Image.Image:
    """Mild elastic warp — paper warping, flexible surface."""
    from scipy.ndimage import gaussian_filter as gf, map_coordinates as mc
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    strength = random.uniform(0.5, 2.0)
    dx = gf(np.random.randn(h, w) * strength, sigma=3)
    dy = gf(np.random.randn(h, w) * strength, sigma=3)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
    y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
    result = np.zeros_like(arr)
    for c in range(3):
        result[:, :, c] = mc(arr[:, :, c], [y_new, x_new], order=1, mode='reflect')
    return Image.fromarray(result.astype(np.uint8))


# =========================================================================
# Handwriting simulation
# =========================================================================

def variable_baseline(img: Image.Image) -> Image.Image:
    """Drifting baseline — handwriting doesn't follow straight lines.

    Applies a smooth random vertical displacement per column,
    simulating natural hand movement across the page.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Generate smooth random displacement (low-frequency noise)
    n_control = random.randint(3, 6)
    control_points = [random.uniform(-2.5, 2.5) for _ in range(n_control)]
    x_positions = np.linspace(0, w - 1, n_control)
    displacements = np.interp(np.arange(w), x_positions, control_points)

    result = np.full_like(arr, arr[0, 0])  # fill with top-left pixel (background)
    for x in range(w):
        shift = int(round(displacements[x]))
        for y in range(h):
            src_y = min(max(y + shift, 0), h - 1)
            result[y, x] = arr[src_y, x]

    return Image.fromarray(result)


def slant(img: Image.Image) -> Image.Image:
    """Random slant — handwriting typically leans left or right.

    Applies a horizontal shear transform. More natural than rotation
    for simulating handwriting angle.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    shear = random.uniform(-0.3, 0.3)  # negative = left lean, positive = right

    result = np.full_like(arr, arr[0, 0])
    for y in range(h):
        offset = int(shear * (y - h / 2))
        for x in range(w):
            src_x = x - offset
            if 0 <= src_x < w:
                result[y, x] = arr[y, src_x]

    return Image.fromarray(result)


def ink_fade(img: Image.Image) -> Image.Image:
    """Ink fading — pen running low, inconsistent ink flow.

    Applies a random gradient that partially fades the text,
    simulating uneven ink distribution.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Detect background color from corners
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    bg = np.median(corners, axis=0)

    # Random fade: either left-to-right, right-to-left, or patchy
    style = random.choice(["lr", "rl", "patchy"])
    if style == "lr":
        fade = np.linspace(1.0, random.uniform(0.3, 0.7), w)
    elif style == "rl":
        fade = np.linspace(random.uniform(0.3, 0.7), 1.0, w)
    else:
        # Patchy: random smooth fade
        n_pts = random.randint(4, 8)
        ctrl = [random.uniform(0.4, 1.0) for _ in range(n_pts)]
        x_pos = np.linspace(0, w - 1, n_pts)
        fade = np.interp(np.arange(w), x_pos, ctrl)

    # Blend toward background where fade is low
    fade = fade[np.newaxis, :, np.newaxis]  # (1, W, 1)
    arr = arr * fade + bg * (1 - fade)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def variable_stroke(img: Image.Image) -> Image.Image:
    """Variable stroke width — pen pressure changes across the word.

    Applies thin/thicken differently across horizontal regions,
    unlike stroke_variation which is uniform.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    if w < 12:
        return img

    # Split into 3-5 vertical strips, each gets different treatment.
    # At least one strip stays unchanged to preserve ink.
    n_strips = random.randint(3, 5)
    strip_w = w // n_strips
    result = arr.copy()
    keep_strip = random.randint(0, n_strips - 1)

    for i in range(n_strips):
        if i == keep_strip:
            continue  # preserve at least one strip
        x_start = i * strip_w
        x_end = min((i + 1) * strip_w, w)
        strip = Image.fromarray(arr[:, x_start:x_end])

        action = random.choice(["thin", "thick", "none"])
        if action == "thin":
            strip = strip.filter(ImageFilter.MinFilter(size=3))
        elif action == "thick":
            strip = strip.filter(ImageFilter.MaxFilter(size=3))

        result[:, x_start:x_end] = np.array(strip)

    return Image.fromarray(result)


def lined_paper(img: Image.Image) -> Image.Image:
    """Lined paper background — horizontal ruled lines behind text.

    Common in handwritten notes, forms, and notebooks.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    line_spacing = random.randint(6, 10)
    line_color = random.uniform(0.7, 0.9)  # light gray
    line_thickness = 1

    for y in range(0, h, line_spacing):
        for dy in range(line_thickness):
            if y + dy < h:
                arr[y + dy, :] = arr[y + dy, :] * line_color + \
                    np.array([200, 200, 230], dtype=np.float32) * (1 - line_color)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Op registry — 25 ops
# =========================================================================

AUGMENT_OPS: list[Callable] = [
    # Quality (4)
    jpeg_compress,
    blur,
    low_resolution,
    photocopy,
    # Lighting (4)
    exposure_jitter,
    uneven_lighting,
    glare,
    striped_shadow,
    # Geometric (3)
    rotation,
    perspective_warp,
    wave_distortion,
    # Document (5)
    bleed_through,
    fold_crease,
    aged_document,
    scanner_edge,
    water_stain,
    # Ink (2)
    stroke_variation,
    smudge,
    # Handwriting (5)
    variable_baseline,
    slant,
    ink_fade,
    variable_stroke,
    lined_paper,
    # Noise & color (3)
    noise,
    color_jitter,
    to_grayscale,
    # Occlusion (1)
    occlusion,
    # Outdoor (1)
    weather_damage,
]

# elastic_distortion needs scipy
try:
    import scipy.ndimage  # noqa: F401
    AUGMENT_OPS.append(elastic_distortion)
except ImportError:
    pass


class RandAugmentOCR:
    """RandAugment-style augmentation for OCR.

    Randomly applies N ops per image. Each op degrades
    realistically but preserves text readability.
    """

    def __init__(self, n_ops: int = 2, p: float = 0.5, ops: list[Callable] | None = None):
        self.n_ops = n_ops
        self.p = p
        self.ops = ops if ops is not None else AUGMENT_OPS

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        ops = random.sample(self.ops, min(self.n_ops, len(self.ops)))
        for op in ops:
            img = op(img)
        return img
