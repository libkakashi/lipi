"""
Training Augmentations for OCR.

Simulates real-world conditions for text in the wild:
  - Documents: scans, photocopies, aged paper, folds
  - Phone captures: perspective, blur, shadows, fingers
  - Signs & banners: outdoor lighting, weather, textured backgrounds
  - Handwriting: ink variation, smudges
  - Scene text: colored/textured backgrounds, 3D shadows, screen artifacts
  - Detection artifacts: imperfect crops, loose/tight bounding boxes
  - Camera sensor: shot noise, read noise, aggressive blur

RandAugment-style: randomly applies N transforms per image.
Each op is designed to degrade but never destroy readability.
"""

import io
import math
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Callable


# =========================================================================
# Shared pixel helpers
# =========================================================================

def _luminance(arr: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an (H, W, 3) float array."""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _soft_bg_mask(arr: np.ndarray, blur_radius: float) -> np.ndarray:
    """Soft [0,1] background mask: bright pixels (top 60% luma) are background,
    with edges softened by a Gaussian blur of the given radius.
    """
    gray = _luminance(arr)
    threshold = np.percentile(gray, 40)
    bg_mask = (gray >= threshold).astype(np.float32)
    mask_img = Image.fromarray((bg_mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(mask_img, dtype=np.float32) / 255.0


def _bg_is_dark(arr: np.ndarray) -> bool:
    """True if the background (median of corner pixels) is dark.

    Ops that assume dark-text-on-light-bg (background replacement, text
    shadow, ink-darkening blends) should skip or flip when this is True.
    """
    h, w = arr.shape[:2]
    corners = np.stack([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]])
    lum = 0.299 * corners[:, 0] + 0.587 * corners[:, 1] + 0.114 * corners[:, 2]
    return float(np.median(lum)) < 110.0


def _border_color(img: Image.Image) -> tuple[int, int, int]:
    """Median color of the image border — used as fill for geometric ops.

    A random light fill on a dark/colored background paints bright wedges
    that never occur in real photos (and gives the model a synthetic cue).
    """
    arr = np.array(img)
    edges = np.concatenate([
        arr[0, :].reshape(-1, 3), arr[-1, :].reshape(-1, 3),
        arr[:, 0].reshape(-1, 3), arr[:, -1].reshape(-1, 3),
    ])
    return tuple(int(v) for v in np.median(edges, axis=0))


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
    """Gaussian, motion, or defocus blur — out of focus, camera shake.

    Severity clamped so text at h=32 stays readable (the old sigma range
    went to 3.5, which destroyed small text; and the old motion-blur path
    used ImageFilter.Kernel with size 7, which PIL rejects — it silently
    no-opped).
    """
    r = random.random()
    if r < 0.5:
        # Gaussian — out of focus
        sigma = random.uniform(0.3, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif r < 0.85:
        # Motion blur — camera shake, mostly horizontal
        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        arr = np.array(img, dtype=np.float32)
        length = random.randint(3, 9)
        axis = 1 if random.random() < 0.7 else 0
        arr = uniform_filter1d(arr, size=length, axis=axis)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:
        # Defocus — box blur approximates a disk kernel
        return img.filter(ImageFilter.BoxBlur(random.randint(1, 2)))


def low_resolution(img: Image.Image) -> Image.Image:
    """Low res — distant photo, thumbnail, cheap camera."""
    w, h = img.size
    scale = random.uniform(0.25, 0.75)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    # Use nearest for very aggressive downscale to simulate pixelation
    upsample = Image.NEAREST if scale < 0.35 else Image.BILINEAR
    return small.resize((w, h), upsample)


def binarize(img: Image.Image) -> Image.Image:
    """1-bit scan / fax — hard threshold, jagged edges, broken strokes.

    A large share of enterprise document input is bilevel (fax, TIFF G4,
    aggressive scanner presets): no anti-aliasing, thin strokes broken by
    the threshold, or dither speckle. Three variants:
      - global: one threshold for the page region
      - adaptive: local-mean threshold (window minus offset) — what real
        scanner binarization does, keeps text under uneven lighting
      - dither: error-diffusion speckle (fax halftone)
    Optionally erodes thin strokes and re-blurs slightly (re-scan of a
    binarized page).
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    style = random.choice(["global", "adaptive", "dither"])

    if style == "global":
        # Threshold between the ink and background modes (an arbitrary
        # percentile can land inside the ink mass on text-dense crops
        # and wipe the text entirely).
        lo, hi = np.percentile(gray, (10, 90))
        t = lo + (hi - lo) * random.uniform(0.35, 0.65)
        binary = (gray > t)
    elif style == "adaptive":
        radius = random.randint(4, 10)
        local_mean = np.array(
            Image.fromarray(gray.astype(np.uint8)).filter(
                ImageFilter.BoxBlur(radius)), dtype=np.float32)
        offset = random.uniform(2, 12)
        binary = (gray > local_mean - offset)
    else:  # dither
        one_bit = img.convert("L").convert("1")  # Floyd-Steinberg
        binary = np.array(one_bit, dtype=bool)

    out = np.where(binary, 255, 0).astype(np.uint8)
    result = Image.fromarray(out).convert("RGB")

    # Broken thin strokes: erode the ink a step
    if random.random() < 0.3:
        result = result.filter(ImageFilter.MaxFilter(size=3))
    # Re-scan softness on top of the hard edges
    if random.random() < 0.4:
        result = result.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))
    return result


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
        arr = arr * mask[:, :, np.newaxis]
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
    """Shadows from blinds or fingers — parallel soft dark bands.

    Bands run across the width for wide line crops (a blind/finger shadow
    falls across the line, not along it). The old version stacked 2-5
    bands along the 32px height, wiping out most of the text.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    across_width = w > 2 * h or (w >= h and random.random() < 0.8)
    dim = w if across_width else h
    num_stripes = random.randint(1, 3)
    stripe_width = random.uniform(0.02, 0.08) * dim
    darkness = random.uniform(0.6, 0.85)

    mask = np.ones(dim, dtype=np.float32)
    coords = np.arange(dim)
    for _ in range(num_stripes):
        pos = random.uniform(0, dim)
        stripe = np.exp(-0.5 * ((coords - pos) / max(stripe_width, 1.0)) ** 2)
        mask *= 1.0 - (1.0 - darkness) * stripe

    if across_width:
        arr = arr * mask[np.newaxis, :, np.newaxis]
    else:
        arr = arr * mask[:, np.newaxis, np.newaxis]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Geometric distortion
# =========================================================================

def rotation(img: Image.Image) -> Image.Image:
    """Slight tilt — not-quite-straight scan or photo."""
    angle = random.uniform(-4, 4)
    bg = _border_color(img)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=bg)


def perspective_warp(img: Image.Image) -> Image.Image:
    """Camera angle — phone held at angle to surface.

    More aggressive than before: real phone photos can have 15-20% warp
    when capturing text from steep angles (menus, signs, whiteboards).
    """
    w, h = img.size
    # 70% mild (3-10%), 30% aggressive (10-20%) for real phone angles
    if random.random() < 0.7:
        s = random.uniform(0.03, 0.10)
    else:
        s = random.uniform(0.10, 0.20)
    tl = (random.uniform(0, s * w), random.uniform(0, s * h))
    tr = (w - random.uniform(0, s * w), random.uniform(0, s * h))
    br = (w - random.uniform(0, s * w), h - random.uniform(0, s * h))
    bl = (random.uniform(0, s * w), h - random.uniform(0, s * h))
    coeffs = _find_perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)], [tl, tr, br, bl])
    bg = _border_color(img)
    result = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=bg)
    return result


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

    if vertical:
        shifts = (amplitude * np.sin(2 * np.pi * frequency * np.arange(w) / w + phase)).astype(np.intp)
        rows = np.clip(np.arange(h)[:, None] + shifts[None, :], 0, h - 1)
        result = arr[rows, np.arange(w)[None, :]]
    else:
        shifts = (amplitude * np.sin(2 * np.pi * frequency * np.arange(h) / h + phase)).astype(np.intp)
        cols = np.clip(np.arange(w)[None, :] + shifts[:, None], 0, w - 1)
        result = arr[np.arange(h)[:, None], cols]

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
    from scipy.ndimage import gaussian_filter, map_coordinates
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    strength = random.uniform(0.5, 2.0)
    dx = gaussian_filter(np.random.randn(h, w) * strength, sigma=3)
    dy = gaussian_filter(np.random.randn(h, w) * strength, sigma=3)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
    y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
    result = np.zeros_like(arr)
    for c in range(3):
        result[:, :, c] = map_coordinates(arr[:, :, c], [y_new, x_new], order=1, mode='reflect')
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

    shifts = np.round(displacements).astype(np.intp)
    rows = np.clip(np.arange(h)[:, None] + shifts[None, :], 0, h - 1)
    result = arr[rows, np.arange(w)[None, :]]

    return Image.fromarray(result)


def slant(img: Image.Image) -> Image.Image:
    """Random slant — handwriting typically leans left or right.

    Applies a horizontal shear transform. More natural than rotation
    for simulating handwriting angle.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    shear = random.uniform(-0.3, 0.3)  # negative = left lean, positive = right

    offsets = (shear * (np.arange(h) - h / 2)).astype(np.intp)
    src_x = np.arange(w)[None, :] - offsets[:, None]
    valid = (src_x >= 0) & (src_x < w)
    src_x = np.clip(src_x, 0, w - 1)
    result = arr[np.arange(h)[:, None], src_x]
    result[~valid] = arr[0, 0]

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
    """Lined paper background — ruled lines behind text.

    Real ruled lines appear roughly once per text height (the text sits
    between them), so a line crop shows a baseline rule and sometimes the
    rule of the line above. The old 6-10px spacing drew 3-5 lines straight
    through the text.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    line_color = random.uniform(0.6, 0.85)
    rule = np.array([200, 200, 230], dtype=np.float32)

    ys = [int(h * random.uniform(0.78, 0.97))]  # baseline rule
    if random.random() < 0.4:
        ys.append(int(h * random.uniform(0.02, 0.18)))  # rule of line above

    for y in ys:
        y = min(max(y, 0), h - 1)
        arr[y, :] = arr[y, :] * line_color + rule * (1 - line_color)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Background & texture (scene text simulation)
# =========================================================================

def textured_background(img: Image.Image) -> Image.Image:
    """Random textured background — simulate text on walls, signs, surfaces.

    Replaces near-white background pixels with a procedural texture,
    keeping the text (dark pixels) intact. This bridges the domain gap
    between clean synthetic renders and real scene text.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if _bg_is_dark(arr):
        return img  # mask assumes dark text on light bg

    # Background = bright pixels (top 60% luma), soft-edged to blend naturally
    bg_mask = _soft_bg_mask(arr, blur_radius=1.5)

    texture_type = random.choice([
        "solid_color", "gradient", "perlin_noise", "stripe", "checker"
    ])

    if texture_type == "solid_color":
        # Random solid color background
        color = np.array([random.randint(60, 240) for _ in range(3)], dtype=np.float32)
        texture = np.full_like(arr, color)

    elif texture_type == "gradient":
        # Color gradient background
        c1 = np.array([random.randint(40, 220) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(40, 220) for _ in range(3)], dtype=np.float32)
        if random.random() < 0.5:
            # Horizontal gradient
            t = np.linspace(0, 1, w)[np.newaxis, :, np.newaxis]
        else:
            # Vertical gradient
            t = np.linspace(0, 1, h)[:, np.newaxis, np.newaxis]
        texture = c1 * (1 - t) + c2 * t

    elif texture_type == "perlin_noise":
        # Procedural noise texture (multi-octave for realism)
        base_color = np.array([random.randint(80, 200) for _ in range(3)], dtype=np.float32)
        noise_amp = random.uniform(20, 60)
        # Low-frequency smooth noise via upscaled random
        small_h, small_w = max(2, h // 8), max(2, w // 8)
        noise_small = np.random.randn(small_h, small_w, 3).astype(np.float32)
        noise_img = Image.fromarray(((noise_small * 127 + 128).clip(0, 255)).astype(np.uint8))
        noise_img = noise_img.resize((w, h), Image.BILINEAR)
        noise_arr = np.array(noise_img, dtype=np.float32) - 128
        texture = base_color + noise_arr * (noise_amp / 127.0)

    elif texture_type == "stripe":
        # Striped pattern (signs, awnings, fabric)
        c1 = np.array([random.randint(60, 220) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(60, 220) for _ in range(3)], dtype=np.float32)
        freq = random.uniform(0.05, 0.2)
        if random.random() < 0.5:
            pattern = (np.sin(np.arange(w) * freq * 2 * math.pi) > 0).astype(np.float32)
            pattern = pattern[np.newaxis, :, np.newaxis]
        else:
            pattern = (np.sin(np.arange(h) * freq * 2 * math.pi) > 0).astype(np.float32)
            pattern = pattern[:, np.newaxis, np.newaxis]
        texture = c1 * pattern + c2 * (1 - pattern)

    else:  # checker
        c1 = np.array([random.randint(80, 220) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(80, 220) for _ in range(3)], dtype=np.float32)
        cell = random.randint(4, 12)
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((yy // cell + xx // cell) % 2).astype(np.float32)
        texture = c1 * checker[:, :, np.newaxis] + c2 * (1 - checker[:, :, np.newaxis])

    texture = np.clip(texture, 0, 255)
    # Composite: background pixels get texture, text pixels stay
    result = arr * (1 - bg_mask[:, :, np.newaxis]) + texture * bg_mask[:, :, np.newaxis]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def colored_background(img: Image.Image) -> Image.Image:
    """Colored or gradient background — text on colored paper, signs, labels.

    Simpler and faster than textured_background; just tints the background
    a random color without pattern.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if _bg_is_dark(arr):
        return img  # mask assumes dark text on light bg

    bg_mask = _soft_bg_mask(arr, blur_radius=1.0)

    # Random background color — biased toward realistic sign/label colors
    palette = [
        (255, 255, 0),    # yellow sign
        (0, 120, 200),    # blue sign
        (200, 50, 50),    # red sign
        (50, 160, 50),    # green sign
        (240, 200, 150),  # beige/cardboard
        (180, 180, 180),  # gray metal
        (100, 80, 60),    # brown/wood
        (255, 200, 200),  # pink
        (200, 220, 255),  # light blue
    ]
    color = np.array(random.choice(palette), dtype=np.float32)
    # Add slight variation
    color = color + np.random.uniform(-20, 20, 3)
    color = np.clip(color, 0, 255)

    bg = np.full_like(arr, color)
    result = arr * (1 - bg_mask[:, :, np.newaxis]) + bg * bg_mask[:, :, np.newaxis]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# =========================================================================
# Crop & boundary (imperfect text detection)
# =========================================================================

def partial_crop_with_transform(
        img: Image.Image) -> tuple[Image.Image, tuple[float, float]]:
    """partial_crop variant that also reports its x-geometry transform.

    Returns (img, (a, b)) where x_new = a * x_old + b maps source-image
    x coordinates to output-image x coordinates, so pixel-space labels
    (segment offsets, per-pixel group labels) can follow the content.
    """
    w, h = img.size
    if w < 8 or h < 8:
        return img, (1.0, 0.0)

    # Crop 5-15% from 1-2 random edges
    n_edges = random.choices([1, 2], weights=[0.6, 0.4])[0]
    edges = random.sample(["left", "right", "top", "bottom"], n_edges)

    left, top, right, bottom = 0, 0, w, h
    for edge in edges:
        if edge == "left":
            left = int(w * random.uniform(0.03, 0.12))
        elif edge == "right":
            right = w - int(w * random.uniform(0.03, 0.12))
        elif edge == "top":
            top = int(h * random.uniform(0.03, 0.15))
        elif edge == "bottom":
            bottom = h - int(h * random.uniform(0.03, 0.15))

    cropped = img.crop((left, top, right, bottom))
    # Resize back to original dimensions
    a = w / (right - left)
    return cropped.resize((w, h), Image.BILINEAR), (a, -left * a)


def partial_crop(img: Image.Image) -> Image.Image:
    """Partial character cropping — real text detectors give imperfect crops.

    Simulates bounding boxes that are slightly too tight, cutting off
    parts of characters at the edges. Common in IIIT5K, IC13, IC15.
    """
    return partial_crop_with_transform(img)[0]


def pad_with_border_with_transform(
        img: Image.Image) -> tuple[Image.Image, tuple[float, float]]:
    """pad_with_border variant that also reports its x-geometry transform.

    Returns (img, (a, b)) with x_new = a * x_old + b, matching
    partial_crop_with_transform's convention.
    """
    w, h = img.size
    pad_frac = random.uniform(0.03, 0.12)

    # Random padding amounts per side
    pad_l = int(w * random.uniform(0, pad_frac))
    pad_r = int(w * random.uniform(0, pad_frac))
    pad_t = int(h * random.uniform(0, pad_frac))
    pad_b = int(h * random.uniform(0, pad_frac))

    # Background color from image corners
    arr = np.array(img)
    bg = tuple(int(v) for v in np.median([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], axis=0))

    new_w = w + pad_l + pad_r
    new_h = h + pad_t + pad_b
    padded = Image.new("RGB", (new_w, new_h), bg)
    padded.paste(img, (pad_l, pad_t))
    # Resize back to original
    a = w / new_w
    return padded.resize((w, h), Image.BILINEAR), (a, pad_l * a)


def pad_with_border(img: Image.Image) -> Image.Image:
    """Add irregular padding/border — detector bbox larger than text.

    Opposite of partial_crop: simulates loose bounding boxes that include
    extra background around the text. Common in real detection pipelines.
    """
    return pad_with_border_with_transform(img)[0]


# =========================================================================
# Screen & display artifacts
# =========================================================================

def screen_artifacts(img: Image.Image) -> Image.Image:
    """Screen/display artifacts — moire patterns, LCD pixel grid, scanlines.

    Simulates text photographed from a screen (common in real-world OCR).
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    effect = random.choice(["moire", "scanline", "pixel_grid"])

    if effect == "moire":
        # Moire pattern from screen interference
        freq1 = random.uniform(0.2, 0.6)
        freq2 = random.uniform(0.2, 0.6)
        angle = random.uniform(0, math.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pattern = np.sin(freq1 * (xx * math.cos(angle) + yy * math.sin(angle)))
        pattern += np.sin(freq2 * (xx * math.cos(angle + 0.5) + yy * math.sin(angle + 0.5)))
        intensity = random.uniform(3, 12)
        arr = arr + pattern[:, :, np.newaxis] * intensity

    elif effect == "scanline":
        # Horizontal scanlines (CRT/interlaced display)
        spacing = random.choice([2, 3])
        darkness = random.uniform(0.85, 0.95)
        for y in range(0, h, spacing):
            arr[y, :] *= darkness

    else:  # pixel_grid
        # LCD sub-pixel grid (visible on phone photos of screens)
        spacing = random.choice([2, 3])
        darkness = random.uniform(0.90, 0.97)
        arr[::spacing, :] *= darkness
        arr[:, ::spacing] *= darkness

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Camera noise models
# =========================================================================

def camera_noise(img: Image.Image) -> Image.Image:
    """Realistic camera sensor noise — shot noise + read noise.

    More realistic than simple Gaussian noise. Shot noise (Poisson)
    is signal-dependent, read noise (Gaussian) is constant.
    Common in low-light phone photos of text.
    """
    arr = np.array(img, dtype=np.float32)

    style = random.choice(["shot", "read", "combined"])

    if style in ("shot", "combined"):
        # Shot noise (Poisson) — brighter pixels get more noise.
        # Gain floor matters: at 0.02 a white pixel is ~5 photons and the
        # image becomes unreadable rainbow noise (label noise for CTC).
        gain = random.uniform(0.08, 0.18)
        scaled = arr * gain
        noisy = np.random.poisson(np.clip(scaled, 0, 255).astype(np.float64))
        arr = noisy.astype(np.float32) / gain

    if style in ("read", "combined"):
        # Read noise (Gaussian, signal-independent)
        sigma = random.uniform(3, 10)
        arr = arr + np.random.normal(0, sigma, arr.shape)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# 3D text / embossed shadow
# =========================================================================

def text_shadow(img: Image.Image) -> Image.Image:
    """Shadow from 3D/embossed text — raised letters cast shadows.

    Common on signs, plaques, building text, car plates.
    Shifts a darkened copy of the text slightly to simulate cast shadow.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if _bg_is_dark(arr):
        return img  # text-mask heuristic assumes dark text on light bg

    # Detect text regions (dark areas)
    gray = _luminance(arr)
    threshold = np.percentile(gray, 30)
    text_mask = (gray < threshold).astype(np.float32)

    # Shadow parameters
    dx = random.choice([-2, -1, 1, 2])
    dy = random.choice([1, 2])
    shadow_intensity = random.uniform(0.15, 0.35)

    # Shift mask to create shadow
    shadow = np.zeros_like(text_mask)
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    shadow[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        text_mask[src_y_start:src_y_end, src_x_start:src_x_end]

    # Remove shadow where text already is (shadow only visible around text)
    shadow = shadow * (1 - text_mask)

    # Blur the shadow for softness
    shadow_img = Image.fromarray((shadow * 255).astype(np.uint8))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    shadow = np.array(shadow_img, dtype=np.float32) / 255.0

    # Apply shadow (darken)
    arr = arr * (1 - shadow[:, :, np.newaxis] * shadow_intensity)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Document decorations — underline, highlighter, table ruling
# =========================================================================

def text_decoration(img: Image.Image) -> Image.Image:
    """Underline or strikethrough — links, form fields, edits, emphasis.

    Text is height-normalized, so the baseline sits around 72-80% of the
    crop; underlines go just below it, strikethrough through the
    x-height band. Spans the full line or a word-sized sub-span.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if _bg_is_dark(arr):
        color = np.array([random.randint(170, 255)] * 3, dtype=np.float32)
    else:
        color = np.array(random.choice([
            (random.randint(0, 60),) * 3,                    # ink
            (30, 60, random.randint(150, 220)),              # link blue
            (random.randint(150, 220), 30, 30),              # red edit
        ]), dtype=np.float32)

    if random.random() < 0.7:
        y = int(h * random.uniform(0.76, 0.90))   # underline
    else:
        y = int(h * random.uniform(0.42, 0.58))   # strikethrough
    thickness = random.choice([1, 1, 2])

    if random.random() < 0.5:
        x0, x1 = 0, w                              # whole line
    else:
        span = random.uniform(0.2, 0.7)            # one word / phrase
        x0 = int(random.uniform(0, 1 - span) * w)
        x1 = min(w, x0 + max(8, int(span * w)))

    alpha = random.uniform(0.7, 1.0)
    ys = slice(max(0, y), min(h, y + thickness))
    arr[ys, x0:x1] = arr[ys, x0:x1] * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def highlighter(img: Image.Image) -> Image.Image:
    """Highlighter marker band — multiply blend so ink stays dark."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if _bg_is_dark(arr):
        return img  # markers don't read on dark backgrounds

    color = np.array(random.choice([
        (255, 235, 60),    # yellow
        (170, 255, 120),   # green
        (255, 160, 200),   # pink
        (120, 230, 255),   # cyan
        (255, 200, 90),    # orange
    ]), dtype=np.float32) / 255.0

    if random.random() < 0.5:
        x0, x1 = 0, w
    else:
        span = random.uniform(0.25, 0.8)
        x0 = int(random.uniform(0, 1 - span) * w)
        x1 = min(w, x0 + max(8, int(span * w)))
    y0 = int(h * random.uniform(0.0, 0.12))
    y1 = int(h * random.uniform(0.85, 1.0))

    strength = random.uniform(0.55, 0.95)
    band = arr[y0:y1, x0:x1]
    arr[y0:y1, x0:x1] = band * (1 - strength) + band * color[None, None, :] * strength
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def table_rules(img: Image.Image) -> Image.Image:
    """Table/form ruling — cell borders at crop edges, occasionally a
    column rule through the text. Ubiquitous in enterprise documents;
    adjacent_line_clutter covers neighboring text but not ruling lines.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 8 or w < 16:
        return img
    dark_bg = _bg_is_dark(arr)
    color = np.array([random.randint(160, 230) if dark_bg
                      else random.randint(20, 110)] * 3, dtype=np.float32)
    alpha = random.uniform(0.6, 1.0)

    def vline(x, t):
        x = max(0, min(w - t, x))
        arr[:, x:x + t] = arr[:, x:x + t] * (1 - alpha) + color * alpha

    def hline(y, t):
        y = max(0, min(h - t, y))
        arr[y:y + t, :] = arr[y:y + t, :] * (1 - alpha) + color * alpha

    drew = False
    if random.random() < 0.65:   # left cell border
        vline(int(w * random.uniform(0, 0.03)), random.choice([1, 2]))
        drew = True
    if random.random() < 0.65:   # right cell border
        vline(int(w * random.uniform(0.97, 1.0)), random.choice([1, 2]))
        drew = True
    if random.random() < 0.25:   # column rule through the crop
        vline(int(w * random.uniform(0.25, 0.75)), 1)
        drew = True
    if random.random() < 0.5:    # row separator above
        hline(int(h * random.uniform(0, 0.06)), random.choice([1, 2]))
        drew = True
    if random.random() < 0.5:    # row separator below
        hline(int(h * random.uniform(0.92, 0.99)), random.choice([1, 2]))
        drew = True
    if not drew:
        vline(0, 1)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# =========================================================================
# Polarity & layout context
# =========================================================================

def polarity_invert(img: Image.Image) -> Image.Image:
    """Light-on-dark text — dark signage, screens in dark mode, LED boards,
    engraved/embossed metal. Without this the model never sees inverted
    polarity, which is a large share of real scene text.
    """
    arr = 255.0 - np.array(img, dtype=np.float32)
    # Slight brightness pull so it isn't always a perfect negative
    if random.random() < 0.5:
        arr *= random.uniform(0.8, 1.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def adjacent_line_clutter(img: Image.Image) -> Image.Image:
    """Slivers of neighboring text lines at the top/bottom edge.

    Real crops from dense documents include the descenders of the line
    above and/or the ascenders of the line below. Approximated by pasting
    scaled, shifted slivers of the image's own text at the edges.
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 12 or w < 16:
        return img
    dark_bg = _bg_is_dark(arr)
    out = arr.copy()

    edges = random.choices([["top"], ["bottom"], ["top", "bottom"]],
                           weights=[0.4, 0.4, 0.2], k=1)[0]
    for edge in edges:
        sliver_h = max(2, int(h * random.uniform(0.08, 0.22)))
        scale = random.uniform(0.7, 1.1)
        src = img.resize((max(8, int(w * scale)), max(8, int(h * scale))),
                         Image.BILINEAR)
        src_arr = np.array(src, dtype=np.float32)
        sh, sw = src_arr.shape[:2]
        sliver_h = min(sliver_h, sh)
        if edge == "top":
            band = src_arr[sh - sliver_h:]      # descenders of line above
        else:
            band = src_arr[:sliver_h]           # ascenders of line below
        band = np.roll(band, random.randint(0, sw), axis=1)
        reps = int(np.ceil(w / band.shape[1]))
        band = np.tile(band, (1, reps, 1))[:, :w]

        rows = slice(0, sliver_h) if edge == "top" else slice(h - sliver_h, h)
        # Ink wins: min-blend on light bg, max-blend on dark bg
        if dark_bg:
            out[rows] = np.maximum(out[rows], band)
        else:
            out[rows] = np.minimum(out[rows], band)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =========================================================================
# Op registry — 34 ops
# Excluded (handwriting-specific, used via style op lists in generate.py):
# variable_baseline, slant, ink_fade, variable_stroke, lined_paper,
# wave_distortion, bleed_through
# =========================================================================

AUGMENT_OPS: list[Callable] = [
    # Quality (5)
    jpeg_compress,
    blur,
    low_resolution,
    photocopy,
    binarize,
    # Lighting (4)
    exposure_jitter,
    uneven_lighting,
    glare,
    striped_shadow,
    # Geometric (2)
    rotation,
    perspective_warp,
    # Document (4)
    fold_crease,
    aged_document,
    scanner_edge,
    water_stain,
    # Ink (2)
    stroke_variation,
    smudge,
    # Noise & color (3)
    noise,
    color_jitter,
    to_grayscale,
    # Occlusion (1)
    occlusion,
    # Outdoor (1)
    weather_damage,
    # Scene text / background (2)
    textured_background,
    colored_background,
    # Crop / boundary (2)
    partial_crop,
    pad_with_border,
    # Screen / display (1)
    screen_artifacts,
    # Camera (2)
    camera_noise,
    text_shadow,
    # Document decorations (3)
    text_decoration,
    highlighter,
    table_rules,
    # Polarity & layout context (2)
    polarity_invert,
    adjacent_line_clutter,
]

# elastic_distortion needs scipy
try:
    import scipy.ndimage  # noqa: F401
    AUGMENT_OPS.append(elastic_distortion)
except ImportError:
    pass


# =========================================================================
# Scenario chains — realistic multi-degradation combinations
# =========================================================================

SCENARIO_CHAINS: list[tuple[str, list[Callable], float]] = [
    # (name, ops_in_order, probability_weight)
    ("phone_document", [
        perspective_warp, uneven_lighting, camera_noise, blur,
    ], 3.0),
    ("phone_sign", [
        colored_background, perspective_warp, glare, camera_noise,
    ], 2.0),
    ("outdoor_sign", [
        textured_background, weather_damage, perspective_warp, exposure_jitter,
    ], 2.0),
    ("old_scan", [
        aged_document, scanner_edge, low_resolution, photocopy,
    ], 2.0),
    ("screenshot", [
        screen_artifacts, jpeg_compress, partial_crop,
    ], 1.5),
    ("photocopy_fax", [
        photocopy, noise, low_resolution, binarize,
    ], 1.5),
    ("book_page", [
        uneven_lighting, fold_crease, blur, camera_noise,
    ], 1.5),
    ("quick_snap", [
        blur, perspective_warp, exposure_jitter, partial_crop,
    ], 2.0),
    ("worn_label", [
        weather_damage, water_stain, low_resolution, noise,
    ], 1.0),
    ("flash_photo", [
        glare, camera_noise, striped_shadow, exposure_jitter,
    ], 1.0),
    ("occluded", [
        occlusion, perspective_warp, camera_noise,
    ], 1.0),
    ("distant_photo", [
        low_resolution, blur, camera_noise, perspective_warp,
    ], 1.5),
    ("dense_document", [
        table_rules, adjacent_line_clutter, uneven_lighting, jpeg_compress,
        blur,
    ], 2.0),
    ("dark_sign", [
        polarity_invert, perspective_warp, glare, camera_noise,
    ], 1.5),
    ("dark_screen", [
        polarity_invert, screen_artifacts, jpeg_compress,
    ], 1.5),
    ("notebook", [
        lined_paper, variable_baseline, rotation, camera_noise,
    ], 0.75),
]

# Name → (ops, weight): lets callers assemble style-specific chain subsets
CHAINS_BY_NAME: dict[str, tuple[list[Callable], float]] = {
    name: (ops, weight) for name, ops, weight in SCENARIO_CHAINS
}


def _pick_scenario(chains: list[tuple[str, list[Callable], float]]) -> list[Callable]:
    """Weighted random selection of a scenario chain."""
    total = sum(w for _, _, w in chains)
    r = random.random() * total
    cumulative = 0.0
    for _, ops, weight in chains:
        cumulative += weight
        if r <= cumulative:
            return ops
    return chains[-1][1]


# Ops that change x-geometry and therefore invalidate pixel-space labels
# (segment offsets, per-pixel group labels). Maps each op to a variant
# returning (img, (a, b)) with x_new = a * x_old + b so callers can keep
# labels aligned with the augmented image. rotation / perspective_warp /
# wave_distortion cause small local displacements but no global shift, so
# they are treated as x-preserving.
X_TRANSFORM_OPS: dict[Callable, Callable] = {
    partial_crop: partial_crop_with_transform,
    pad_with_border: pad_with_border_with_transform,
}


class RandAugmentOCR:
    """Augmentation for OCR: mix of random ops and realistic scenario chains.

    50% of augmented samples get a scenario chain (realistic
    multi-degradation), 50% get N random independent ops (diversity).
    `chains` restricts which scenario chains apply (e.g. per data style);
    None means all chains, [] disables the chain branch entirely.
    """

    def __init__(self, n_ops: int = 2, p: float = 0.5,
                 ops: list[Callable] | None = None,
                 chains: list[tuple[str, list[Callable], float]] | None = None):
        self.n_ops = n_ops
        self.p = p
        self.ops = ops if ops is not None else AUGMENT_OPS
        self.chains = chains if chains is not None else SCENARIO_CHAINS

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.apply_with_transform(img)[0]

    def apply_with_transform(
            self, img: Image.Image) -> tuple[Image.Image, tuple[float, float]]:
        """Apply augmentation, returning (img, (a, b)) with x_new = a*x_old + b.

        The composed x-affine covers every applied op that shifts content
        horizontally (X_TRANSFORM_OPS); all other ops contribute identity.
        Callers that hold pixel-space labels must remap them by (a, b).
        """
        if random.random() > self.p:
            return img, (1.0, 0.0)

        if self.chains and random.random() < 0.5:
            # Scenario chain: apply all ops in a realistic combination
            ops = _pick_scenario(self.chains)
        else:
            # Random ops: original RandAugment behavior
            ops = random.sample(self.ops, min(self.n_ops, len(self.ops)))

        a, b = 1.0, 0.0
        for op in ops:
            with_transform = X_TRANSFORM_OPS.get(op)
            if with_transform is not None:
                img, (oa, ob) = with_transform(img)
                # Compose: x → oa*(a*x + b) + ob
                a, b = oa * a, oa * b + ob
            else:
                img = op(img)
        return img, (a, b)
