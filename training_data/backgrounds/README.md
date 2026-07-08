# Background image bank

Drop real background photos here (any of `.jpg .jpeg .png .webp .bmp`,
subdirectories fine): paper/document scans, walls, wood, metal, fabric,
storefronts, outdoor surfaces.

The `photo_background` augmentation op composites rendered text onto
random crops of these images (SynthText-style) — real surface statistics
that procedural textures can't match. While this directory is empty the
op silently falls back to procedural textures, so nothing breaks without
it; realism just improves when it's populated.

Good sources: DTD (Describable Textures Dataset), your own photos of
paper/walls/signs, or crops from any photo collection with the text
regions avoided.
