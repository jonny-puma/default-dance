#!/usr/bin/env python3
"""
GIF-to-braille processing pipeline.

Converts dance.gif into clean braille frames by:
1. Color-space segmentation: the character is warm (R>B), the background/glow is blue (B>R)
2. Use the "warmth" channel (R-B) as a natural mask
3. Multiply warmth mask with luminance diff for clean character isolation
4. Crop, resize, histogram stretch, braille encode

Saves debug images at each stage.
Run: .venv/bin/python process.py
"""

from PIL import Image, ImageFilter
import base64
import json
import os
import stat
import zlib

GIF_PATH = "dance.gif"
DEBUG_DIR = "debug"
OUTPUT_PATH = "ascii_frames.json"
PLAYER_TEMPLATE = "player.py"
SCRIPT_OUTPUT = "../default-dance"

BRAILLE_CHAR_W = 60
FRAME_STEP = 1  # use every frame

# Braille dot bit positions
DOT_MAP = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


def load_gif():
    im = Image.open(GIF_PATH)
    print(f"GIF: {im.size[0]}x{im.size[1]}, {im.n_frames} frames")
    return im


def compute_background(im):
    """Mean of all frames as background estimate (grayscale)."""
    w, h = im.size
    pixel_sums = [0] * (w * h)
    for i in range(im.n_frames):
        im.seek(i)
        pixels = list(im.convert("L").getdata())
        for j in range(w * h):
            pixel_sums[j] += pixels[j]
    bg = [p // im.n_frames for p in pixel_sums]
    bg_img = Image.new("L", (w, h))
    bg_img.putdata(bg)
    bg_img.save(os.path.join(DEBUG_DIR, "01_background.png"))
    print(f"Background computed")
    return bg_img


def extract_frames(im, bg_img):
    """
    For each frame:
    - Compute luminance diff from background
    - Compute warmth channel: max(0, R - B) — high for character, ~0 for blue bg/glow
    - Combine: warmth acts as a mask, diff provides detail
    """
    w, h = im.size
    bg = list(bg_img.getdata())

    frames = []
    for i in range(0, im.n_frames, FRAME_STEP):
        im.seek(i)

        # Get RGB and grayscale
        rgb = list(im.convert("RGB").getdata())
        gray = list(im.convert("L").getdata())

        result = []
        for j in range(w * h):
            r, g, b = rgb[j]

            # Coolness: how much bluer than red. Background/glow is strongly
            # blue (B >> R, coolness high). The character ranges from warm
            # skin (R >> B) to neutral-dark clothing (R ≈ B, coolness ≈ 0).
            # Only reject pixels that are distinctly blue.
            coolness = b - r

            # Luminance diff from background
            diff = abs(gray[j] - bg[j])

            if coolness < 15 and diff > 3:
                result.append(min(255, diff))
            else:
                result.append(0)

        result_img = Image.new("L", (w, h))
        result_img.putdata(result)
        frames.append(result_img)

    # Save debug
    for idx in [0, 10, 17, 30]:
        if idx < len(frames):
            # Save the combined result
            scaled = [min(255, p * 3) for p in frames[idx].getdata()]
            vis = Image.new("L", (w, h))
            vis.putdata(scaled)
            vis.save(os.path.join(DEBUG_DIR, f"02_color_masked_{idx}.png"))

    # Also save just the warmth channel for one frame for debugging
    im.seek(50)
    rgb = list(im.convert("RGB").getdata())
    warmth_px = [min(255, max(0, r - b) * 3) for r, g, b in rgb]
    warmth_img = Image.new("L", (w, h))
    warmth_img.putdata(warmth_px)
    warmth_img.save(os.path.join(DEBUG_DIR, "02_warmth_channel.png"))

    print(f"Extracted {len(frames)} frames with color-space segmentation")
    return frames


def find_crop_and_resize(frames):
    """Find union bounding box across all frames, crop and resize to braille dot grid."""
    w, h = frames[0].size
    threshold = 3

    global_min_x, global_max_x = w, 0
    global_min_y, global_max_y = h, 0
    for frame in frames:
        px = list(frame.getdata())
        for y in range(h):
            for x in range(w):
                if px[y * w + x] > threshold:
                    global_min_x = min(global_min_x, x)
                    global_max_x = max(global_max_x, x)
                    global_min_y = min(global_min_y, y)
                    global_max_y = max(global_max_y, y)

    pad = 3
    global_min_x = max(0, global_min_x - pad)
    global_max_x = min(w - 1, global_max_x + pad)
    global_min_y = max(0, global_min_y - pad)
    global_max_y = min(h - 1, global_max_y + pad)
    crop_box = (global_min_x, global_min_y, global_max_x + 1, global_max_y + 1)
    crop_w = global_max_x - global_min_x + 1
    crop_h = global_max_y - global_min_y + 1
    print(f"Crop: ({global_min_x},{global_min_y})-({global_max_x},{global_max_y}) = {crop_w}x{crop_h}")

    dot_w = BRAILLE_CHAR_W * 2
    dot_h = int(dot_w * crop_h / crop_w)
    dot_h = ((dot_h + 3) // 4) * 4
    print(f"Braille grid: {BRAILLE_CHAR_W}x{dot_h // 4} chars = {dot_w}x{dot_h} dots")

    resized = []
    for frame in frames:
        cropped = frame.crop(crop_box)
        r = cropped.resize((dot_w, dot_h), Image.LANCZOS)
        resized.append(r)

    # Save debug
    for idx in [0, 10, 17, 30]:
        if idx < len(resized):
            scaled = [min(255, p * 4) for p in resized[idx].getdata()]
            vis = Image.new("L", resized[idx].size)
            vis.putdata(scaled)
            vis.save(os.path.join(DEBUG_DIR, f"03_cropped_{idx}.png"))

    return resized, dot_w, dot_h


def stretch_contrast(frames, dot_w, dot_h):
    """Histogram stretch for maximum contrast."""
    # Collect all non-zero pixel values
    all_vals = []
    for frame in frames:
        for p in frame.getdata():
            if p > 1:
                all_vals.append(p)
    all_vals.sort()

    if not all_vals:
        print("WARNING: no non-zero pixels found!")
        return frames

    lo = all_vals[int(len(all_vals) * 0.03)]
    hi = all_vals[int(len(all_vals) * 0.97)]
    print(f"Stretch: lo={lo}, hi={hi} (from {len(all_vals)} pixels)")

    # Map non-zero pixels to [out_lo, 255] so the dimmest character pixels
    # still land in the first visible density level (not blank).
    out_lo = 28  # 255/9 ≈ first density bucket boundary
    out_range = 255 - out_lo

    stretched = []
    for frame in frames:
        px = list(frame.getdata())
        new_px = []
        for p in px:
            if p <= 1:
                new_px.append(0)
            else:
                v = int((p - lo) / max(1, hi - lo) * out_range + out_lo)
                new_px.append(max(out_lo, min(255, v)))
        img = Image.new("L", (dot_w, dot_h))
        img.putdata(new_px)
        stretched.append(img)

    # Save debug
    for idx in [0, 10, 17, 30]:
        if idx < len(stretched):
            stretched[idx].save(os.path.join(DEBUG_DIR, f"04_stretched_{idx}.png"))

    return stretched


def encode_braille(frames, dot_w, dot_h):
    """Convert grayscale frames to braille character lines."""
    char_w = dot_w // 2
    char_h = dot_h // 4
    threshold = 50  # binary threshold on stretched values

    all_frames = []
    for frame in frames:
        px = list(frame.getdata())
        lines = []
        for char_row in range(char_h):
            line = ""
            for char_col in range(char_w):
                code = 0
                for dy in range(4):
                    for dx in range(2):
                        x = char_col * 2 + dx
                        y = char_row * 4 + dy
                        if x < dot_w and y < dot_h:
                            if px[y * dot_w + x] > threshold:
                                code |= DOT_MAP[dy][dx]
                line += chr(0x2800 + code)
            lines.append(line)
        all_frames.append(lines)

    print(f"Encoded {len(all_frames)} braille frames ({char_w}x{char_h} chars)")
    return all_frames


# Density ramp: space (empty) → light → heavy. Characters chosen for
# increasing visual density when rendered in a monospace terminal font.
DENSITY_RAMP = " .·:;+*#%@"


def encode_density(frames, dot_w, dot_h):
    """Convert grayscale frames to density-mapped character lines.

    Each character cell covers a CELL_W x CELL_H pixel block. The average
    brightness of that block selects a character from DENSITY_RAMP.
    """
    cell_w, cell_h = 2, 4  # same cell size as braille for consistent framing
    char_w = dot_w // cell_w
    char_h = dot_h // cell_h
    n_levels = len(DENSITY_RAMP) - 1  # subtract 1 for the space (background)

    all_frames = []
    for frame in frames:
        px = list(frame.getdata())
        lines = []
        for char_row in range(char_h):
            line = ""
            for char_col in range(char_w):
                total = 0
                count = 0
                for dy in range(cell_h):
                    for dx in range(cell_w):
                        x = char_col * cell_w + dx
                        y = char_row * cell_h + dy
                        if x < dot_w and y < dot_h:
                            total += px[y * dot_w + x]
                            count += 1
                avg = total / count if count else 0
                idx = int(avg / 255 * n_levels)
                idx = min(idx, n_levels)
                line += DENSITY_RAMP[idx]
            lines.append(line)
        all_frames.append(lines)

    print(f"Encoded {len(all_frames)} density frames ({char_w}x{char_h} chars)")
    return all_frames


def generate_script(braille_frames):
    """Generate the standalone default-dance script from player.py template."""
    raw = json.dumps(braille_frames).encode()
    compressed = zlib.compress(raw)
    encoded = base64.b64encode(compressed).decode()

    # Wrap the base64 string into quoted lines
    data_lines = []
    for i in range(0, len(encoded), 76):
        data_lines.append(f'    "{encoded[i:i+76]}"')
    data_str = "\n".join(data_lines)

    template = open(PLAYER_TEMPLATE).read()
    script = template.replace("%%DATA%%", data_str)

    with open(SCRIPT_OUTPUT, "w") as f:
        f.write(script)
    os.chmod(SCRIPT_OUTPUT, os.stat(SCRIPT_OUTPUT).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Generated {SCRIPT_OUTPUT} ({os.path.getsize(SCRIPT_OUTPUT)} bytes)")


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    im = load_gif()

    print("\n--- Step 1: Background ---")
    bg = compute_background(im)

    print("\n--- Step 2: Color-space extraction ---")
    frames = extract_frames(im, bg)

    print("\n--- Step 3: Crop & resize ---")
    resized, dot_w, dot_h = find_crop_and_resize(frames)

    print("\n--- Step 4: Stretch contrast ---")
    stretched = stretch_contrast(resized, dot_w, dot_h)

    print("\n--- Step 5: Encode density ---")
    braille_frames = encode_density(stretched, dot_w, dot_h)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(braille_frames, f)
    print(f"\nSaved {len(braille_frames)} frames to {OUTPUT_PATH}")

    print("\n--- Step 6: Generate script ---")
    generate_script(braille_frames)

    # Preview
    print("\nPreview frame 17:")
    for line in braille_frames[17]:
        print(line)


if __name__ == "__main__":
    main()
