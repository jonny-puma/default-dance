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
import json
import os

GIF_PATH = "dance.gif"
DEBUG_DIR = "debug"
OUTPUT_PATH = "ascii_frames.json"

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

            # Warmth: how much warmer than blue. Character is warm, bg is blue.
            warmth = max(0, r - b)

            # Luminance diff from background
            diff = abs(gray[j] - bg[j])

            # Binary mask: if pixel has any warmth at all, it's character.
            # Even dark clothing (R=30, B=25) gives warmth=5.
            # The blue bg always has B >> R, so warmth=0.
            if warmth >= 3 and diff > 3:
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

    stretched = []
    for frame in frames:
        px = list(frame.getdata())
        new_px = []
        for p in px:
            if p <= 1:
                new_px.append(0)
            else:
                v = int((p - lo) / max(1, hi - lo) * 255)
                new_px.append(max(0, min(255, v)))
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

    print("\n--- Step 5: Encode braille ---")
    braille_frames = encode_braille(stretched, dot_w, dot_h)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(braille_frames, f)
    print(f"\nSaved {len(braille_frames)} frames to {OUTPUT_PATH}")

    # Preview
    print("\nPreview frame 17:")
    for line in braille_frames[17]:
        print(line)


if __name__ == "__main__":
    main()
