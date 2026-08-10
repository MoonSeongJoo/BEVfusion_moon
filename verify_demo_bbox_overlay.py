import os
import csv
import glob
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# User configuration
# ============================================================

IMG_DIR = "./data/nuscenes/demo/demo_frames_src"
LAB_DIR = "./data/nuscenes/demo/demo_labels_src"
OUT_DIR = "./data/nuscenes/demo/demo_bbox_verify"

MAX_FRAMES_TO_CONTACT_SHEET = 100
CONTACT_COLS = 4
THUMB_W = 480
THUMB_H = 270

# Label format:
# Class x1 y1 x2 y2 confidence
# Example:
# Car 426 512 812 743 1.00


# ============================================================
# Utility
# ============================================================

CLASS_COLORS = {
    "Car": (0, 255, 80),
    "Pedestrian": (255, 210, 0),
    "Cone": (255, 120, 0),
    "Cyclist": (0, 200, 255),
    "Unknown": (255, 255, 255),
}


def get_color(cls):
    return CLASS_COLORS.get(cls, CLASS_COLORS["Unknown"])


def safe_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def read_label(label_path):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                print(f"[WARN] invalid label line: {label_path}:{line_no}: {line}")
                continue

            cls = parts[0]
            try:
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                conf = float(parts[5])
            except ValueError:
                print(f"[WARN] parse error: {label_path}:{line_no}: {line}")
                continue

            boxes.append({
                "cls": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": conf,
                "line": line,
            })

    return boxes


def validate_box(box, w, h):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

    errors = []

    if x2 <= x1:
        errors.append("x2<=x1")
    if y2 <= y1:
        errors.append("y2<=y1")
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        errors.append("negative_coord")
    if x1 >= w or x2 >= w or y1 >= h or y2 >= h:
        errors.append("out_of_image")
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        errors.append("too_small")

    return errors


def draw_overlay(img, boxes, frame_name):
    draw = ImageDraw.Draw(img)
    font = safe_font(20)
    small_font = safe_font(16)

    w, h = img.size

    # Header
    header_h = 38
    draw.rectangle([0, 0, w, header_h], fill=(0, 0, 0))
    draw.text((10, 8), f"{frame_name} | image={w}x{h} | boxes={len(boxes)}",
              fill=(255, 255, 255), font=small_font)

    for idx, box in enumerate(boxes):
        cls = box["cls"]
        color = get_color(cls)

        x1 = int(round(box["x1"]))
        y1 = int(round(box["y1"]))
        x2 = int(round(box["x2"]))
        y2 = int(round(box["y2"]))
        conf = box["conf"]

        errors = validate_box(box, w, h)

        # Clip for drawing
        dx1 = max(0, min(x1, w - 1))
        dy1 = max(0, min(y1, h - 1))
        dx2 = max(0, min(x2, w - 1))
        dy2 = max(0, min(y2, h - 1))

        # Invalid bbox는 빨간색
        if errors:
            color = (255, 0, 0)

        # Box line
        line_w = 4 if not errors else 6
        for k in range(line_w):
            draw.rectangle([dx1 - k, dy1 - k, dx2 + k, dy2 + k],
                           outline=color)

        label = f"{idx}:{cls} {conf:.2f}"
        if errors:
            label += " " + ",".join(errors)

        # Label background
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_x = dx1
        label_y = max(0, dy1 - th - 8)

        draw.rectangle([label_x, label_y, label_x + tw + 10, label_y + th + 8],
                       fill=color)
        draw.text((label_x + 5, label_y + 4), label,
                  fill=(0, 0, 0), font=font)

    return img


def make_contact_sheet(overlay_paths, out_path):
    if not overlay_paths:
        return

    selected = overlay_paths[:MAX_FRAMES_TO_CONTACT_SHEET]
    rows = (len(selected) + CONTACT_COLS - 1) // CONTACT_COLS

    sheet_w = CONTACT_COLS * THUMB_W
    sheet_h = rows * THUMB_H

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))

    for i, p in enumerate(selected):
        img = Image.open(p).convert("RGB")
        img = img.resize((THUMB_W, THUMB_H))

        x = (i % CONTACT_COLS) * THUMB_W
        y = (i // CONTACT_COLS) * THUMB_H
        sheet.paste(img, (x, y))

    sheet.save(out_path, quality=95)


def make_html(overlay_files, out_html):
    with open(out_html, "w") as f:
        f.write("""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>nuScenes BBox Verification</title>
<style>
body { background:#111827; color:white; font-family:Arial; margin:20px; }
h1 { font-size:28px; }
.grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:20px; }
.card { background:#1f2937; padding:12px; border-radius:10px; }
img { width:100%; border:1px solid #94a3b8; }
.name { margin:8px 0; color:#cbd5e1; }
</style>
</head>
<body>
<h1>nuScenes GT 2D BBox Verification</h1>
<p>Red boxes indicate invalid or suspicious coordinates.</p>
<div class="grid">
""")
        for p in overlay_files:
            name = os.path.basename(p)
            rel = os.path.join("overlay", name)
            f.write(f"""
<div class="card">
  <div class="name">{name}</div>
  <img src="{rel}">
</div>
""")
        f.write("""
</div>
</body>
</html>
""")


# ============================================================
# Main
# ============================================================

def main():
    overlay_dir = os.path.join(OUT_DIR, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "frame_*.jpg")))

    if not image_paths:
        print("[ERROR] no frame_*.jpg found in", IMG_DIR)
        return

    summary_rows = []
    overlay_paths = []

    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LAB_DIR, base + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = read_label(label_path)

        invalid_count = 0
        for b in boxes:
            if validate_box(b, w, h):
                invalid_count += 1

        overlay = draw_overlay(img, boxes, base)

        out_img = os.path.join(overlay_dir, base + "_overlay.jpg")
        overlay.save(out_img, quality=95)

        overlay_paths.append(out_img)

        summary_rows.append({
            "frame": base,
            "image_path": img_path,
            "label_path": label_path,
            "image_w": w,
            "image_h": h,
            "num_boxes": len(boxes),
            "invalid_boxes": invalid_count,
        })

        print(f"[OK] {base}: image={w}x{h}, boxes={len(boxes)}, invalid={invalid_count}")

    # CSV summary
    summary_csv = os.path.join(OUT_DIR, "verify_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "image_path",
                "label_path",
                "image_w",
                "image_h",
                "num_boxes",
                "invalid_boxes",
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Contact sheet
    contact_path = os.path.join(OUT_DIR, "contact_sheet.jpg")
    make_contact_sheet(overlay_paths, contact_path)

    # HTML gallery
    html_path = os.path.join(OUT_DIR, "index.html")
    make_html(overlay_paths, html_path)

    print("")
    print("[DONE]")
    print("overlay dir  :", overlay_dir)
    print("summary csv  :", summary_csv)
    print("contact sheet:", contact_path)
    print("html gallery :", html_path)
    print("")
    print("Open:")
    print(f"  xdg-open {contact_path}")
    print(f"  xdg-open {html_path}")


if __name__ == "__main__":
    main()