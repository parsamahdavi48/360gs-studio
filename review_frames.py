#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageTk
except Exception as e:  # pragma: no cover - environment-dependent import
    Image = None
    ImageTk = None
    _PIL_IMPORT_ERROR = e
else:
    _PIL_IMPORT_ERROR = None

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:  # pragma: no cover - environment-dependent import
    tk = None
    ttk = None
    messagebox = None
    _TK_IMPORT_ERROR = e
else:
    _TK_IMPORT_ERROR = None

if Image is not None and hasattr(Image, "Resampling"):
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
else:
    RESAMPLE_BILINEAR = Image.BILINEAR if Image is not None else None


class ReviewApp:
    def __init__(self, root: tk.Tk, scene_dir: Path, csv_path: Path) -> None:
        self.root = root
        self.scene_dir = scene_dir
        self.csv_path = csv_path

        self.rows = self._load_rows(csv_path)
        if not self.rows:
            raise RuntimeError(f"No rows found in {csv_path}")

        self.index = 0
        self.current_photo = None

        self._build_ui()
        self._bind_keys()
        self._render_current()

    def _load_rows(self, path: Path) -> List[Dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            decision = row.get("decision", "keep").strip().lower()
            row["decision"] = "drop" if decision == "drop" else "keep"
        return rows

    def _build_ui(self) -> None:
        self.root.title("Frame Review")
        self.root.geometry("1200x820")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.X)

        self.title_var = tk.StringVar()
        ttk.Label(top, textvariable=self.title_var, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)

        self.decision_var = tk.StringVar()
        self.decision_label = ttk.Label(top, textvariable=self.decision_var, foreground="#1f6f1f", font=("Segoe UI", 11, "bold"))
        self.decision_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(main, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(10, 10))

        info_row = ttk.Frame(main)
        info_row.pack(fill=tk.X)
        self.info_var = tk.StringVar()
        ttk.Label(info_row, textvariable=self.info_var).pack(side=tk.LEFT)

        button_row = ttk.Frame(main)
        button_row.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_row, text="Prev (Left)", command=self.prev_row).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Next (Right)", command=self.next_row).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Toggle Keep/Drop (Space)", command=self.toggle_decision).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Save (S)", command=self.save).pack(side=tk.RIGHT)

        hint = ttk.Label(
            main,
            text="Keys: Left/Right=move, Space=toggle keep/drop, S=save, Q=quit",
            foreground="#555555",
        )
        hint.pack(anchor="w", pady=(8, 0))

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _: self.prev_row())
        self.root.bind("<Right>", lambda _: self.next_row())
        self.root.bind("<space>", lambda _: self.toggle_decision())
        self.root.bind("s", lambda _: self.save())
        self.root.bind("S", lambda _: self.save())
        self.root.bind("q", lambda _: self.quit())
        self.root.bind("Q", lambda _: self.quit())

    def _current_row(self) -> Dict[str, str]:
        return self.rows[self.index]

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def _fit_size(self, width: int, height: int, max_w: int, max_h: int) -> Tuple[int, int]:
        ratio = min(max_w / max(width, 1), max_h / max(height, 1), 1.0)
        return max(1, int(width * ratio)), max(1, int(height * ratio))

    def _render_current(self) -> None:
        row = self._current_row()
        seq = int(row.get("seq", self.index + 1))
        total = len(self.rows)

        image_rel = row.get("output_file", "")
        image_path = self.scene_dir / image_rel

        self.title_var.set(f"{seq}/{total}  {image_rel}")
        decision = row.get("decision", "keep")
        self.decision_var.set(f"Decision: {decision.upper()}")

        color = "#b30000" if decision == "drop" else "#1f6f1f"
        self.decision_label.configure(foreground=color)

        info_text = (
            f"orig={row.get('original_index', '-')}, final={row.get('final_index', '-')}, "
            f"ts={row.get('timestamp_sec', '-')}, status={row.get('status', '-')}, "
            f"blur(orig/final)={row.get('blur_score_original', '-')}/{row.get('blur_score_final', '-')}, "
            f"change(orig/final)={row.get('change_score_original', '-')}/{row.get('change_score_final', '-')}"
        )
        self.info_var.set(info_text)

        self.canvas.delete("all")
        try:
            img = self._load_image(image_path)
            canvas_w = max(self.canvas.winfo_width(), 100)
            canvas_h = max(self.canvas.winfo_height(), 100)
            fit_w, fit_h = self._fit_size(img.width, img.height, canvas_w - 20, canvas_h - 20)
            resized = img.resize((fit_w, fit_h), RESAMPLE_BILINEAR)
            self.current_photo = ImageTk.PhotoImage(resized)
            x = canvas_w // 2
            y = canvas_h // 2
            self.canvas.create_image(x, y, image=self.current_photo, anchor=tk.CENTER)
        except Exception as e:
            self.canvas.create_text(20, 20, anchor=tk.NW, fill="#ffffff", text=str(e))

    def prev_row(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def next_row(self) -> None:
        if self.index < len(self.rows) - 1:
            self.index += 1
            self._render_current()

    def toggle_decision(self) -> None:
        row = self._current_row()
        row["decision"] = "drop" if row.get("decision", "keep") == "keep" else "keep"
        self._render_current()

    def save(self) -> None:
        if not self.rows:
            return

        fieldnames = list(self.rows[0].keys())
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        keep_count = sum(1 for r in self.rows if r.get("decision") != "drop")
        drop_count = len(self.rows) - keep_count
        messagebox.showinfo("Saved", f"Updated {self.csv_path}\nkeep={keep_count}, drop={drop_count}")

    def quit(self) -> None:
        self.root.destroy()


def ensure_gui_deps() -> None:
    missing = []
    if Image is None or ImageTk is None:
        missing.append(f"Pillow (import failed: {_PIL_IMPORT_ERROR})")
    if tk is None or ttk is None or messagebox is None:
        missing.append(f"tkinter (import failed: {_TK_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing GUI dependencies: " + "; ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review extracted frames and edit keep/drop decisions.")
    parser.add_argument(
        "scene_dir",
        nargs="?",
        default=".",
        help="Scene directory containing selected_frames.csv and images/",
    )
    parser.add_argument(
        "--csv",
        default="selected_frames.csv",
        help="CSV filename under scene_dir (default=selected_frames.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).resolve()
    csv_path = scene_dir / args.csv

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    root = tk.Tk()
    try:
        app = ReviewApp(root, scene_dir, csv_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    root.mainloop()


if __name__ == "__main__":
    main()
