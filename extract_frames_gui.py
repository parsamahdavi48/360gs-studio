#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
except Exception as e:  # pragma: no cover - environment-dependent import
    tk = None
    filedialog = None
    ttk = None
    messagebox = None
    _TK_IMPORT_ERROR = e
else:
    _TK_IMPORT_ERROR = None


class ExtractFramesGUI:
    def __init__(self, root: tk.Tk, initial_scene_dir: Optional[str] = None) -> None:
        self.root = root
        self.root.title("Frame Extractor")
        self.root.geometry("1080x800")

        self.base_dir = Path(__file__).resolve().parent

        self.log_queue: Queue = Queue()
        self.current_thread: Optional[threading.Thread] = None
        self.is_running = False

        self.input_video = tk.StringVar()
        self.scene_dir = tk.StringVar(value=initial_scene_dir or ".")
        self.mode = tk.StringVar(value="change")

        self.interval_sec = tk.StringVar(value="0.5")
        self.change_threshold = tk.StringVar(value="0.04")
        self.min_gap_sec = tk.StringVar(value="0.25")
        self.max_gap_sec = tk.StringVar(value="2.0")

        self.analysis_width = tk.StringVar(value="960")
        self.blur_percentile = tk.StringVar(value="25.0")
        self.blur_window_frames = tk.StringVar(value="0")

        self.image_ext = tk.StringVar(value="jpg")
        self.jpg_quality = tk.StringVar(value="2")

        self.ffmpeg_bin = tk.StringVar(value="ffmpeg")
        self.ffprobe_bin = tk.StringVar(value="ffprobe")

        self.metashape_output = tk.StringVar(value="metashape_images")
        self.scene_dir.trace_add("write", lambda *_: self._refresh_action_buttons())

        self._build_ui()
        self._update_mode_widgets()
        self._refresh_action_buttons()
        self.root.after(100, self._poll_log_queue)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        form = ttk.Frame(container)
        form.pack(fill=tk.X)

        r = 0
        ttk.Label(form, text="Input Video").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.input_video, width=95).grid(row=r, column=1, sticky=tk.EW, pady=4)
        ttk.Button(form, text="Browse", command=self._browse_video).grid(row=r, column=2, padx=(6, 0), pady=4)

        r += 1
        ttk.Label(form, text="Scene Directory").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.scene_dir, width=95).grid(row=r, column=1, sticky=tk.EW, pady=4)
        ttk.Button(form, text="Browse", command=self._browse_scene_dir).grid(row=r, column=2, padx=(6, 0), pady=4)

        r += 1
        ttk.Label(form, text="Mode").grid(row=r, column=0, sticky=tk.W, pady=4)
        mode_frame = ttk.Frame(form)
        mode_frame.grid(row=r, column=1, sticky=tk.W, pady=4)
        ttk.Radiobutton(mode_frame, text="Change-Based", value="change", variable=self.mode, command=self._update_mode_widgets).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Fixed Interval", value="fixed", variable=self.mode, command=self._update_mode_widgets).pack(side=tk.LEFT, padx=(14, 0))

        r += 1
        ttk.Label(form, text="Fixed Interval (sec)").grid(row=r, column=0, sticky=tk.W, pady=4)
        self.interval_entry = ttk.Entry(form, textvariable=self.interval_sec, width=12)
        self.interval_entry.grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Change Threshold").grid(row=r, column=0, sticky=tk.W, pady=4)
        self.change_threshold_entry = ttk.Entry(form, textvariable=self.change_threshold, width=12)
        self.change_threshold_entry.grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Min Gap (sec)").grid(row=r, column=0, sticky=tk.W, pady=4)
        self.min_gap_entry = ttk.Entry(form, textvariable=self.min_gap_sec, width=12)
        self.min_gap_entry.grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Max Gap (sec)").grid(row=r, column=0, sticky=tk.W, pady=4)
        self.max_gap_entry = ttk.Entry(form, textvariable=self.max_gap_sec, width=12)
        self.max_gap_entry.grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Analysis Width").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.analysis_width, width=12).grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Blur Percentile").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.blur_percentile, width=12).grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Blur Window (frames)").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.blur_window_frames, width=12).grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Image Ext").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(form, textvariable=self.image_ext, values=["jpg", "png"], width=10, state="readonly").grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="JPEG Quality (ffmpeg -q:v)").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.jpg_quality, width=12).grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="ffmpeg").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.ffmpeg_bin, width=25).grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="ffprobe").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.ffprobe_bin, width=25).grid(row=r, column=1, sticky=tk.W, pady=4)

        r += 1
        ttk.Label(form, text="Metashape Output Folder").grid(row=r, column=0, sticky=tk.W, pady=4)
        ttk.Entry(form, textvariable=self.metashape_output, width=25).grid(row=r, column=1, sticky=tk.W, pady=4)

        form.columnconfigure(1, weight=1)

        button_row = ttk.Frame(container)
        button_row.pack(fill=tk.X, pady=(12, 8))

        self.run_button = ttk.Button(button_row, text="1) Run Extraction", command=self._run_extraction)
        self.run_button.pack(side=tk.LEFT)

        self.review_button = ttk.Button(button_row, text="2) Open Review GUI", command=self._open_review, state=tk.DISABLED)
        self.review_button.pack(side=tk.LEFT, padx=(8, 0))

        self.export_button = ttk.Button(button_row, text="3) Export Keep Frames", command=self._export_keep, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(button_row, textvariable=self.status_var).pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(container, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=16, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scroll.set)

    def _set_running(self, running: bool) -> None:
        self.is_running = running
        if running:
            self.status_var.set("Running")
            self.run_button.configure(state=tk.DISABLED)
        else:
            self.status_var.set("Idle")
            self.run_button.configure(state=tk.NORMAL)
        self._refresh_action_buttons()

    def _refresh_action_buttons(self) -> None:
        if self.is_running:
            state = tk.DISABLED
        else:
            csv_path = Path(self.scene_dir.get().strip() or ".") / "selected_frames.csv"
            state = tk.NORMAL if csv_path.exists() else tk.DISABLED
        self.review_button.configure(state=state)
        self.export_button.configure(state=state)

    def _append_log(self, text: str) -> None:
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "done":
                    rc = int(item[1])
                    phase = item[2]
                    self._set_running(False)
                    if rc == 0:
                        self._append_log(f"[{phase}] completed successfully")
                        self.status_var.set(f"Done: {phase}")
                    else:
                        self._append_log(f"[{phase}] failed (exit={rc})")
                        self.status_var.set(f"Failed: {phase}")
                    continue

                self._append_log(str(item))
        except Empty:
            pass

        self.root.after(100, self._poll_log_queue)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_video.set(path)

    def _browse_scene_dir(self) -> None:
        path = filedialog.askdirectory(title="Select scene directory")
        if path:
            self.scene_dir.set(path)

    def _update_mode_widgets(self) -> None:
        is_fixed = self.mode.get() == "fixed"
        fixed_state = tk.NORMAL if is_fixed else tk.DISABLED
        change_state = tk.DISABLED if is_fixed else tk.NORMAL

        self.interval_entry.configure(state=fixed_state)
        self.change_threshold_entry.configure(state=change_state)
        self.min_gap_entry.configure(state=change_state)
        self.max_gap_entry.configure(state=change_state)

    def _start_subprocess(self, cmd: list[str], phase: str) -> None:
        if self.is_running:
            messagebox.showwarning("Busy", "Another process is running.")
            return

        self._set_running(True)
        self._append_log("$ " + " ".join(cmd))

        def worker() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None

                for line in proc.stdout:
                    self.log_queue.put(line.rstrip())

                rc = proc.wait()
            except Exception as e:
                self.log_queue.put(f"Error: {e}")
                rc = 1

            self.log_queue.put(("done", rc, phase))

        self.current_thread = threading.Thread(target=worker, daemon=True)
        self.current_thread.start()

    def _build_extract_cmd(self) -> list[str]:
        input_video = self.input_video.get().strip()
        scene_dir = self.scene_dir.get().strip()

        if not input_video:
            raise ValueError("Input video is required")
        if not Path(input_video).exists():
            raise ValueError(f"Input video not found: {input_video}")
        if not scene_dir:
            raise ValueError("Scene directory is required")

        script = self.base_dir / "extract_frames.py"
        if not script.exists():
            raise FileNotFoundError(f"extract_frames.py not found: {script}")

        cmd = [
            sys.executable,
            str(script),
            input_video,
            scene_dir,
            "--mode",
            self.mode.get(),
            "--analysis-width",
            self.analysis_width.get().strip(),
            "--blur-percentile",
            self.blur_percentile.get().strip(),
            "--blur-window-frames",
            self.blur_window_frames.get().strip(),
            "--image-ext",
            self.image_ext.get().strip(),
            "--jpg-quality",
            self.jpg_quality.get().strip(),
            "--ffmpeg",
            self.ffmpeg_bin.get().strip(),
            "--ffprobe",
            self.ffprobe_bin.get().strip(),
        ]

        if self.mode.get() == "fixed":
            cmd.extend(["--interval-sec", self.interval_sec.get().strip()])
        else:
            cmd.extend(
                [
                    "--change-threshold",
                    self.change_threshold.get().strip(),
                    "--min-gap-sec",
                    self.min_gap_sec.get().strip(),
                    "--max-gap-sec",
                    self.max_gap_sec.get().strip(),
                ]
            )

        return cmd

    def _run_extraction(self) -> None:
        try:
            cmd = self._build_extract_cmd()
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        self._start_subprocess(cmd, "extract")

    def _open_review(self) -> None:
        scene_dir = self.scene_dir.get().strip() or "."
        script = self.base_dir / "review_frames.py"
        csv_path = Path(scene_dir) / "selected_frames.csv"

        if not script.exists():
            messagebox.showerror("Error", f"review_frames.py not found: {script}")
            return
        if not csv_path.exists():
            messagebox.showerror("Error", f"CSV not found: {csv_path}")
            return

        cmd = [sys.executable, str(script), scene_dir]
        try:
            subprocess.Popen(cmd)
            self._append_log("[review] launched")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_keep(self) -> None:
        scene_dir = self.scene_dir.get().strip() or "."
        output_name = self.metashape_output.get().strip() or "metashape_images"

        script = self.base_dir / "apply_frame_decisions.py"
        csv_path = Path(scene_dir) / "selected_frames.csv"
        if not script.exists():
            messagebox.showerror("Error", f"apply_frame_decisions.py not found: {script}")
            return
        if not csv_path.exists():
            messagebox.showerror("Error", f"CSV not found: {csv_path}")
            return

        should_clean = messagebox.askyesno(
            "Clean Output",
            f"Clean existing image files in '{output_name}' before export?",
            default=messagebox.YES,
        )

        cmd = [
            sys.executable,
            str(script),
            scene_dir,
            "--output",
            output_name,
        ]
        if should_clean:
            cmd.append("--clean-output")

        self._start_subprocess(cmd, "export")


def ensure_gui_deps() -> None:
    if tk is None or ttk is None or filedialog is None or messagebox is None:
        raise RuntimeError(f"tkinter is required to run this GUI: {_TK_IMPORT_ERROR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI wrapper for extract_frames.py and review/export workflow.")
    parser.add_argument(
        "--scene-dir",
        default=".",
        help="Initial scene directory shown in GUI",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_gui_deps()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    root = tk.Tk()
    app = ExtractFramesGUI(root, initial_scene_dir=args.scene_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
