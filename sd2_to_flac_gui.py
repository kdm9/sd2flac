#!/usr/bin/env python3
"""
sd2_to_flac_gui.py — Tkinter GUI for sd2_to_flac.

Provides directory choosers, option widgets, a Start button, and a
scrolling log text box that captures all logger output in real time.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from sd2_to_flac import ALL_CANDIDATES, PROBE_BYTES, main as sd2_main, log as sd2_log


# ── Logging handler that writes to a Tk Text widget ─────────────────────────

class TextHandler(logging.Handler):
    """Logging handler that appends records to a tkinter Text widget."""

    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + "\n"
        # Schedule the insert on the main thread
        self.text_widget.after(0, self._append, msg)

    def _append(self, msg):
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)


# ── GUI ──────────────────────────────────────────────────────────────────────

class SD2ToFlacApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SD2 → FLAC Converter")
        self.minsize(640, 560)
        self._build_ui()
        self._running = False

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # ── Input directory ──────────────────────────────────────────
        row = 0
        tk.Label(self, text="Input directory:").grid(
            row=row, column=0, sticky=tk.W, **pad)
        self.input_var = tk.StringVar()
        tk.Entry(self, textvariable=self.input_var, width=50).grid(
            row=row, column=1, sticky=tk.EW, **pad)
        tk.Button(self, text="Browse…", command=self._browse_input).grid(
            row=row, column=2, **pad)

        # ── Output directory ─────────────────────────────────────────
        row += 1
        tk.Label(self, text="Output directory:").grid(
            row=row, column=0, sticky=tk.W, **pad)
        self.output_var = tk.StringVar()
        tk.Entry(self, textvariable=self.output_var, width=50).grid(
            row=row, column=1, sticky=tk.EW, **pad)
        tk.Button(self, text="Browse…", command=self._browse_output).grid(
            row=row, column=2, **pad)

        # ── Sample rate ──────────────────────────────────────────────
        row += 1
        tk.Label(self, text="Sample rate (Hz):").grid(
            row=row, column=0, sticky=tk.W, **pad)
        self.sr_var = tk.StringVar(value="")
        sr_entry = tk.Entry(self, textvariable=self.sr_var, width=12)
        sr_entry.grid(row=row, column=1, sticky=tk.W, **pad)
        tk.Label(self, text="(leave blank for auto / 44100)").grid(
            row=row, column=2, sticky=tk.W, **pad)

        # ── Encoding ────────────────────────────────────────────────
        row += 1
        tk.Label(self, text="Encoding:").grid(
            row=row, column=0, sticky=tk.W, **pad)
        encoding_choices = ["Auto (try all)"] + list(ALL_CANDIDATES.keys())
        self.encoding_var = tk.StringVar(value=encoding_choices[0])
        enc_combo = ttk.Combobox(
            self, textvariable=self.encoding_var,
            values=encoding_choices, state="readonly", width=20,
        )
        enc_combo.grid(row=row, column=1, sticky=tk.W, **pad)

        # ── Probe bytes ─────────────────────────────────────────────
        row += 1
        tk.Label(self, text="Probe bytes:").grid(
            row=row, column=0, sticky=tk.W, **pad)
        self.probe_var = tk.StringVar(value=str(PROBE_BYTES))
        tk.Entry(self, textvariable=self.probe_var, width=12).grid(
            row=row, column=1, sticky=tk.W, **pad)

        # ── Checkboxes ──────────────────────────────────────────────
        row += 1
        self.xattr_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Read xattr / resource fork metadata",
                       variable=self.xattr_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, **pad)

        row += 1
        self.dry_run_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Dry run (don't actually convert)",
                       variable=self.dry_run_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, **pad)

        row += 1
        self.verbose_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Verbose logging",
                       variable=self.verbose_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, **pad)

        row += 1
        self.include_input_dirname_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Include input directory name in output folder",
                       variable=self.include_input_dirname_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, **pad)

        # ── Start button ────────────────────────────────────────────
        row += 1
        self.start_btn = tk.Button(
            self, text="Start", command=self._on_start,
            bg="#4CAF50", fg="white", font=("Helvetica", 14, "bold"),
            padx=20, pady=6,
        )
        self.start_btn.grid(row=row, column=0, columnspan=3, pady=12)

        # ── Log output ──────────────────────────────────────────────
        row += 1
        tk.Label(self, text="Log:").grid(
            row=row, column=0, sticky=tk.W, **pad)

        row += 1
        log_frame = tk.Frame(self)
        log_frame.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, **pad)
        self.log_text = tk.Text(log_frame, height=16, state=tk.DISABLED,
                                wrap=tk.WORD, font=("Courier", 11))
        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Make the log area expand when the window is resized
        self.columnconfigure(1, weight=1)
        self.rowconfigure(row, weight=1)

        # ── Attach logging handler ──────────────────────────────────
        handler = TextHandler(self.log_text)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        sd2_log.addHandler(handler)

    # ── Callbacks ────────────────────────────────────────────────────────

    def _browse_input(self):
        d = filedialog.askdirectory(title="Select input directory")
        if d:
            self.input_var.set(d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_var.set(d)

    def _on_start(self):
        if self._running:
            return

        # Clear log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        input_dir = self.input_var.get().strip()
        if not input_dir:
            self._log_msg("ERROR: Please select an input directory.\n")
            return

        output_dir = self.output_var.get().strip() or None

        sr_text = self.sr_var.get().strip()
        sample_rate = None
        if sr_text:
            try:
                sample_rate = int(sr_text)
            except ValueError:
                self._log_msg("ERROR: Sample rate must be an integer.\n")
                return

        enc_choice = self.encoding_var.get()
        if enc_choice.startswith("Auto"):
            encoding = None
        else:
            encoding = [enc_choice]

        try:
            probe_bytes = int(self.probe_var.get().strip())
        except ValueError:
            self._log_msg("ERROR: Probe bytes must be an integer.\n")
            return

        xattr = self.xattr_var.get()
        dry_run = self.dry_run_var.get()
        verbose = self.verbose_var.get()
        include_input_dirname = self.include_input_dirname_var.get()

        self._running = True
        self.start_btn.configure(state=tk.DISABLED, text="Running…")

        thread = threading.Thread(
            target=self._run_conversion,
            kwargs=dict(
                input_dir=input_dir,
                output_dir=output_dir,
                sample_rate=sample_rate,
                encoding=encoding,
                xattr=xattr,
                probe_bytes=probe_bytes,
                dry_run=dry_run,
                verbose=verbose,
                include_input_dirname=include_input_dirname,
            ),
            daemon=True,
        )
        thread.start()

    def _run_conversion(self, **kwargs):
        try:
            ok, fail = sd2_main(**kwargs)
            summary = f"\n{'='*40}\nFinished: {ok} succeeded, {fail} failed.\n"
            self.log_text.after(0, self._log_msg, summary)
        except Exception as exc:
            self.log_text.after(0, self._log_msg,
                                f"\nERROR: {exc}\n")
        finally:
            self.after(0, self._conversion_done)

    def _conversion_done(self):
        self._running = False
        self.start_btn.configure(state=tk.NORMAL, text="Start")

    def _log_msg(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


# ── Entry point ──────────────────────────────────────────────────────────────

def gui():
    """Launch the Tkinter GUI."""
    app = SD2ToFlacApp()
    app.mainloop()


if __name__ == "__main__":
    gui()
