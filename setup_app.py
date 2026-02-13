"""
setup_app.py — py2app build script for the SD2-to-FLAC GUI.

Usage:
    pip install py2app
    python setup_app.py py2app

This produces  dist/SD2 to FLAC.app  which bundles Python, static-ffmpeg,
and all dependencies into a self-contained macOS application.
"""

import os
import subprocess
import sys
from setuptools import setup

# Ensure static_ffmpeg binaries are downloaded before bundling.
# We locate them and include them as data files so they end up inside the .app.
FFMPEG_DATA_FILES = []
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    # Find the actual binary paths
    ffmpeg_path = subprocess.check_output(["which", "ffmpeg"]).decode().strip()
    ffprobe_path = subprocess.check_output(["which", "ffprobe"]).decode().strip()
    if ffmpeg_path and ffprobe_path:
        # Place them in the MacOS directory inside the bundle so they're on PATH
        FFMPEG_DATA_FILES.append(("", [ffmpeg_path, ffprobe_path]))
except Exception as exc:
    print(f"Warning: could not locate static_ffmpeg binaries: {exc}",
          file=sys.stderr)

APP = ["sd2_to_flac_gui.py"]

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "SD2 to FLAC",
        "CFBundleDisplayName": "SD2 to FLAC Converter",
        "CFBundleIdentifier": "com.sd2toflac.app",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "10.13",
    },
    "packages": [
        "static_ffmpeg",
    ],
    "includes": [
        "sd2_to_flac",
        "sd2_to_flac_gui",
    ],
}

# Only set iconfile if one actually exists
ICON_FILE = "icon.icns"
if os.path.isfile(ICON_FILE):
    OPTIONS["iconfile"] = ICON_FILE

setup(
    name="SD2 to FLAC",
    app=APP,
    data_files=FFMPEG_DATA_FILES,
    options={"py2app": OPTIONS},
)
