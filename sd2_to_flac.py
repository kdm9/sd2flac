#!/usr/bin/env python3
"""
sd2_to_flac.py — Convert Sound Designer II (.sd2) split-stereo files to FLAC.

Scans a directory (recursively) for matching .L.sd2 / .R.sd2 pairs, guesses
the PCM encoding (bit depth + endianness) by analysing sample statistics,
then shells out to ffmpeg to merge and encode as FLAC.

On macOS / HFS+ volumes the original SD2 resource fork may still be
attached.  Use --xattr to attempt to read sample rate, bit depth, and
channel info from the 'com.apple.ResourceFork' extended attribute.

Usage (CLI):
    sd2-to-flac INPUT_DIR [OUTPUT_DIR] [OPTIONS]

Usage (library):
    from sd2_to_flac import main
    main(input_dir="./raw_audio", output_dir="./converted", sample_rate=48000)

Requirements:  Python 3.7+.  ffmpeg is provided automatically via static-ffmpeg.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from statistics import mean, variance

# ── Candidate encodings ──────────────────────────────────────────────────────

ALL_CANDIDATES = {
    "16be": {"label": "16-bit big-endian",    "bps": 2, "endian": "big",    "bits": 16},
    "16le": {"label": "16-bit little-endian", "bps": 2, "endian": "little", "bits": 16},
    "24be": {"label": "24-bit big-endian",    "bps": 3, "endian": "big",    "bits": 24},
    "24le": {"label": "24-bit little-endian", "bps": 3, "endian": "little", "bits": 24},
    "32be": {"label": "32-bit big-endian",    "bps": 4, "endian": "big",    "bits": 32},
    "32le": {"label": "32-bit little-endian", "bps": 4, "endian": "little", "bits": 32},
}

DEFAULT_CANDIDATES = ["16be", "16le", "24be", "24le", "32be", "32le"]

# Bytes to read from each file for analysis
PROBE_BYTES = 65536

log = logging.getLogger("sd2_to_flac")
log.setLevel(logging.DEBUG)


# ── Decoding helpers ─────────────────────────────────────────────────────────

def decode_16(data, big_endian):
    """Decode raw bytes as 16-bit signed PCM."""
    n = len(data) // 2
    fmt = "{}{}h".format(">" if big_endian else "<", n)
    return list(struct.unpack(fmt, data[: n * 2]))


def decode_32(data, big_endian):
    """Decode raw bytes as 32-bit signed PCM."""
    n = len(data) // 4
    fmt = "{}{}i".format(">" if big_endian else "<", n)
    return list(struct.unpack(fmt, data[: n * 4]))


def decode_24(data, big_endian):
    """Decode raw bytes as 24-bit signed PCM (no native struct support)."""
    n = len(data) // 3
    samples = []
    for i in range(n):
        off = i * 3
        b0, b1, b2 = data[off], data[off + 1], data[off + 2]
        if big_endian:
            val = (b0 << 16) | (b1 << 8) | b2
        else:
            val = (b2 << 16) | (b1 << 8) | b0
        if val & 0x800000:
            val -= 0x1000000
        samples.append(val)
    return samples


def decode_samples(data, candidate):
    """Decode raw bytes into integer samples for a given candidate encoding."""
    bps = candidate["bps"]
    big = candidate["endian"] == "big"
    usable = len(data) - (len(data) % bps)
    if usable < bps * 128:
        return []
    chunk = data[:usable]
    if bps == 2:
        return decode_16(chunk, big)
    elif bps == 3:
        return decode_24(chunk, big)
    elif bps == 4:
        return decode_32(chunk, big)
    return []


# ── Scoring ──────────────────────────────────────────────────────────────────

def autocorrelation(samples, lag):
    """Compute normalised autocorrelation at a given lag."""
    n = len(samples)
    if n <= lag:
        return 0.0
    m = mean(samples)
    var = sum((s - m) ** 2 for s in samples) / n
    if var == 0:
        return 0.0
    cov = sum((samples[i] - m) * (samples[i + lag] - m)
              for i in range(n - lag)) / (n - lag)
    return cov / var


def byte_entropy(data, offset, stride):
    """
    Compute Shannon entropy of bytes at positions offset, offset+stride,
    offset+2*stride, ... within data.  Returns bits.
    """
    counts = [0] * 256
    n = 0
    i = offset
    while i < len(data):
        counts[data[i]] += 1
        n += 1
        i += stride
    if n == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def sub_sample_independence(data, candidate):
    """
    Test whether the upper and lower halves of each sample behave
    independently.  If data is really N-bit read as 2N-bit, the upper
    and lower N-bit halves are actually independent samples, so their
    cross-correlation is near zero.  True 2N-bit data has strongly
    correlated upper/lower parts.

    Returns a score in [0, 1] where higher means MORE independent
    (i.e. more likely to be an aliased smaller bit-depth).
    """
    bps = candidate["bps"]
    if bps < 4:
        return 0.0  # Only relevant for 32-bit candidates

    big = candidate["endian"] == "big"
    n = len(data) // bps
    if n < 128:
        return 0.0

    # Split each 32-bit sample into its high-16 and low-16 parts
    uppers = []
    lowers = []
    for i in range(min(n, 4096)):
        off = i * bps
        if big:
            hi = struct.unpack_from(">h", data, off)[0]
            lo = struct.unpack_from(">h", data, off + 2)[0]
        else:
            lo = struct.unpack_from("<h", data, off)[0]
            hi = struct.unpack_from("<h", data, off + 2)[0]
        uppers.append(hi)
        lowers.append(lo)

    # Cross-correlation between upper and lower halves
    if len(uppers) < 4:
        return 0.0
    mu = mean(uppers)
    ml = mean(lowers)
    vu = sum((u - mu) ** 2 for u in uppers)
    vl = sum((l - ml) ** 2 for l in lowers)
    if vu == 0 or vl == 0:
        return 0.0
    cov = sum((uppers[i] - mu) * (lowers[i] - ml) for i in range(len(uppers)))
    cross_corr = cov / math.sqrt(vu * vl)

    # Low cross-correlation → independent halves → likely aliased
    return max(0.0, 1.0 - abs(cross_corr))


def byte_entropy_uniformity(data, candidate):
    """
    In true N-bit audio, different byte positions within a sample have
    different entropy (MSB is lower entropy, LSB is higher).  When
    smaller-bit data is read at a larger frame size, the byte positions
    show more uniform entropy because they're actually from independent
    samples.

    Returns a score where higher = more uniform = more likely aliased.
    """
    bps = candidate["bps"]
    if bps < 3:
        return 0.0  # Only meaningful for 24-bit and 32-bit

    # Compute entropy of each byte position within the sample frame
    entropies = []
    for b in range(bps):
        entropies.append(byte_entropy(data, b, bps))

    if len(entropies) < 2:
        return 0.0

    # Measure how uniform the entropies are across byte positions.
    # In true wide audio, MSB entropy < LSB entropy (more variation).
    # In aliased data, all positions have similar entropy.
    ent_range = max(entropies) - min(entropies)
    # Normalise: 8 bits max entropy, so max possible range is ~8
    return max(0.0, 1.0 - ent_range / 4.0)


def compute_score(samples, data, candidate):
    """
    Compute a composite score for a candidate encoding.  Lower = better.

    Components:
    1. Normalised delta variance (smoothness) — primary signal
    2. Autocorrelation ratio penalty — lag-1 should be > lag-2 for true framing
    3. Sub-sample independence penalty — detects N-bit aliased as 2N-bit
    4. Byte entropy uniformity penalty — detects uniform byte entropy from aliasing
    5. Bit-depth prior — slight preference for smaller bit depths (Occam's razor)
    """
    n = len(samples)
    if n < 128:
        return float("inf")

    # --- (1) Normalised delta variance ---
    # Use a subsample for speed on very long probes
    cap = min(n, 8192)
    sub = samples[:cap]
    deltas = [sub[i + 1] - sub[i] for i in range(len(sub) - 1)]
    lo, hi = min(sub), max(sub)
    val_range = hi - lo
    if val_range == 0:
        return float("inf")
    try:
        dvar = variance(deltas)
    except Exception:
        return float("inf")
    norm_delta = dvar / (val_range * val_range)

    # --- (2) Autocorrelation ratio ---
    ac1 = autocorrelation(sub, 1)
    ac2 = autocorrelation(sub, 2)
    ac_penalty = 0.0
    if ac1 < 0.5:
        ac_penalty += 0.3 * (0.5 - ac1)
    if ac1 < ac2 and ac2 > 0:
        ac_penalty += 0.2 * (ac2 - ac1)

    # --- (3) Sub-sample independence (32-bit candidates only) ---
    indep = sub_sample_independence(data, candidate)
    indep_penalty = 0.5 * indep

    # --- (4) Byte entropy uniformity (24/32-bit candidates) ---
    ent_uniform = byte_entropy_uniformity(data, candidate)
    ent_penalty = 0.3 * max(0.0, ent_uniform - 0.7)

    # --- (5) Bit-depth prior (very mild Occam's razor) ---
    bits = candidate["bits"]
    prior = {16: 0.0, 24: 0.001, 32: 0.005}.get(bits, 0.005)

    score = norm_delta + ac_penalty + indep_penalty + ent_penalty + prior

    log.debug(
        "    %-24s  delta=%.4e  ac1=%.3f  ac2=%.3f  "
        "ac_pen=%.4f  indep=%.4f  ent_pen=%.4f  total=%.4e",
        candidate["label"], norm_delta, ac1, ac2,
        ac_penalty, indep_penalty, ent_penalty, score,
    )

    return score


# ── Encoding detection ───────────────────────────────────────────────────────

def guess_encoding(file_paths, candidates, probe_bytes=PROBE_BYTES):
    """
    Jointly guess the encoding across one or more files (e.g. L + R).
    Returns the candidate dict with the lowest combined score.
    """
    best_candidate = None
    best_score = float("inf")

    raw_data = {}
    for fp in file_paths:
        with open(str(fp), "rb") as f:
            raw_data[fp] = f.read(probe_bytes)

    for cand in candidates:
        total = 0.0
        valid = True
        for fp in file_paths:
            samples = decode_samples(raw_data[fp], cand)
            if not samples:
                valid = False
                break
            total += compute_score(samples, raw_data[fp], cand)
        if not valid:
            continue
        avg = total / len(file_paths)
        if avg < best_score:
            best_score = avg
            best_candidate = cand

    return best_candidate


# ── xattr / resource fork parsing ────────────────────────────────────────────

def read_xattr(filepath, attr_name):
    """Read an extended attribute from a file.  Returns bytes or None."""
    if hasattr(os, "getxattr"):
        try:
            return os.getxattr(str(filepath), attr_name)
        except OSError:
            return None

    try:
        result = subprocess.run(
            ["xattr", "-px", attr_name, str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            hex_str = result.stdout.decode("ascii", errors="ignore")
            hex_str = re.sub(r"\s+", "", hex_str)
            return bytes.fromhex(hex_str)
    except FileNotFoundError:
        pass

    return None


def parse_sd2_resource_fork(rfork_data):
    """
    Attempt to extract audio parameters from an SD2 resource fork.

    Returns a dict with any of: sample_rate, bits, channels (or empty dict).
    """
    info = {}

    if len(rfork_data) < 16:
        return info

    try:
        data_off, map_off, data_len, map_len = struct.unpack_from(
            ">IIII", rfork_data, 0
        )
    except struct.error:
        return info

    log.debug("  Resource fork: data_off=%d map_off=%d data_len=%d map_len=%d",
              data_off, map_off, data_len, map_len)

    common_rates = {44100.0, 48000.0, 96000.0, 88200.0, 22050.0,
                    32000.0, 16000.0, 11025.0, 8000.0, 176400.0, 192000.0}

    for i in range(0, len(rfork_data) - 7):
        try:
            val = struct.unpack_from(">d", rfork_data, i)[0]
            if val in common_rates:
                info["sample_rate"] = int(val)
                log.debug("  Found sample rate %.0f at offset %d (float64)", val, i)
                break
        except struct.error:
            continue

    if "sample_rate" not in info:
        for i in range(0, len(rfork_data) - 9):
            try:
                ext_bytes = rfork_data[i:i + 10]
                sign = (ext_bytes[0] >> 7) & 1
                exponent = ((ext_bytes[0] & 0x7F) << 8) | ext_bytes[1]
                mantissa = struct.unpack_from(">Q", ext_bytes, 2)[0]
                if exponent == 0 or exponent == 0x7FFF:
                    continue
                f = mantissa / (1 << 63) * (2.0 ** (exponent - 16383))
                if sign:
                    f = -f
                if f in common_rates:
                    info["sample_rate"] = int(f)
                    log.debug("  Found sample rate %.0f at offset %d (80-bit)", f, i)
                    break
            except (struct.error, OverflowError, ZeroDivisionError):
                continue

    valid_bits = {8, 16, 24, 32}
    for i in range(0, len(rfork_data) - 1):
        try:
            val = struct.unpack_from(">H", rfork_data, i)[0]
            if val in valid_bits:
                if i >= map_off or i < data_off:
                    info.setdefault("bits", val)
                    log.debug("  Found candidate bit depth %d at offset %d", val, i)
                    break
        except struct.error:
            continue

    return info


def try_xattr_metadata(filepath):
    """
    Try to read SD2 metadata from the file's extended attributes.
    Returns a dict with any discovered parameters.
    """
    info = {}

    rfork = read_xattr(filepath, "com.apple.ResourceFork")
    if rfork:
        log.debug("  Found resource fork xattr (%d bytes) on %s",
                  len(rfork), filepath.name)
        info.update(parse_sd2_resource_fork(rfork))
    else:
        rfork_path = str(filepath) + "/..namedfork/rsrc"
        try:
            with open(rfork_path, "rb") as f:
                rfork = f.read()
            if rfork:
                log.debug("  Found named fork (%d bytes) on %s",
                          len(rfork), filepath.name)
                info.update(parse_sd2_resource_fork(rfork))
        except (OSError, IOError):
            log.debug("  NO resource fork xattr on %s", filepath.name)
            pass

    return info


# ── File discovery ───────────────────────────────────────────────────────────

def find_pairs(input_dir):
    """
    Recursively find .L.sd2 / .R.sd2 pairs (and solo mono .sd2 files).

    Returns a dict keyed by ``(relative_dir, base_name)`` where
    *relative_dir* is a :class:`Path` relative to *input_dir* (or ``"."``
    for the root) and *base_name* is the stem without the ``.L`` / ``.R``
    suffix.  Values are dicts ``{"L": path, "R": path}`` with either key
    optional.
    """
    sd2_lr = re.compile(r"^(.+)\.(L|R)(\.sd2)?$", re.IGNORECASE)
    pairs = {}

    input_dir = Path(input_dir)

    for entry in sorted(input_dir.rglob("*")):
        if not entry.is_file():
            continue
        m = sd2_lr.match(entry.name)
        if not m:
            continue
        base = m.group(1)
        channel = m.group(2).upper()
        # Compute the directory path relative to input_dir
        try:
            rel_dir = entry.parent.relative_to(input_dir)
        except ValueError:
            rel_dir = Path(".")
        key = (rel_dir, base)
        pairs.setdefault(key, {})
        pairs[key][channel] = entry

    return pairs


# ── ffmpeg conversion ────────────────────────────────────────────────────────

def ffmpeg_raw_format(encoding):
    """Return the ffmpeg -f format string for a raw PCM encoding."""
    bits = encoding["bits"]
    endian = "be" if encoding["endian"] == "big" else "le"
    return "s{}{}".format(bits, endian)


def convert_pair(left, right, output, encoding, sample_rate, dry_run=False):
    """
    Use ffmpeg to merge L + R raw PCM into a stereo FLAC.
    If only one channel exists, produce a mono FLAC.
    """
    raw_fmt = ffmpeg_raw_format(encoding)
    channels = [p for p in (left, right) if p is not None]
    if not channels:
        return False

    cmd = ["ffmpeg", "-y"]

    for ch_path in channels:
        cmd.extend([
            "-f", raw_fmt,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", str(ch_path),
        ])

    if len(channels) == 2:
        cmd.extend([
            "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]",
            "-map", "[aout]",
        ])

    cmd.extend(["-c:a", "flac", "-compression_level", "12", str(output)])

    log.info("  cmd: %s", " ".join(cmd))

    if dry_run:
        return True

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            log.error("ffmpeg failed for %s:\n%s", output.name,
                      result.stderr.decode("utf-8", errors="replace"))
            return False
        return True
    except FileNotFoundError:
        log.error("ffmpeg not found on PATH.")
        return False


# ── Main (library entry point) ───────────────────────────────────────────────

def _ensure_ffmpeg():
    """Put static-ffmpeg's bundled binaries on PATH if available."""
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        log.debug("static_ffmpeg: paths added")
    except Exception as exc:
        log.warning("Could not initialise static_ffmpeg (%s); "
                    "falling back to system ffmpeg.", exc)


def main(
    input_dir,
    output_dir=None,
    sample_rate=None,
    encoding=None,
    xattr=False,
    probe_bytes=PROBE_BYTES,
    dry_run=False,
    verbose=False,
    include_input_dirname=False,
):
    """
    Convert split-stereo .sd2 files to FLAC.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing .L.sd2 / .R.sd2 files (searched recursively).
    output_dir : str, Path, or None
        Output directory.  The subdirectory structure under *input_dir* is
        mirrored here.  If ``None``, FLAC files are written next to the
        original .sd2 files.
    sample_rate : int or None
        Sample rate in Hz.  Overrides xattr-detected rate.
        Defaults to 44100 if not detected.
    encoding : list of str, or None
        Candidate encoding key(s) from ALL_CANDIDATES.
        If one value is given it is used directly; if multiple are given
        the best is auto-selected.  ``None`` means try all.
    xattr : bool
        Attempt to read metadata from extended attributes / resource fork.
    probe_bytes : int
        Bytes to read per file for encoding detection.
    dry_run : bool
        Print ffmpeg commands without running them.
    verbose : bool
        Enable debug-level logging.
    include_input_dirname : bool
        When *output_dir* is set, include the basename of *input_dir* as
        an extra directory level under *output_dir*.  For example, if
        *input_dir* is ``/data/audio1`` and *output_dir* is ``/out``, a
        file at ``day1/rec.L.sd2`` would be written to
        ``/out/audio1/day1/rec.flac`` instead of ``/out/day1/rec.flac``.
        Ignored when *output_dir* is ``None``.

    Returns
    -------
    tuple of (int, int)
        (ok_count, fail_count)
    """
    input_dir = Path(input_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)

    # Configure logging level (handler should already be attached by caller)
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    _ensure_ffmpeg()

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        return (0, 0)

    # Resolve candidate list
    if encoding:
        candidate_list = [ALL_CANDIDATES[k] for k in encoding]
    else:
        candidate_list = [ALL_CANDIDATES[k] for k in DEFAULT_CANDIDATES]

    # Discover file pairs (recursively)
    pairs = find_pairs(input_dir)
    if not pairs:
        log.error("No .L.sd2 / .R.sd2 files found in %s", input_dir)
        return (0, 0)

    log.info("Found %d file pair(s) in %s (recursive)", len(pairs), input_dir)
    log.info("Candidate encodings: %s\n",
             ", ".join(c["label"] for c in candidate_list))

    ok_count = 0
    fail_count = 0

    for (rel_dir, base_name), channels in sorted(pairs.items()):
        left = channels.get("L")
        right = channels.get("R")
        ch_desc = "stereo" if (left and right) else "mono"

        display_name = str(rel_dir / base_name) if str(rel_dir) != "." else base_name
        log.info("── %s (%s) ──", display_name, ch_desc)

        if left and right:
            ls, rs = left.stat().st_size, right.stat().st_size
            if ls != rs:
                log.warning("  Warning: L/R sizes differ (L=%d, R=%d bytes)",
                            ls, rs)

        # ── xattr metadata ───────────────────────────────────────────
        xattr_info = {}
        if xattr:
            for fp in [left, right]:
                if fp is not None:
                    xi = try_xattr_metadata(fp)
                    if xi:
                        log.info("  xattr metadata from %s: %s", fp.name, xi)
                        for k, v in xi.items():
                            xattr_info.setdefault(k, v)

        # ── Determine encoding ───────────────────────────────────────
        if len(candidate_list) == 1:
            enc = candidate_list[0]
            log.info("  Encoding (specified): %s", enc["label"])
        else:
            active_candidates = candidate_list
            if "bits" in xattr_info:
                xbits = xattr_info["bits"]
                narrowed = [c for c in candidate_list if c["bits"] == xbits]
                if narrowed:
                    log.info("  xattr bit depth %d narrows candidates to: %s",
                             xbits,
                             ", ".join(c["label"] for c in narrowed))
                    active_candidates = narrowed

            probe_files = [p for p in (left, right) if p is not None]
            log.info("  Detecting encoding from %d channel(s)...",
                     len(probe_files))
            enc = guess_encoding(probe_files, active_candidates, probe_bytes)
            if enc is None:
                log.error("  Could not determine encoding — skipping.")
                fail_count += 1
                continue
            log.info("  Detected: %s", enc["label"])

        # ── Determine sample rate ────────────────────────────────────
        if sample_rate:
            sr = sample_rate
        elif "sample_rate" in xattr_info:
            sr = xattr_info["sample_rate"]
            log.info("  Sample rate from xattr: %d Hz", sr)
        else:
            sr = 44100
        log.info("  Sample rate: %d Hz", sr)

        # ── Determine output path ────────────────────────────────────
        if output_dir is not None:
            # Mirror the subdirectory structure under output_dir
            if include_input_dirname:
                dest_dir = output_dir / input_dir.name / rel_dir
            else:
                dest_dir = output_dir / rel_dir
        else:
            # No output dir given — write next to the originals
            any_file = left if left is not None else right
            dest_dir = any_file.parent

        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / (base_name + ".flac")

        # ── Convert ──────────────────────────────────────────────────
        if convert_pair(left, right, out_path, enc, sr, dry_run=dry_run):
            action = "Would create" if dry_run else "Created"
            log.info("  %s: %s\n", action, out_path)
            ok_count += 1
        else:
            log.error("  Failed: %s\n", out_path)
            fail_count += 1

    log.info("Done. %d succeeded, %d failed.", ok_count, fail_count)
    return (ok_count, fail_count)


# ── CLI entry point ──────────────────────────────────────────────────────────

def cli():
    """Command-line interface — parses sys.argv and calls main()."""
    parser = argparse.ArgumentParser(
        description="Convert split-stereo .sd2 files to FLAC via ffmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s ./raw_audio ./converted
  %(prog)s ./raw_audio --sample-rate 48000
  %(prog)s ./raw_audio --encoding 24be 16be
  %(prog)s ./raw_audio --xattr
  %(prog)s ./raw_audio --dry-run --verbose
  %(prog)s ./raw_audio ./out --include-input-dirname
""",
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Directory containing .L.sd2 / .R.sd2 files (searched recursively)")
    parser.add_argument(
        "output_dir", type=Path, nargs="?", default=None,
        help="Output directory (default: write next to originals)")
    parser.add_argument(
        "-r", "--sample-rate", type=int, default=None,
        help="Sample rate in Hz.  Overrides xattr-detected rate.  "
             "Default 44100 if not detected.")
    parser.add_argument(
        "--encoding", nargs="+", default=None, metavar="ENC",
        choices=list(ALL_CANDIDATES.keys()),
        help="Candidate encoding(s) to try.  Choices: %(choices)s.  Default: all.")
    parser.add_argument(
        "--xattr", action="store_true",
        help="Attempt to read sample rate and bit depth from the file's "
             "extended attributes / resource fork (macOS / HFS+).")
    parser.add_argument(
        "--probe-bytes", type=int, default=PROBE_BYTES,
        help="Bytes to read per file for encoding detection "
             "(default: %(default)s)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print ffmpeg commands without running them")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show debug output including per-candidate scores")
    parser.add_argument(
        "--include-input-dirname", action="store_true",
        help="Include the input directory's basename as an extra level "
             "in the output directory structure.  Only effective when "
             "an output directory is specified.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    _ok, fail = main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        encoding=args.encoding,
        xattr=args.xattr,
        probe_bytes=args.probe_bytes,
        dry_run=args.dry_run,
        verbose=args.verbose,
        include_input_dirname=args.include_input_dirname,
    )
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    cli()
