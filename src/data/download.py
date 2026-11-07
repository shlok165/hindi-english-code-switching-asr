"""Fetch the MUCS 2021 Subtask-2 Hindi-English archives from OpenSLR.

Standard library only, so it runs before the virtualenv exists.

Single connections are slow on uncached byte ranges (0.02-0.7 MB/s measured on
2026-11-07; recently read ranges look far faster, so short speed tests mislead).
Throughput scales with connection count, so archives are fetched as parallel
byte ranges across all mirrors, and a stalled range moves to the next mirror.

OpenSLR publishes no checksums here. Sizes are the Content-Length reported by
www.openslr.org, and the gzip CRC is checked before extraction because one
archive can be assembled from several hosts.

    python -m src.data.download                     # test + blind
    python -m src.data.download --only train
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import queue
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MUCS_DATA_ROOT", ROOT / "data"))
RAW = DATA / "raw"
EXTRACTED = DATA / "interim" / "extracted"
SUMS = ROOT / "docs" / "SHA256SUMS"

MIRRORS = (
    "https://www.openslr.org",
    "https://openslr.trmal.net",
    "https://openslr.elda.org",
)
PATH = "/resources/104/"
ARCHIVES = {
    "test": ("Hindi-English_test.tar.gz", 443_929_204),
    "blind": ("subtask2_blindtest_wReadme.tar.gz", 1_008_263_116),
    "train": ("Hindi-English_train.tar.gz", 7_332_628_101),
}
CHUNK = 1 << 20
RANGE_BYTES = 8 << 20
WORKERS = 12
MIN_RATE = 30_000  # bytes/s; slower ranges are retried on another mirror
GRACE_SECONDS = 15
MAX_ATTEMPTS = 40


def fetch_range(url: str, start: int, end: int, part: Path) -> None:
    """Fetch bytes start..end inclusive and write them into part."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    data = bytearray()
    began = time.time()
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise OSError(f"{url} ignored the Range header")
        while block := response.read(1 << 16):
            data += block
            elapsed = time.time() - began
            if elapsed > GRACE_SECONDS and len(data) / elapsed < MIN_RATE:
                raise TimeoutError(f"stalled at {len(data) / elapsed / 1e3:.0f} KB/s")
    if len(data) != end - start + 1:
        raise OSError(f"short range: {len(data)} of {end - start + 1} bytes")
    with part.open("r+b") as out:
        out.seek(start)
        out.write(data)


def download(name: str) -> Path:
    """Download one archive, resuming from any ranges already on disk."""
    filename, expected = ARCHIVES[name]
    dest = RAW / filename
    RAW.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size != expected:
            raise SystemExit(f"{dest} has the wrong size; delete it and retry")
        return dest

    part = dest.with_name(filename + ".part")
    journal = dest.with_name(filename + ".ranges")
    if not part.exists():
        with part.open("wb") as handle:
            handle.truncate(expected)
    done = set()
    if journal.exists():
        done = {int(line) for line in journal.read_text(encoding="utf-8").split()}

    starts = [s for s in range(0, expected, RANGE_BYTES) if s not in done]
    pending: queue.Queue[tuple[int, int]] = queue.Queue()
    for start in starts:
        pending.put((start, 0))
    lock = threading.Lock()
    remaining = [len(starts)]
    failed: list[str] = []
    began = time.time()
    total = len(starts)

    def worker(index: int) -> None:
        mirror = index % len(MIRRORS)
        while True:
            try:
                start, attempt = pending.get(timeout=2)
            except queue.Empty:
                with lock:
                    if remaining[0] == 0 or failed:
                        return
                continue
            end = min(start + RANGE_BYTES, expected) - 1
            url = MIRRORS[mirror] + PATH + filename
            try:
                fetch_range(url, start, end, part)
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                mirror = (mirror + 1) % len(MIRRORS)
                if attempt + 1 >= MAX_ATTEMPTS:
                    with lock:
                        failed.append(f"range {start}: {error}")
                    return
                pending.put((start, attempt + 1))
                continue
            with lock:
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(f"{start}\n")
                remaining[0] -= 1
                finished = total - remaining[0]
                if finished % 10 == 0 or remaining[0] == 0:
                    rate = finished * RANGE_BYTES / (time.time() - began) / 1e6
                    print(f"  {filename}: {finished}/{total} ranges, {rate:.2f} MB/s", flush=True)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed:
        raise SystemExit(f"[{name}] failed: {failed[0]}")
    part.replace(dest)
    journal.unlink()
    return dest


def verify_gzip(path: Path) -> None:
    """Read the whole stream; gzip raises if the CRC or length is wrong."""
    with gzip.open(path, "rb") as stream:
        while stream.read(16 * CHUNK):
            pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def record_checksum(path: Path) -> str:
    """Append path's digest to docs/SHA256SUMS, or fail if it has changed."""
    digest = sha256(path)
    SUMS.parent.mkdir(parents=True, exist_ok=True)
    known = {}
    if SUMS.exists():
        for line in SUMS.read_text(encoding="utf-8").splitlines():
            value, _, filename = line.partition("  ")
            known[filename] = value
    if path.name in known and known[path.name] != digest:
        raise SystemExit(f"{path.name}: checksum differs from {SUMS}")
    if path.name not in known:
        with SUMS.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{digest}  {path.name}\n")
    return digest


def extract(path: Path, name: str) -> Path:
    """Unpack into its own directory, refusing unsafe archive members."""
    target = EXTRACTED / name
    marker = target / ".extracted"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(target, filter="data")
    marker.write_text(path.name + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=ARCHIVES, default=["test", "blind"])
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    for name in args.only:
        path = download(name)
        verify_gzip(path)
        print(f"[{name}] gzip CRC ok, sha256 {record_checksum(path)}", flush=True)
        if not args.no_extract:
            print(f"[{name}] extracted to {extract(path, name)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
