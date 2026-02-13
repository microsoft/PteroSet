"""Download and extract the Humboldt - AI4G Bioacoustics Dataset from Zenodo.

Downloads individual files using parallel chunked HTTP range requests.

Usage:

    python download_data.py
    python download_data.py --output-dir ./my_data --keep-zip
    python download_data.py --workers 16
"""

import argparse
import hashlib
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ── Configuration ────────────────────────────────────────────────────────────

RECORD_ID = "18563039"
ZENODO_FILES_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files"
DOWNLOAD_DIR = Path(__file__).resolve().parent

DEFAULT_WORKERS = 8
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per parallel chunk
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds, doubles each attempt
REQUEST_TIMEOUT = 120  # seconds

log = logging.getLogger(__name__)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Return a :class:`requests.Session` with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ── Zenodo API ───────────────────────────────────────────────────────────────

def get_file_list() -> list[dict]:
    """Fetch the list of files in the Zenodo record."""
    resp = _make_session().get(ZENODO_FILES_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["entries"]


# ── Download ─────────────────────────────────────────────────────────────────

def _download_range(url: str, start: int, end: int, filepath: str) -> int:
    """Download a byte range into *filepath* at the correct offset.

    Retries up to ``MAX_RETRIES`` times with exponential back-off on any
    network or I/O error.
    """
    session = _make_session()
    headers = {"Range": f"bytes={start}-{end}"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            bytes_written = 0
            with open(filepath, "r+b") as fh:
                fh.seek(start)
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    bytes_written += len(chunk)
            return bytes_written
        except (requests.exceptions.RequestException, IOError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(
                "Chunk %d–%d failed (attempt %d/%d): %s — retrying in %ds",
                start, end, attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    return 0  # unreachable


def download_file_parallel(
    url: str,
    dest: Path,
    filename: str,
    total_size: int,
    num_workers: int,
) -> Path:
    """Download *url* into *dest/filename* using parallel HTTP range requests."""
    file_path = dest / filename

    with open(file_path, "wb") as fh:
        fh.truncate(total_size)

    ranges = [
        (start, min(start + CHUNK_SIZE - 1, total_size - 1))
        for start in range(0, total_size, CHUNK_SIZE)
    ]

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar, \
         ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_download_range, url, s, e, str(file_path)): (s, e)
            for s, e in ranges
        }
        for future in as_completed(futures):
            pbar.update(future.result())

    return file_path


def download_file_simple(url: str, dest: Path, filename: str) -> Path:
    """Download *url* into *dest/filename* in a single stream (small files)."""
    file_path = dest / filename
    resp = _make_session().get(url, stream=True, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(file_path, "wb") as fh, \
         tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            pbar.update(len(chunk))

    return file_path


# ── Post-download ────────────────────────────────────────────────────────────

def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Return ``True`` if *filepath*'s MD5 matches *expected_md5*."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract *zip_path* into *dest*."""
    print(f"Extracting {zip_path.name} → {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print(f"Extraction of {zip_path.name} complete.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Humboldt - AI4G Bioacoustics Dataset from Zenodo.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DOWNLOAD_DIR),
        help=f"Directory to save and extract data (default: {DOWNLOAD_DIR}).",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep ZIP files after extraction.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel connections per file (default: {DEFAULT_WORKERS}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list for Zenodo record {RECORD_ID} ...")
    files = get_file_list()
    print(f"Found {len(files)} file(s):")
    for entry in files:
        print(f"  • {entry['key']}  ({entry['size'] / (1024 ** 2):.1f} MB)")

    for entry in files:
        filename: str = entry["key"]
        size: int = entry["size"]
        content_url: str = entry["links"]["content"]
        expected_md5: str = entry["checksum"].removeprefix("md5:")

        print(f"\nDownloading {filename} ({size / (1024 ** 3):.2f} GB) ...")

        if size > CHUNK_SIZE:
            file_path = download_file_parallel(content_url, dest, filename, size, args.workers)
        else:
            file_path = download_file_simple(content_url, dest, filename)

        print(f"Verifying checksum for {filename} ...")
        if verify_checksum(file_path, expected_md5):
            print(f"Checksum OK.")
        else:
            log.warning("Checksum mismatch for %s!", filename)

        if filename.endswith(".zip"):
            extract_zip(file_path, dest)
            if not args.keep_zip:
                file_path.unlink()
                print(f"Removed {filename}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
