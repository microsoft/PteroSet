"""Download and extract PteroSet from Zenodo.

Downloads individual files using parallel chunked HTTP range requests.

Usage:

    python download_data.py
    python download_data.py --output-dir ./my_data --keep-zip
    python download_data.py --workers 16

"""

import argparse
import csv
import hashlib
import logging
import shutil
import time
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ── Configuration ────────────────────────────────────────────────────────────

# Zenodo "concept" record ID: stays constant across all versions of the
# dataset, so resolving it via /versions/latest always gets the newest one.
CONCEPT_RECORD_ID = "18554918"

# The results reported in the paper were obtained from the 48 kHz audio of the
# first dataset version. 
PAPER_AUDIO_RECORD_ID = "18563039"
PAPER_AUDIO_ZIP = "audios_48khz.zip"
PAPER_AUDIO_SAMPLE_RATE = 48000

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

def get_file_list(zenodo_files_url) -> list[dict]:
    """Fetch the list of files in the Zenodo record."""
    resp = _make_session().get(zenodo_files_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["entries"]


def get_latest_files_url(concept_record_id: str) -> str:
    """Resolve the files URL of the latest version of a Zenodo record series."""
    resp = _make_session().get(
        f"https://zenodo.org/api/records/{concept_record_id}/versions/latest",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["links"]["files"]


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


def download_entry(entry: dict, dest: Path, workers: int, keep_zip: bool) -> None:
    """Download one Zenodo file entry into *dest*, verify it and extract ZIPs."""
    filename: str = entry["key"]
    size: int = entry["size"]
    content_url: str = entry["links"]["content"]
    expected_md5: str = entry["checksum"].removeprefix("md5:")

    print(f"\nDownloading {filename} ({size / (1024 ** 3):.2f} GB) ...")

    if size > CHUNK_SIZE:
        file_path = download_file_parallel(content_url, dest, filename, size, workers)
    else:
        file_path = download_file_simple(content_url, dest, filename)

    print(f"Verifying checksum for {filename} ...")
    if verify_checksum(file_path, expected_md5):
        print(f"Checksum OK.")
    else:
        log.warning("Checksum mismatch for %s!", filename)

    if filename.endswith(".zip"):
        extract_zip(file_path, dest)
        if not keep_zip:
            file_path.unlink()
            print(f"Removed {filename}.")


# ── 48 kHz audio (paper results) ─────────────────────────────────────────────

def read_audio_projects(dest: Path) -> dict[str, str]:
    """Return ``{audio_file: project_name}`` from the metadata*.csv in *dest*."""
    candidates = sorted(dest.glob("metadata*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metadata*.csv file found in {dest}")
    with open(candidates[0], newline="", encoding="utf-8-sig") as fh:
        return {row["audio_file"]: row["project_name"] for row in csv.DictReader(fh)}


def _wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate()


def check_no_other_audio(dest: Path, audio_projects: dict[str, str]) -> None:
    """Fail if a project folder already holds audio at another sample rate.

    Mixing 48 kHz and 192 kHz files in the same folders would silently change
    the spectrograms, so the existing audio has to be moved away first.
    """
    for audio_file, project in audio_projects.items():
        existing = dest / project / audio_file
        if existing.exists() and _wav_sample_rate(existing) != PAPER_AUDIO_SAMPLE_RATE:
            raise SystemExit(
                f"{existing} already exists with a sample rate of "
                f"{_wav_sample_rate(existing)} Hz. Move or rename the existing "
                f"audio folders in {dest} before downloading the 48 kHz audio."
            )


def place_audio_in_project_folders(
    dest: Path, audio_projects: dict[str, str]
) -> None:
    """Move the flat 48 kHz WAVs into ``<dest>/<project>/``.

    The 48 kHz release stores every WAV in a single ``audios_48khz/`` folder,
    while the pipeline expects ``<project>/<audio_file>``.
    """
    flat_dir = dest / Path(PAPER_AUDIO_ZIP).stem
    missing = [f for f in audio_projects if not (flat_dir / f).exists()
               and not (dest / audio_projects[f] / f).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} audio file(s) listed in the metadata are missing "
            f"from {flat_dir} (e.g. {missing[0]})."
        )

    moved = 0
    for audio_file, project in audio_projects.items():
        source = flat_dir / audio_file
        target = dest / project / audio_file
        if not source.exists():
            continue  # already placed by a previous run
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved += 1

    if flat_dir.is_dir() and not any(flat_dir.iterdir()):
        flat_dir.rmdir()
    print(f"Placed {moved} audio file(s) into their project folders in {dest}.")


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
    parser.add_argument(
        "--record_id",
        type=str,
        default=None,
        help=f"Zenodo record ID (default: last version).",
    )
    parser.add_argument(
        "--audio",
        type=str,
        choices=["192khz", "48khz"],
        default="192khz",
        help=(
            "Audio to download. '192khz' (default) gets the original recordings "
            "of the selected record. '48khz' gets the 48 kHz audio of record "
            f"{PAPER_AUDIO_RECORD_ID} instead."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if args.record_id is None:
        print("Fetching file list from Zenodo's latest version ...")
        zenodo_files_url = get_latest_files_url(CONCEPT_RECORD_ID)
    else:
        print(f"Fetching file list for Zenodo record {args.record_id} ...")
        zenodo_files_url = f"https://zenodo.org/api/records/{args.record_id}/files"

    files = get_file_list(zenodo_files_url)

    paper_audio = None
    if args.audio == "48khz":
        # The metadata tells which ZIPs hold the 192 kHz audio (<project>.zip)
        # and where each 48 kHz WAV belongs, so it has to be downloaded first.
        metadata_entries = [
            e for e in files
            if e["key"].startswith("metadata") and e["key"].endswith(".csv")
        ]
        if not metadata_entries:
            raise SystemExit("The Zenodo record has no metadata*.csv file.")
        for entry in metadata_entries:
            download_entry(entry, dest, args.workers, args.keep_zip)
        audio_projects = read_audio_projects(dest)
        check_no_other_audio(dest, audio_projects)

        project_zips = {f"{project}.zip" for project in audio_projects.values()}
        files = [
            e for e in files
            if e["key"] not in project_zips and e not in metadata_entries
        ]

        paper_files_url = (
            f"https://zenodo.org/api/records/{PAPER_AUDIO_RECORD_ID}/files"
        )
        paper_audio = next(
            e for e in get_file_list(paper_files_url) if e["key"] == PAPER_AUDIO_ZIP
        )
        print(
            f"Using the 48 kHz audio from Zenodo record {PAPER_AUDIO_RECORD_ID} "
            f"instead of {', '.join(sorted(project_zips))}."
        )

    print(f"Found {len(files)} file(s):")
    for entry in files + ([paper_audio] if paper_audio else []):
        print(f"  • {entry['key']}  ({entry['size'] / (1024 ** 2):.1f} MB)")

    for entry in files:
        download_entry(entry, dest, args.workers, args.keep_zip)

    if paper_audio is not None:
        download_entry(paper_audio, dest, args.workers, args.keep_zip)
        place_audio_in_project_folders(dest, audio_projects)

    print("\nDone.")


if __name__ == "__main__":
    main()
