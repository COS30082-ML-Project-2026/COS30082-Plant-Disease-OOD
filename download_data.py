"""Download and extract DPD-Small from a public Hugging Face dataset repo.

Hugging Face Dataset:
https://huggingface.co/datasets/print1Username/COS30082-Machine-Learning-Project.dataset

Install: python -m pip install -U huggingface_hub py7zr tqdm
Run:     python download_data.py

The archive must contain three CSVs and DPD_Small_dataset/ at its root.
A manifest is saved after successful extraction so future runs can skip it.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

REPO_ID = "print1Username/COS30082-Machine-Learning-Project.dataset"
ARCHIVE_NAME = "DPD_Small.7z"
DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST = DATA_DIR / ".dpd_small_manifest.json"
CSV_NAMES = (
    "DPD_Small_train.csv",
    "DPD_Small_test_seen.csv",
    "DPD_Small_test_unseen.csv",
)
IMAGE_DIR = "DPD_Small_dataset"


def validate_structure(root: Path) -> None:
    """Check required files and ensure at least one image exists."""
    missing = [name for name in CSV_NAMES if not (root / name).is_file() or (root / name).stat().st_size == 0]
    images = root / IMAGE_DIR
    if not images.is_dir() or not any(p.is_file() for p in images.rglob("*")):
        missing.append(IMAGE_DIR + "/ (nonempty)")
    if missing:
        raise ValueError("Missing or empty dataset entries: " + ", ".join(missing))


def manifest_files(root: Path) -> dict[str, int]:
    paths = [root / name for name in CSV_NAMES]
    paths.extend(p for p in (root / IMAGE_DIR).rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in paths}


def already_complete() -> bool:
    if not MANIFEST.is_file():
        return False
    try:
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if saved.get("repo_id") != REPO_ID or saved.get("archive") != ARCHIVE_NAME:
            return False
        validate_structure(DATA_DIR)
        files = saved.get("files")
        return isinstance(files, dict) and bool(files) and files == manifest_files(DATA_DIR)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def safe_archive_names(names: list[str]) -> None:
    for name in names:
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or ":" in path.parts[0]):
            raise ValueError(f"Unsafe archive path: {name!r}")


def extract_archive(archive_path: Path, destination: Path) -> None:
    import py7zr
    from tqdm import tqdm

    class ProgressCallback:
        def __init__(self, bar):
            self.bar = bar

        def report_start_preparation(self, *args):
            pass

        def report_start(self, *args):
            pass

        def report_update(self, *args):
            pass

        def report_end(self, *args):
            self.bar.update(1)

        def report_postprocess(self, *args):
            pass

    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        names = archive.getnames()
        safe_archive_names(names)
        file_count = sum(not name.endswith("/") for name in names)
        with tqdm(total=file_count, desc="Extracting", unit="file") as bar:
            archive.extractall(path=destination, callback=ProgressCallback(bar))


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
        import py7zr  # noqa: F401; fail early with installation guidance
        import tqdm  # noqa: F401
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print("Install: python -m pip install -U huggingface_hub py7zr tqdm", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if already_complete():
        print(f"Dataset already complete in {DATA_DIR}; skipping download and extraction.")
        return 0

    archive_path = DATA_DIR / ARCHIVE_NAME
    try:
        print(f"Downloading {ARCHIVE_NAME} from {REPO_ID} ...", flush=True)
        # The HF client displays a byte-based download progress bar.
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=ARCHIVE_NAME,
            repo_type="dataset",
            local_dir=DATA_DIR,
            token=False,
        )
        archive_path = Path(downloaded)

        # Extract into a temporary folder so a failed extraction does not
        # masquerade as a complete dataset on the next run.
        with tempfile.TemporaryDirectory(prefix="dpd_extract_", dir=DATA_DIR) as temp:
            staging = Path(temp)
            extract_archive(archive_path, staging)
            validate_structure(staging)
            extracted = manifest_files(staging)
            for name in (*CSV_NAMES, IMAGE_DIR):
                target = DATA_DIR / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                shutil.move(str(staging / name), str(target))

        MANIFEST.write_text(
            json.dumps({"repo_id": REPO_ID, "archive": ARCHIVE_NAME,
                        "files": extracted}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        archive_path.unlink(missing_ok=True)
        print(f"\nDone. Dataset ready at: {DATA_DIR}")
        return 0
    except Exception as exc:
        print(f"\nDataset setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check repository access, archive contents, free disk space, and network.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
