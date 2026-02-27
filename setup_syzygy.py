import os
import urllib.request
import sys

RAW_FILES = [
    "KBBvK",
    "KBvKB",
    "KNPvKN",
    "KPvK",
    "KQPvKN",
    "KQvKB",
    "KRNvK",
    "KRvK",
    "KBNvK",
    "KBvKN",
    "KNvK",
    "KPvKP",
    "KQPvKQ",
    "KQvKN",
    "KRPvK",
    "KRvKB",
    "KBPvK",
    "KBvKP",
    "KNvKN",
    "KQBvK",
    "KQPvKR",
    "KQvKP",
    "KRPvKB",
    "KRvKN",
    "KBPvKB",
    "KNNvK",
    "KNvKP",
    "KQNvK",
    "KQQvK",
    "KQvKQ",
    "KRPvKN",
    "KRvKP",
    "KBPvKN",
    "KNPvK",
    "KPPvK",
    "KQPvK",
    "KQRvK",
    "KQvKR",
    "KRPvKR",
    "KRvKR",
    "KBvK",
    "KNPvKB",
    "KPPvKP",
    "KQPvKB",
    "KQvK",
    "KRBvK",
    "KRRvK",
]

BASE_URL = "https://tablebase.sesse.net/syzygy/3-4-5/"
TARGET_DIR = os.path.join("engine", "assets", "syzygy")


def download_file(filename, folder):
    """Checks if file exists locally. If not, downloads it."""
    dest_path = os.path.join(TARGET_DIR, folder, filename)

    if os.path.exists(dest_path):
        print(f"  [Skip] {filename} (Already exists)")
        return

    url = BASE_URL + filename
    print(f"  [Down] {filename}...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(" OK")
    except Exception as e:
        print(f" Error: {e}")


def main():
    print(f"Checking {len(RAW_FILES) * 2} Syzygy files (WDL + DTZ)...")

    wdl_dir = os.path.join(TARGET_DIR, "wdl")
    dtz_dir = os.path.join(TARGET_DIR, "dtz")
    os.makedirs(wdl_dir, exist_ok=True)
    os.makedirs(dtz_dir, exist_ok=True)

    for base_name in RAW_FILES:
        download_file(f"{base_name}.rtbw", "wdl")
        download_file(f"{base_name}.rtbz", "dtz")

    print("\nSyzygy setup complete.")


if __name__ == "__main__":
    main()
