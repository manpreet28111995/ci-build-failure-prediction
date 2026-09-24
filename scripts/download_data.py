#!/usr/bin/env python3
"""
download_data.py - TravisTorrent Figshare Dataset Downloader & Project Extractor.

Downloads the official TravisTorrent dump from Figshare and verifies its md5.
Build level extraction is done by scripts/extract_projects.py.
"""

import os
import sys
import gzip
import shutil
import hashlib
import argparse
import urllib.request
from typing import List, Optional

FIGSHARE_FILES = {
    "2016_snapshot": {
        "id": 34303127,
        "name": "travistorrent_5_3_2016.csv.gz",
        "url": "https://ndownloader.figshare.com/files/34303127",
        "size_bytes": 81902127,
        "md5": "4713d8b73b667a2c375b89c20f531fc0"
    },
    "2017_final": {
        "id": 34303124,
        "name": "final-2017-01-25.csv.gz",
        "url": "https://ndownloader.figshare.com/files/34303124",
        "size_bytes": 264607174,
        "md5": "5d359345fecf0bf47be6edb10b5aae58"
    }
}


def compute_md5(filepath: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(url: str, dest_path: str, expected_md5: Optional[str] = None):
    print(f"Downloading from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)

    import ssl
    ctx = ssl._create_unverified_context()

    temp_path = dest_path + ".tmp"
    with urllib.request.urlopen(req, context=ctx) as resp, open(temp_path, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb:.1f} MB / {total_mb:.1f} MB)")
                sys.stdout.flush()

    print("\nDownload complete.")
    if expected_md5:
        print("Verifying MD5 checksum...")
        computed = compute_md5(temp_path)
        if computed != expected_md5:
            os.remove(temp_path)
            raise ValueError(f"MD5 mismatch! Expected {expected_md5}, got {computed}")
        print("MD5 checksum verified successfully.")

    os.replace(temp_path, dest_path)


def extract_project(csv_gz_path: str, output_csv_path: str, project_names: List[str], max_rows: Optional[int] = None):
    """
    Stream a gzipped CSV and filter for specific gh_project_name values.
    Saves extracted records directly to output_csv_path.
    """
    print(f"Extracting projects {project_names} from {csv_gz_path}...")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    matched_count = 0
    total_scanned = 0
    
    with gzip.open(csv_gz_path, "rt", encoding="utf-8", errors="replace") as gz_in, \
         open(output_csv_path, "w", encoding="utf-8") as out_f:
        
        header = gz_in.readline()
        out_f.write(header)
        header_cols = [c.strip().strip('"') for c in header.split(",")]
        
        try:
            proj_idx = header_cols.index("gh_project_name")
        except ValueError:
            raise ValueError("gh_project_name column not found in header!")
            
        for line in gz_in:
            total_scanned += 1
            cols = line.split(",")
            if len(cols) > proj_idx:
                pname = cols[proj_idx].strip().strip('"')
                if pname in project_names:
                    out_f.write(line)
                    matched_count += 1
                    if max_rows and matched_count >= max_rows:
                        break
                        
            if total_scanned % 100000 == 0:
                print(f"Scanned {total_scanned} rows, extracted {matched_count} matching records...")
                
    print(f"Finished. Extracted {matched_count} rows across {total_scanned} scanned builds into {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract TravisTorrent data.")
    parser.add_argument("--version", choices=["2016_snapshot", "2017_final"], default="2016_snapshot")
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--extract_projects", nargs="*", default=[], help="optional raw row level extraction of named projects")
    parser.add_argument("--processed_dir", default="data/processed")
    args = parser.parse_args()

    meta = FIGSHARE_FILES[args.version]
    dest_path = os.path.join(args.output_dir, meta["name"])
    
    if not os.path.exists(dest_path):
        download_file(meta["url"], dest_path, meta["md5"])
    else:
        print(f"File already exists at {dest_path}")
        
    for proj in args.extract_projects:
        safe_name = proj.replace("/", "_")
        proj_out = os.path.join(args.processed_dir, f"{safe_name}.csv")
        extract_project(dest_path, proj_out, [proj])


if __name__ == "__main__":
    main()
