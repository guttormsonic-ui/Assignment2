import os
import sys
import tarfile
import urllib.request
import hashlib

BASE_URL   = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
OUTPUT_DIR = "D:\\ML\\A2\\Pet"          

FILES = {
    "images.tar.gz":      "5c4f3ee8e5d25df40f4fd59a7f44e54c",   
    "annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f",
}



def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def reporthook(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(int(count * block_size * 100 / total_size), 100)
    bar = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct:3d}%", end="", flush=True)


def download(url: str, dest: str):
    print(f"  Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress bar


def safe_extract(tar_path: str, output_dir: str):
    """
    Extract a tar.gz, skipping or truncating any member whose resolved
    path would be too long (common on Windows / some Linux FS configs).
    Also strips leading './' or path traversal components for safety.
    """
    os.makedirs(output_dir, exist_ok=True)
    skipped = []

    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        total   = len(members)

        for i, member in enumerate(members, 1):
            print(f"\r  Extracting {i}/{total} ...", end="", flush=True)
            parts = member.name.replace("\\", "/").split("/")
            parts = [p for p in parts if p and p != ".." and p != "."]
            safe_name = os.path.join(*parts) if parts else None

            if safe_name is None:
                skipped.append(member.name)
                continue

            # ── Check total path length (Windows MAX_PATH = 260) ─────────────
            full_path = os.path.join(output_dir, safe_name)
            if len(full_path) > 255:
                # Try to shorten the filename while keeping extension
                dirname  = os.path.dirname(full_path)
                basename = os.path.basename(full_path)
                name, ext = os.path.splitext(basename)
                short    = name[:20] + "_trunc" + ext
                full_path = os.path.join(dirname, short)

            member.name = safe_name   # rewrite the member name in-place

            try:
                tf.extract(member, path=output_dir)
            except (OSError, ValueError) as exc:
                skipped.append(f"{member.name}: {exc}")

    print()  # newline after progress
    return skipped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, expected_md5 in FILES.items():
        url       = BASE_URL + filename
        dest_path = os.path.join(OUTPUT_DIR, filename)

        # ── Download ──────────────────────────────────────────────────────────
        if os.path.exists(dest_path):
            print(f"  {filename} already downloaded, verifying checksum ...")
        else:
            download(url, dest_path)

        # ── Verify checksum ───────────────────────────────────────────────────
        actual = md5(dest_path)
        if actual != expected_md5:
            print(f"  WARNING: checksum mismatch for {filename}!")
            print(f"    expected: {expected_md5}")
            print(f"    actual  : {actual}")
            answer = input("  Continue anyway? [y/N] ").strip().lower()
            if answer != "y":
                sys.exit(1)
        else:
            print(f"  Checksum OK.")

        # ── Extract ───────────────────────────────────────────────────────────
        print(f"  Extracting {filename} → {OUTPUT_DIR}/")
        skipped = safe_extract(dest_path, OUTPUT_DIR)

        if skipped:
            print(f"  Skipped {len(skipped)} file(s) due to path issues:")
            for s in skipped[:10]:
                print(f"    • {s}")
            if len(skipped) > 10:
                print(f"    … and {len(skipped) - 10} more.")
        else:
            print("  All files extracted successfully.")
        print()

    print("Done! Dataset is ready in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()