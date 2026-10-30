"""Download the Figshare brain tumor dataset (Cheng et al., 3,064 T1-CE MRI slices with tumor masks).
No Kaggle account needed. ~880 MB.

If the automatic download is blocked, download manually from
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
and put the zip at data/raw/figshare_1512427.zip, then re-run this script.
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://figshare.com/ndownloader/articles/1512427/versions/5"


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    raw = Path(args.out)
    raw.mkdir(parents=True, exist_ok=True)
    outer = raw / "figshare_1512427.zip"

    if not outer.exists():
        print(f"Downloading {URL}")
        tmp = outer.with_suffix(".part")
        download(URL, tmp)
        tmp.rename(outer)
    else:
        print(f"Found {outer}, skipping download")

    stage = raw / "_outer"
    with zipfile.ZipFile(outer) as z:
        z.extractall(stage)

    mat_dir = raw / "mat"
    mat_dir.mkdir(exist_ok=True)
    inner = sorted(stage.rglob("*.zip"))
    for zp in inner:
        print(f"Extracting {zp.name}")
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".mat"):
                    with z.open(name) as src, open(mat_dir / Path(name).name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    # some mirrors ship the .mat files directly
    for mp in stage.rglob("*.mat"):
        if mp.name != "cvind.mat":
            shutil.copy(mp, mat_dir / mp.name)
    shutil.rmtree(stage, ignore_errors=True)

    n = len(list(mat_dir.glob("*.mat")))
    print(f"Done: {n} .mat files in {mat_dir} (expected 3064)")


if __name__ == "__main__":
    main()
