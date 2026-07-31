#!/usr/bin/env python3
"""Download a large public Google Drive file reliably (LODGE/EDGE checkpoints, Jukebox priors).

Why this exists: ``gdown`` on the pods is a broken 6.1.0 build (no ``--fuzzy``, and its CLI aborts
large downloads with "Gdown can't. Please check connections and permissions."). For big files Google
serves an intermediate **"Virus scan warning"** HTML page containing a confirm form; the real bytes
only come back after that form is submitted (carrying the session cookie + a ``uuid``/``confirm``
token). This script parses that form with the stdlib and follows it, so it works with zero extra deps
and survives Google's periodic tweaks to the flow.

Usage:
    python scripts/download_gdrive.py <file_id> <output_path>
    python scripts/download_gdrive.py 13Yp__EPAw0EjrSS898X5FtSQGmveBykA pretrained_models.tar.gz

Exit codes: 0 on success, 2 if the response was HTML (not the file) or the download failed.
"""
from __future__ import annotations

import http.cookiejar
import re
import shutil
import sys
import urllib.parse
import urllib.request


def download(file_id: str, output: str) -> int:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", "Mozilla/5.0")]

    first = op.open(f"https://drive.google.com/uc?export=download&id={file_id}")
    head = first.read(200_000)
    ctype = first.headers.get("Content-Type", "")

    # Small direct file (no virus-scan interstitial): stream straight to disk.
    if "text/html" not in ctype:
        with open(output, "wb") as f:
            f.write(head)
            shutil.copyfileobj(first, f, 1024 * 1024)
        print(f"OK {output} (direct, {ctype})", flush=True)
        return 0

    page = head.decode("utf-8", "ignore")
    m = re.search(r'action="([^"]+download[^"]*)"', page)
    action = m.group(1).replace("&amp;", "&") if m else "https://drive.usercontent.google.com/download"
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', page))
    fields.setdefault("id", file_id)
    fields.setdefault("export", "download")
    fields.setdefault("confirm", "t")
    url = action + "?" + urllib.parse.urlencode(fields)

    r = op.open(url)
    if "text/html" in r.headers.get("Content-Type", ""):
        sys.stderr.write("ERROR: still HTML after confirm (quota exceeded or private file?)\n")
        return 2
    with open(output, "wb") as f:
        shutil.copyfileobj(r, f, 1024 * 1024)
    print(f"OK {output} ({r.headers.get('Content-Type')})", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: download_gdrive.py <file_id> <output_path>\n")
        return 2
    return download(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
