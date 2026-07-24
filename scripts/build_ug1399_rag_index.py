#!/usr/bin/env python3
"""Build a BM25-ready UG1399 RAG index from PDF, HTML, or plain text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from c2hls_rag import CHUNK_SIZE, OVERLAP, chunk_text  # noqa: E402

DEFAULT_OUT = REPO / "artifacts" / "rag" / "ug1399"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "PDF source requires the optional 'pypdf' package. "
            "Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def _read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return parser.get_text()


def read_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in (".html", ".htm"):
        return _read_html(path)
    return path.read_text(encoding="utf-8", errors="replace")


def build_index(
    *,
    source: Path,
    out: Path,
    chunk_size: int,
    overlap: int,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"source not found: {source}")

    text = read_source(source)
    chunk_texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    out.mkdir(parents=True, exist_ok=True)
    chunks_path = out / "chunks.jsonl"
    meta_path = out / "index_meta.json"

    with chunks_path.open("w", encoding="utf-8") as fh:
        for i, chunk in enumerate(chunk_texts):
            record = {"id": f"c{i}", "text": chunk}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "engine": "bm25",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "source": str(source.resolve()),
        "source_sha256": _sha256_file(source),
        "n_chunks": len(chunk_texts),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="UG1399 source file (.txt, .html, .htm, or .pdf)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output index directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in characters (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help=f"Chunk overlap in characters (default: {OVERLAP})",
    )
    args = parser.parse_args(argv)

    meta = build_index(
        source=args.source,
        out=args.out,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(
        f"Built RAG index: {args.out.resolve()} "
        f"({meta['n_chunks']} chunks, engine={meta['engine']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
