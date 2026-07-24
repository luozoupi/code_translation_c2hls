#!/usr/bin/env python3
"""Build RAG2 opt/repair BM25 indexes from knowledge_repo PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

from c2hls_rag import CHUNK_SIZE, OVERLAP, chunk_text  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_ug1399_rag_index",
    REPO / "scripts" / "build_ug1399_rag_index.py",
)
_ug = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ug)
read_source = _ug.read_source  # type: ignore[attr-defined]

DEFAULT_KR = REPO / "artifacts" / "rag" / "knowledge_repo"
DEFAULT_OPT_OUT = REPO / "artifacts" / "rag" / "rag2_opt"
DEFAULT_REPAIR_OUT = REPO / "artifacts" / "rag" / "rag2_repair"

OPT_SOURCES = (
    "ug1399-vitis-hls-en-us-2024.1.pdf",
    "ug902-vivado-high-level-synthesis.pdf",
)
REPAIR_SOURCES = (
    "ug1399-vitis-hls-en-us-2024.1.pdf",
    "bug_database.pdf",
    "pragma_bug_database.pdf",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_multi_source_index(
    *,
    sources: list[Path],
    out: Path,
    policy: str,
    chunk_size: int,
    overlap: int,
) -> dict:
    missing = [p for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing RAG2 sources:\n  " + "\n  ".join(str(p) for p in missing)
        )

    out.mkdir(parents=True, exist_ok=True)
    chunks_path = out / "chunks.jsonl"
    meta_path = out / "index_meta.json"

    records: list[dict] = []
    source_meta: list[dict] = []
    with chunks_path.open("w", encoding="utf-8") as fh:
        for source in sources:
            text = read_source(source)
            chunk_texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            source_meta.append(
                {
                    "path": str(source.resolve()),
                    "name": source.name,
                    "sha256": _sha256_file(source),
                    "n_chunks": len(chunk_texts),
                }
            )
            for piece in chunk_texts:
                rec = {
                    "id": f"c{len(records)}",
                    "text": piece,
                    "source": source.name,
                }
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "engine": "bm25",
        "policy": policy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "n_chunks": len(records),
        "sources": source_meta,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-repo", type=Path, default=DEFAULT_KR)
    parser.add_argument("--opt-out", type=Path, default=DEFAULT_OPT_OUT)
    parser.add_argument("--repair-out", type=Path, default=DEFAULT_REPAIR_OUT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    args = parser.parse_args(argv)

    kr: Path = args.knowledge_repo
    opt_sources = [kr / name for name in OPT_SOURCES]
    repair_sources = [kr / name for name in REPAIR_SOURCES]

    opt_meta = build_multi_source_index(
        sources=opt_sources,
        out=args.opt_out,
        policy="opt",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    repair_meta = build_multi_source_index(
        sources=repair_sources,
        out=args.repair_out,
        policy="repair",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(
        f"Built RAG2 opt: {args.opt_out.resolve()} ({opt_meta['n_chunks']} chunks)\n"
        f"Built RAG2 repair: {args.repair_out.resolve()} ({repair_meta['n_chunks']} chunks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
