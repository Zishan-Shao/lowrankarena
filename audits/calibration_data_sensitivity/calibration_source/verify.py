#!/usr/bin/env python3
"""Verify the released method-independent calibration inputs without NumPy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
from pathlib import Path


TOKENIZER_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
EXPECTED = {
    "wikitext": "72b25ac4b3e08f9fbbabc71e00f3bb5ec32a60a3ae96be67289808f25d6669ce",
    "wikitext_draws/s0": "395153e0199b10ddd1dad782f59fc8a9a0f694eea53429f07fcead46902268ba",
    "wikitext_draws/s1": "79f90ff9c25a75b1fe6565d6b6ff2e178e168ef65564076066f08316cb0c6b5f",
    "c4": "59443b6b553b4ba9920415b7a5fdfcf248201789fe53e01b71432db446aa87d5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum_file(directory: Path) -> int:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError(f"missing {checksum_path}")
    checked = 0
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected_hash, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"malformed {checksum_path}:{line_number}") from error
        relative = relative.lstrip("*")
        target = directory / relative
        if not target.is_file():
            raise ValueError(f"checksum target is missing: {target}")
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise ValueError(f"checksum mismatch: {target}")
        checked += 1
    return checked


def read_npy_payload(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a NumPy file: {path}")
        major, minor = handle.read(2)
        if (major, minor) == (1, 0):
            header_length = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_length = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"unsupported NumPy format {major}.{minor}: {path}")
        encoding = "utf-8" if major == 3 else "latin1"
        header = ast.literal_eval(handle.read(header_length).decode(encoding).strip())
        payload = handle.read()
    if not isinstance(header, dict):
        raise ValueError(f"invalid NumPy header: {path}")
    return header, payload


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record: {path}:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            records.append(record)
    return records


def verify_input(root: Path, relative: str, expected_tensor_hash: str) -> tuple[int, str]:
    directory = root / relative
    checksum_count = verify_checksum_file(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != "lowrankarena_indexed_calibration_manifest_v1":
        raise ValueError(f"unexpected schema in {manifest_path}")
    dataset = manifest.get("dataset", {})
    selection = manifest.get("selection", {})
    tokenizer = manifest.get("tokenizer", {})
    artifacts = manifest.get("artifacts", {})
    if dataset.get("split") != "train":
        raise ValueError(f"non-train calibration source in {manifest_path}")
    if selection.get("sample_count") != 256 or selection.get("sequence_length") != 2048:
        raise ValueError(f"unexpected tensor dimensions in {manifest_path}")
    if selection.get("method_loader_or_seed_used") is not False:
        raise ValueError(f"method-dependent input selection in {manifest_path}")
    if selection.get("prng") is not None:
        raise ValueError(f"unexpected PRNG selection in {manifest_path}")
    if tokenizer.get("revision") != TOKENIZER_REVISION:
        raise ValueError(f"unexpected tokenizer revision in {manifest_path}")
    if tokenizer.get("add_special_tokens") is not False:
        raise ValueError(f"special-token policy mismatch in {manifest_path}")
    if artifacts.get("input_ids_int64le_sha256") != expected_tensor_hash:
        raise ValueError(f"published tensor hash mismatch in {manifest_path}")

    input_path = directory / str(artifacts.get("input_ids", ""))
    header, payload = read_npy_payload(input_path)
    if header.get("descr") not in ("<i8", "=i8"):
        raise ValueError(f"expected little-endian int64 tensor: {input_path}")
    if header.get("fortran_order") is not False or header.get("shape") != (256, 2048):
        raise ValueError(f"expected C-order int64[256,2048]: {input_path}")
    expected_bytes = 256 * 2048 * 8
    if len(payload) != expected_bytes:
        raise ValueError(f"unexpected payload size in {input_path}: {len(payload)}")
    if hashlib.sha256(payload).hexdigest() != expected_tensor_hash:
        raise ValueError(f"tensor-byte hash mismatch: {input_path}")

    row_hashes = artifacts.get("ordered_sample_hashes")
    if not isinstance(row_hashes, list) or len(row_hashes) != 256:
        raise ValueError(f"missing ordered row hashes in {manifest_path}")
    row_bytes = 2048 * 8
    for index, expected_row_hash in enumerate(row_hashes):
        row = payload[index * row_bytes : (index + 1) * row_bytes]
        if hashlib.sha256(row).hexdigest() != expected_row_hash:
            raise ValueError(f"row {index} hash mismatch: {input_path}")

    selected_path = directory / str(artifacts.get("selected_corpus", ""))
    indices_path = directory / str(artifacts.get("source_indices", ""))
    selected_rows = load_jsonl(selected_path)
    source_rows = load_jsonl(indices_path)
    if len(selected_rows) != 256 or len(source_rows) != 256:
        raise ValueError(f"expected 256 records in {selected_path} and {indices_path}")
    if [row.get("sample_index") for row in source_rows] != list(range(256)):
        raise ValueError(f"source indices are not in sample order: {indices_path}")
    source_key = "article_ordinal" if "article_ordinal" in source_rows[0] else "source_row_index"
    if len({row.get(source_key) for row in source_rows}) != 256:
        raise ValueError(f"source units are not distinct: {indices_path}")
    for index, (selected, source, row_hash) in enumerate(
        zip(selected_rows, source_rows, row_hashes)
    ):
        if selected.get("sample_index") != index:
            raise ValueError(f"selected corpus is not in sample order: {selected_path}")
        if source.get("input_ids_int64le_sha256") != row_hash:
            raise ValueError(f"source row {index} token hash mismatch: {indices_path}")
        if selected.get("raw_text_sha256") != source.get("raw_text_sha256"):
            raise ValueError(f"source row {index} raw-text hash mismatch")

    return checksum_count, str(dataset.get("name"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_root", type=Path, help="downloaded calibration_corpora/v1")
    args = parser.parse_args()

    total_checksums = 0
    for relative, tensor_hash in EXPECTED.items():
        checked, dataset = verify_input(args.calibration_root, relative, tensor_hash)
        total_checksums += checked
        print(f"verified {relative}: dataset={dataset}, checksums={checked}, tensor={tensor_hash}")
    print(
        f"PASS: verified {len(EXPECTED)} method-independent inputs and "
        f"{total_checksums} file checksums"
    )


if __name__ == "__main__":
    main()
