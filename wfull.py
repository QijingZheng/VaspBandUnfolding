#!/usr/bin/env python3

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScreenedPotentialData:
    ngvector: int
    is_full: bool
    head: np.ndarray
    wing: np.ndarray
    cwing: np.ndarray
    matrix: np.ndarray


def _read_fortran_record(handle) -> bytes:
    marker = handle.read(4)
    if len(marker) != 4:
        raise EOFError("Unexpected EOF while reading Fortran record marker")
    nbytes = struct.unpack("<i", marker)[0]
    payload = handle.read(nbytes)
    if len(payload) != nbytes:
        raise EOFError("Unexpected EOF while reading Fortran record payload")
    end_marker = handle.read(4)
    if len(end_marker) != 4:
        raise EOFError("Unexpected EOF while reading Fortran record trailer")
    if struct.unpack("<i", end_marker)[0] != nbytes:
        raise ValueError("Fortran record marker mismatch")
    return payload


def find_screened_potential_file(search_dirs: list[Path], index: int) -> Path | None:
    app = f"{index:04d}"
    for dpath in search_dirs:
        full_path = dpath / f"WFULL{app}.tmp"
        if full_path.is_file():
            return full_path
        diag_path = dpath / f"W{app}.tmp"
        if diag_path.is_file():
            return diag_path
    return None


def read_screened_potential(path: str | Path) -> ScreenedPotentialData:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        header = np.frombuffer(_read_fortran_record(handle), dtype=np.int32)
        if header.size != 2:
            raise ValueError(f"Expected two int32 values in {file_path}, found {header.size}")
        n1 = int(header[0])
        n2 = int(header[1])

        head = np.frombuffer(_read_fortran_record(handle), dtype=np.complex128)
        wing = np.frombuffer(_read_fortran_record(handle), dtype=np.complex128)
        cwing = np.frombuffer(_read_fortran_record(handle), dtype=np.complex128)
        matrix_payload = _read_fortran_record(handle)

    if head.size != 9:
        raise ValueError(f"Expected 9 complex head entries in {file_path}, found {head.size}")
    head_mat = np.asarray(head.reshape((3, 3)), dtype=np.complex128)

    if wing.size % 3 != 0 or cwing.size % 3 != 0:
        raise ValueError(f"Expected wing records in multiples of 3 complex values in {file_path}")
    wing_mat = np.asarray(wing.reshape((-1, 3)), dtype=np.complex128)
    cwing_mat = np.asarray(cwing.reshape((-1, 3)), dtype=np.complex128)

    if n2 == 0:
        diag = np.frombuffer(matrix_payload, dtype=np.complex128)
        if diag.size != n1:
            raise ValueError(f"Expected {n1} diagonal entries in {file_path}, found {diag.size}")
        matrix = np.asarray(np.diag(diag), dtype=np.complex128)
        return ScreenedPotentialData(ngvector=n1, is_full=False, head=head_mat, wing=wing_mat, cwing=cwing_mat, matrix=matrix)

    matrix = np.frombuffer(matrix_payload, dtype=np.complex128)
    if matrix.size != n1 * n1:
        raise ValueError(f"Expected {n1 * n1} complex entries in {file_path}, found {matrix.size}")
    matrix = np.asarray(matrix.reshape((n1, n1)), dtype=np.complex128)
    return ScreenedPotentialData(ngvector=n1, is_full=True, head=head_mat, wing=wing_mat, cwing=cwing_mat, matrix=matrix)


def read_screened_potential_diag(path: str | Path) -> np.ndarray:
    return np.diag(read_screened_potential(path).matrix).copy()