"""Expectimax dla całego levelu (35 klocków, 14 linii) z tabelami na dysku.

Model polityki:
* dopóki do zbicia zostało więcej niż 4 linie, stos ma najwyżej 4 wiersze i nie
  ma w nim przykrytych pustych pól,
* gdy zostają najwyżej 4 linie, plansza jest blokiem PC; w ostatnich
  ``HOLE_PIECES`` klockach dozwolone są nawisy i wsunięcia.

Wartość to dokładne prawdopodobieństwo wygranej najlepszej gry w tym modelu,
czyli dolne ograniczenie szansy przy dowolnej grze.
"""
from __future__ import annotations

if __name__ == "__main__":
    from bootstrap import use_project_venv

    use_project_venv(__file__)

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numba import njit, prange

from bag import FULL_BAG, bag_size_after, build_bag_tables
from engine import (
    MAX_SUCC,
    Move,
    find_moves,
    key_of,
    placements_into,
    segments_ok,
    successors_into,
)

TOTAL_PIECES = 35
TOTAL_LINES = 14
HOLE_PIECES = 5
CACHE_VERSION = 1
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"

Progress = Callable[[float, str], None]


def print_progress(fraction: float, message: str) -> None:
    print(f"PROGRESS {fraction:.4f} {message}", flush=True)


def file_progress(path: str | Path) -> Progress:
    """Postęp do pliku: okienkowy .exe nie ma standardowego wyjścia."""

    def report(fraction: float, message: str) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{fraction:.4f} {message}\n")

    return report


@njit(nogil=True, inline="always")
def popcount(value: int) -> int:
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count


@njit(nogil=True)
def is_flat(field: int) -> bool:
    """Prawda, gdy żadne puste pole nie jest przykryte kostką."""
    for y in range(3):
        lower = (field >> (10 * y)) & 1023
        upper = (field >> (10 * (y + 1))) & 1023
        if (upper & ~lower) != 0:
            return False
    return True


@njit(nogil=True)
def step_into(key: int, t: int, piece: int, out: np.ndarray) -> int:
    """Dozwolone stany po położeniu klocka numer ``t``."""
    field = key & 1099511627775
    lines = (140 - 4 * t + popcount(field)) // 10
    holes_allowed = (34 - t) <= HOLE_PIECES
    if lines <= 4:
        count = successors_into(key, piece, out)
        if holes_allowed:
            return count
        kept = 0
        for i in range(count):
            if is_flat(out[i] & 1099511627775):
                out[kept] = out[i]
                kept += 1
        return kept
    raw = placements_into(field, 4, piece, out)
    kept = 0
    for i in range(raw):
        new_field = out[i] & 1099511627775
        new_lines = lines - (4 - (out[i] >> 40))
        if new_lines <= 4:
            if (new_field >> (10 * new_lines)) != 0 or not segments_ok(new_field, new_lines):
                continue
            if not holes_allowed and not is_flat(new_field):
                continue
            out[kept] = new_field | (new_lines << 40)
        else:
            if not is_flat(new_field):
                continue
            out[kept] = new_field | (4 << 40)
        kept += 1
    return kept


@njit(nogil=True, parallel=True)
def _expand(keys: np.ndarray, t: int) -> np.ndarray:
    n = keys.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        total = 0
        for piece in range(7):
            total += step_into(keys[i], t, piece, buffer)
        offsets[i + 1] = total
    for i in range(n):
        offsets[i + 1] += offsets[i]
    result = np.empty(offsets[n], dtype=np.int64)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        position = offsets[i]
        for piece in range(7):
            count = step_into(keys[i], t, piece, buffer)
            for j in range(count):
                result[position + j] = buffer[j]
            position += count
    return result


@njit(nogil=True, parallel=True)
def _alive_mask(keys: np.ndarray, next_keys: np.ndarray, t: int) -> np.ndarray:
    n = keys.shape[0]
    m = next_keys.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        found = False
        for piece in range(7):
            count = step_into(keys[i], t, piece, buffer)
            for j in range(count):
                index = np.searchsorted(next_keys, buffer[j])
                if index < m and next_keys[index] == buffer[j]:
                    found = True
                    break
            if found:
                break
        mask[i] = found
    return mask


@njit(nogil=True, parallel=True)
def _build_edges(keys: np.ndarray, next_keys: np.ndarray, t: int):
    n = keys.shape[0]
    m = next_keys.shape[0]
    offsets = np.zeros(n * 7 + 1, dtype=np.int64)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        for piece in range(7):
            count = step_into(keys[i], t, piece, buffer)
            kept = 0
            for j in range(count):
                index = np.searchsorted(next_keys, buffer[j])
                if index < m and next_keys[index] == buffer[j]:
                    kept += 1
            offsets[i * 7 + piece + 1] = kept
    for i in range(n * 7):
        offsets[i + 1] += offsets[i]
    edges = np.empty(offsets[n * 7], dtype=np.int32)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        for piece in range(7):
            count = step_into(keys[i], t, piece, buffer)
            position = offsets[i * 7 + piece]
            for j in range(count):
                index = np.searchsorted(next_keys, buffer[j])
                if index < m and next_keys[index] == buffer[j]:
                    edges[position] = index
                    position += 1
    return offsets, edges


@njit(nogil=True, parallel=True, fastmath=True)
def _dp_layer(offsets, edges, next_values, per_piece, trans, draws, next_width, n):
    """W(f, c, B) = średnia po następnym klocku z maksimum po ruchach klocka c."""
    values = np.zeros((n, 7 * per_piece), dtype=np.float32)
    inverse = 1.0 / draws
    for i in prange(n):
        best = np.empty(next_width, dtype=np.float32)
        for piece in range(7):
            start = offsets[i * 7 + piece]
            stop = offsets[i * 7 + piece + 1]
            if start == stop:
                continue
            best[:] = 0.0
            for q in range(start, stop):
                row = next_values[edges[q]]
                for z in range(next_width):
                    if row[z] > best[z]:
                        best[z] = row[z]
            for b in range(per_piece):
                j = piece * per_piece + b
                total = 0.0
                for d in range(draws):
                    total += best[trans[j, d]]
                values[i, j] = total * inverse
    return values


def _expand_unique(keys: np.ndarray, t: int, chunk: int = 100_000) -> np.ndarray:
    parts = [np.unique(_expand(keys[i:i + chunk], t)) for i in range(0, len(keys), chunk)]
    while len(parts) > 1:
        parts = [
            np.union1d(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i]
            for i in range(0, len(parts), 2)
        ]
    return parts[0] if parts else np.empty(0, dtype=np.int64)


def _states_paths(cache: Path) -> tuple[Path, Path]:
    return cache / f"keys_v{CACHE_VERSION}.npy", cache / f"key_offsets_v{CACHE_VERSION}.npy"


def _values_paths(cache: Path, offset: int) -> tuple[Path, Path]:
    stem = f"values_v{CACHE_VERSION}_o{offset}"
    return cache / f"{stem}.u16", cache / f"{stem}.meta.npz"


def build_states(cache: Path = CACHE_DIR, progress: Progress = print_progress) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    layers: list[np.ndarray | None] = [np.array([key_of(0, 4)], dtype=np.int64)]
    for t in range(TOTAL_PIECES):
        layers.append(_expand_unique(layers[t], t))
        progress(0.85 * (t + 1) / TOTAL_PIECES, f"Stany: klocek {t + 1}/{TOTAL_PIECES}")
    alive: list[np.ndarray | None] = [None] * (TOTAL_PIECES + 1)
    alive[TOTAL_PIECES] = layers[TOTAL_PIECES]
    for t in range(TOTAL_PIECES - 1, -1, -1):
        alive[t] = layers[t][_alive_mask(layers[t], alive[t + 1], t)]
        layers[t] = None
        progress(0.85 + 0.15 * (TOTAL_PIECES - t) / TOTAL_PIECES, "Odrzucanie martwych stanów")
    sizes = np.array([len(layer) for layer in alive], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    keys_path, offsets_path = _states_paths(cache)
    temp = keys_path.with_suffix(".tmp.npy")
    np.save(temp, np.concatenate(alive))
    os.replace(temp, keys_path)
    temp = offsets_path.with_suffix(".tmp.npy")
    np.save(temp, offsets)
    os.replace(temp, offsets_path)


def build_values(offset: int = 0, cache: Path = CACHE_DIR, progress: Progress = print_progress) -> None:
    keys_path, offsets_path = _states_paths(cache)
    keys = np.load(keys_path, mmap_mode="r")
    key_offsets = np.load(offsets_path)
    tables = build_bag_tables()
    data_path, meta_path = _values_paths(cache, offset)
    temp_path = data_path.with_suffix(".tmp")
    widths = np.array(
        [7 * int(tables.per_piece[bag_size_after(t, offset)]) for t in range(TOTAL_PIECES + 1)],
        dtype=np.int64,
    )
    value_offsets = np.zeros(TOTAL_PIECES + 1, dtype=np.int64)
    scales = np.ones(TOTAL_PIECES + 1, dtype=np.float64)
    position = 0
    with open(temp_path, "wb") as handle:
        current = np.ones((1, int(widths[TOTAL_PIECES])), dtype=np.float32)
        np.full(current.size, 65535, dtype=np.uint16).tofile(handle)
        value_offsets[TOTAL_PIECES] = 0
        position = current.size
        next_keys = np.asarray(keys[key_offsets[TOTAL_PIECES]:key_offsets[TOTAL_PIECES + 1]])
        for t in range(TOTAL_PIECES - 1, -1, -1):
            layer_keys = np.asarray(keys[key_offsets[t]:key_offsets[t + 1]])
            edge_offsets, edges = _build_edges(layer_keys, next_keys, t)
            s = bag_size_after(t, offset)
            current = _dp_layer(
                edge_offsets, edges, current, int(tables.per_piece[s]), tables.trans[s],
                int(tables.trans_len[s]), current.shape[1], len(layer_keys),
            )
            scale = float(current.max()) or 1.0
            scales[t] = scale
            value_offsets[t] = position
            for start in range(0, len(current), 200_000):
                part = current[start:start + 200_000]
                np.clip(np.ceil(part / scale * 65535.0), 0, 65535).astype(np.uint16).tofile(handle)
            position += current.size
            next_keys = layer_keys
            progress((TOTAL_PIECES - t) / TOTAL_PIECES, f"Wartości: klocek {t + 1}/{TOTAL_PIECES}")
    os.replace(temp_path, data_path)
    temp_meta = meta_path.with_suffix(".tmp.npz")
    np.savez(temp_meta, widths=widths, offsets=value_offsets, scales=scales)
    os.replace(temp_meta, meta_path)


class BagMismatch(ValueError):
    """Wskazany klocek nie może wyjść z aktualnego stanu worka."""


@dataclass(frozen=True)
class MoveEval:
    move: Move
    probability: float


@dataclass(frozen=True)
class Tables:
    offset: int
    values: np.ndarray
    widths: np.ndarray
    offsets: np.ndarray
    scales: np.ndarray


class Solver:
    """Odczyt tabel i ocena ruchów dla konkretnej pozycji."""

    def __init__(self, cache: Path = CACHE_DIR):
        self.cache = Path(cache)
        self.bag = build_bag_tables()
        self._keys: np.ndarray | None = None
        self._key_offsets: np.ndarray | None = None
        self._tables: dict[int, Tables] = {}

    def states_ready(self) -> bool:
        return all(path.exists() for path in _states_paths(self.cache))

    def values_ready(self, offset: int) -> bool:
        return self.states_ready() and all(path.exists() for path in _values_paths(self.cache, offset))

    def ensure(self, offset: int = 0, progress: Progress = print_progress) -> None:
        if not self.states_ready():
            build_states(self.cache, progress)
        if not self.values_ready(offset):
            build_values(offset, self.cache, progress)

    def _load_states(self) -> None:
        if self._keys is None:
            keys_path, offsets_path = _states_paths(self.cache)
            self._keys = np.load(keys_path, mmap_mode="r")
            self._key_offsets = np.load(offsets_path)

    def tables(self, offset: int) -> Tables:
        if offset not in self._tables:
            self._load_states()
            data_path, meta_path = _values_paths(self.cache, offset)
            meta = np.load(meta_path)
            self._tables[offset] = Tables(
                offset,
                np.memmap(data_path, dtype=np.uint16, mode="r"),
                meta["widths"],
                meta["offsets"],
                meta["scales"],
            )
        return self._tables[offset]

    def flat_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        self._load_states()
        return np.asarray(self._keys), self._key_offsets

    @staticmethod
    def lines_remaining(field: int, t: int) -> int | None:
        total = 4 * (TOTAL_PIECES - t) + bin(int(field)).count("1")
        return total // 10 if total % 10 == 0 else None

    @staticmethod
    def offset_for(t: int, bag_after_current: int) -> int:
        return (6 - bin(bag_after_current).count("1") - t) % 7

    def state_value(self, key: int, t: int, piece: int, bag_after: int, offset: int) -> float:
        tables = self.tables(offset)
        start, stop = int(self._key_offsets[t]), int(self._key_offsets[t + 1])
        layer = self._keys[start:stop]
        index = int(np.searchsorted(layer, key))
        if index >= len(layer) or int(layer[index]) != key:
            return 0.0
        s = bag_size_after(t, offset)
        column = int(self.bag.index[s, piece, bag_after])
        if column < 0:
            raise BagMismatch("Stan worka nie pasuje do numeru ruchu")
        raw = int(tables.values[int(tables.offsets[t]) + index * int(tables.widths[t]) + column])
        return raw / 65535.0 * float(tables.scales[t])

    def _after_move(self, key: int, t: int, nxt: int | None, bag_after_current: int, offset: int) -> float:
        if t == TOTAL_PIECES:
            return 1.0 if key == 0 else 0.0
        pool = bag_after_current or FULL_BAG
        if nxt is not None:
            if not (pool >> nxt) & 1:
                raise BagMismatch("Tego klocka nie ma już w worku")
            return self.state_value(key, t, nxt, pool & ~(1 << nxt), offset)
        drawn = [p for p in range(7) if (pool >> p) & 1]
        return sum(self.state_value(key, t, p, pool & ~(1 << p), offset) for p in drawn) / len(drawn)

    def analyze(self, field: int, t: int, current: int, nxt: int | None, bag_after_current: int) -> list[MoveEval]:
        """Ruchy klocka ``current`` od najlepszego; ``nxt=None`` gdy next nieznany."""
        lines = self.lines_remaining(field, t)
        if lines is None or lines <= 0 or int(field) >> 40:
            return []
        offset = self.offset_for(t, bag_after_current)
        self.tables(offset)
        frame = min(lines, 4)
        if int(field) >> (10 * frame):
            return []
        results = []
        for move in find_moves(field, frame, current):
            new_lines = lines - (frame - move.result_rows)
            new_rows = min(new_lines, 4)
            if int(move.result_field) >> (10 * new_rows):
                continue
            key = key_of(move.result_field, new_rows)
            results.append(MoveEval(move, self._after_move(key, t + 1, nxt, bag_after_current, offset)))
        results.sort(key=lambda e: (-e.probability, e.move.is_tuck, e.move.orientation, e.move.x))
        return results

    def level_value(self, first_piece: int | None = None, offset: int = 0) -> float:
        """Szansa wygranej z pustej planszy, średnio albo dla pierwszego klocka."""
        self.tables(offset)
        s = bag_size_after(0, offset)
        masks = []
        for piece in range(7):
            for b in range(int(self.bag.per_piece[s])):
                masks.append((piece, int(self.bag.masks[s, piece, b])))
        if first_piece is not None:
            masks = [m for m in masks if m[0] == first_piece]
        key = key_of(0, 4)
        return sum(self.state_value(key, 0, p, b, offset) for p, b in masks) / len(masks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Budowa tabel solvera")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--offset", type=int, default=0, choices=range(7))
    args = parser.parse_args()
    solver = Solver()
    if args.build:
        started = time.time()
        solver.ensure(args.offset)
        print(f"DONE {time.time() - started:.0f}s", flush=True)
    if solver.values_ready(args.offset):
        print(f"P(level) offset={args.offset}: {solver.level_value(offset=args.offset) * 100:.3f}%")
    else:
        sys.exit("Brak tabel: uruchom z --build")


if __name__ == "__main__":
    main()
