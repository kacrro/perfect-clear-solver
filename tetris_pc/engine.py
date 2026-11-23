"""Silnik planszy: bitboardy, generowanie ruchów, zbijanie linii.

Pole planszy to bit ``10 * y + x``; ``y = 0`` to dolny wiersz, ``x = 0`` lewa
kolumna. Blok PC ma najwyżej 4 wiersze, więc stan mieści się w 40 bitach.
Klucz stanu w solverze to ``field | (L << 40)``, gdzie L to liczba wierszy,
które zostały do zbicia w bieżącym bloku.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

BOARD_W = 10
MAX_L = 4
ROW_MASK = (1 << BOARD_W) - 1
KEY_SHIFT = 40
FIELD_MASK = (1 << KEY_SHIFT) - 1
MAX_SUCC = 96

PIECES = "IOTSZJL"
PIECE_INDEX = {name: i for i, name in enumerate(PIECES)}
NUM_PIECES = 7

# Orientacje w kolejności SRS: 0 (spawn), R, 2, L. Wiersze od góry do dołu.
SHAPES: dict[str, tuple[tuple[str, ...], ...]] = {
    "I": (("####",), ("#", "#", "#", "#")),
    "O": (("##", "##"),),
    "T": ((".#.", "###"), ("#.", "##", "#."), ("###", ".#."), (".#", "##", ".#")),
    "S": ((".##", "##."), ("#.", "##", ".#")),
    "Z": (("##.", ".##"), (".#", "##", "#.")),
    "J": (("#..", "###"), ("##", "#.", "#."), ("###", "..#"), (".#", ".#", "##")),
    "L": (("..#", "###"), ("#.", "#.", "##"), ("###", "#.."), ("##", ".#", ".#")),
}
ORIENTATION_NAMES = ("0", "R", "2", "L")


def _build_tables():
    num = np.zeros(NUM_PIECES, dtype=np.int64)
    mask = np.zeros((NUM_PIECES, 4), dtype=np.int64)
    width = np.zeros((NUM_PIECES, 4), dtype=np.int64)
    height = np.zeros((NUM_PIECES, 4), dtype=np.int64)
    for p, name in enumerate(PIECES):
        num[p] = len(SHAPES[name])
        for o, rows in enumerate(SHAPES[name]):
            width[p, o] = len(rows[0])
            height[p, o] = len(rows)
            bits = 0
            for r, text in enumerate(rows):
                y = len(rows) - 1 - r
                for x, ch in enumerate(text):
                    if ch == "#":
                        bits |= 1 << (BOARD_W * y + x)
            mask[p, o] = bits
    return num, mask, width, height


NUM_ORI, ORI_MASK, ORI_W, ORI_H = _build_tables()


@njit(nogil=True, inline="always")
def make_key(field: int, rows: int) -> int:
    return field | (rows << 40)


@njit(nogil=True, inline="always")
def _flood_row(seeds: int, free: int) -> int:
    current = seeds & free
    while True:
        grown = (current | (current << 1) | (current >> 1)) & free
        if grown == current:
            return current
        current = grown


@njit(nogil=True)
def clear_rows(board: int, rows: int):
    """Zbija pełne wiersze; zwraca (plansza, liczba pozostałych wierszy)."""
    result = 0
    kept = 0
    for y in range(rows):
        row = (board >> (10 * y)) & 1023
        if row != 1023:
            result |= row << (10 * kept)
            kept += 1
    return result, kept


@njit(nogil=True)
def placements_into(field: int, rows: int, piece: int, out: np.ndarray) -> int:
    """Zapisuje w ``out`` unikalne klucze stanów po położeniu klocka.

    Obrót jest możliwy tylko nad stosem. Dalej działa BFS warstwami po y:
    w każdej warstwie klocek przesuwa się w bok po wolnych pozycjach, potem
    schodzi niżej. Klocek musi w całości leżeć w ``rows`` wierszach bloku.
    """
    count = 0
    for o in range(NUM_ORI[piece]):
        shape = ORI_MASK[piece, o]
        h = ORI_H[piece, o]
        max_x = 10 - ORI_W[piece, o]
        reach = (1 << (max_x + 1)) - 1
        y = rows
        while y >= 0:
            free_below = 0
            if y > 0:
                for x in range(max_x + 1):
                    if ((shape << (x + 10 * (y - 1))) & field) == 0:
                        free_below |= 1 << x
            landing = reach & ~free_below
            if y + h <= rows:
                bits = landing
                while bits:
                    x = 0
                    probe = bits
                    while (probe & 1) == 0:
                        probe >>= 1
                        x += 1
                    bits &= bits - 1
                    new_field, new_rows = clear_rows(field | (shape << (x + 10 * y)), rows)
                    key = new_field | (new_rows << 40)
                    duplicate = False
                    for i in range(count):
                        if out[i] == key:
                            duplicate = True
                            break
                    if not duplicate:
                        out[count] = key
                        count += 1
            if y == 0:
                break
            reach = _flood_row(reach & free_below, free_below)
            y -= 1
    return count


@njit(nogil=True)
def segments_ok(field: int, rows: int) -> bool:
    """Szybki, bezpieczny test: każdy segment pustych pól dzieli się przez 4.

    Dwie sąsiednie kolumny należą do jednego segmentu tylko wtedy, gdy w tym
    samym wierszu obie mają puste pole. Klocek nie może przejść między nimi w
    inny sposób, także po zbiciu linii, bo wiersze nie przesuwają się w bok.
    """
    if rows == 0:
        return field == 0
    occupied = field & ((1 << (10 * rows)) - 1)
    empty = ~occupied & ((1 << (10 * rows)) - 1)
    linked = 0
    for y in range(rows):
        row = (empty >> (10 * y)) & 1023
        linked |= row & (row >> 1)
    size = 0
    for x in range(10):
        for y in range(rows):
            size += (empty >> (10 * y + x)) & 1
        if ((linked >> x) & 1) == 0:
            if size % 4 != 0:
                return False
            size = 0
    return size % 4 == 0


@njit(nogil=True)
def successors_into(key: int, piece: int, out: np.ndarray) -> int:
    """Następniki stanu po klocku, z odrzuceniem martwych segmentów."""
    field = key & 1099511627775
    rows = key >> 40
    raw = placements_into(field, rows, piece, out)
    kept = 0
    for i in range(raw):
        candidate = out[i]
        if segments_ok(candidate & 1099511627775, candidate >> 40):
            out[kept] = candidate
            kept += 1
    return kept


@njit(nogil=True, parallel=True)
def expand_keys(keys: np.ndarray) -> np.ndarray:
    """Wszystkie następniki listy stanów dla wszystkich 7 klocków (z duplikatami)."""
    n = keys.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        total = 0
        for piece in range(7):
            total += successors_into(keys[i], piece, buffer)
        offsets[i + 1] = total
    for i in range(n):
        offsets[i + 1] += offsets[i]
    result = np.empty(offsets[n], dtype=np.int64)
    for i in prange(n):
        buffer = np.empty(MAX_SUCC, dtype=np.int64)
        position = offsets[i]
        for piece in range(7):
            c = successors_into(keys[i], piece, buffer)
            for j in range(c):
                result[position + j] = buffer[j]
            position += c
    return result


# --- API wygodne dla Pythona -------------------------------------------------

def key_of(field: int, rows: int) -> int:
    return int(field) | (int(rows) << KEY_SHIFT)


def split_key(key: int) -> tuple[int, int]:
    key = int(key)
    return key & FIELD_MASK, key >> KEY_SHIFT


def placements(field: int, rows: int, piece: int | str) -> list[tuple[int, int]]:
    """Unikalne wyniki położenia klocka jako (plansza, pozostałe wiersze)."""
    if isinstance(piece, str):
        piece = PIECE_INDEX[piece]
    out = np.empty(MAX_SUCC, dtype=np.int64)
    count = placements_into(int(field), int(rows), int(piece), out)
    return [split_key(k) for k in out[:count]]


def board_from_text(text: str) -> int:
    """Plansza z tekstu; pierwszy wiersz to góra, ``#`` to kostka."""
    lines = [ln.strip() for ln in text.strip().splitlines()]
    field = 0
    for r, line in enumerate(lines):
        y = len(lines) - 1 - r
        for x, ch in enumerate(line):
            if ch in "#X":
                field |= 1 << (BOARD_W * y + x)
    return field


def board_to_text(field: int, rows: int = MAX_L) -> str:
    lines = []
    for y in range(rows - 1, -1, -1):
        lines.append("".join("#" if (field >> (10 * y + x)) & 1 else "." for x in range(10)))
    return "\n".join(lines)


def piece_cells(piece: int, orientation: int, x: int, y: int) -> list[tuple[int, int]]:
    """Lista pól (kolumna, wiersz) zajmowanych przez klocek."""
    shape = int(ORI_MASK[piece, orientation])
    cells = []
    for row in range(4):
        for col in range(4):
            if (shape >> (10 * row + col)) & 1:
                cells.append((x + col, y + row))
    return cells


@dataclass(frozen=True)
class Move:
    """Konkretny ruch: pozycja końcowa i ścieżka od stanu nad stosem."""

    piece: int
    orientation: int
    x: int
    y: int
    drop_x: int
    path: tuple[tuple[str, int], ...]
    result_field: int
    result_rows: int

    @property
    def cells(self) -> list[tuple[int, int]]:
        return piece_cells(self.piece, self.orientation, self.x, self.y)

    @property
    def is_tuck(self) -> bool:
        return any(step == "side" for step, _ in self.path)


def _collides(field: int, piece: int, orientation: int, x: int, y: int) -> bool:
    return (int(ORI_MASK[piece, orientation]) << (x + 10 * y)) & field != 0


def find_moves(field: int, rows: int, piece: int | str) -> list[Move]:
    """Ruchy z najprostszą ścieżką dla każdej unikalnej planszy wynikowej.

    Preferowane są ruchy bez wsuwania: najmniej kroków bocznych pod stosem.
    """
    if isinstance(piece, str):
        piece = PIECE_INDEX[piece]
    field = int(field)
    best: dict[tuple[int, int], tuple[tuple[int, int, int], Move]] = {}
    for o in range(int(NUM_ORI[piece])):
        max_x = BOARD_W - int(ORI_W[piece, o])
        h = int(ORI_H[piece, o])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {}
        queue: deque[tuple[int, int]] = deque()
        for x in range(max_x + 1):
            parents[(x, rows)] = None
            queue.append((x, rows))
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x, y - 1), (x - 1, y), (x + 1, y)):
                if nx < 0 or nx > max_x or ny < 0 or (nx, ny) in parents:
                    continue
                if _collides(field, piece, o, nx, ny):
                    continue
                parents[(nx, ny)] = (x, y)
                queue.append((nx, ny))
        for (x, y) in parents:
            if y + h > rows:
                continue
            if y > 0 and not _collides(field, piece, o, x, y - 1):
                continue
            chain = [(x, y)]
            while parents[chain[-1]] is not None:
                chain.append(parents[chain[-1]])
            chain.reverse()
            path = _compress_path(chain)
            new_field, new_rows = clear_rows(field | (int(ORI_MASK[piece, o]) << (x + 10 * y)), rows)
            side_steps = sum(abs(amount) for step, amount in path if step == "side")
            score = (side_steps, o, x)
            target = (int(new_field), int(new_rows))
            move = Move(piece, o, x, y, chain[0][0], tuple(path), target[0], target[1])
            if target not in best or score < best[target][0]:
                best[target] = (score, move)
    return [entry[1] for entry in best.values()]


def _compress_path(chain: list[tuple[int, int]]) -> list[tuple[str, int]]:
    """Zamienia listę pozycji na kroki ``drop`` i ``side`` (znak = kierunek)."""
    steps: list[tuple[str, int]] = []
    for (ax, ay), (bx, by) in zip(chain, chain[1:]):
        kind, amount = ("drop", ay - by) if by != ay else ("side", bx - ax)
        if steps and steps[-1][0] == kind:
            steps[-1] = (kind, steps[-1][1] + amount)
        else:
            steps.append((kind, amount))
    return steps
