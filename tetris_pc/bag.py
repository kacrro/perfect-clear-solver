"""Model losowania 7-bag oraz tablice przejść worka używane przez solver."""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from engine import NUM_PIECES, PIECES

FULL_BAG = (1 << NUM_PIECES) - 1


def bag_size_after(t: int, offset: int = 0) -> int:
    """Ile klocków zostało w worku po klocku numer ``t`` (licząc od 0).

    ``offset`` to liczba klocków worka, które wyszły przed startem levelu.
    """
    return 6 - (t + offset) % 7


def remove_piece(mask: int, piece: int) -> int:
    """Worek po wyciągnięciu klocka; pusty worek zastępuje nowy pełny."""
    current = mask if mask else FULL_BAG
    return current & ~(1 << piece)


def mask_to_text(mask: int) -> str:
    return "".join(name for i, name in enumerate(PIECES) if (mask >> i) & 1)


def text_to_mask(text: str) -> int:
    mask = 0
    for ch in text.upper():
        mask |= 1 << PIECES.index(ch)
    return mask


class BagGenerator:
    """Generator sekwencji 7-bag."""

    def __init__(self, seed: int | None = None, offset: int = 0):
        self._rng = random.Random(seed)
        self._queue: list[int] = []
        if offset:
            self._refill()
            del self._queue[:offset]

    def _refill(self) -> None:
        pieces = list(range(NUM_PIECES))
        self._rng.shuffle(pieces)
        self._queue.extend(pieces)

    def next(self) -> int:
        if not self._queue:
            self._refill()
        return self._queue.pop(0)

    def take(self, count: int) -> list[int]:
        return [self.next() for _ in range(count)]


@dataclass(frozen=True)
class BagTables:
    """Indeksowanie kombinacji (klocek, worek po nim) dla każdego rozmiaru worka.

    Dla rozmiaru ``s`` wiersz wartości ma ``7 * C(6, s)`` pól. ``index[s, c, B]``
    daje numer pola. ``trans[s]`` wskazuje pola następnej warstwy dla każdego
    możliwego kolejnego klocka, a ``trans_len[s]`` mówi, ile ich jest.
    """

    per_piece: np.ndarray
    index: np.ndarray
    masks: np.ndarray
    trans: tuple[np.ndarray, ...]
    trans_len: np.ndarray


def build_bag_tables() -> BagTables:
    per_piece = np.zeros(7, dtype=np.int64)
    index = -np.ones((7, NUM_PIECES, 128), dtype=np.int32)
    masks = -np.ones((7, NUM_PIECES, 20), dtype=np.int32)
    for s in range(7):
        for c in range(NUM_PIECES):
            others = [p for p in range(NUM_PIECES) if p != c]
            subsets = [sum(1 << p for p in combo) for combo in combinations(others, s)]
            per_piece[s] = len(subsets)
            for b, mask in enumerate(sorted(subsets)):
                index[s, c, mask] = c * len(subsets) + b
                masks[s, c, b] = mask
    trans = []
    trans_len = np.zeros(7, dtype=np.int64)
    for s in range(7):
        draws = s if s else NUM_PIECES
        next_s = s - 1 if s else 6
        trans_len[s] = draws
        table = np.zeros((NUM_PIECES * int(per_piece[s]), draws), dtype=np.int32)
        for c in range(NUM_PIECES):
            for b in range(int(per_piece[s])):
                mask = int(masks[s, c, b])
                pool = mask if s else FULL_BAG
                drawn = [p for p in range(NUM_PIECES) if (pool >> p) & 1]
                row = c * int(per_piece[s]) + b
                for slot, n in enumerate(drawn):
                    table[row, slot] = index[next_s, n, pool & ~(1 << n)]
        trans.append(table)
    return BagTables(per_piece, index, masks, tuple(trans), trans_len)
