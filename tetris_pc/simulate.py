"""Monte Carlo: porównanie polityk na losowych sekwencjach 7-bag."""
from __future__ import annotations

if __name__ == "__main__":
    from bootstrap import use_project_venv

    use_project_venv(__file__)

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import get_num_threads, njit, prange

from bag import build_bag_tables
from engine import MAX_SUCC, placements_into
from solver import Solver, popcount, step_into

POLICIES = {"optimal": 0, "greedy": 1, "random": 2}
POLICY_NAMES = {"optimal": "optymalna", "greedy": "zachłanna", "random": "losowa"}


@njit(nogil=True, inline="always")
def _xorshift(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


@njit(nogil=True)
def _deal(state: int, offset: int, pieces: np.ndarray, after: np.ndarray) -> int:
    """Losuje 36 klocków; ``after[i]`` to worek po klocku ``i``."""
    order = np.empty(7, dtype=np.int64)
    filled = 0
    first = True
    while filled < pieces.shape[0]:
        for i in range(7):
            order[i] = i
        for i in range(6, 0, -1):
            state = _xorshift(state)
            j = state % (i + 1)
            order[i], order[j] = order[j], order[i]
        for position in range(offset if first else 0, 7):
            if filled == pieces.shape[0]:
                break
            remaining = 0
            for k in range(position + 1, 7):
                remaining |= 1 << order[k]
            pieces[filled] = order[position]
            after[filled] = remaining
            filled += 1
        first = False
    return state


@njit(nogil=True)
def _legal_into(key: int, t: int, piece: int, out: np.ndarray) -> int:
    """Wszystkie legalne ruchy bez ograniczeń modelu (dla polityk bazowych)."""
    field = key & 1099511627775
    lines = (140 - 4 * t + popcount(field)) // 10
    if lines <= 4:
        return placements_into(field, lines, piece, out)
    raw = placements_into(field, 4, piece, out)
    kept = 0
    for i in range(raw):
        new_field = out[i] & 1099511627775
        new_lines = lines - (4 - (out[i] >> 40))
        new_rows = min(new_lines, 4)
        if (new_field >> (10 * new_rows)) != 0:
            continue
        out[kept] = new_field | (new_rows << 40)
        kept += 1
    return kept


@njit(nogil=True)
def _greedy_score(field: int) -> int:
    """Mniej znaczy lepiej: dziury, suma wysokości i nierówność powierzchni."""
    holes = 0
    aggregate = 0
    bumpiness = 0
    previous = -1
    for x in range(10):
        height = 0
        for y in range(3, -1, -1):
            if (field >> (10 * y + x)) & 1:
                height = y + 1
                break
        for y in range(height):
            if not (field >> (10 * y + x)) & 1:
                holes += 1
        aggregate += height
        if previous >= 0:
            bumpiness += abs(height - previous)
        previous = height
    return 50 * holes + aggregate + 2 * bumpiness


@njit(nogil=True)
def _play(game, seed, policy, offset, keys, key_offsets, values, value_offsets, widths, bag_index):
    """Zwraca (wygrana, dotarcie do ostatnich 4 linii, liczba decyzji)."""
    pieces = np.empty(36, dtype=np.int64)
    after = np.empty(36, dtype=np.int64)
    buffer = np.empty(MAX_SUCC, dtype=np.int64)
    state = (((game + 1) * 2654435761 + (seed + 1) * 40503) & 0xFFFFFFFF) | 1
    for _ in range(8):
        state = _xorshift(state)
    state = _deal(state, offset, pieces, after)
    key = 4 << 40
    reached = False
    decisions = 0
    for t in range(35):
        field = key & 1099511627775
        if (140 - 4 * t + popcount(field)) // 10 <= 4:
            reached = True
        decisions += 1
        if policy == 0:
            count = step_into(key, t, pieces[t], buffer)
        else:
            count = _legal_into(key, t, pieces[t], buffer)
        if count == 0:
            return 0, reached, decisions
        chosen = -1
        if policy == 0:
            if t == 34:
                for i in range(count):
                    if buffer[i] == 0:
                        chosen = i
            else:
                start = key_offsets[t + 1]
                stop = key_offsets[t + 2]
                s = 6 - (t + 1 + offset) % 7
                column = bag_index[s, pieces[t + 1], after[t + 1]]
                best = 0
                for i in range(count):
                    index = start + np.searchsorted(keys[start:stop], buffer[i])
                    if index < stop and keys[index] == buffer[i]:
                        value = values[value_offsets[t + 1] + (index - start) * widths[t + 1] + column]
                        if value > best:
                            best = value
                            chosen = i
            if chosen < 0:
                return 0, reached, decisions
        elif policy == 1:
            best_score = 1 << 40
            for i in range(count):
                score = _greedy_score(buffer[i] & 1099511627775)
                if score < best_score:
                    best_score = score
                    chosen = i
        else:
            state = _xorshift(state)
            chosen = state % count
        key = buffer[chosen]
    return int(key == 0), reached, decisions


@njit(nogil=True, parallel=True)
def _run_batch(first, count, seed, policy, offset, keys, key_offsets, values, value_offsets, widths, bag_index):
    wins = np.zeros(count, dtype=np.int8)
    reached = np.zeros(count, dtype=np.int8)
    decisions = np.zeros(count, dtype=np.int8)
    for g in prange(count):
        w, r, d = _play(first + g, seed, policy, offset, keys, key_offsets, values, value_offsets, widths, bag_index)
        wins[g] = w
        reached[g] = r
        decisions[g] = d
    return int(wins.sum()), int(reached.sum()), int(decisions.astype(np.int64).sum())


@dataclass
class SimResult:
    policy: str
    games: int = 0
    wins: int = 0
    reached: int = 0
    decisions: int = 0
    elapsed: float = 0.0
    threads: int = 1

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def margin(self) -> float:
        """Połowa 95% przedziału ufności dla procentu wygranych."""
        if not self.games:
            return 0.0
        p = self.win_rate
        return 1.96 * (p * (1 - p) / self.games) ** 0.5

    @property
    def block_rate(self) -> float:
        """Skuteczność końcowego bloku: wygrane wśród gier, które do niego doszły."""
        return self.wins / self.reached if self.reached else 0.0

    @property
    def reached_rate(self) -> float:
        return self.reached / self.games if self.games else 0.0

    @property
    def decision_us(self) -> float:
        return self.elapsed * self.threads / self.decisions * 1e6 if self.decisions else 0.0


def simulate(
    policy: str,
    games: int,
    seed: int = 1,
    offset: int = 0,
    batch: int = 2000,
    on_progress: Callable[[SimResult], bool | None] | None = None,
    solver: Solver | None = None,
) -> SimResult:
    """Rozgrywa partie partiami; ``on_progress`` zwraca False, aby przerwać."""
    tables = build_bag_tables()
    if policy == "optimal":
        solver = solver or Solver()
        data = solver.tables(offset)
        keys, key_offsets = solver.flat_arrays()
        values, value_offsets, widths = np.asarray(data.values), data.offsets, data.widths
    else:
        keys = np.empty(1, dtype=np.int64)
        keys.setflags(write=False)
        values = np.empty(1, dtype=np.uint16)
        values.setflags(write=False)
        key_offsets = value_offsets = widths = np.zeros(37, dtype=np.int64)
    result = SimResult(policy, threads=get_num_threads())
    started = time.perf_counter()
    while result.games < games:
        size = min(batch, games - result.games)
        wins, reached, decisions = _run_batch(
            result.games, size, seed, POLICIES[policy], offset, keys, key_offsets,
            values, value_offsets, widths, tables.index,
        )
        result.games += size
        result.wins += wins
        result.reached += reached
        result.decisions += decisions
        result.elapsed = time.perf_counter() - started
        if on_progress is not None and on_progress(result) is False:
            break
    return result


def format_result(result: SimResult) -> str:
    return (
        f"{POLICY_NAMES[result.policy]:<10} gry: {result.games:,}  "
        f"level: {result.win_rate * 100:.3f}% ±{result.margin * 100:.3f}  "
        f"doszło do ostatnich 4 linii: {result.reached_rate * 100:.2f}%  "
        f"końcowy blok 4 linii: {result.block_rate * 100:.2f}%  "
        f"decyzja: {result.decision_us:.1f} µs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Symulacja polityk")
    parser.add_argument("--games", type=int, default=100_000)
    parser.add_argument("--policy", choices=[*POLICIES, "all"], default="all")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0, choices=range(7))
    args = parser.parse_args()
    names = list(POLICIES) if args.policy == "all" else [args.policy]
    for name in names:
        print(format_result(simulate(name, args.games, args.seed, args.offset)), flush=True)


if __name__ == "__main__":
    main()
