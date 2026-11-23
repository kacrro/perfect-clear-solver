import random
import time

import numpy as np
import pytest

from bag import FULL_BAG, BagGenerator, remove_piece
from engine import MAX_SUCC, PIECE_INDEX, key_of
from simulate import simulate
from solver import TOTAL_PIECES, BagMismatch, Solver, is_flat, step_into


@pytest.fixture(scope="module")
def solver():
    instance = Solver()
    if not instance.values_ready(0):
        pytest.skip("Brak tabel: uruchom python solver.py --build")
    return instance


def test_flat_detection():
    assert is_flat(0b1111)
    assert not is_flat(1 << 10)


def test_rules_keep_flat_stack_early():
    out = np.empty(MAX_SUCC, dtype=np.int64)
    count = step_into(key_of(0, 4), 0, PIECE_INDEX["T"], out)
    assert count > 0
    assert all(is_flat(int(k) & ((1 << 40) - 1)) for k in out[:count])


def test_bellman_at_level_start(solver):
    """Wartość startu = średnia po nexcie z najlepszego ruchu."""
    current = PIECE_INDEX["T"]
    bag = remove_piece(FULL_BAG, current)
    stored = solver.state_value(key_of(0, 4), 0, current, bag, 0)
    options = [p for p in range(7) if (bag >> p) & 1]
    backed_up = np.mean([solver.analyze(0, 0, current, n, bag)[0].probability for n in options])
    assert backed_up == pytest.approx(stored, rel=1e-3)


def test_moves_are_sorted(solver):
    current = PIECE_INDEX["L"]
    bag = remove_piece(FULL_BAG, current)
    values = [e.probability for e in solver.analyze(0, 0, current, PIECE_INDEX["J"], bag)]
    assert values == sorted(values, reverse=True) and values[0] > 0


def test_impossible_next_is_reported(solver):
    current = PIECE_INDEX["T"]
    bag = remove_piece(FULL_BAG, current)
    with pytest.raises(BagMismatch):
        solver.analyze(0, 0, current, current, bag)


def test_following_best_moves_can_win_and_values_are_consistent(solver):
    wins = 0
    for seed in range(300):
        generator = BagGenerator(seed)
        sequence = generator.take(TOTAL_PIECES + 1)
        field, bag = 0, FULL_BAG
        for t in range(TOTAL_PIECES):
            bag = remove_piece(bag, sequence[t])
            nxt = sequence[t + 1] if t < TOTAL_PIECES - 1 else None
            moves = solver.analyze(field, t, sequence[t], nxt, bag)
            if not moves or moves[0].probability <= 0:
                break
            field = moves[0].move.result_field
        else:
            assert field == 0
            wins += 1
    assert 0.08 < wins / 300 < 0.26


def test_simulation_matches_exact_value(solver):
    result = simulate("optimal", 400_000, seed=11, solver=solver)
    assert abs(result.win_rate - solver.level_value()) < 3 * max(result.margin, 1e-3)


def test_suggestion_latency(solver):
    rng = random.Random(5)
    slowest = 0.0
    for _ in range(40):
        generator = BagGenerator(rng.randrange(10**6))
        sequence = generator.take(TOTAL_PIECES + 1)
        field, bag = 0, FULL_BAG
        for t in range(TOTAL_PIECES):
            bag = remove_piece(bag, sequence[t])
            nxt = sequence[t + 1] if t < TOTAL_PIECES - 1 else None
            started = time.perf_counter()
            moves = solver.analyze(field, t, sequence[t], nxt, bag)
            slowest = max(slowest, time.perf_counter() - started)
            if not moves or moves[0].probability <= 0:
                break
            field = moves[0].move.result_field
    assert slowest < 0.5
