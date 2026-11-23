import numpy as np
import pytest

from bag import FULL_BAG, BagGenerator, build_bag_tables, remove_piece, text_to_mask
from engine import (
    MAX_SUCC,
    PIECE_INDEX,
    PIECES,
    board_from_text,
    clear_rows,
    find_moves,
    key_of,
    placements,
    segments_ok,
    split_key,
    successors_into,
)


@pytest.mark.parametrize("piece,count", [("I", 17), ("O", 9), ("T", 34), ("S", 17), ("Z", 17), ("J", 34), ("L", 34)])
def test_hard_drop_counts_on_empty_board(piece, count):
    assert len(placements(0, 4, piece)) == count


def test_tuck_under_overhang_is_found():
    field = board_from_text("""
        ..........
        ..........
        ###.......
        ..........
    """)
    target = field | 0b1111
    assert (target, 4) in placements(field, 4, "I")
    move = next(m for m in find_moves(field, 4, "I") if m.result_field == target)
    assert move.is_tuck
    assert (move.x, move.y) == (0, 0)


def test_closed_pocket_is_rejected():
    field = board_from_text("""
        ..........
        ..........
        #####.....
        ....#.....
    """)
    target = field | 0b1111
    assert (target, 4) not in placements(field, 4, "I")
    assert all(m.result_field != target for m in find_moves(field, 4, "I"))


def test_no_rotation_under_overhang():
    # T w orientacji L pasuje do szczeliny, ale wejście wymagałoby obrotu pod
    # nawisem w kolumnie 0, więc takiego ruchu nie wolno wygenerować.
    field = board_from_text("""
        ..........
        #.#.......
        ..#.......
        #.#.......
    """)
    t_left = board_from_text("""
        .#........
        ##........
        .#........
    """)
    assert field & t_left == 0
    assert (field | t_left, 4) not in placements(field, 4, "T")


def test_clear_rows_drops_upper_rows():
    board = board_from_text("""
        ..........
        #.........
        ##########
        .#........
    """)
    result, rows = clear_rows(board, 4)
    assert rows == 3
    assert result == board_from_text("""
        ..........
        #.........
        .#........
    """)


def test_placement_completes_line():
    field = board_from_text("######....")
    assert (0, 0) in placements(field, 1, "I")


def test_piece_cannot_stick_out_of_block():
    assert all(rows == 2 for _, rows in placements(0, 2, "O"))
    assert not any(f >> 20 for f, _ in placements(0, 2, "T"))
    assert len([1 for f, _ in placements(0, 2, "I")]) == 7


def _solve(sequence: str, rows: int, pruned: bool) -> bool:
    buffer = np.empty(MAX_SUCC, dtype=np.int64)
    seen: set[tuple[int, int]] = set()

    def search(depth: int, key: int) -> bool:
        field, left = split_key(key)
        if left == 0:
            return field == 0
        if depth == len(sequence) or (depth, key) in seen:
            return False
        seen.add((depth, key))
        piece = PIECE_INDEX[sequence[depth]]
        if pruned:
            count = successors_into(key, piece, buffer)
            options = [int(k) for k in buffer[:count]]
        else:
            options = [key_of(f, r) for f, r in placements(field, left, piece)]
        return any(search(depth + 1, option) for option in options)

    return search(0, key_of(0, rows))


def test_known_sequence_has_four_line_perfect_clear():
    assert _solve("ZLIJTSOIZT", 4, pruned=True)


@pytest.mark.parametrize("sequence,expected", [("IIOJJ", True), ("IIOLL", True), ("IOTSZ", False)])
def test_pruning_keeps_two_line_results(sequence, expected):
    assert _solve(sequence, 2, pruned=False) is expected
    assert _solve(sequence, 2, pruned=True) is expected


def test_segments_reject_odd_pocket():
    field = board_from_text("""
        ##.#######
        ##.#######
    """)
    assert not segments_ok(field, 2)
    assert segments_ok(board_from_text("""
        ##..######
        ##..######
    """), 2)


def test_bag_never_repeats_within_seven():
    generator = BagGenerator(seed=7)
    for _ in range(500):
        assert sorted(generator.take(7)) == list(range(7))


def test_bag_offset_starts_mid_bag():
    generator = BagGenerator(seed=3, offset=4)
    assert len(set(generator.take(3))) == 3
    assert sorted(generator.take(7)) == list(range(7))


def test_bag_sample_from_plan_is_valid():
    sample = "ZLIJTSO IZTJOSL TIZSOJL TJSIOZL".split()
    assert all(sorted(group) == sorted(PIECES) for group in sample)


def test_remove_piece_refills_empty_bag():
    assert remove_piece(0, PIECE_INDEX["T"]) == FULL_BAG & ~text_to_mask("T")
    assert remove_piece(text_to_mask("T"), PIECE_INDEX["T"]) == 0


def test_bag_tables_point_to_valid_rows():
    tables = build_bag_tables()
    for s in range(7):
        next_s = s - 1 if s else 6
        limit = 7 * int(tables.per_piece[next_s])
        assert tables.trans[s].shape == (7 * int(tables.per_piece[s]), int(tables.trans_len[s]))
        assert tables.trans[s].min() >= 0 and tables.trans[s].max() < limit
