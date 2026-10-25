from tools.run_lora_post_surgery_study import _ordered_test_rows


def test_test_row_order_is_preserved_without_a_sampling_seed() -> None:
    rows = tuple((index, f"row-{index}") for index in range(20))

    assert _ordered_test_rows(rows, None) is rows


def test_test_row_shuffle_is_deterministic_and_preserves_every_row() -> None:
    rows = tuple((index, f"row-{index}") for index in range(20))

    first = _ordered_test_rows(rows, 2026)
    repeated = _ordered_test_rows(rows, 2026)

    assert first == repeated
    assert first != rows
    assert set(first) == set(rows)
