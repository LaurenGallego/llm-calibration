import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from madcal.analysis import load_frame
from madcal.storage import ResultStore, RunStatus, Variant

from .conftest import single_agent_debates, write_run


def test_an_empty_store_gives_an_empty_frame(store: ResultStore):
    frame = load_frame(store)
    assert frame.rows.num_rows == 0
    assert frame.excluded_runs == ()


def test_a_run_that_never_finished_is_excluded_and_named(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]), status=RunStatus.RUNNING)
    frame = load_frame(store)
    assert frame.rows.num_rows == 0
    assert frame.excluded_runs == ("run-1",)


def test_a_failed_run_is_excluded(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]), status=RunStatus.FAILED)
    frame = load_frame(store)
    assert frame.rows.num_rows == 0
    assert frame.excluded_runs == ("run-1",)


def test_a_complete_run_is_read(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8), (False, 0.6)]))
    frame = load_frame(store)
    assert frame.rows.num_rows == 4
    assert frame.excluded_runs == ()


def test_system_rows_survive_the_null_safe_join(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8), (False, 0.6)]))
    levels = load_frame(store).rows.column("level").to_pylist()
    assert levels.count("system") == 2
    assert levels.count("agent") == 2


def test_system_confidence_is_read_from_the_system_signal(store: ResultStore):
    write_run(store, "run-1", {"q0": ([(0, 0, True, 0.25)], (True, 0.75))})
    rows = load_frame(store).rows.to_pydict()
    by_level = dict(zip(rows["level"], rows["confidence"], strict=True))
    assert by_level["agent"] == 0.25
    assert by_level["system"] == 0.75


def test_an_unparseable_confidence_is_a_null_and_keeps_its_graded_row(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, None)]))
    rows = load_frame(store).rows.to_pydict()
    assert rows["confidence"] == [None, None]
    assert rows["graded_correct"] == [True, True]


def test_an_unparseable_answer_is_graded_incorrect_but_keeps_its_true_null(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(None, 0.9)]))
    rows = load_frame(store).rows.to_pydict()
    assert rows["correct"] == [None, None]
    assert rows["graded_correct"] == [False, False]


def test_stub_rows_are_refused_by_default(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]), variant=Variant.STUB)
    with pytest.raises(ValueError, match="stub"):
        load_frame(store)


def test_stub_rows_are_readable_when_explicitly_allowed(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]), variant=Variant.STUB)
    assert load_frame(store, allow_stub=True).rows.num_rows == 2


def test_an_unfinished_stub_run_is_excluded_before_the_stub_check(store: ResultStore):
    write_run(
        store,
        "run-1",
        single_agent_debates([(True, 0.8)]),
        variant=Variant.STUB,
        status=RunStatus.RUNNING,
    )
    assert load_frame(store).rows.num_rows == 0


def test_two_complete_runs_are_read_together(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]))
    write_run(store, "run-2", single_agent_debates([(False, 0.2)]), config_hash="cfg-b")
    assert load_frame(store).rows.num_rows == 4


def test_a_finished_run_is_read_while_an_unfinished_one_beside_it_is_not(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]))
    write_run(
        store,
        "run-2",
        single_agent_debates([(False, 0.2)]),
        config_hash="cfg-b",
        status=RunStatus.RUNNING,
    )
    frame = load_frame(store)
    assert frame.rows.num_rows == 2
    assert frame.excluded_runs == ("run-2",)


def test_a_duplicated_signal_row_is_refused_rather_than_inflating_the_frame(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8)]))
    signals = sorted((store.root / "signals").rglob("*.parquet"))[0]
    table = pq.read_table(signals)
    pq.write_table(pa.concat_tables([table, table]), signals)
    with pytest.raises(ValueError, match="changed the row count"):
        load_frame(store)


def test_a_graded_row_with_no_confidence_row_at_all_is_refused(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.8), (False, 0.4)]))
    signals = sorted((store.root / "signals").rglob("*.parquet"))[0]
    table = pq.read_table(signals)
    pq.write_table(table.slice(0, table.num_rows - 1), signals)
    with pytest.raises(ValueError, match="no confidence row"):
        load_frame(store)
