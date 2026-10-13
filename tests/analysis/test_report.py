import pytest

from madcal.analysis import (
    ACCURACY,
    ANALYSIS_STATUS,
    ANSWER_POLICY,
    CALIBRATION_POLICY,
    REPORT_SCHEMA,
    build_report,
    to_arrow,
)
from madcal.metrics import metric_registry
from madcal.storage import ResultStore, Variant

from .conftest import Debates, single_agent_debates, write_run

RESAMPLES = 50


def report(store: ResultStore, allow_stub: bool = False):
    return build_report(store, n_resamples=RESAMPLES, seed=0, allow_stub=allow_stub)


def bounds(row):
    assert row.ci_low is not None and row.ci_high is not None
    return row.ci_low, row.ci_high


def find(rows, metric: str, level: str = "agent"):
    matches = [row for row in rows if row.metric == metric and row.level == level]
    assert len(matches) == 1, f"expected one {metric} row at {level}, got {len(matches)}"
    return matches[0]


def test_every_registered_metric_is_reported_on_every_cell(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (False, 0.2)]))
    rows = report(store)
    for level in ("agent", "system"):
        reported = {row.metric for row in rows if row.level == level}
        assert reported == {ACCURACY, *metric_registry.names()}


def test_a_system_level_cell_is_reported_beside_the_agent_cell(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (False, 0.2)]))
    assert {row.level for row in report(store)} == {"agent", "system"}


def test_accuracy_scores_an_unparseable_answer_incorrect(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (True, 0.9), (None, 0.9)]))
    assert find(report(store), ACCURACY).estimate == pytest.approx(2 / 3)


def test_accuracy_counts_every_row_including_those_without_a_confidence(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (True, None), (False, None)]))
    accuracy = find(report(store), ACCURACY)
    assert accuracy.estimate == pytest.approx(2 / 3)
    assert accuracy.n_rows == 3


def test_a_calibration_metric_drops_rows_with_no_confidence(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 1.0), (True, 1.0), (False, None)]))
    brier = find(report(store), "brier")
    assert brier.n_analysable == 2
    assert brier.estimate == pytest.approx(0.0)


def test_retention_reports_the_share_of_rows_a_calibration_metric_could_use(store: ResultStore):
    write_run(
        store,
        "run-1",
        single_agent_debates([(True, 0.5), (True, None), (False, None), (False, 0.5)]),
    )
    brier = find(report(store), "brier")
    assert brier.n_rows == 4
    assert brier.n_analysable == 2
    assert brier.retention == pytest.approx(0.5)


def test_a_calibration_metric_uses_the_graded_label_so_an_unparsed_answer_counts_wrong(
    store: ResultStore,
):
    write_run(store, "run-1", single_agent_debates([(None, 1.0), (None, 1.0)]))
    brier = find(report(store), "brier")
    assert brier.n_analysable == 2
    assert brier.estimate == pytest.approx(1.0)


def test_accuracy_and_a_calibration_metric_report_different_denominators(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.5), (True, None)]))
    rows = report(store)
    assert find(rows, ACCURACY).n_rows == 2
    assert find(rows, "brier").n_analysable == 1


def test_a_one_class_cell_reports_no_auroc_and_says_why(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (True, 0.4), (True, 0.7)]))
    auroc = find(report(store), "auroc")
    assert auroc.estimate is None
    assert auroc.reason == "undefined_on_cell"


def test_a_one_class_cell_records_the_counts_that_explain_the_gap(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (True, 0.4), (True, 0.7)]))
    auroc = find(report(store), "auroc")
    assert (auroc.n_correct, auroc.n_incorrect) == (3, 0)


def test_a_two_class_cell_reports_an_auroc(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (False, 0.1)]))
    auroc = find(report(store), "auroc")
    assert auroc.estimate == pytest.approx(1.0)
    assert auroc.reason is None


def test_a_cell_where_no_row_has_a_confidence_reports_a_null_with_a_reason(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, None), (False, None)]))
    brier = find(report(store), "brier")
    assert brier.estimate is None
    assert brier.reason == "no_rows"
    assert find(report(store), ACCURACY).estimate == pytest.approx(0.5)


def test_every_row_is_labelled_exploratory_and_names_both_policies(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (False, 0.2)]))
    for row in report(store):
        assert row.analysis_status == ANALYSIS_STATUS
        assert row.answer_policy == ANSWER_POLICY
        assert row.calibration_policy == CALIBRATION_POLICY


def test_cells_are_split_by_protocol(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9)]), protocol="vanilla")
    write_run(
        store,
        "run-2",
        single_agent_debates([(False, 0.9)]),
        protocol="best_of_n",
        config_hash="cfg-b",
    )
    protocols = {row.protocol for row in report(store)}
    assert protocols == {"vanilla", "best_of_n"}


def test_cells_are_split_by_round(store: ResultStore):
    debates: Debates = {
        f"q{index}": ([(0, 0, True, 0.9), (1, 0, False, 0.3)], (True, 0.9)) for index in range(2)
    }
    write_run(store, "run-1", debates, n_agents=1, n_rounds=1)
    rounds = {row.round for row in report(store) if row.level == "agent"}
    assert rounds == {0, 1}


def test_cells_are_split_by_agent(store: ResultStore):
    debates: Debates = {
        f"q{index}": ([(0, 0, True, 0.9), (0, 1, False, 0.3)], (True, 0.9)) for index in range(2)
    }
    write_run(store, "run-1", debates, n_agents=2, n_rounds=0)
    agents = {row.agent_id for row in report(store) if row.level == "agent"}
    assert agents == {"agent_0", "agent_1"}


def test_a_cell_counts_questions_not_rows(store: ResultStore):
    debates: Debates = {
        f"q{index}": ([(0, 0, True, 0.9), (0, 1, True, 0.3)], (True, 0.9)) for index in range(3)
    }
    write_run(store, "run-1", debates, n_agents=2, n_rounds=0)
    system = find(report(store), ACCURACY, level="system")
    assert (system.n_questions, system.n_rows) == (3, 3)


def test_stub_rows_are_refused_before_any_number_is_computed(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9)]), variant=Variant.STUB)
    with pytest.raises(ValueError, match="stub"):
        report(store)


def test_an_empty_store_reports_nothing(store: ResultStore):
    assert report(store) == []


def test_the_report_converts_to_a_table_matching_its_schema(store: ResultStore):
    write_run(store, "run-1", single_agent_debates([(True, 0.9), (False, 0.2)]))
    rows = report(store)
    table = to_arrow(rows)
    assert table.schema == REPORT_SCHEMA
    assert table.num_rows == len(rows)


def test_an_interval_brackets_its_point_estimate(store: ResultStore):
    write_run(
        store,
        "run-1",
        single_agent_debates([(index % 3 != 0, 0.1 * (index % 10)) for index in range(30)]),
    )
    accuracy = find(report(store), ACCURACY)
    low, high = bounds(accuracy)
    assert accuracy.estimate is not None
    assert low <= accuracy.estimate <= high
