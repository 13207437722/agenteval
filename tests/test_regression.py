"""回放状态迁移"""

from agenteval.models import CaseResult, GroundTruth
from agenteval.regression import collect_bad_cases, compare_runs, pick_guard_cases, replay_report


def res(tid, label):
    return CaseResult(trace_id=tid, turn_id=0, config="x", final_label=label)


LABELS = {(t, 0): GroundTruth(trace_id=t, turn_id=0, label=l) for t, l in
          [("a", "pass"), ("b", "fail"), ("c", "pass"), ("d", "fail")]}


def test_transitions():
    old = [res("a", "fail"), res("b", "pass"), res("c", "pass"), res("d", "fail")]
    new = [res("a", "pass"), res("b", "pass"), res("c", "fail"), res("d", "fail")]
    assert collect_bad_cases(old, LABELS) == ["a", "b"]
    assert set(pick_guard_cases(old, LABELS, n=2)) <= {"c", "d"}
    t = compare_runs(old, new, LABELS)
    assert t == {"fixed": ["a"], "still_bad": ["b"], "regressed": ["c"], "stable": ["d"]}
    text = replay_report(t, "old", "new", ["a", "b"], ["c", "d"], old, new, LABELS)
    assert "修复：1/2" in text and "regressed" in text
