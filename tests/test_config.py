import pytest

from etod.config import load_config, parse_overrides


def test_default_loads():
    cfg = load_config()
    assert cfg.detector.backend == "ultralytics"
    assert cfg["event_trigger"]["reference"] == "previous"   # paper-faithful default
    assert cfg["event_trigger"]["max_skip"] is None


def test_experiment_overrides_and_cli():
    cfg = load_config("configs/experiments/synthetic_smoke.yaml", ["detector.conf=0.5", "event_trigger.thresholds=[1,2]"])
    assert cfg["dataset"]["type"] == "synthetic"
    assert cfg["detector"]["conf"] == 0.5
    assert cfg["event_trigger"]["thresholds"] == [1, 2]


def test_typo_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["detector.confidence=0.5"])


def test_parse_overrides_nested():
    assert parse_overrides(["a.b.c=3"]) == {"a": {"b": {"c": 3}}}


@pytest.mark.parametrize("name", ["ua_detrac_quick", "ua_detrac_paper", "custom_video", "synthetic_smoke"])
def test_all_experiment_configs_valid(name):
    load_config(f"configs/experiments/{name}.yaml")
