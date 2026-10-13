import pytest

from madcal import cli

GPUS = "0, GPU-a, 40000\n1, GPU-b, 3\n2, GPU-c, 5\n3, GPU-d, 2000\n"


def test_a_gpu_with_a_compute_process_or_memory_in_use_is_not_idle():
    assert cli.idle_gpus(GPUS, "GPU-b\n") == [2]


def test_every_gpu_is_idle_on_an_empty_machine():
    assert cli.idle_gpus("0, GPU-a, 0\n1, GPU-b, 1\n", "") == [0, 1]


@pytest.fixture
def smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    outputs = {"--query-gpu=index,uuid,memory.used": GPUS, "--query-compute-apps=gpu_uuid": ""}
    monkeypatch.setattr(cli, "_smi", lambda *query: outputs[query[0]])


def test_two_processes_claim_different_idle_gpus(smi, monkeypatch, tmp_path):
    assert cli.claim_gpu(tmp_path) == "1"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert cli.claim_gpu(tmp_path) == "2"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit, match="no idle GPU"):
        cli.claim_gpu(tmp_path)


def test_a_device_set_by_the_scheduler_is_kept(smi, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert cli.claim_gpu(tmp_path) == "3"


def test_no_gpu_is_claimed_without_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.claim_gpu(tmp_path) is None
