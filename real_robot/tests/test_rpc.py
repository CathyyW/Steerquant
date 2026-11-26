import numpy as np
import pytest

from fastwam_steerquant.robot_async_client import W4A8Async50PolicyClient
from fastwam_steerquant.robot_async_server import PackedRobotPolicy, serve_connection


def diagnostics():
    return {
        "backend": "cutlass_w4a8",
        "activation_bits": 8,
        "rotation": "none",
        "sparse_enabled": False,
        "deployment_sha256": "expected",
        "task": "pack",
        "action_coordinates": "physical",
        "kernel_audit": {
            "verified_sites": 600,
            "native_calls": 6000,
            "cuda_kernel_verified": True,
            "tile64_kernels": 6000,
        },
        "runtime_settings": {
            "w4a8_tile": "64",
            "cuda_graph": True,
            "fuse_block": False,
            "fused_modules": 0,
        },
        "cuda_graph": True,
        "summary": {"denoise_calls": 10, "finite": True, "graph_replays": 10},
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("backend", "reference"),
        ("activation_bits", 4),
        ("deployment_sha256", "other"),
        ("task", "stack"),
        ("action_coordinates", "normalized"),
    ],
)
def test_client_rejects_wrong_runtime_identity(key, value):
    client = W4A8Async50PolicyClient.__new__(W4A8Async50PolicyClient)
    client.expected_sha256 = "expected"
    client.expected_task = "pack"
    info = diagnostics()
    client.validate_diagnostics(info)
    info[key] = value
    with pytest.raises(RuntimeError):
        client.validate_diagnostics(info, require_summary=False)


def test_rpc_returns_physical_action_without_transform():
    class Policy:
        observation = None
        summary = None

        def infer(self, observation):
            assert observation == {"frame": 7}
            return np.full((32, 14), 0.125, dtype=np.float32)

        def diagnostics(self):
            return {"ok": True, **diagnostics()}

    class Connection:
        def __init__(self):
            self.requests = iter([
                {"op": "infer", "obs": {"frame": 7}},
                {"op": "reset"},
                {"op": "shutdown"},
            ])
            self.responses = []

        def recv(self):
            return next(self.requests)

        def send(self, response):
            self.responses.append(response)

    connection = Connection()
    assert serve_connection(Policy(), connection)
    assert all(response["ok"] for response in connection.responses)
    np.testing.assert_array_equal(
        connection.responses[0]["action"],
        np.full((32, 14), 0.125, dtype=np.float32),
    )


def test_server_reports_live_client_timing_contract():
    policy = PackedRobotPolicy.__new__(PackedRobotPolicy)
    policy.config = {
        "wam": {
            "ckpt": "deployment.pt",
            "quantization": {
                "task": "pack",
                "deployment_sha256": "expected",
                "cuda_graph": True,
            },
        },
        "runtime": {"action_chunk_size": 32, "rospy_rate": 40},
    }
    policy.policy = type("Policy", (), {"runtime_settings": {}})()
    policy.kernel_audit = {}
    policy.summary = None

    info = policy.diagnostics()

    assert info["control_hz"] == 40
    assert info["latency_window_ms"] == 775
