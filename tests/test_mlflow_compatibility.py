"""Exercise Goblet's actual tracking statements without requiring DGL/CUDA.

The training code supplies scalar losses to MLflow; these checks execute its
tracking statements with deterministic scalar fixtures and the real MLflow API.
They do not run neural-network training or mock the MLflow client.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Model" / "train.py"
REFERENCE = Path(__file__).with_name("mlflow_reference.json")
LEGACY = Path(__file__).with_name("legacy_mlruns")


def source_tracking_statements():
    tree = ast.parse(SOURCE.read_text())
    train = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "train")
    before, metrics, after = [], [], []
    for statement in train.body:
        if isinstance(statement, ast.For):
            metrics += [s for s in statement.body if is_tracking_call(s)]
        elif is_tracking_call(statement):
            name = statement.value.func.attr
            (after if name in ("log_artifact", "end_run") else before).append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            # Include the compatibility opt-in from the real training code.
            call = statement.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "setdefault":
                if call.args and isinstance(call.args[0], ast.Constant):
                    if call.args[0].value == "MLFLOW_ALLOW_FILE_STORE":
                        before.append(statement)
    return before, metrics, after


def is_tracking_call(statement):
    return (isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "mlflow")


def execute_statements(statements, namespace):
    code = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(code, str(SOURCE), "exec"), namespace)


def replay_tracking(directory):
    directory = Path(directory)
    cfg = {"train": {"batch_size": 256, "lr": 0.0001, "epoch": 2,
                     "data_size": 1000000, "log_dir": "/log/",
                     "log_filename": "pretrain-ver2.txt"}}
    artifact = directory / "log" / "pretrain-ver2.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("pretrain-ver2.txt")
    namespace = {"mlflow": mlflow, "np": np, "cfg": cfg, "os": os,
                 "hydra": SimpleNamespace(utils=SimpleNamespace(
                     get_original_cwd=lambda: str(directory)))}
    before, metrics, after = source_tracking_statements()
    execute_statements(before, namespace)
    run_id = mlflow.active_run().info.run_id
    for step in [1, 2]:
        namespace["step"] = step
        for index, key in enumerate(["train_loss", "train_loss_node", "train_loss_edge",
                    "train_acc_node", "train_acc_edge", "test_loss",
                    "test_loss_node", "test_loss_edge", "test_acc_node",
                    "test_acc_edge"]):
            namespace[key] = [(0.5 + index * 0.05) / step,
                              (0.25 + index * 0.025) / step]
        execute_statements(metrics, namespace)
    execute_statements(after, namespace)
    return run_id


def snapshot(client, run_id):
    run = client.get_run(run_id)
    histories = {key: [{"step": metric.step, "value": metric.value}
                      for metric in client.get_metric_history(run_id, key)]
                 for key in sorted(run.data.metrics)}
    artifacts = client.list_artifacts(run_id)
    downloaded = client.download_artifacts(run_id, "pretrain-ver2.txt")
    return {"params": run.data.params, "metrics": run.data.metrics,
            "histories": histories, "status": run.info.status,
            "artifact_names": [artifact.path for artifact in artifacts],
            "artifact_sha256": hashlib.sha256(Path(downloaded).read_bytes()).hexdigest()}


class MLflowCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(REFERENCE.read_text())
        self.environment = patch.dict(os.environ)
        self.environment.start()
        os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        self.addCleanup(self.environment.stop)
        self.addCleanup(mlflow.end_run)

    def test_training_tracking_matches_old_version(self):
        with tempfile.TemporaryDirectory() as directory:
            run_id = replay_tracking(directory)
            self.assertEqual(snapshot(mlflow.tracking.MlflowClient(), run_id),
                             self.reference["snapshot"])

    def test_existing_2_11_3_runs_read_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / "mlruns"
            shutil.copytree(LEGACY, store)
            run_id = self.reference["legacy_run_id"]
            # The fixture has portable placeholders instead of machine paths.
            for meta in store.rglob("meta.yaml"):
                value = yaml.safe_load(meta.read_text())
                if "artifact_location" in value:
                    value["artifact_location"] = (store / "0").as_uri()
                if "artifact_uri" in value:
                    value["artifact_uri"] = (store / "0" / run_id / "artifacts").as_uri()
                with meta.open("w") as handle:
                    yaml.safe_dump(value, handle)
            # Execute Goblet's own setup, before interacting with old runs.
            namespace = {"mlflow": mlflow, "os": os,
                         "hydra": SimpleNamespace(utils=SimpleNamespace(
                             get_original_cwd=lambda: directory))}
            before, _, _ = source_tracking_statements()
            setup = [s for s in before if not (is_tracking_call(s)
                     and s.value.func.attr in ("start_run", "log_param"))]
            execute_statements(setup, namespace)
            client = mlflow.tracking.MlflowClient()
            self.assertEqual(snapshot(client, run_id), self.reference["snapshot"])
            run = client.get_run(run_id)
            self.assertEqual({"experiment_id": run.info.experiment_id,
                              "start_time": run.info.start_time,
                              "end_time": run.info.end_time},
                             self.reference["legacy_metadata"])
            for key, expected in self.reference["legacy_metric_timestamps"].items():
                self.assertEqual([point.timestamp for point in
                                  client.get_metric_history(run_id, key)], expected)
            # Existing runs remain resumable; original metrics are retained.
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("resume_check", 0.125, step=3)
            self.assertEqual(client.get_run(run_id).data.metrics["resume_check"], 0.125)
            for key in self.reference["snapshot"]["metrics"]:
                self.assertEqual(client.get_run(run_id).data.metrics[key],
                                 self.reference["snapshot"]["metrics"][key])

    @unittest.skipIf(mlflow.__version__.startswith("2."),
                     "The file-store opt-in was introduced in MLflow 3")
    def test_explicit_file_store_opt_out_is_respected(self):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "false"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(mlflow.exceptions.MlflowException):
                replay_tracking(directory)
        self.assertEqual(os.environ["MLFLOW_ALLOW_FILE_STORE"], "false")


if __name__ == "__main__":
    unittest.main()
