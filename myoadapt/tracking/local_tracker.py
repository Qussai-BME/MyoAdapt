"""
local_tracker.py — JSON-based experiment tracker (no MLflow required)
======================================================================
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class LocalTracker:
    """
    Lightweight experiment tracker that writes JSON files.

    Use case: research environments without MLflow, CI runs, edge devices.

    Each experiment gets a directory; each run gets a JSON file with:
    - config
    - metrics
    - parameters
    - artifacts (paths)
    - status
    """

    def __init__(self, root_dir: Union[str, Path] = "./experiments"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_id: Optional[str] = None
        self.current_experiment: Optional[str] = None
        self.run_data: Dict[str, Any] = {}

    def set_experiment(self, name: str) -> LocalTracker:
        self.current_experiment = name
        (self.root_dir / name).mkdir(parents=True, exist_ok=True)
        return self

    def start_run(self, run_name: Optional[str] = None) -> LocalTracker:
        if not self.current_experiment:
            self.set_experiment("default")
        self.current_run_id = run_name or f"run_{uuid.uuid4().hex[:8]}"
        self.run_data = {
            "run_id": self.current_run_id,
            "experiment": self.current_experiment,
            "run_name": run_name,
            "start_time": time.time(),
            "status": "running",
            "params": {},
            "metrics": {},
            "artifacts": [],
        }
        return self

    def log_params(self, params: Dict[str, Any]) -> None:
        self.run_data["params"].update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        entry = {"value": float(value), "step": step or 0, "time": time.time()}
        if key not in self.run_data["metrics"]:
            self.run_data["metrics"][key] = []
        self.run_data["metrics"][key].append(entry)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, path: Union[str, Path]) -> None:
        self.run_data["artifacts"].append(str(path))

    def end_run(self, status: str = "completed") -> Path:
        self.run_data["end_time"] = time.time()
        self.run_data["elapsed_seconds"] = (
            self.run_data["end_time"] - self.run_data["start_time"]
        )
        self.run_data["status"] = status

        out_path = (self.root_dir / self.current_experiment /
                     f"{self.current_run_id}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.run_data, f, indent=2, default=str)
        logger.info(f"Run logged → {out_path}")
        self.current_run_id = None
        self.run_data = {}
        return out_path

    def list_runs(self, experiment: Optional[str] = None) -> list:
        exp = experiment or self.current_experiment
        if not exp:
            return []
        runs = []
        for f in (self.root_dir / exp).glob("*.json"):
            with open(f) as fp:
                runs.append(json.load(fp))
        return runs

    def get_best_run(self, metric: str, mode: str = "max") -> Optional[Dict]:
        runs = self.list_runs()
        if not runs:
            return None
        scored = []
        for r in runs:
            if metric in r.get("metrics", {}):
                values = r["metrics"][metric]
                last = values[-1]["value"] if values else None
                if last is not None:
                    scored.append((last, r))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=(mode == "max"))
        return scored[0][1]
