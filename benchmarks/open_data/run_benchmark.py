"""Run feature selection on open datasets and score what it produced.

For each (dataset, pipeline) pair this measures three things that a selection
run has to be judged on together:

* **cost** -- wall clock and peak RSS, under the same 8-core / 50 GB budget the
  work cluster gives the job;
* **reduction** -- how many features survived;
* **quality** -- ROC-AUC of the same LightGBM trained on the selection versus
  on every feature, scored on the ``test`` split the pipeline never reads.

A pipeline that halves the feature count and loses 0.002 AUC is a good trade; a
pipeline that halves it and loses 0.05 is not, and only the third measurement
tells them apart.

    python -m benchmarks.open_data.run_benchmark --datasets madelon --pipelines cheap
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from benchmarks.open_data import datasets as ds
from benchmarks.open_data.evaluate import (
    Evaluation,
    evaluate_feature_set,
    selection_quality,
    summarize,
)
from fmlib.feature_selection import (
    FeatureSchema,
    FeatureSelectionConfig,
    FeatureSelectionPipeline,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MEMORY_LIMIT_GB = 50
CPU_BUDGET = 8


@dataclass
class RunRecord:
    """One (dataset, pipeline) outcome."""

    dataset: str
    pipeline: str
    status: str
    n_features_in: int = 0
    n_features_out: int = 0
    seconds: float = 0.0
    peak_rss_gb: float = 0.0
    steps: list[dict[str, Any]] = field(default_factory=list)
    evaluation: Optional[dict[str, Any]] = None
    ground_truth: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self: RunRecord) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "dataset": self.dataset,
            "pipeline": self.pipeline,
            "status": self.status,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "seconds": round(self.seconds, 2),
            "peak_rss_gb": round(self.peak_rss_gb, 2),
            "steps": self.steps,
            "evaluation": self.evaluation,
            "ground_truth": self.ground_truth,
            "error": self.error,
        }


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def apply_memory_cap(limit_gb: int = MEMORY_LIMIT_GB) -> None:
    """Cap the address space so an overrun fails here, not on the cluster.

    The benchmark exists partly to find out which datasets fit the budget, so
    exceeding it has to be an error rather than a slow swap.
    """
    limit = limit_gb * 1024**3
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    logger.info("address space capped at %d GB", limit_gb)


def load_pipeline_config(name: str, *, dataset: str) -> FeatureSelectionConfig:
    """Load a pipeline YAML and stamp it with this dataset's cache identity."""
    payload = _read_yaml(CONFIG_DIR / f"{name}.yaml")
    cache = payload.setdefault("statistics", {}).setdefault("cache", {})
    if cache.get("enabled"):
        cache["dataset_id"] = f"{dataset}:{name}"
        cache["path"] = str(RESULTS_DIR / "cache" / f"{dataset}.json")
    return FeatureSelectionConfig.from_dict(payload)


def _read_yaml(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        msg = f"{path}: YAML root must be a mapping."
        raise ValueError(msg)
    return payload


def build_schema(meta: dict[str, Any]) -> FeatureSchema:
    """Build the pipeline schema from prepared-dataset metadata."""
    return FeatureSchema(
        categorical=tuple(meta["categorical"]),
        continuous=tuple(meta["continuous"]),
        target=meta["target"],
        task_type=meta["task_type"],
        time=meta["time"],
    )


def run_one(
    dataset: str,
    pipeline: str,
    *,
    frames: dict[str, pd.DataFrame],
    meta: dict[str, Any],
    output_dir: Path,
    skip_evaluation: bool = False,
) -> RunRecord:
    """Run one pipeline on one dataset and score the result."""
    schema = build_schema(meta)
    record = RunRecord(
        dataset=dataset,
        pipeline=pipeline,
        status="ok",
        n_features_in=len(schema.candidate_features()),
    )
    try:
        config = load_pipeline_config(pipeline, dataset=dataset)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        record.status = "config_error"
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    run_dir = output_dir / dataset / pipeline
    _clear_run_dir(run_dir)

    started = time.perf_counter()
    try:
        # The pipeline never sees `test`: it is reserved for scoring below.
        result = FeatureSelectionPipeline(config).fit_select(
            schema=schema,
            datasets={"train": frames["train"], "valid": frames["valid"]},
            output_dir=run_dir,
        )
    except Exception as exc:  # noqa: BLE001 - a failing pipeline is a result
        record.status = "failed"
        record.seconds = time.perf_counter() - started
        record.peak_rss_gb = peak_rss_gb()
        record.error = f"{type(exc).__name__}: {exc}"
        logger.warning("%s/%s failed: %s", dataset, pipeline, record.error)
        logger.debug(traceback.format_exc())
        return record

    record.seconds = time.perf_counter() - started
    record.peak_rss_gb = peak_rss_gb()
    record.n_features_out = len(result.selected_features)
    record.steps = _step_summary(result)
    record.ground_truth = selection_quality(
        result.selected_features,
        meta.get("informative_features", []),
        meta.get("drift_features", []),
    )

    if not skip_evaluation and result.selected_features:
        try:
            record.evaluation = evaluate_feature_set(
                frames,
                result.selected_features,
                target=meta["target"],
                categorical=meta["categorical"],
                label=pipeline,
                n_jobs=CPU_BUDGET,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001 - keep the selection result
            record.status = "eval_failed"
            record.error = f"{type(exc).__name__}: {exc}"
    return record


def _clear_run_dir(run_dir: Path) -> None:
    """Empty a run directory so a re-run's artifacts are the only ones in it.

    ``SelectionResult.save`` never overwrites -- it appends ``_1``, ``_2`` --
    which is right for a production run and wrong here: after a second sweep
    the directory holds two generations side by side and reading
    ``final_results.json`` silently gives the older one.
    """
    if run_dir.exists():
        for path in sorted(run_dir.glob("*.json")):
            path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)


def _step_summary(result: Any) -> list[dict[str, Any]]:
    """How many features each step removed, in order."""
    counts: dict[str, int] = {}
    for item in result.dropped_features:
        key = f"{item.stage}.{item.method}"
        counts[key] = counts.get(key, 0) + 1
    return [{"step": key, "dropped": value} for key, value in counts.items()]


def run_dataset(
    dataset: str,
    pipelines: list[str],
    *,
    output_dir: Path,
    skip_baseline: bool = False,
) -> list[RunRecord]:
    """Run every pipeline on one dataset, plus the all-features baseline."""
    frames, meta = ds.load(dataset)
    logger.info(
        "%s: %d train rows, %d features (%d categorical), %.2f%% positive",
        dataset,
        len(frames["train"]),
        meta["n_features"],
        len(meta["categorical"]),
        100 * meta["positive_rate"]["train"],
    )

    records: list[RunRecord] = []
    if not skip_baseline:
        baseline = RunRecord(
            dataset=dataset,
            pipeline="baseline",
            status="ok",
            n_features_in=meta["n_features"],
            n_features_out=meta["n_features"],
        )
        all_features = list(meta["categorical"]) + list(meta["continuous"])
        baseline.evaluation = evaluate_feature_set(
            frames,
            all_features,
            target=meta["target"],
            categorical=meta["categorical"],
            label="baseline",
            n_jobs=CPU_BUDGET,
        ).to_dict()
        baseline.peak_rss_gb = peak_rss_gb()
        records.append(baseline)

    for pipeline in pipelines:
        records.append(
            run_one(
                dataset,
                pipeline,
                frames=frames,
                meta=meta,
                output_dir=output_dir,
            ),
        )
    return records


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--pipelines", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--tag", default="run", help="name for this sweep's files")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--no-memory-cap",
        action="store_true",
        help="do not cap the address space at 50 GB",
    )
    args = parser.parse_args()

    if not args.no_memory_cap:
        apply_memory_cap()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    for dataset in args.datasets:
        records.extend(
            run_dataset(
                dataset,
                args.pipelines,
                output_dir=output_dir,
                skip_baseline=args.skip_baseline,
            ),
        )

    payload = [record.to_dict() for record in records]
    json_path = output_dir / f"{args.tag}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    table = _flat_table(records)
    csv_path = output_dir / f"{args.tag}.csv"
    table.to_csv(csv_path, index=False)
    logger.info("wrote %s and %s", json_path, csv_path)
    print(table.to_string(index=False))  # noqa: T201 - this is a CLI report


def _flat_table(records: list[RunRecord]) -> pd.DataFrame:
    """One row per run, with the downstream numbers inlined."""
    rows = []
    for record in records:
        evaluation = record.evaluation or {}
        rows.append(
            {
                "dataset": record.dataset,
                "pipeline": record.pipeline,
                "status": record.status,
                "features_in": record.n_features_in,
                "features_out": record.n_features_out,
                "kept_%": (
                    round(100 * record.n_features_out / record.n_features_in, 1)
                    if record.n_features_in
                    else None
                ),
                "select_s": round(record.seconds, 1),
                "peak_rss_gb": round(record.peak_rss_gb, 2),
                "auc_test": evaluation.get("auc_test"),
                "auc_valid": evaluation.get("auc_valid"),
                "fit_s": evaluation.get("fit_seconds"),
                **record.ground_truth,
                "error": record.error[:80],
            },
        )
    frame = pd.DataFrame(rows)
    if "auc_test" in frame and not frame.empty:
        for _dataset, part in frame.groupby("dataset"):
            baseline = part.loc[part["pipeline"] == "baseline", "auc_test"]
            if not baseline.empty and pd.notna(baseline.iloc[0]):
                frame.loc[part.index, "auc_delta"] = (
                    part["auc_test"] - float(baseline.iloc[0])
                ).round(5)
    return frame


__all__ = ["Evaluation", "RunRecord", "main", "run_dataset", "run_one", "summarize"]

if __name__ == "__main__":
    main()
