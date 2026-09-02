"""Run feature selection on open datasets and score what it produced.

For each (dataset, pipeline, seed) triple this measures three things that a
selection run has to be judged on together:

* **cost** -- wall clock and peak RSS, under the same 8-core / 50 GB budget the
  work cluster gives the job;
* **reduction** -- how many features survived;
* **quality** -- ROC-AUC of the same LightGBM trained on the selection versus
  on every feature, scored on the ``test`` split the pipeline never reads.

A pipeline that halves the feature count and loses 0.002 AUC is a good trade; a
pipeline that halves it and loses 0.05 is not, and only the third measurement
tells them apart.

The seed picks the train/valid/test split, the pipeline's own sampling and the
downstream model's seed, so repeating a sweep over several seeds gives the
spread that says which differences are real::

    python -m benchmarks.open_data.run_benchmark \
        --datasets madelon --pipelines cheap --seeds 42 7 2024 --workers 6

A pipeline may carry parameter overrides after ``@``, so a threshold sweep
needs no new config files::

    --pipelines 'only_iv@threshold=0.005' 'only_iv@threshold=0.05'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    """One (dataset, pipeline, seed) outcome."""

    dataset: str
    pipeline: str
    status: str
    seed: int = ds.DEFAULT_SEED
    n_features_in: int = 0
    n_features_out: int = 0
    seconds: float = 0.0
    peak_rss_gb: float = 0.0
    n_jobs: int = CPU_BUDGET
    steps: list[dict[str, Any]] = field(default_factory=list)
    evaluation: Optional[dict[str, Any]] = None
    ground_truth: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self: RunRecord) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "dataset": self.dataset,
            "pipeline": self.pipeline,
            "seed": self.seed,
            "status": self.status,
            "n_features_in": self.n_features_in,
            "n_features_out": self.n_features_out,
            "seconds": round(self.seconds, 2),
            "peak_rss_gb": round(self.peak_rss_gb, 2),
            "n_jobs": self.n_jobs,
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
    exceeding it has to be an error rather than a slow swap. Under
    ``--workers N`` the cap is per worker and the parent divides the budget, so
    the processes together stay inside it.
    """
    limit = limit_gb * 1024**3
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    logger.info("address space capped at %d GB", limit_gb)


def parse_pipeline_spec(spec: str) -> tuple[str, dict[str, str]]:
    """Split ``name@key=value,key=value`` into a config name and overrides.

    Args:
        spec: Pipeline specification as written on the command line.

    Returns:
        The config file's stem and the raw (unparsed) override mapping.

    Raises:
        ValueError: If an override is not in ``key=value`` form.
    """
    name, _, tail = spec.partition("@")
    overrides: dict[str, str] = {}
    for chunk in filter(None, (part.strip() for part in tail.split(","))):
        key, sep, value = chunk.partition("=")
        if not sep:
            msg = f"{spec!r}: override {chunk!r} is not key=value."
            raise ValueError(msg)
        overrides[key.strip()] = value.strip()
    return name.strip(), overrides


def load_pipeline_config(
    spec: str,
    *,
    dataset: str,
    seed: int = ds.DEFAULT_SEED,
) -> FeatureSelectionConfig:
    """Load a pipeline YAML, apply its overrides and stamp it with the seed."""
    name, overrides = parse_pipeline_spec(spec)
    payload = _read_yaml(CONFIG_DIR / f"{name}.yaml")
    _apply_overrides(payload, overrides)
    payload.setdefault("execution", {})["seed"] = seed

    cache = payload.setdefault("statistics", {}).setdefault("cache", {})
    if cache.get("enabled"):
        cache["dataset_id"] = f"{dataset}:{spec}:s{seed}"
        cache["path"] = str(RESULTS_DIR / "cache" / f"{dataset}_s{seed}.json")
    return FeatureSelectionConfig.from_dict(payload)


def _apply_overrides(payload: dict[str, Any], overrides: dict[str, str]) -> None:
    """Patch a config payload in place with ``key=value`` command-line overrides.

    ``iv.threshold`` addresses a step's parameter, ``execution.seed`` a
    top-level section, and a bare ``threshold`` the sole step of a
    single-method config -- which is what the per-method sweeps use.
    """
    import yaml

    steps = _step_params(payload)
    for key, raw in overrides.items():
        value = yaml.safe_load(raw)
        head, _, rest = key.partition(".")
        if rest and head in steps:
            _set_path(steps[head], rest, value)
        elif not rest and len(steps) == 1:
            _set_path(next(iter(steps.values())), key, value)
        else:
            _set_path(payload, key, value)


def _step_params(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map each method in ``order`` to its (materialised) parameter mapping."""
    found: dict[str, dict[str, Any]] = {}
    for step in payload.get("order") or []:
        if not isinstance(step, dict):
            continue
        for method, params in list(step.items()):
            if params is None:
                params = {}
                step[method] = params
            if isinstance(params, dict):
                found[method] = params
    return found


def _set_path(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign ``value`` at a dotted path, creating intermediate mappings."""
    keys = dotted.split(".")
    for key in keys[:-1]:
        nxt = mapping.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            mapping[key] = nxt
        mapping = nxt
    mapping[keys[-1]] = value


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
    seed: int = ds.DEFAULT_SEED,
    n_jobs: int = CPU_BUDGET,
    skip_evaluation: bool = False,
) -> RunRecord:
    """Run one pipeline on one dataset and score the result."""
    schema = build_schema(meta)
    record = RunRecord(
        dataset=dataset,
        pipeline=pipeline,
        status="ok",
        seed=seed,
        n_jobs=n_jobs,
        n_features_in=len(schema.candidate_features()),
    )
    try:
        config = load_pipeline_config(pipeline, dataset=dataset, seed=seed)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        record.status = "config_error"
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    run_dir = output_dir / dataset / f"s{seed}" / _safe_name(pipeline)
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
        logger.warning("%s/%s/s%d failed: %s", dataset, pipeline, seed, record.error)
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
                seed=seed,
                n_jobs=n_jobs,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001 - keep the selection result
            record.status = "eval_failed"
            record.error = f"{type(exc).__name__}: {exc}"
    return record


def _safe_name(pipeline: str) -> str:
    """Turn a pipeline spec into something usable as a directory name."""
    return pipeline.replace("/", "_").replace(" ", "")


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
    seed: int = ds.DEFAULT_SEED,
    n_jobs: int = CPU_BUDGET,
    skip_baseline: bool = False,
) -> list[RunRecord]:
    """Run every pipeline on one dataset and seed, plus the all-features baseline."""
    frames, meta = ds.load(dataset, seed)
    logger.info(
        "%s (seed %d): %d train rows, %d features (%d categorical), %.2f%% positive",
        dataset,
        seed,
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
            seed=seed,
            n_jobs=n_jobs,
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
            seed=seed,
            n_jobs=n_jobs,
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
                seed=seed,
                n_jobs=n_jobs,
            ),
        )
    return records


def _run_unit(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker entry point: every pipeline for one (dataset, seed) pair.

    Grouping by dataset and seed means the parquet files are read once per
    unit rather than once per run, and keeps each worker's resident set to a
    single copy of one dataset.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
    )
    if payload["memory_gb"]:
        apply_memory_cap(payload["memory_gb"])
    records = run_dataset(
        payload["dataset"],
        payload["pipelines"],
        output_dir=Path(payload["output_dir"]),
        seed=payload["seed"],
        n_jobs=payload["n_jobs"],
        skip_baseline=payload["skip_baseline"],
    )
    return [record.to_dict() for record in records]


def _set_thread_env(threads: int) -> None:
    """Pin the numeric libraries to ``threads`` so N workers share 8 cores.

    This has to happen before a worker is spawned: OpenMP reads the thread
    count once at load, and a worker that missed it would take the whole
    machine and make every timing in the sweep meaningless.
    """
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--pipelines",
        nargs="+",
        required=True,
        help="config name, optionally with @key=value overrides",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[ds.DEFAULT_SEED],
        help="one split, sampling and model seed per value",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="run (dataset, seed) units in parallel; the CPU budget is divided",
    )
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--tag", default="run", help="name for this sweep's files")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--no-memory-cap",
        action="store_true",
        help="do not cap the address space at 50 GB",
    )
    args = parser.parse_args()

    workers = max(1, args.workers)
    n_jobs = max(1, CPU_BUDGET // workers)
    memory_gb = 0 if args.no_memory_cap else max(1, MEMORY_LIMIT_GB // workers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    units = [
        {
            "dataset": dataset,
            "seed": seed,
            "pipelines": list(args.pipelines),
            "output_dir": str(output_dir),
            "skip_baseline": args.skip_baseline,
            "n_jobs": n_jobs,
            "memory_gb": memory_gb,
        }
        for dataset in args.datasets
        for seed in args.seeds
    ]

    logger.info(
        "%d units over %d worker(s): %d thread(s) and %d GB each",
        len(units),
        workers,
        n_jobs,
        memory_gb or MEMORY_LIMIT_GB,
    )
    payload = _execute(units, workers=workers, n_jobs=n_jobs, memory_gb=memory_gb)

    json_path = output_dir / f"{args.tag}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    table = _flat_table(payload)
    csv_path = output_dir / f"{args.tag}.csv"
    table.to_csv(csv_path, index=False)
    logger.info("wrote %s and %s", json_path, csv_path)
    print(table.to_string(index=False))  # noqa: T201 - this is a CLI report


def _execute(
    units: list[dict[str, Any]],
    *,
    workers: int,
    n_jobs: int,
    memory_gb: int,
) -> list[dict[str, Any]]:
    """Run every unit, in this process or across a pool, and collect the rows."""
    if workers == 1:
        _set_thread_env(n_jobs)
        if memory_gb:
            apply_memory_cap(memory_gb)
        rows: list[dict[str, Any]] = []
        for unit in units:
            rows.extend(_run_unit({**unit, "memory_gb": 0}))  # capped already
        return rows

    import multiprocessing as mp

    # Spawned workers inherit the environment as it stands now, so the thread
    # pinning has to be in place before the pool is built.
    _set_thread_env(n_jobs)
    rows = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {pool.submit(_run_unit, unit): unit for unit in units}
        for future in as_completed(futures):
            unit = futures[future]
            done += 1
            try:
                rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - one unit must not sink the sweep
                logger.error(
                    "unit %s/s%d died: %s: %s",
                    unit["dataset"],
                    unit["seed"],
                    type(exc).__name__,
                    exc,
                )
            logger.info(
                "%d/%d units done (%s seed %d)",
                done,
                len(units),
                unit["dataset"],
                unit["seed"],
            )
    return rows


def _flat_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per run, with the downstream numbers inlined.

    ``auc_delta`` is taken against the baseline of the *same* seed: the split
    moves with the seed, so a cross-seed comparison would fold split noise into
    the selection effect.
    """
    flat = []
    for record in rows:
        evaluation = record.get("evaluation") or {}
        flat.append(
            {
                "dataset": record["dataset"],
                "pipeline": record["pipeline"],
                "seed": record["seed"],
                "status": record["status"],
                "features_in": record["n_features_in"],
                "features_out": record["n_features_out"],
                "kept_%": (
                    round(100 * record["n_features_out"] / record["n_features_in"], 1)
                    if record["n_features_in"]
                    else None
                ),
                "select_s": round(record["seconds"], 1),
                "peak_rss_gb": record["peak_rss_gb"],
                "n_jobs": record["n_jobs"],
                "auc_test": evaluation.get("auc_test"),
                "auc_valid": evaluation.get("auc_valid"),
                "fit_s": evaluation.get("fit_seconds"),
                **record.get("ground_truth", {}),
                "error": record.get("error", "")[:80],
            },
        )
    frame = pd.DataFrame(flat)
    if "auc_test" in frame and not frame.empty:
        for _key, part in frame.groupby(["dataset", "seed"]):
            baseline = part.loc[part["pipeline"] == "baseline", "auc_test"]
            if not baseline.empty and pd.notna(baseline.iloc[0]):
                frame.loc[part.index, "auc_delta"] = (
                    part["auc_test"] - float(baseline.iloc[0])
                ).round(5)
    return frame.sort_values(["dataset", "pipeline", "seed"], ignore_index=True)


__all__ = [
    "Evaluation",
    "RunRecord",
    "load_pipeline_config",
    "main",
    "parse_pipeline_spec",
    "run_dataset",
    "run_one",
    "summarize",
]

if __name__ == "__main__":
    main()
