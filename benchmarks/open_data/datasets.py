"""Open binary-classification datasets for the feature-selection benchmark.

Every dataset is fetched once from OpenML, normalised, split, and cached as
parquet so later runs cost nothing. The ladder spans the two axes that matter
for feature selection -- how many rows and how many columns -- plus one
synthetic set whose informative features are known, so selection quality can be
measured against ground truth rather than only against downstream AUC.

Run ``python -m benchmarks.open_data.datasets --all`` to materialise the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MONTH_COLUMN = "month_part"
SPLIT_COLUMN = "split_type"
TARGET_COLUMN = "target"
N_MONTHS = 12
DEFAULT_SEED = 42


def data_root() -> Path:
    """Cache directory for prepared datasets (``FS_BENCH_DATA`` overrides)."""
    return Path(
        os.environ.get("FS_BENCH_DATA", "~/rusakov/data/fs_benchmark"),
    ).expanduser()


@dataclass(frozen=True)
class DatasetSpec:
    """One benchmark dataset.

    Args:
        name: Short identifier used on the command line and in results.
        source: ``openml:<id>`` or ``synthetic``.
        note: What this dataset exercises that the others do not.
        max_rows: Cap applied with a stratified sample before splitting, so a
            run stays inside the 50 GB / 8-core budget the work cluster has.
        positive_label: Raw target value mapped to 1. ``None`` picks the rarer
            class, which matches the response-modelling convention.
        drift_features: Features whose distribution is shifted in the later
            months (synthetic sets only) -- ground truth for PSI.
        informative: Number of genuinely informative features (synthetic only).
    """

    name: str
    source: str
    note: str
    max_rows: Optional[int] = None
    positive_label: Optional[Any] = None
    drift_features: int = 0
    informative: int = 0
    n_rows: int = 0
    n_features: int = 0


REGISTRY: dict[str, DatasetSpec] = {
    "madelon": DatasetSpec(
        name="madelon",
        source="openml:1485",
        note="NIPS 2003 FS challenge: 500 columns, only ~20 carry signal.",
    ),
    "bioresponse": DatasetSpec(
        name="bioresponse",
        source="openml:4134",
        note="Widest real set here: 1776 numeric molecular descriptors.",
    ),
    "kdd09_appetency": DatasetSpec(
        name="kdd09_appetency",
        source="openml:1111",
        note="Closest to the bank profile: mixed types, 70% missing, 1.8% positives.",
    ),
    "santander": DatasetSpec(
        name="santander",
        source="openml:42395",
        note="200k rows x 200 anonymised numeric features, mildly imbalanced.",
    ),
    "porto_seguro": DatasetSpec(
        name="porto_seguro",
        source="openml:42742",
        note="Largest by rows (595k); insurance claims with real categorical mix.",
    ),
    "synth_wide": DatasetSpec(
        name="synth_wide",
        source="synthetic",
        note="Known answer: 40 informative of 600, plus 30 features that drift by month.",
        informative=40,
        drift_features=30,
    ),
}


def prepared_dir(name: str) -> Path:
    """Directory holding the prepared parquet splits for ``name``."""
    return data_root() / name


def is_prepared(name: str) -> bool:
    """Whether every artifact for ``name`` is already on disk."""
    directory = prepared_dir(name)
    return (directory / "meta.json").exists() and all(
        (directory / f"{split}.parquet").exists()
        for split in ("train", "valid", "test")
    )


def load(name: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load prepared splits and metadata, preparing them first if needed.

    Args:
        name: Registry key.

    Returns:
        Tuple of ``{"train"/"valid"/"test": frame}`` and the metadata mapping.
    """
    if not is_prepared(name):
        prepare(name)
    directory = prepared_dir(name)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    frames = {
        split: pd.read_parquet(directory / f"{split}.parquet")
        for split in ("train", "valid", "test")
    }
    return frames, meta


def prepare(name: str, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Fetch, normalise, split and cache one dataset.

    Args:
        name: Registry key.
        seed: Seed for sampling, the month assignment and the split.

    Returns:
        The metadata mapping written next to the parquet files.
    """
    spec = REGISTRY[name]
    directory = prepared_dir(name)
    directory.mkdir(parents=True, exist_ok=True)

    if spec.source == "synthetic":
        frame, categorical, continuous = _build_synthetic(spec, seed=seed)
    else:
        frame, categorical, continuous = _fetch_openml(spec, seed=seed)

    # A builder that injects drift has already tied its features to a period;
    # re-rolling the months here would break exactly the link under test.
    if MONTH_COLUMN not in frame.columns:
        frame = _assign_months(frame, seed=seed)
    splits = _split(frame, seed=seed)

    for split, part in splits.items():
        part.to_parquet(directory / f"{split}.parquet", index=False)

    meta = {
        "name": spec.name,
        "source": spec.source,
        "note": spec.note,
        "categorical": categorical,
        "continuous": continuous,
        "target": TARGET_COLUMN,
        "time": MONTH_COLUMN,
        "task_type": "binary_classification",
        "n_rows": {split: int(len(part)) for split, part in splits.items()},
        "n_features": len(categorical) + len(continuous),
        "positive_rate": {
            split: round(float(part[TARGET_COLUMN].mean()), 6)
            for split, part in splits.items()
        },
        "informative_features": (
            [f"inf_{index}" for index in range(spec.informative)]
            if spec.informative
            else []
        ),
        "drift_features": (
            [f"drift_{index}" for index in range(spec.drift_features)]
            if spec.drift_features
            else []
        ),
        "seed": seed,
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "prepared %s: %d train / %d valid / %d test rows, %d features",
        name,
        meta["n_rows"]["train"],
        meta["n_rows"]["valid"],
        meta["n_rows"]["test"],
        meta["n_features"],
    )
    return meta


def _fetch_openml(
    spec: DatasetSpec,
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Download one OpenML dataset and normalise it to the benchmark shape."""
    from sklearn.datasets import fetch_openml

    data_id = int(spec.source.split(":", 1)[1])
    cache = data_root() / "_openml"
    cache.mkdir(parents=True, exist_ok=True)
    logger.info("fetching OpenML %d (%s)", data_id, spec.name)
    bunch = fetch_openml(
        data_id=data_id,
        as_frame=True,
        parser="auto",
        data_home=str(cache),
    )
    frame = bunch.frame.copy()
    target_name = bunch.target.name if bunch.target is not None else None
    if target_name is None or target_name not in frame.columns:
        msg = f"{spec.name}: OpenML did not name a target column."
        raise ValueError(msg)

    target = _binarize(frame.pop(target_name), spec.positive_label)
    frame = _drop_degenerate(frame)
    frame[TARGET_COLUMN] = target
    frame = frame.loc[frame[TARGET_COLUMN].notna()].reset_index(drop=True)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")

    if spec.max_rows is not None and len(frame) > spec.max_rows:
        frame = _stratified_sample(frame, spec.max_rows, seed=seed)

    categorical, continuous = _split_roles(frame)
    return frame, categorical, continuous


def _binarize(series: pd.Series, positive_label: Optional[Any]) -> pd.Series:
    """Map a two-class target onto 0/1, defaulting to "rarer class is positive"."""
    values = series.dropna().unique()
    if len(values) != 2:
        msg = f"expected a binary target, found {len(values)} distinct values"
        raise ValueError(msg)
    if positive_label is None:
        counts = series.value_counts()
        positive_label = counts.index[-1]
    return (series == positive_label).astype("float64").where(series.notna())


def _drop_degenerate(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove all-null and single-valued columns before they reach the pipeline.

    These are exactly what `null_rate` and `constants` exist to catch, so
    leaving them in would measure the filters against columns no real feature
    store would ship. Anything subtler stays.
    """
    keep = [
        name
        for name in frame.columns
        if frame[name].notna().any() and frame[name].nunique(dropna=True) > 1
    ]
    return frame.loc[:, keep]


def _split_roles(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Classify columns into categorical and continuous candidates."""
    categorical: list[str] = []
    continuous: list[str] = []
    for name in frame.columns:
        if name in {TARGET_COLUMN, MONTH_COLUMN, SPLIT_COLUMN}:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            continuous.append(name)
        else:
            categorical.append(name)
    return categorical, continuous


def _stratified_sample(frame: pd.DataFrame, max_rows: int, *, seed: int) -> pd.DataFrame:
    """Bound rows while preserving the target ratio."""
    fraction = max_rows / len(frame)
    sampled = (
        frame.groupby(TARGET_COLUMN, group_keys=False, observed=True)
        .apply(lambda part: part.sample(frac=fraction, random_state=seed))
        .reset_index(drop=True)
    )
    return sampled


def _assign_months(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Attach a synthetic period column.

    None of the open datasets ship a usable time axis, so months are assigned
    at random. That is deliberate and has to be read the right way: on these
    datasets PSI and the out-of-time split measure *cost*, not benefit, because
    there is no real drift to find. `synth_wide` is the exception -- it injects
    drift into named features, and that is where PSI is scored.
    """
    rng = np.random.default_rng(seed)
    frame = frame.copy()
    frame[MONTH_COLUMN] = rng.integers(1, N_MONTHS + 1, len(frame)).astype("int16")
    return frame


def _split(frame: pd.DataFrame, *, seed: int) -> dict[str, pd.DataFrame]:
    """Stratified 60/20/20 train/valid/test split."""
    from sklearn.model_selection import train_test_split

    train, holdout = train_test_split(
        frame,
        test_size=0.4,
        stratify=frame[TARGET_COLUMN],
        random_state=seed,
    )
    valid, test = train_test_split(
        holdout,
        test_size=0.5,
        stratify=holdout[TARGET_COLUMN],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _build_synthetic(
    spec: DatasetSpec,
    *,
    seed: int,
    n_rows: int = 120_000,
    n_noise: int = 530,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a set whose informative and drifting features are both known.

    Structure:

    * ``inf_*``   -- informative, from ``make_classification``;
    * ``drift_*`` -- pure noise, but shifted in the later months, so PSI should
      drop them and no relevance filter should;
    * ``noise_*`` -- pure noise, including a few high-null and quasi-constant
      columns so the cheap statistics filters have something real to remove;
    * ``cat_*``   -- categorical, one of them informative.
    """
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(seed)
    features, target = make_classification(
        n_samples=n_rows,
        n_features=spec.informative,
        n_informative=spec.informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[0.93, 0.07],
        flip_y=0.01,
        class_sep=0.8,
        random_state=seed,
    )
    frame = pd.DataFrame(
        features,
        columns=[f"inf_{index}" for index in range(spec.informative)],
    )
    for index in range(spec.drift_features):
        frame[f"drift_{index}"] = rng.normal(size=n_rows)
    for index in range(n_noise):
        frame[f"noise_{index}"] = rng.normal(size=n_rows)

    # A handful of columns that the cheap filters are supposed to catch.
    frame["noise_0"] = np.where(rng.random(n_rows) < 0.97, np.nan, frame["noise_0"])
    frame["noise_1"] = np.where(rng.random(n_rows) < 0.999, 1.0, 0.0)
    frame["noise_2"] = frame["noise_3"] * 1.0001 + rng.normal(scale=1e-4, size=n_rows)

    frame["cat_informative"] = pd.Series(
        np.where(target == 1, rng.choice(["a", "b"], n_rows, p=[0.8, 0.2]),
                 rng.choice(["a", "b"], n_rows, p=[0.3, 0.7])),
    )
    frame["cat_noise"] = rng.choice(list("xyz"), n_rows)
    frame[TARGET_COLUMN] = target.astype("int8")

    frame[MONTH_COLUMN] = rng.integers(1, N_MONTHS + 1, n_rows).astype("int16")
    # Shift the drift columns in the last third of the period.
    late = frame[MONTH_COLUMN] >= 9
    for index in range(spec.drift_features):
        column = f"drift_{index}"
        frame.loc[late, column] = frame.loc[late, column] + 3.0

    categorical, continuous = _split_roles(frame)
    return frame, categorical, continuous


def main() -> None:
    """Prepare one or more datasets from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="dataset names; empty with --all")
    parser.add_argument("--all", action="store_true", help="prepare every dataset")
    parser.add_argument("--force", action="store_true", help="re-prepare if cached")
    args = parser.parse_args()

    names = list(REGISTRY) if args.all else args.names
    if not names:
        parser.error("pass dataset names or --all")
    for name in names:
        if not args.force and is_prepared(name):
            logger.info("%s already prepared at %s", name, prepared_dir(name))
            continue
        prepare(name)


if __name__ == "__main__":
    main()
