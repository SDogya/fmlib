"""Collapse a multi-seed sweep into mean +- std, so differences can be read.

A single run tells you what happened on one split. It cannot tell you whether
a 0.002 AUC gap is the filter or the split, and on these datasets that is
exactly the size of most of the gaps. This averages each (dataset, pipeline)
over its seeds and reports the spread next to the mean.

``auc_delta`` is already computed per seed against that seed's own baseline, so
averaging it is meaningful: its standard deviation is the noise floor a
selection effect has to clear.

    python -m benchmarks.open_data.aggregate results/per_method.csv --markdown
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Number of seeds below which a standard deviation is not worth printing.
MIN_SEEDS_FOR_SPREAD = 2


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Average each (dataset, pipeline) over seeds and keep the spread.

    Args:
        frame: Flat per-run table as written by ``run_benchmark``.

    Returns:
        One row per (dataset, pipeline), ordered by dataset then mean AUC.
    """
    ok = frame.loc[frame["status"].isin({"ok", "eval_failed"})]
    grouped = ok.groupby(["dataset", "pipeline"], dropna=False)
    summary = grouped.agg(
        seeds=("seed", "nunique"),
        features_in=("features_in", "max"),
        features_out_mean=("features_out", "mean"),
        features_out_std=("features_out", "std"),
        kept_pct=("kept_%", "mean"),
        select_s=("select_s", "mean"),
        peak_rss_gb=("peak_rss_gb", "max"),
        auc_mean=("auc_test", "mean"),
        auc_std=("auc_test", "std"),
        delta_mean=("auc_delta", "mean"),
        delta_std=("auc_delta", "std"),
    ).reset_index()

    for column in ("informative_recall", "informative_precision", "drifting_kept"):
        if column in ok.columns and ok[column].notna().any():
            summary[column] = grouped[column].mean().to_numpy()

    summary["significant"] = _is_significant(summary)
    numeric = summary.select_dtypes("number").columns
    summary[numeric] = summary[numeric].round(5)
    return summary.sort_values(
        ["dataset", "delta_mean"],
        ascending=[True, False],
        ignore_index=True,
    )


def _is_significant(summary: pd.DataFrame) -> pd.Series:
    """Whether the mean delta clears twice its own spread across seeds.

    A crude test, and deliberately so: with three seeds anything finer would
    pretend to a precision the sample does not have. It answers one question --
    is this gap bigger than the split noise sitting underneath it.
    """
    enough = summary["seeds"] >= MIN_SEEDS_FOR_SPREAD
    spread = summary["delta_std"].fillna(0.0)
    return enough & (summary["delta_mean"].abs() > 2 * spread) & (spread > 0)


def to_markdown(summary: pd.DataFrame) -> str:
    """Render the summary as the table used in ``RESULTS.md``."""
    lines = [
        "| датасет | пайплайн | осталось | % | время, с | AUC test | Δ AUC (± std) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, part in summary.groupby("dataset", sort=False):
        first = True
        for row in part.itertuples():
            name = f"**{dataset}**" if first else ""
            first = False
            delta = (
                "—"
                if pd.isna(row.delta_mean)
                else f"{row.delta_mean:+.4f} ± {0.0 if pd.isna(row.delta_std) else row.delta_std:.4f}"
            )
            lines.append(
                f"| {name} | {row.pipeline} | {row.features_out_mean:.0f} "
                f"| {row.kept_pct:.1f} | {row.select_s:.1f} "
                f"| {row.auc_mean:.5f} | {delta} |",
            )
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="per-run CSV files to combine")
    parser.add_argument("--out", default="", help="write the summary CSV here")
    parser.add_argument("--markdown", action="store_true", help="print a table too")
    args = parser.parse_args()

    frame = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    summary = aggregate(frame)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        logger.info("wrote %s", args.out)

    print(summary.to_string(index=False))  # noqa: T201 - this is a CLI report
    if args.markdown:
        print()  # noqa: T201
        print(to_markdown(summary))  # noqa: T201


__all__ = ["aggregate", "main", "to_markdown"]

if __name__ == "__main__":
    main()
