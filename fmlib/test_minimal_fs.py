#!/usr/bin/env python3
"""Minimal feature selection test script.

This script demonstrates the feature selection pipeline with a simple example.
It creates a small DataFrame with one high-null feature and verifies that
the pipeline correctly drops it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths to find the fmlib package
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / ".."))

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame

from fmlib.feature_selection import (
    FeatureSchema,
    FeatureSelectionConfig,
    FeatureSelectionPipeline,
)
from fmlib.feature_selection.config import (
    NullRateConfig,
    PipelineStepConfig,
    StatisticsConfig,
)


def create_test_spark_session() -> SparkSession:
    """Create a local Spark session for testing."""
    return (
        SparkSession.builder.appName("feature-selection-test")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def create_test_dataframe(spark: SparkSession) -> DataFrame:
    """Create a small test DataFrame with known null rates.

    Features:
    - feature_ok: 10% nulls (should be kept with threshold=0.5)
    - feature_bad: 80% nulls (should be dropped with threshold=0.5)
    - feature_poor: 95% nulls (should be dropped with threshold=0.5)
    - target: target column
    """
    np.random.seed(42)
    n_rows = 100

    data = {
        "feature_ok": np.concatenate([np.random.randn(90), [np.nan] * 10]),
        "feature_bad": np.concatenate([[np.nan] * 80, np.random.randn(20)]),
        "feature_poor": np.concatenate([[np.nan] * 95, [1.0] * 5]),
        "target": np.random.randint(0, 2, n_rows),
        "id_col": range(n_rows),
    }

    pdf = pd.DataFrame(data)
    return spark.createDataFrame(pdf)


def main() -> None:
    """Run minimal feature selection test."""
    print("=" * 60)
    print("Minimal Feature Selection Test")
    print("=" * 60)

    # 1. Create Spark session
    print("\n1. Creating Spark session...")
    spark = create_test_spark_session()
    print("   Spark session created successfully")

    # 2. Create test DataFrame
    print("\n2. Creating test DataFrame...")
    df = create_test_dataframe(spark)
    print("   DataFrame created with columns:", df.columns)
    print("   Row count:", df.count())

    # Show null rates for verification
    print("\n   Null rates in original DataFrame:")
    for col in df.columns:
        null_pct = df.filter(df[col].isNull()).count() / df.count() * 100
        print(f"   - {col}: {null_pct:.1f}% nulls")

    # 3. Configure feature selection using proper dataclass objects
    print("\n3. Configuring FeatureSelectionPipeline...")
    config = FeatureSelectionConfig(
        order=(PipelineStepConfig("null_rate", {"threshold": 0.5}),),
        statistics=StatisticsConfig(
            null_rate=NullRateConfig(threshold=0.5),
        ),
    )
    print("   Config created with null_rate threshold=0.5")

    pipeline = FeatureSelectionPipeline(config)
    print("   Pipeline created")

    # 4. Define feature schema
    print("\n4. Defining FeatureSchema...")
    schema = FeatureSchema(
        categorical=["id_col"],
        continuous=["feature_ok", "feature_bad", "feature_poor"],
        target="target",
        task_type="binary_classification",
        time=None,
        fold=None,
        id_columns=["id_col"],
    )
    print("   Schema defined:")
    print(f"   - categorical: {schema.categorical}")
    print(f"   - continuous: {schema.continuous}")
    print(f"   - target: {schema.target}")

    # 5. Run feature selection
    print("\n5. Running feature selection...")
    result = pipeline.fit_select(
        spark=spark,
        datasets={"train": df},
        schema=schema,
        output_dir=None,
    )
    print("   Feature selection completed!")

    # 6. Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\nSelected features ({len(result.selected_features)}):")
    for feat in result.selected_features:
        print(f"  - {feat}")

    print(f"\nDropped features ({len(result.dropped_features)}):")
    for dropped in result.dropped_features:
        print(f"  - {dropped.feature}")
        print(f"    Stage: {dropped.stage}")
        print(f"    Method: {dropped.method}")
        print(f"    Reason: {dropped.reason}")
        print(f"    Value: {dropped.value}")
        print(f"    Threshold: {dropped.threshold}")

    # 7. Verify expected behavior
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    expected_dropped = {"feature_bad", "feature_poor"}
    actual_dropped = {d.feature for d in result.dropped_features}

    if expected_dropped == actual_dropped:
        print("\n✓ SUCCESS: Pipeline correctly dropped features with high null rates!")
        print(f"  Expected to drop: {expected_dropped}")
        print(f"  Actually dropped: {actual_dropped}")
        print(f"  Kept features: {set(result.selected_features)}")
    else:
        print("\n✗ MISMATCH in dropped features")
        print(f"  Expected: {expected_dropped}")
        print(f"  Actual: {actual_dropped}")

    # Save result to JSON
    print("\n6. Saving result to JSON...")
    output_path = Path("pd.json")
    result.save(output_path)
    print(f"   Result saved to {output_path.resolve()}")

    # Clean up
    spark.stop()
    print("\nDone!")


if __name__ == "__main__":
    main()
