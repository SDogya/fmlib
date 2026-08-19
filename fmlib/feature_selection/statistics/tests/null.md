## Null-rate tests

Spark cases use the session `spark` fixture (`createDataFrame` / `agg` / `collect`). There is no FakeSparkDataFrame and no `skipif` for missing pyspark: the fixture fails the run if Spark cannot start.

### Pandas

- Empty candidates do not read `datasets`.
- Unsupported train types raise `ExecutionError`.
- Threshold is strict (`>`): rate equal to the threshold is kept.
- Mixed nulls (`None`, `NaN`, `pd.NA`) and empty frames.
- Verbose emit: `backend="pandas"`, `n_rows=len(frame)`.

### Spark (real session)

- Rates on a Spark frame match pandas `isna().mean()` on the same rows.
- `DoubleType` / `FloatType`: both SQL null and NaN count as missing.
- Integer / string / boolean / timestamp: only SQL null (via `count`).
- Dotted and spaced column names are quoted.
- Empty schema (0 rows) returns rate `0.0` and drops nothing.
- Missing schema columns raise `ExecutionError`.
- Spark input must not call the pandas backend.
- Verbose emit: `backend="spark"`, `n_rows=None`.
- Aggregation failure wraps the root cause (`java_exception` or `__cause__`) via a monkeypatch of real `DataFrame.collect`.

### Helpers

- `_quoted_col` produces a real `pyspark.sql.Column` with backticks.
- `_root_cause` is a pure exception helper (no Spark double).
- `_is_spark_dataframe` is true for a real Spark DataFrame and false for pandas / third-party objects.
