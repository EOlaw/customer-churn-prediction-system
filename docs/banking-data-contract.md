# Banking CSV contract — `banking-csv-contract/1.0.0`

Defined in [churn/ml/contract.py](../churn/ml/contract.py) and enforced by [churn/ml/data.py](../churn/ml/data.py). The version string is stored on every imported dataset, quality report, training report, and model registration. Change any rule below only together with a version bump.

## Columns

| Column | Type | Rule |
| --- | --- | --- |
| `CustomerId` | identifier | non-empty, ≤ 120 chars, unique after stripping surrounding whitespace; never a feature |
| `CreditScore` | integer | 300 – 850 |
| `Geography` | category | exactly `France`, `Germany`, `Spain` (case-sensitive) |
| `Gender` | category | exactly `Female`, `Male` |
| `Age` | integer | 18 – 100 |
| `Tenure` | integer | 0 – 10 |
| `Balance` | float | 0 – 10,000,000 |
| `NumOfProducts` | integer | 1 – 4 |
| `HasCrCard`, `IsActiveMember` | integer | 0 or 1 |
| `EstimatedSalary` | float | 0 – 10,000,000 |
| `Exited` | integer | 0 or 1; each class needs ≥ 10 rows; file needs ≥ 50 rows |

"Integer" accepts integer-valued text such as `41.0` but rejects `35.5`. Bounds are inclusive. Empty cells, whitespace-only cells, and standard null tokens (`NA`, `NaN`, `null`) are *missing*.

## File-level rules

| Rule | Outcome |
| --- | --- |
| File larger than 50 MiB, not UTF-8 (a UTF-8 BOM is accepted), empty, header-only | error |
| Repeated header names; any row whose field count differs from the header | error (`malformed_csv` lists row numbers; pandas would otherwise silently shift columns) |
| Missing required column | error |
| `RowNumber`, `Surname` | `info`: ignored, never stored or used. `Surname` is personal data |
| `Complain`, `Satisfaction Score`, `Card Type`, `Point Earned` | **error** (`leakage_suspect_column`): see the [leakage review](banking-leakage-review.md) |
| Any other unexpected column | warning: excluded from storage and features |
| Missing value in a required column | error. Nothing is imputed at import |
| Duplicate `CustomerId` | error |
| Rows repeating another row's features and target under a different ID | warning (`duplicate_feature_rows`) |

## Quality report

`inspect_csv` collects *every* problem rather than stopping at the first, and never drops, repairs, or imputes a row. The report (JSON) holds: `status` (`passed`, `passed_with_warnings`, `failed`), `source_kind` and a human label, the file SHA-256 and size, row count, per-column missing counts and observed min/max, target counts and prevalence, duplicate counts, and an `issues` list. Each issue has `code`, `severity`, `column`, `count`, `message`, and up to five `example_rows` (1-based data-row numbers, header excluded). Records are located by row number; identifier values are not echoed. The report contains no timestamps, so identical input always yields an identical report and `quality_report_sha256`.

```bash
python -m churn.ml.data path/to/file.csv --source-kind external --output reports/quality.json
# exit code 0 = passed / passed_with_warnings, 1 = failed
```

## Not covered by this contract

The contract checks representation, not truth. It cannot detect plausible but wrong values, label noise, selection bias, or whether a field was observable before the exit. See the leakage review.
