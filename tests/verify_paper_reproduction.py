#!/usr/bin/env python3
"""Verify public paper-reproduction outputs against committed reference results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_FILES = (
    "expected_predictions.csv",
    "expected_fold_metrics.csv",
    "expected_selected_pipelines.csv",
    "expected_selected_features.csv",
)

RESULT_FILES = {
    "expected_predictions.csv": "predictions.csv",
    "expected_fold_metrics.csv": "fold_metrics.csv",
    "expected_selected_pipelines.csv": "selected_pipelines.csv",
    "expected_selected_features.csv": "selected_features.csv",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Compare outputs produced by run_grouped_nested_validation.py "
            "with the committed paper-reproduction reference results."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results",
        help="Directory containing outputs from the public nested-validation script.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=repo_root / "tests" / "reference_results",
        help="Directory containing committed expected paper-reproduction outputs.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for numeric comparison (default: 1e-10).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance for numeric comparison (default: 1e-10).",
    )
    parser.add_argument(
        "--strict-exact",
        action="store_true",
        help="Require exact numeric equality instead of tolerance-based equality.",
    )
    return parser.parse_args()


def compare_numeric(
    expected: pd.Series,
    actual: pd.Series,
    *,
    strict_exact: bool,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    exp = expected.to_numpy(dtype=float)
    act = actual.to_numpy(dtype=float)

    if exp.shape != act.shape:
        return False, float("inf")

    both_nan = np.isnan(exp) & np.isnan(act)
    one_nan = np.isnan(exp) ^ np.isnan(act)
    if np.any(one_nan):
        return False, float("inf")

    valid = ~both_nan
    if not np.any(valid):
        return True, 0.0

    diff = np.abs(exp[valid] - act[valid])
    max_diff = float(np.max(diff))

    if strict_exact:
        ok = np.array_equal(exp[valid], act[valid])
    else:
        ok = bool(
            np.allclose(
                exp[valid],
                act[valid],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )

    return ok, max_diff


def compare_text(expected: pd.Series, actual: pd.Series) -> bool:
    exp = expected.fillna("<NA>").astype(str).to_numpy()
    act = actual.fillna("<NA>").astype(str).to_numpy()
    return bool(np.array_equal(exp, act))


def compare_csv(
    expected_path: Path,
    actual_path: Path,
    *,
    strict_exact: bool,
    atol: float,
    rtol: float,
) -> bool:
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)

    print(f"\n{actual_path.name}")

    if list(expected.columns) != list(actual.columns):
        print("  FAIL: column schema/order differs")
        print(f"  expected: {list(expected.columns)}")
        print(f"  actual:   {list(actual.columns)}")
        return False

    if len(expected) != len(actual):
        print(
            f"  FAIL: row count differs "
            f"(expected {len(expected)}, actual {len(actual)})"
        )
        return False

    print(f"  rows: {len(actual)}")
    print(f"  columns: {len(actual.columns)}")

    file_ok = True

    for column in expected.columns:
        exp_col = expected[column]
        act_col = actual[column]

        exp_numeric = pd.api.types.is_numeric_dtype(exp_col)
        act_numeric = pd.api.types.is_numeric_dtype(act_col)

        if exp_numeric and act_numeric:
            ok, max_diff = compare_numeric(
                exp_col,
                act_col,
                strict_exact=strict_exact,
                atol=atol,
                rtol=rtol,
            )
            if not ok:
                print(
                    f"  FAIL: {column} "
                    f"(numeric max |difference| = {max_diff:.17g})"
                )
                file_ok = False
        else:
            if not compare_text(exp_col, act_col):
                print(f"  FAIL: {column} (text/categorical mismatch)")
                file_ok = False

    if file_ok:
        print("  PASS")

    return file_ok


def main() -> int:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    reference_dir = args.reference_dir.resolve()

    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        return 2

    if not reference_dir.is_dir():
        print(f"ERROR: reference directory not found: {reference_dir}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("PAPER REPRODUCTION VERIFICATION")
    print("=" * 72)
    print(f"Results:   {results_dir}")
    print(f"Reference: {reference_dir}")

    if args.strict_exact:
        print("Numeric comparison: exact")
    else:
        print(
            "Numeric comparison: "
            f"rtol={args.rtol:g}, atol={args.atol:g}"
        )

    overall_ok = True

    for expected_name in REFERENCE_FILES:
        actual_name = RESULT_FILES[expected_name]

        expected_path = reference_dir / expected_name
        actual_path = results_dir / actual_name

        if not expected_path.is_file():
            print(f"\nFAIL: missing reference file: {expected_path}")
            overall_ok = False
            continue

        if not actual_path.is_file():
            print(f"\nFAIL: missing result file: {actual_path}")
            overall_ok = False
            continue

        ok = compare_csv(
            expected_path,
            actual_path,
            strict_exact=args.strict_exact,
            atol=args.atol,
            rtol=args.rtol,
        )
        overall_ok = overall_ok and ok

    print("\n" + "=" * 72)
    if overall_ok:
        print("PAPER REPRODUCTION VERIFICATION: PASS")
        print("=" * 72)
        return 0

    print("PAPER REPRODUCTION VERIFICATION: FAIL")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
