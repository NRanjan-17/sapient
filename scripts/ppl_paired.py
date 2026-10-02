#!/usr/bin/env python3
"""
Paired comparison of two perplexity runs that scored identical tokens.

Each input file should contain one floating-point number per line: the
negative log-likelihood (in nats) of one scored token. Files are produced
by `sapient eval-ppl --dump-nll <file>`.

IMPORTANT: The two files MUST come from runs that scored the SAME tokens
in the SAME order. This means both runs must use:
- The same model and tokenizer
- The same text file / evaluation corpus
- The same context window size (--ctx)
- The same chunk configuration (--chunks)

The script computes:
- Per-file perplexity (exp of mean NLL)
- Paired differences (B - A for each token)
- Sample standard deviation and standard error of the difference
- 95% confidence interval for the ratio ppl_b / ppl_a
- Statistical significance via z-score

Example:
  python3 scripts/ppl_paired.py baseline.nll candidate.nll
  python3 scripts/ppl_paired.py baseline.nll candidate.nll --json

Exit codes:
  0 = success
  2 = invalid input (mismatched lengths, empty files, parse errors)
  1 = unexpected error
"""

import argparse
import json
import math
import sys


def read_nll_file(path):
    """Read a file of NLL values (one per line), skipping blanks."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(2)

    values = []
    for line in lines:
        line = line.strip()
        if not line:  # skip blank lines
            continue
        try:
            values.append(float(line))
        except ValueError:
            print(f"Error parsing line in {path}: '{line}' is not a valid float", file=sys.stderr)
            sys.exit(2)

    return values


def compute_sample_std(values):
    """Compute sample standard deviation (n-1 denominator)."""
    n = len(values)
    if n < 2:
        return 0.0

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def main():
    parser = argparse.ArgumentParser(
        description="Paired comparison of two perplexity runs.",
        epilog="Both files must contain NLL values from identical token sequences."
    )
    parser.add_argument("file_a", help="Baseline NLL file")
    parser.add_argument("file_b", help="Candidate NLL file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Read both files
    a_values = read_nll_file(args.file_a)
    b_values = read_nll_file(args.file_b)

    # Check validity
    if not a_values or not b_values:
        print("Error: one or both files are empty", file=sys.stderr)
        sys.exit(2)

    if len(a_values) != len(b_values):
        print(f"Error: file lengths differ ({len(a_values)} vs {len(b_values)})", file=sys.stderr)
        sys.exit(2)

    n = len(a_values)

    # Compute perplexities
    mean_a = sum(a_values) / n
    mean_b = sum(b_values) / n
    ppl_a = math.exp(mean_a)
    ppl_b = math.exp(mean_b)

    # Compute paired differences
    diffs = [b_values[i] - a_values[i] for i in range(n)]
    mean_d = sum(diffs) / n
    sd_d = compute_sample_std(diffs)
    se_d = sd_d / math.sqrt(n) if n > 0 else 0.0

    # Ratio and confidence interval
    ratio = math.exp(mean_d)
    z = mean_d / se_d if se_d > 0 else 0.0

    # 95% CI for the ratio: exp(mean_d ± 1.96*se_d)
    ci_lower = math.exp(mean_d - 1.96 * se_d)
    ci_upper = math.exp(mean_d + 1.96 * se_d)

    # Percentage change and CI
    pct = (ratio - 1) * 100
    pct_lo = (ci_lower - 1) * 100
    pct_hi = (ci_upper - 1) * 100

    # Verdict
    if ci_upper < 1.0:
        verdict = "B better"
    elif ci_lower > 1.0:
        verdict = "B worse"
    else:
        verdict = "no significant difference"

    if args.json:
        output = {
            "n": n,
            "ppl_a": ppl_a,
            "ppl_b": ppl_b,
            "ratio": ratio,
            "ratio_lo": ci_lower,
            "ratio_hi": ci_upper,
            "pct": pct,
            "pct_lo": pct_lo,
            "pct_hi": pct_hi,
            "z": z,
            "verdict": verdict
        }
        print(json.dumps(output))
    else:
        print(f"tokens: {n}")
        print(f"perplexity A: {ppl_a:.4f}   B: {ppl_b:.4f}")
        print(f"B vs A: {pct:+.3f}%  (95% interval {pct_lo:+.3f}% .. {pct_hi:+.3f}%)")
        print(f"verdict: {verdict}  (z = {z:.2f})")


if __name__ == "__main__":
    main()
