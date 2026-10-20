"""Replay the numerical Pareto analysis without plotting or model fitting."""

import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from . import generation, prediction, selection, explanation, verification
from .common import ROOT, DATA, read, sha
from ..data import check_environment, check_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "generate", "predict", "select", "explain", "verify"],
        default="all",
    )
    parser.add_argument("--reference", type=Path, help="Optional matching numerical reference directory")
    args = parser.parse_args()
    check_environment()
    check_inputs()
    output = args.output.resolve()
    if output == ROOT or any(
        output.is_relative_to(ROOT / p) for p in ["data", "models", "src"]
    ):
        parser.error(
            "Choose a separate results directory, outside data, models and source code."
        )
    for name in ["search", "results", "checks", "inputs"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    stages = {
        "generate": generation.run,
        "predict": prediction.run,
        "select": selection.run,
        "explain": explanation.run,
        "verify": verification.run,
    }
    with threadpool_limits(limits=1):
        for name, function in stages.items():
            if args.stage in ("all", name):
                print(f"{name}: {output}", flush=True)
                if name == "verify":
                    function(output, reference=args.reference)
                else:
                    function(output)


if __name__ == "__main__":
    main()
