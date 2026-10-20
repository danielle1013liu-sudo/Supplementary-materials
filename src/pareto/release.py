"""Save a compact, checksummed snapshot of a verified current Pareto replay."""

import argparse
import gzip
from pathlib import Path
import shutil
from .common import DATA, ROOT, SEEDS, dump, read, sha


def validate_snapshot():
    release = read(DATA / "CURRENT_RELEASE.json")
    if release["accepted_seeds"] != SEEDS:
        raise ValueError("Pareto snapshot does not use the accepted model panel")
    for name, expected in release["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Accepted source differs from Pareto snapshot: {name}")
    for name, expected in release["snapshot_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed Pareto snapshot: {name}")
    return ROOT / release["snapshot_directory"]


def run(source):
    source = Path(source).resolve()
    report = read(source / "checks/NUMERICAL_REPRODUCTION.json")
    provenance = read(source / "checks/MODEL_AND_DATA_VERIFICATION.json")
    if not report["current_model_predictions_replayed"] or report["accepted_seeds"] != SEEDS:
        raise ValueError("A complete current-release numerical verification is required")
    for name, expected in provenance["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Accepted source changed after verification: {name}")
    destination = DATA / "current"
    checksums = {}
    for folder in ("inputs", "search", "results", "checks"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
        for path in sorted((source / folder).iterdir()):
            if not path.is_file():
                continue
            target = destination / folder / path.name
            if path.suffix == ".csv" and path.stat().st_size > 1024 * 1024:
                target = target.with_suffix(target.suffix + ".gz")
                with path.open("rb") as stream, target.open("wb") as saved:
                    with gzip.GzipFile(fileobj=saved, mode="wb", filename="", mtime=0) as compressed:
                        shutil.copyfileobj(stream, compressed)
            else:
                shutil.copyfile(path, target)
            checksums[target.relative_to(ROOT).as_posix()] = sha(target)
    release = {
        "release": "IL10 crosslink-seven revision; accepted Validation-selected RF median panel",
        "accepted_seeds": SEEDS,
        "snapshot_directory": destination.relative_to(ROOT).as_posix(),
        "reference_directory": (destination / "results").relative_to(ROOT).as_posix(),
        "historical_reference_directory": "data/pareto/reference",
        "historical_reference_scope": "Previous IL10 release; its fixed model lock and results are archival, not the current Pareto result",
        "source_sha256": provenance["source_sha256"],
        "snapshot_sha256": checksums,
        "replay_command": "python -m src.pareto --output outputs/pareto --reference data/pareto/current/results",
        "figures_command": "python -m src.pareto.plotting --input outputs/pareto/results --output results/pareto",
        "counts": read(source / "results/PARETO_SUMMARY.json"),
        "numeric_verification": report,
        "prediction_batch_cache_included": False,
    }
    dump(DATA / "CURRENT_RELEASE.json", release)
    validate_snapshot()
    print(f"Current Pareto snapshot: {len(checksums)} files -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/pareto")
    run(parser.parse_args().input)
