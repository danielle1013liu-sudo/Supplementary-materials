"""Draw the all-ten-descriptor Pareto front and representative SHAP heatmaps.

Run after numerical replay with ``--input outputs/all10_replay/results``, or
omit --input to draw the checked reference tables shipped with the repository.
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .common import DATA, FEATURES, ROOT, csv, dump, sha, table
from ..hydrogel_immune_ml.public_encoding import (
    PUBLIC_VERSION,
    decode_public_input,
    encode_public_input,
    fields,
)

XY = ["tnf_mean_pg_ml", "il10_mean_pg_ml"]
PINK, EDGE = "#D99CB6", "#9B778A"
STARS = ["#5F7295", "#735C84", "#AA8931"]
NAMES = {field["machine_column"]: field["feature"] for field in fields()}
NAMES.update(
    modulus_value_kPa="Modulus",
    pore_size_um="Pore size",
    culture_or_exposure_duration_h="Exposure time",
)


def source_path(source, name):
    path = source / name
    return path if path.exists() else path.with_suffix(path.suffix + ".gz")


def save(fig, output, stem):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{extension}", dpi=600, facecolor="white")
    plt.close(fig)


def public_profiles(profiles):
    """Export current publication letters without changing the model inputs."""
    coded = encode_public_input(profiles)
    restored = decode_public_input(coded, PUBLIC_VERSION)
    pd.testing.assert_frame_equal(restored, profiles, check_exact=True)
    return coded


def plot_front(eligible, retained, representatives, output):
    fig = plt.figure(figsize=(4.6, 4.6 * 736 / 940))
    ax = fig.add_axes([164 / 940, 105 / 736, 743 / 940, 604 / 736])
    ax.set(
        xlim=(0, eligible[XY[0]].max() * 1.055),
        ylim=(0, eligible[XY[1]].max() * 1.065),
    )
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.set_xlabel("Predicted TNF-α (pg/mL)", fontsize=9.7, labelpad=3.5, fontweight="bold")
    ax.set_ylabel("Predicted IL-10 (pg/mL)", fontsize=9.7, labelpad=4, fontweight="bold")
    ax.tick_params(labelsize=8.5, length=3, width=.65)
    ax.spines[["top", "right"]].set_visible(False)
    gutter = 14 / 940 * fig.get_figwidth() * 72
    ax.spines["left"].set_position(("outward", gutter))
    extension = gutter / (ax.get_position().width * fig.get_figwidth() * 72) * ax.get_xlim()[1]
    ax.spines["bottom"].set_bounds(-extension, ax.get_xlim()[1])
    frame = Rectangle((150 / 940, 105 / 736), 757 / 940, 604 / 736, transform=fig.transFigure)
    cloud = ax.scatter(
        eligible[XY[0]], eligible[XY[1]], s=3, facecolors="#EAEAEA",
        alpha=.015, edgecolors="none", zorder=1, rasterized=True,
    )
    front = ax.scatter(
        retained[XY[0]], retained[XY[1]], s=15, facecolors=PINK,
        edgecolors=EDGE, linewidths=.45, zorder=4,
    )
    collections = [cloud, front]
    for color, row in zip(STARS, representatives.itertuples(index=False)):
        collections.append(ax.scatter(
            row.tnf_mean_pg_ml, row.il10_mean_pg_ml, s=62, marker="*",
            facecolors=color, edgecolors="#504D55", linewidths=.7, zorder=7,
        ))
    for collection in collections:
        collection.set_clip_path(frame)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=3, markerfacecolor="#D9D9D9",
               markeredgecolor="none", alpha=.45, label="Candidates"),
        Line2D([], [], marker="o", linestyle="", markersize=3.1, markerfacecolor=PINK,
               markeredgecolor=EDGE, markeredgewidth=.45, label="Pareto front"),
        Line2D([], [], marker="*", linestyle="", markersize=5.6, markerfacecolor="white",
               markeredgecolor="#504D55", markeredgewidth=.7, label="Representative designs"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.9,
              handletextpad=.55, handlelength=1.05, labelspacing=.22, borderaxespad=.35)
    np.testing.assert_array_equal(front.get_offsets().data, retained[XY].to_numpy())
    limits = {"x": list(ax.get_xlim()), "y": list(ax.get_ylim())}
    save(fig, output, "Fig10_Pareto_final")
    return limits


def plot_shap(heat, representatives, output):
    order = [FEATURES[i] for i in [3, 0, 1, 4, 5, 6, 7, 8, 9, 2]]
    ids = representatives.candidate_id.tolist()
    cmap = LinearSegmentedColormap.from_list("signed_SHAP", ["#3B75AF", "#FAFAFA", "#D65A76"])
    fig = plt.figure(figsize=(9.2, 3.9))
    checks = []
    for j, target in enumerate(("tnf", "il10")):
        part = heat.loc[heat.target.eq(target)]
        matrix = part.pivot(index="descriptor_name", columns="candidate_id", values="grouped_SHAP_pg_ml")
        matrix = matrix.loc[order, ids].to_numpy()
        for candidate in ids:
            group = part.loc[part.candidate_id.eq(candidate)]
            if len(group) != 10 or set(group.descriptor_name) != set(FEATURES):
                raise ValueError(f"Incomplete grouped SHAP descriptors: {target}, {candidate}")
            np.testing.assert_allclose(
                group.expected_value_pg_ml.iloc[0] + group.grouped_SHAP_pg_ml.sum(),
                group.prediction_pg_ml.iloc[0], atol=1e-7, rtol=0,
            )
        ax = fig.add_axes([.18 + j * .49, .14, .235, .74])
        cax = fig.add_axes([.435 + j * .49, .21, .012, .61])
        bound = max(50, np.ceil(np.abs(matrix).max() / 50) * 50)
        mesh = ax.pcolormesh(np.arange(4), np.arange(11), matrix, cmap=cmap,
                             norm=TwoSlopeNorm(0, vmin=-bound, vmax=bound), shading="flat")
        ax.set(ylim=(10, 0), xticks=np.arange(3) + .5, xticklabels=representatives.display_label,
               yticks=np.arange(10) + .5, yticklabels=[NAMES[field] for field in order])
        ax.tick_params(length=0, pad=4, labelsize=8.7)
        for row in range(10):
            for col in range(3):
                ax.text(col + .5, row + .5, str(int(round(matrix[row, col]))),
                        ha="center", va="center", fontsize=8.3,
                        color="white" if abs(matrix[row, col]) > .57 * bound else "#222222")
        ax.spines[:].set_visible(False)
        ax.set_title({"tnf": "TNF-α", "il10": "IL-10"}[target], fontsize=11, pad=9)
        colorbar = fig.colorbar(mesh, cax=cax)
        colorbar.set_label("Grouped SHAP contribution (pg/mL)", fontsize=8, fontweight="bold")
        colorbar.ax.tick_params(labelsize=7.5, width=.5)
        colorbar.outline.set_linewidth(.5)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        colorbar.solids.set_linewidth(.8)
        fig.text(.025 + j * .49, .965, "ab"[j], fontsize=14, va="top")
        checks.append({"target": target, "color_limits": [-bound, bound],
                       "column_ids": ids, "row_order": order})
    save(fig, output, "Fig11_Pareto_SHAP_final")
    return checks


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == ROOT or any(output.is_relative_to(ROOT / name) for name in ("data", "models", "src")):
        raise ValueError("Choose a separate output directory outside data, models and source code")
    output.mkdir(parents=True, exist_ok=True)
    names = ["ALL_CANDIDATES_AND_PREDICTIONS.csv", "FINAL_PARETO_PROFILES.csv",
             "FINAL_PARETO_REPRESENTATIVES.csv", "FINAL_REPRESENTATIVE_SHAP.csv"]
    paths = [source_path(source, name) for name in names]
    candidates, retained, representatives, heat = [csv(path) for path in paths]
    eligible = candidates.loc[candidates.eligible].copy()
    if representatives.display_label.tolist() != ["P1", "P2", "P3"]:
        raise ValueError("Expected the current P1, P2, P3 representative mapping")
    for profiles in (retained, representatives):
        values = eligible.set_index("candidate_id").loc[profiles.candidate_id, XY]
        np.testing.assert_array_equal(values.to_numpy(), profiles[XY].to_numpy())
    if not set(representatives.candidate_id).issubset(retained.candidate_id):
        raise ValueError("A representative is missing from the final front")
    try:
        font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
        font = "Times New Roman"
    except ValueError:
        font = "DejaVu Serif"
    with mpl.rc_context({"font.family": font, "font.size": 8.5, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white"}):
        limits = plot_front(eligible, retained, representatives, output)
        heat_checks = plot_shap(heat, representatives, output)
    layers = []
    for label, frame in [("Candidates", eligible), ("Pareto front", retained),
                         ("Representative designs", representatives)]:
        part = frame[["candidate_id"] + XY].copy()
        part["layer"] = label
        part["display_label"] = part.candidate_id.map(representatives.set_index("candidate_id").display_label).fillna("")
        layers.append(part)
    pd.concat(layers, ignore_index=True).to_csv(
        output / "Fig10_Pareto_final_source.csv.gz", index=False, float_format="%.17g",
        lineterminator="\n", compression={"method": "gzip", "mtime": 0},
    )
    table(output / "Fig11_Pareto_SHAP_final_source.csv", heat)
    table(output / "Pareto_profiles_public_codes.csv", public_profiles(retained))
    table(output / "Pareto_representatives_public_codes.csv", public_profiles(representatives))
    report = {
        "publication_codebook_version": PUBLIC_VERSION,
        "eligible_candidates": len(eligible), "final_profiles": len(retained),
        "objective_locations": len(retained[XY].drop_duplicates()),
        "representative_ids": dict(zip(representatives.display_label, representatives.candidate_id)),
        "axis_limits": limits, "heatmaps": heat_checks, "font": font,
        "source_sha256": {path.name: sha(path) for path in paths},
        "cloud_rasterized": True, "predictions_recalculated": False,
    }
    dump(output / "FIGURE_NUMERICAL_CHECKS.json", report)
    print(f"Figures 10 and 11: {len(retained)} profiles at {report['objective_locations']} locations -> {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
