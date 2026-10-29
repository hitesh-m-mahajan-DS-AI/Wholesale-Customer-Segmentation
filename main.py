"""Describe public wholesale customers using one fixed three-cluster model."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip"
SOURCE_SHA256 = "c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef"
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
SEED = 42
N_CLUSTERS = 3


def validate_spending(data):
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing spending columns: {sorted(missing)}")
    values = data[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Spending must be finite, complete and nonnegative.")
    return data[FEATURES]


def load_data(path=None):
    path = Path(path) if path is not None else ROOT / "data" / "wholesale_customers.csv"
    if not path.exists():
        with urlopen(SOURCE_URL, timeout=30) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                raw = archive.read("Wholesale customers data.csv")
        if sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded data differs from the audited source.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source checksum mismatch; restore the bundled UCI CSV.")
    data = pd.read_csv(path)
    if data.shape != (440, 8) or data.isna().any().any():
        raise ValueError("Expected 440 complete source rows and eight columns.")
    validate_spending(data)
    if not data["Channel"].isin([1, 2]).all() or not data["Region"].isin([1, 2, 3]).all():
        raise ValueError("Unknown channel or region codes.")
    audit = {"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
             "rows": len(data), "missing_values": int(data.isna().sum().sum()),
             "exact_duplicate_rows": int(data.duplicated().sum()),
             "feature_minima": data[FEATURES].min().astype(int).to_dict(),
             "feature_maxima": data[FEATURES].max().astype(int).to_dict(),
             "feature_medians": data[FEATURES].median().to_dict(),
             "channel_counts": data["Channel"].value_counts().sort_index().to_dict(),
             "region_counts": data["Region"].value_counts().sort_index().to_dict()}
    data.insert(0, "customer_id", [f"ROW-{i:03d}" for i in range(1, len(data) + 1)])
    # A display-only summary, NOT an extra clustering input.
    data["total_category_spending"] = data[FEATURES].sum(axis=1)
    return data, audit


def fit_segments(data):
    values = validate_spending(data)
    model = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20, algorithm="lloyd")),
    ])
    model.fit(values)
    transformed = model[:-1].transform(values)
    raw_labels = model.named_steps["kmeans"].labels_
    customers = data.copy()
    # Labels are arbitrary identifiers, not an ordering from worse to better.
    customers["cluster"] = raw_labels + 1
    customers["silhouette"] = silhouette_samples(transformed, raw_labels, metric="euclidean")
    score = float(silhouette_score(transformed, raw_labels, metric="euclidean"))
    return model, customers, transformed, score


def profile_segments(customers):
    groups = customers.groupby("cluster")
    medians = groups[FEATURES].median()
    means = groups[FEATURES].mean()
    overall = customers[FEATURES].median()
    # Actual source medians are positive; zero reference values remain undefined.
    median_ratios = medians.divide(overall.replace(0, np.nan), axis="columns")
    summary = groups.agg(customers=("customer_id", "size"),
                         mean_total_spending=("total_category_spending", "mean"),
                         median_total_spending=("total_category_spending", "median"),
                         mean_silhouette=("silhouette", "mean"))
    summary["share_pct"] = 100 * summary["customers"] / len(customers)
    return summary, medians, means, median_ratios


def save_charts(customers, summary, ratios, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#007F78", 2: "#D97732", 3: "#6B71A5"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for cluster, group in customers.groupby("cluster"):
        axes[0].scatter(np.log1p(group["Fresh"]), np.log1p(group["Grocery"]), color=colors[cluster],
                        alpha=0.65, s=22, label=f"Cluster {cluster} (n={len(group)})")
    axes[0].set(title="Two-category view of six-feature clusters", xlabel="ln(1 + Fresh spending)",
                ylabel="ln(1 + Grocery spending)")
    axes[0].legend(fontsize=8)
    bars = axes[1].bar([f"Cluster {i}" for i in summary.index], summary["customers"],
                       color=[colors[i] for i in summary.index])
    axes[1].bar_label(bars)
    axes[1].set(title="Every customer assigned once", ylabel="Customers")
    axes[1].set_ylim(0, summary["customers"].max() * 1.15)
    fig.savefig(output / "segmentation_overview.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 4), layout="constrained")
    heat = np.log2(ratios.to_numpy())
    limit = max(1.0, float(np.nanmax(np.abs(heat))))
    image = axis.imshow(heat, cmap="BrBG", vmin=-limit, vmax=limit, aspect="auto")
    labels = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents / Paper", "Delicatessen"]
    axis.set_xticks(range(6), labels)
    axis.set_yticks(range(len(ratios)), [f"Cluster {i}" for i in ratios.index])
    axis.set_title("Median annual spending / whole-sample category median")
    for row in range(len(ratios)):
        for col in range(6):
            axis.text(col, row, f"{ratios.iloc[row, col]:.2f}x", ha="center", va="center", fontsize=10,
                      bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    fig.colorbar(image, ax=axis, label="log2 ratio; 0 = whole-sample median")
    fig.savefig(output / "cluster_profiles.png", dpi=160)
    plt.close(fig)


def main():
    start = time.perf_counter()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    data, audit = load_data()
    model, customers, transformed, score = fit_segments(data)
    summary, medians, means, ratios = profile_segments(customers)
    customers.to_csv(output / "customer_segments.csv", index=False)
    summary.to_csv(output / "cluster_summary.csv")
    medians.to_csv(output / "cluster_medians.csv")
    means.to_csv(output / "cluster_means.csv")
    ratios.to_csv(output / "median_ratios.csv")
    for column in ["Channel", "Region"]:
        counts = pd.crosstab(customers["cluster"], customers[column])
        counts.to_csv(output / f"{column.lower()}_counts.csv")
        (100 * counts.divide(counts.sum(axis=1), axis=0)).to_csv(output / f"{column.lower()}_percentages.csv")
    pd.DataFrame(transformed, columns=FEATURES).assign(customer_id=data["customer_id"]).to_csv(
        output / "transformed_features.csv", index=False)
    pd.DataFrame(model.named_steps["kmeans"].cluster_centers_, columns=FEATURES,
                 index=pd.Index([1, 2, 3], name="cluster")).to_csv(output / "standardized_log_centers.csv")
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    save_charts(customers, summary, ratios, output)
    metrics = {"customers": len(customers), "features": FEATURES, "n_clusters": N_CLUSTERS,
               "random_state": SEED, "n_init": 20, "silhouette_score": score,
               "inertia_standardized_log_space": float(model.named_steps["kmeans"].inertia_),
               "iterations": int(model.named_steps["kmeans"].n_iter_),
               "cluster_sizes": {str(key): int(value) for key, value in summary["customers"].items()},
               "negative_silhouette_customers": int((customers["silhouette"] < 0).sum()),
               "evaluation": "Internal full-cohort geometry in six standardized log-spending features, not predictive accuracy or held-out validation.",
               "note": "Three clusters fixed in advance. Levels and patterns, not pure proportional mix; labels are arbitrary.",
               "run_seconds": round(time.perf_counter() - start, 3)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(summary.round(3).to_string())
    print("\nMedian category spending (original monetary units):\n", medians.to_string())
    print(f"\nSilhouette (internal, full cohort): {score:.6f}")
    print("Negative silhouette customers:", metrics["negative_silhouette_customers"])


if __name__ == "__main__":
    main()
