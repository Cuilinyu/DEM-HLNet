"""Paper-style scatter plot for the held-out elevation errors."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def save_scatter(prediction, reference, metrics, output, title):
    lower = min(prediction.min(), reference.min()) - 2
    upper = max(prediction.max(), reference.max()) + 3
    edges = np.arange(lower, upper, 0.65)
    counts, _, _ = np.histogram2d(prediction, reference, bins=(edges, edges))
    x_bins = np.searchsorted(edges, prediction) - 1
    y_bins = np.searchsorted(edges, reference) - 1
    frequency = gaussian_filter(counts, 1.3)[x_bins, y_bins] / len(prediction)
    order = frequency.argsort()
    limits = np.array([-20.0, 20.0])

    fig, ax = plt.subplots(figsize=(14, 10), dpi=210)
    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.17, top=0.89)
    points = ax.scatter(
        prediction[order], reference[order], c=frequency[order], cmap="RdBu_r",
        s=61, alpha=0.82, linewidths=0.3, edgecolors="face", zorder=2,
    )
    ax.plot(limits, limits, "--", color="red", linewidth=4.1, dashes=(5, 2.4),
            label="1:1 line", zorder=3)
    if np.ptp(prediction) > 0:
        slope, intercept = np.polyfit(prediction, reference, 1)
        ax.plot(limits, slope * limits + intercept, color="black",
                linewidth=4.1, label="Regression Line", zorder=4)
    ticks = np.arange(-20, 21, 10)
    ax.set(xlim=limits, ylim=limits, xticks=ticks, yticks=ticks)
    ax.set_title(title, fontsize=33, pad=20)
    ax.set_xlabel("Predicted height error (m)", fontfamily="serif", fontsize=31, labelpad=15)
    ax.set_ylabel("True height error (m)", fontfamily="serif", fontsize=31, labelpad=14)
    ax.tick_params(axis="both", labelsize=32, length=10, width=2.1, pad=9)
    for spine in ax.spines.values():
        spine.set_linewidth(2.1)
    ax.grid(linestyle="--", color="#d8d8d8", alpha=0.52, linewidth=1.5, zorder=0)
    ax.legend(loc="upper left", fontsize=27, frameon=False, handlelength=2.3, handletextpad=0.7)

    r = metrics["pearson_r"]
    labels = [
        rf"$r = {r:.2f}$" if r is not None else "r = N/A",
        rf"$MAE = {metrics['mae_m']:.2f}$",
        rf"$RMSE = {metrics['rmse_m']:.2f}$",
        rf"$SD = {metrics['sd_m']:.2f}$",
    ]
    ax.text(0.984, 0.032, "\n".join(labels), transform=ax.transAxes, fontsize=27,
            ha="right", va="bottom", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2))
    colorbar = fig.colorbar(points, ax=ax, pad=0.017, fraction=0.046, extend="both")
    colorbar.set_label("frequency", fontsize=30, labelpad=13)
    colorbar.ax.tick_params(labelsize=24, length=8, width=2, pad=7)
    colorbar.outline.set_linewidth(2)
    fig.savefig(output, dpi=210)
    plt.close(fig)
