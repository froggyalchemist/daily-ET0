# TODO:  move plotting functions here
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, PercentFormatter


# ============================================================
# formatting helpers for relative (%) vs. absolute panels
# ============================================================
def _fold_change_formatter(value, _pos=None):
    """Turn a %-change value (e.g. 400, meaning +400%) into a fold-change
    label (e.g. '×5'). Floors at ×0 since a relative change can't go below
    -100% (you can't lose more than everything you started with) -- so
    every value <= -100% collapses to the same '×0' label. That's the
    tradeoff of this style: it reads intuitively near 0, but throws away
    resolution on the negative side of a wide colorbar like panel A's
    (vmin=-1000). Percent style keeps that resolution; pick per panel.
    """
    fold = max(1 + value / 100.0, 0.0)
    return f"×{fold:g}"


def _format_edge_label(value, anom_type, relative_style):
    if anom_type == "relative":
        return _fold_change_formatter(value) if relative_style == "fold" else f"{value:.0f}%"
    return f"{value:.3g}"


# ============================================================
# 4. plotting helper
# ============================================================
def plot_panel(ax, plot_grid, title, vmin, vmax, cmap, extend="max", label="",
                anom_type="absolute", relative_style="percent"):
    """
    anom_type : "absolute" or "relative"
        "absolute": behaves exactly like the original function -- plain
        numeric ticks, title suffixed with `unit` (e.g. "mm day-1").
        "relative": `plot_grid` is assumed to already be a %-change (i.e.
        multiplied by 100 upstream, as frequency/intensity/duration are
        here). Colorbar + histogram ticks are relabeled as percent (default)
        or, if relative_style="fold", as a fold-change like "×5". The title
        is suffixed with "[% change]" instead of `unit`, so a physical unit
        never gets attached to a relative quantity by mistake -- this was
        the actual bug in panels A-C of the original figure (titled
        "event/year", "mm/day", "days/event" while plotting %-change data).
    relative_style : "percent" or "fold", only used when anom_type="relative".
    """
    if anom_type not in ("absolute", "relative"):
        raise ValueError("anom_type must be 'absolute' or 'relative'")

    LON, LAT = plot_grid.lon.values, plot_grid.lat.values
    plot_grid = plot_grid.values
    im = ax.pcolormesh(
        LON, LAT, plot_grid,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax
    )

    ax.set_extent([-180, 180, -65, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, color="0.9", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="gray")
    ax.coastlines(linewidth=1.5)
    ax.text(
        0.02, 1.05, label,
        transform=ax.transAxes,
        fontsize=25,
        fontweight='bold',
        va='top',
        ha='left'
    )
    # ========================================================
    # inset colorbar
    # ========================================================
    cax = inset_axes(
        ax,
        width="35%",
        height="3%",
        loc='lower center',
        bbox_to_anchor=(0.15, 0.08, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    cb = plt.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend=extend
    )

    # FIX: this is the actual "-100% / ×10 instead of a bare number" ask.
    if anom_type == "relative":
        if relative_style == "fold":
            cb.ax.xaxis.set_major_formatter(FuncFormatter(_fold_change_formatter))
        else:
            cb.ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    # anom_type == "absolute": leave the default numeric formatter as-is

    cb.ax.tick_params(labelsize=12)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.outline.set_linewidth(1)
    cb.outline.set_edgecolor("black")
    cax.patch.set_alpha(0.8)

    ax.set_title(title, fontsize=20)

    # ========================================================
    # inset histogram (area-weighted)
    # ========================================================
    if np.ndim(LAT) == 1:
        lat2d = np.repeat(LAT[:, None], plot_grid.shape[1], axis=1)
    else:
        lat2d = LAT

    weights = np.cos(np.deg2rad(lat2d))

    # FIX: the next three lines were duplicated verbatim right below
    # themselves in the original (no effect on the result, just wasted
    # cycles) -- removed the redundant copy.
    valid_mask = np.isfinite(plot_grid) & np.isfinite(weights)
    valid_data = plot_grid[valid_mask].flatten()
    valid_weights = weights[valid_mask].flatten()

    # Don't drop out-of-range values -- clip them into the edge bins instead
    global_mean = np.sum(valid_data * valid_weights) / np.sum(valid_weights)
    weighted_squared_diff = np.sum(valid_weights * (valid_data - global_mean) ** 2) / np.sum(valid_weights)
    global_std = np.sqrt(weighted_squared_diff)
    valid_data = np.clip(valid_data, vmin, vmax)

    if len(valid_data) > 0:
        normalized_weights = valid_weights / np.sum(valid_weights)

        inset_ax = inset_axes(
            ax,
            width="12%",
            height="15%",
            loc="lower left",
            bbox_to_anchor=(0.1, 0.155, 1.1, 1.3),
            bbox_transform=ax.transAxes
        )

        bins = np.linspace(vmin, vmax, 22)

        hist_vals, bin_edges, patches = inset_ax.hist(
            valid_data,
            bins=bins,
            weights=normalized_weights,
            density=True,
            edgecolor="black",
            linewidth=1.2
        )

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for patch, center in zip(patches, bin_centers):
            patch.set_facecolor(cmap(norm(center)))

        inset_ax.tick_params(labelsize=9)
        inset_ax.get_yaxis().set_visible(False)

        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("gray")
            spine.set_alpha(0.5)

        inset_ax.set_xlim(vmin, vmax)
        xticks = np.linspace(vmin, vmax, 3)  # left, center, right
        inset_ax.set_xticks(xticks)

        # FIX: the mu/sigma box stays in percent even when
        # relative_style="fold" -- the mean/stdev of a fold-change isn't a
        # standard, easily-interpreted quantity, so the summary stats stay
        # in % and the x5-style fold labels are reserved for the tick marks.
        mean_str = f"{global_mean:.4g}%" if anom_type == "relative" else f"{global_mean:0.4g}"
        std_str = f"{global_std:.4g}%" if anom_type == "relative" else f"{global_std:0.4g}"
        formatted_mean = mean_str.replace("-", "−")
        formatted_std = std_str.replace("-", "−")
        inset_ax.set_facecolor("0.9")
        inset_ax.set_title(
            f"μ = {formatted_mean}\nσ = {formatted_std}",
            fontsize=12,
            ha="left",
            x=0
        )
        xtick_labels = [
            _format_edge_label(vmin, anom_type, relative_style),
            _format_edge_label(xticks[1], anom_type, relative_style),
            ">" + _format_edge_label(vmax, anom_type, relative_style),
        ]
        inset_ax.set_xticklabels(xtick_labels)
    ax.gridlines(
        draw_labels=False,
        linewidth=0.4,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )

    return im
