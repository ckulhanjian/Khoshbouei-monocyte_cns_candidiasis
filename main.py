# imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from scipy import stats

# METADATA
CELL_TYPES  = ["Microglia", "Neutrophils", "Macrophages"]
GENOTYPES   = ["WT", "KO"]
GENO_COLORS = {"WT": "blue", "KO": "orange"}
ANIMAL_MARKERS = {1: "o", 2: "s", 3: "^"}

# STYLE 
FONT        = "Arial"
CONTROL_COLOR = "#808080"   # grey — Control (WT)
MONOCYTE_COLOR = "#fd810a"  # orange — Monocyte De
GENO_COLORS = {"WT": CONTROL_COLOR, "KO": MONOCYTE_COLOR}
GENO_LABELS = {"WT": "Control", "KO": "Monocyte Depleted"}

# UNITS
FUNGAL_UNIT = "Fungal Volume (mm³)"
IMMUNE_UNIT_INNER = "Cell Density (1000 cells/100 μm circle)"
IMMUNE_UNIT_OUTER = "Cell Density (1000 cells/200 μm circle)"

# make directories if they don't exist
def make_dir():
    os.makedirs("tables", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/bars", exist_ok=True)
    os.makedirs("plots/heatmaps", exist_ok=True)
    os.makedirs("plots/scatters", exist_ok=True)

# read excel file
def read_file(filename):
    try:
        df = pd.read_excel(filename)
    except:
        print("Error reading excel file.")
    return df

'''
load, clean and merge fungal and immune data into a single dataframe for analysis
- read both files
- clean immune counts (remove commas, convert to numeric)
- reshape immune data from wide to long format (1 row per brain region + immune cell type
- merge fungal and immune data on common keys
'''
def load(fungal_file, immune_file):
    fungal = read_file(fungal_file)
    immune = read_file(immune_file)

    # remove commas in number values for immune counts
    for col in ["Microglia", "Neutrophils", "Macrophages"]:
        immune[col] = immune[col].astype(str).str.replace(",", "", regex=False)
        immune[col] = pd.to_numeric(immune[col], errors="coerce")

    # immune file -> 1 row becomes 3 rows
    # wide to long
    '''
    row 1   KO inner Microglia
    row 2   KO inner Neutophils
    row 3   KO inner Macrophages
    '''
    # melt — keep brain_area so we can filter inner/outer later
    imm_long = immune.melt(
        id_vars=["Brain #", "brain_region", "batch", "brain_area", "genotype"],
        value_vars=["Microglia", "Neutrophils", "Macrophages"],
        var_name="immune_cell",
        value_name="immune_count",
    )

    # clean up inner or outer with whitespaces
    imm_long["brain_area"] = imm_long["brain_area"].str.strip()

    # merge on the 1-to-1 key (no averaging)
    merged = fungal.merge(
        imm_long,
        on=["Brain #", "batch", "brain_region", "genotype", "immune_cell"]
    )

    # safety check in case of non numeric chars
    merged["volume"]       = pd.to_numeric(merged["volume"],       errors="coerce") 
    merged["immune_count"] = pd.to_numeric(merged["immune_count"], errors="coerce")
    merged = merged.dropna(subset=["volume", "immune_count"]) # remove na

    return merged

def spearman(x, y):
    """Returns (rho, p) or (nan, nan) if fewer than 3 points."""
    if len(x) < 3:
        return np.nan, np.nan
    return stats.spearmanr(x, y)

def pearson(x,y):
    if len(x) < 3:
        return np.nan, np.nan
    return stats.pearsonr(x, y)

def sig_stars(p):
    """Convert p-value to significance stars string."""
    if pd.isna(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# individual brain regions heatmap
def plot_heatmap(region_data, region, save=True):
    """
    Single heatmap figure with WT (left) and KO (right) panels.
    Each cell = Pearson ρ between fungal volume and immune count.
    Yellow * = p < 0.05.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.suptitle(f"{region} — Pearson r Heatmap\n(fungal volume vs immune count)",
                fontsize=12, fontweight="bold", y=1.05)

    for ax, geno in zip(axes, GENOTYPES):
        # ax.set_facecolor(PANEL)
        sub = region_data[region_data["genotype"] == geno]

        rho_vals, pval_vals = [], []
        for cell in CELL_TYPES:
            s = sub[sub["immune_cell"] == cell]
            rho, p = pearson(s["volume"].values, s["immune_count"].values)
            rho_vals.append(rho)
            pval_vals.append(p)

        rho_df = pd.DataFrame([rho_vals], columns=CELL_TYPES, index=["r^2"])

        sns.heatmap(
            rho_df.astype(float), ax=ax,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            vmin=-1, vmax=1,
            annot=True, fmt=".2f",
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=1, linecolor="#2a2a4a",
            cbar_kws={"shrink": 0.8, "label": "r^2"},
        )

        # Mark significant cells
        for j, p in enumerate(pval_vals):
            if pd.notna(p) and p < 0.05:
                ax.text(j + 0.5, 0.15, "*", ha="center", va="center",
                        fontsize=18, color="yellow", fontweight="bold")

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params( labelsize=8)

        ax.set_title(geno, fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(labelsize=9)
        ax.set_xticklabels(CELL_TYPES, rotation=20, ha="right")
        ax.set_yticklabels([], rotation=0)
        ax.set_ylabel("")

    plt.tight_layout()
    if save:
        fname = f"plots/heatmaps/{region.replace('/', '_')}_heatmap.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")
    return fig

# style scatter
def style_scatter_ax(ax, cell, area):
    """
    Apply reference-image styling to a single scatter axis.
    Call this after plotting points and regression lines.
    """
    immune_unit = IMMUNE_UNIT_INNER if area == "Inner" else IMMUNE_UNIT_OUTER

    ax.set_title(cell, fontsize=10, fontweight="bold",
                 fontfamily=FONT, color="black", pad=6)
    ax.set_xlabel(FUNGAL_UNIT, fontsize=9,
                  fontfamily=FONT, color="black", labelpad=6)
    ax.set_ylabel(immune_unit, fontsize=9,
                  fontfamily=FONT, color="black", labelpad=6)

    # match reference — remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="both", labelsize=8, color="black", labelcolor="black")
    ax.set_facecolor("white")

# scatter plots per brain region
def plot_scatter(region_data, region, save=True, inner=True):
    area = "Inner" if inner else "Outer"
    region_data = region_data[region_data["brain_area"] == area].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{region} — {area}",
                 fontsize=12, fontweight="bold",
                 fontfamily=FONT, color="black", y=1.01)

    for ax, cell in zip(axes, CELL_TYPES):
        r2_annotations = {}

        for geno in GENOTYPES:
            sub = region_data[
                (region_data["genotype"] == geno) &
                (region_data["immune_cell"] == cell)
            ].dropna(subset=["volume", "immune_count"])

            sub = sub.copy()
            sub["volume"] /= 1000
            sub["immune_count"] /= 1000

            color = GENO_COLORS[geno]
            label = GENO_LABELS[geno]

            ax.scatter(sub["volume"], sub["immune_count"],
                       color=color, label=label,
                       s=50, alpha=0.8, edgecolors="none")

            if len(sub) >= 3:
                m, b = np.polyfit(sub["volume"], sub["immune_count"], 1)
                x_line = np.linspace(sub["volume"].min(), sub["volume"].max(), 100)
                ax.plot(x_line, m * x_line + b, color=color,
                        linewidth=1.5, linestyle="-", alpha=0.9)  # solid line

                rho, p = pearson(sub["volume"].values, sub["immune_count"].values)
                r2_annotations[geno] = (rho ** 2, sig_stars(p))

        style_scatter_ax(ax, cell, area)

        wt_r2, wt_stars = r2_annotations.get("WT", (float("nan"), ""))
        ko_r2, ko_stars = r2_annotations.get("KO", (float("nan"), ""))

        wt_text = f"r²={wt_r2:.2f} {wt_stars}".strip()
        ko_text = f"r²={ko_r2:.2f} {ko_stars}".strip()

        # Both centered, WT slightly left of center, KO slightly right
        ax.text(0.35, .98, wt_text,
                transform=ax.transAxes,
                ha="right", va="bottom",
                color=GENO_COLORS["WT"],
                fontsize=9, fontweight="bold", fontfamily=FONT)

        ax.text(0.65, .98, ko_text,
                transform=ax.transAxes,
                ha="left", va="bottom",
                color=GENO_COLORS["KO"],
                fontsize=9, fontweight="bold", fontfamily=FONT)

    plt.tight_layout()

    if save:
        fname = f"plots/scatters/{area}/{region.replace('/', '_')}_scatterplot.svg"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white", format='svg')
        plt.close(fig)
    return fig

# total heatmap
def heatmap(data, save=True, inner=True):
    distance = "Inner" if inner == True else "Outer"
    if inner:
        data = data[data["brain_area"] == distance]

    wt = data[data["genotype"] == "WT"]
    ko = data[data["genotype"] == "KO"]

    wt_corr = (
        wt.groupby(["brain_region", "immune_cell"])[["volume", "immune_count"]]
          .apply(lambda x: x["volume"].corr(x["immune_count"], method="pearson"))
          .unstack()
    )

    ko_corr = (
        ko.groupby(["brain_region", "immune_cell"])[["volume", "immune_count"]]
          .apply(lambda x: x["volume"].corr(x["immune_count"], method="pearson"))
          .unstack()
    )

    # wt_corr **= 2  # convert to r²
    # ko_corr **= 2  # convert to r²

    heatmap_style(wt_corr, ko_corr, save, distance)

# style heatmap
def heatmap_style(wt_corr, ko_corr, save=True, distance="Inner"):
    # color palette
    cmap = sns.diverging_palette(
        222, 21,          # hue: 0=grey-ish, 30=orange
        s=80, l=50,
        as_cmap=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Pearson r: Fungal Volume (μm³) vs Immune Cell Count ({distance})",
                 fontsize=15, fontweight="bold",
                 fontfamily=FONT, color="black", y=0.98)

    for ax, matrix, title in zip(
        [ax1, ax2],
        [wt_corr, ko_corr],
        ["", ""]
    ):
        sns.heatmap(
            matrix, ax=ax,
            cmap=cmap,
            vmin=-1, vmax=1,
            annot=True, fmt=".2f",
            annot_kws={"size": 10, "weight": "normal",
                       "color": "#444444", "family": FONT},
            linewidths=0.6, linecolor="#eeeeee",
            cbar_kws={"label": "Pearson's ρ", "shrink": 0.8}
        )

        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color("black")
        cbar.ax.yaxis.label.set_fontsize(10)
        cbar.ax.yaxis.label.set_fontweight("bold")
        cbar.ax.yaxis.label.set_fontfamily(FONT)
        cbar.ax.tick_params(labelsize=8, colors="black")
        cbar.ax.yaxis.set_tick_params(labelcolor="black")
        cbar.outline.set_edgecolor("black")

        # axes labels and ticks
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=13, fontweight="bold",
                     fontfamily=FONT, color="black", pad=10)
        ax.set_xlabel("Immune Cell Type", fontsize=10, fontweight="bold",
                      fontfamily=FONT, color="black", labelpad=8)
        ax.set_ylabel("Brain Region", fontsize=10, fontweight="bold",
                      fontfamily=FONT, color="black", labelpad=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right",
                           fontsize=9, color="black", fontfamily=FONT)
        
    plt.tight_layout()
    if save:
        print(f"Saving heatmap_{distance}...")
        fname = f"plots/heatmaps/heatmap_{distance}.svg"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white", format='svg')
        plt.close(fig)
    return fig

def main():
    make_dir()
    fungal_file = "cleaned_data/fungal_volumes.xlsx"
    immune_file = "cleaned_data/immune_cell_counts.xlsx"
    data = load(fungal_file, immune_file)
    
    # 1. plot heatmap
    heatmap(data, inner=True)
    heatmap(data, inner=False)
    
    # 2. correlation plots
    regions = sorted(data["brain_region"].unique())
    for region in regions:
        print(f"Plotting: {region}")
        region_data = data[data["brain_region"] == region]
        # plot_heatmap(region_data, region)
        plot_scatter(region_data, region, inner=True)   # Inner plot
        plot_scatter(region_data, region, inner=False)  # Outer plot

if __name__== "__main__":
    main()