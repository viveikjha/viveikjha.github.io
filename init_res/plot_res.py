import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from astropy.cosmology import FlatLambdaCDM

# Define the file path - change this to your CSV file path
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_results_dp1_optical.csv")

# Read the CSV file
df = pd.read_csv(csv_file)

# Define the physical parameters to plot (median columns only, excluding object_id and redshift)
# We'll pair each parameter with a nice label for the plot
params = {
    'redshift': r'Redshift',
    'log10M_\\star_M_\\odot_median': r'$\log_{10}(M_{\star}/M_{\odot})$',
    'log10Z_Z_\\odot_median': r'$\log_{10}(Z/Z_{\odot})$',
    'tau\\,Gyr_median': r'$\tau$ (Gyr)',
    't_{\\mathrm{age\\,Gyr_median': r'$t_{\mathrm{age}}$ (Gyr)',
    'tau_{\\mathrm{dust_median': r'$\tau_{\mathrm{dust}}$',
    'SFR_M_\\odot_yr{-1_median': r'SFR ($M_{\odot}\,\mathrm{yr}^{-1}$)',
    'sSFR_yr{-1_median': r'sSFR ($\mathrm{yr}^{-1}$)',
    'A_v_mag_median': r'$A_V$ (mag)',
}

# Filter to only columns that exist in the dataframe
available_params = {}
for col_key, label in params.items():
    matching_cols = [c for c in df.columns if col_key in c or c == col_key]
    if matching_cols:
        available_params[matching_cols[0]] = label
    else:
        print(f"Warning: Column matching '{col_key}' not found in the CSV file.")

n_params = len(available_params)
if n_params == 0:
    print("No matching columns found. Available columns are:")
    print(df.columns.tolist())
    sys.exit(1)

# Determine grid size
ncols = 3
nrows = int(np.ceil(n_params / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.atleast_2d(axes)
axes_flat = axes.flatten()

param_bins = {
    'redshift': 40,
    'log10M_\\star_M_\\odot_median': 30,
    'log10Z_Z_\\odot_median': 30,
    'tau\\,Gyr_median': 25,
    't_{\\mathrm{age\\,Gyr_median': 25,
    'tau_{\\mathrm{dust_median': 25,
    'SFR_M_\\odot_yr{-1_median': 35,
    'sSFR_yr{-1_median': 35,
    'A_v_mag_median': 40,
}
for idx, (col, label) in enumerate(available_params.items()):
    n_bins = 20
    for key, bins in param_bins.items():
        if key in col or col == key:
            n_bins = bins
            break

    data = df[col].dropna()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'axes.linewidth': 1.2,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data, bins=20, color='steelblue', edgecolor='white', linewidth=0.6, alpha=0.85)
    ax.set_xlabel(label, fontsize=16, labelpad=8)
    ax.set_ylabel('Count', fontsize=16, labelpad=8)
    ax.set_title(f'Distribution of {label}', fontsize=17, pad=12)
    ax.tick_params(axis='both', which='major', labelsize=13, length=5, width=1.1, direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8, direction='in', top=True, right=True)
    ax.minorticks_on()

    median_val = data.median()
    mean_val = data.mean()
    ax.axvline(median_val, color='crimson', linestyle='--', linewidth=2.0, label=f'Median: {median_val:.3f}')
    ax.axvline(mean_val, color='darkorange', linestyle='-', linewidth=2.0, label=f'Mean: {mean_val:.3f}')
    ax.legend(fontsize=13, framealpha=0.9, edgecolor='gray', frameon=True)

    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    safe_col = col.replace('\\', '_').replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hist_{safe_col}.png"),
                dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # --- M* - SFR plot (Star-Forming Main Sequence) ---
    mass_key = 'log10M_\\star_M_\\odot_median'
    sfr_key = 'SFR_M_\\odot_yr{-1_median'

    mass_col = [c for c in df.columns if mass_key in c or c == mass_key]
    sfr_col = [c for c in df.columns if sfr_key in c or c == sfr_key]

    if mass_col and sfr_col:
        mass_col = mass_col[0]
        sfr_col = sfr_col[0]

        df_ms = df[[mass_col, sfr_col, 'redshift']].dropna() if 'redshift' in df.columns else df[[mass_col, sfr_col]].dropna()
        df_ms = df_ms[df_ms[sfr_col] > 0].copy()
        df_ms['log10_SFR'] = np.log10(df_ms[sfr_col])

        x = df_ms[mass_col].values
        y = df_ms['log10_SFR'].values

        # Binned medians (stars)
        n_bins = 10
        x_bins = np.linspace(np.percentile(x, 2), np.percentile(x, 98), n_bins + 1)
        x_centers, y_medians, y_errs = [], [], []
        for i in range(n_bins):
            mask = (x >= x_bins[i]) & (x < x_bins[i + 1])
            if mask.sum() >= 5:
                x_centers.append(0.5 * (x_bins[i] + x_bins[i + 1]))
                y_medians.append(np.median(y[mask]))
                y_errs.append(np.std(y[mask]) / np.sqrt(mask.sum()))

        x_centers = np.array(x_centers)
        y_medians = np.array(y_medians)
        y_errs = np.array(y_errs)

        # Linear fit through binned medians
        fit_coeffs = np.polyfit(x_centers, y_medians, 1)
        fit_line = np.poly1d(fit_coeffs)
        x_fit = np.linspace(x_centers.min(), x_centers.max(), 200)

        fig2, ax2 = plt.subplots(figsize=(9, 7))

        # Scatter plot of observed points in grey
        ax2.scatter(x, y, color='grey', alpha=0.4, s=50, zorder=2, label='Observed data')

        # Binned medians connected with a line
        ax2.errorbar(x_centers, y_medians, yerr=y_errs,
                     fmt='*', color='royalblue', markersize=20, markeredgecolor='black',
                     markeredgewidth=0.7, ecolor='royalblue', elinewidth=1.5,
                     capsize=3, zorder=10, label='Binned median ± SE')
        ax2.plot(x_centers, y_medians, color='crimson', linewidth=1.5,
                 linestyle='-', alpha=0.7, zorder=9)

        # Linear fit line
        

        ax2.set_xlabel(r'$\log_{10}(M_{\star}/M_{\odot})$', fontsize=14)
        ax2.set_ylabel(r'$\log_{10}(\mathrm{SFR}\;/\;M_{\odot}\,\mathrm{yr}^{-1})$', fontsize=14)
        ax2.set_title(r'$M_{\star}$ – SFR Relation (Star-Forming Main Sequence)', fontsize=15)
        ax2.tick_params(axis='both', labelsize=11, direction='in', top=True, right=True)
        ax2.minorticks_on()
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax2.legend(fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mstar_sfr_relation.png"),
                    dpi=200, bbox_inches='tight')
        plt.show()
        plt.close(fig2)
        print("Plot saved as 'mstar_sfr_relation.png'")
    else:
        missing = []
        if not mass_col:
            missing.append("stellar mass")
        if not sfr_col:
            missing.append("SFR")
        print(f"Warning: Could not create M*-SFR plot. Missing column(s): {', '.join(missing)}")

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Cosmology — consistent with your YAML
# ---------------------------------------------------------------------------
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.30)

# ---------------------------------------------------------------------------
# Speagle et al. (2014) main sequence
# ---------------------------------------------------------------------------
def speagle2014_ms(log_mstar: np.ndarray, redshift: float,
                   cosmo: FlatLambdaCDM) -> np.ndarray:
    """
    Speagle et al. (2014) star-forming main sequence.

    Parameters
    ----------
    log_mstar : np.ndarray
        log10(M* / Msun)
    redshift : float
        Redshift at which to evaluate the relation.
    cosmo : FlatLambdaCDM
        Cosmology instance for age calculation.

    Returns
    -------
    np.ndarray
        log10(SFR / Msun yr^-1)
    """
    t = cosmo.age(redshift).value          # Age of universe in Gyr
    log_sfr = (0.84 - 0.026 * t) * log_mstar - (6.51 - 0.11 * t)
    return log_sfr

# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------
z_median      = 0.01
sigma_speagle = 0.2                        # Speagle et al. intrinsic scatter [dex]

log_mstar_grid = np.linspace(9.0, 12.0, 200)
log_sfr_ms     = speagle2014_ms(log_mstar_grid, z_median, cosmo)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# --- Hexbin of your data (replace df with your actual dataframe) ----------
hb = ax.hexbin(
    df_ms[mass_col], df_ms['log10_SFR'],
    gridsize=30,
    mincnt=2,
    cmap="Blues_r",
    linewidths=0.3,
)
cb = fig.colorbar(hb, ax=ax, pad=0.02)
cb.set_label("Count", fontsize=12)

# --- Speagle et al. (2014) relation at z = 0.68 ---------------------------
ax.plot(
    log_mstar_grid, log_sfr_ms,
    color="crimson",
    lw=2.0,
    ls="--",
    label=f"Speagle et al. (2014), $z = {z_median}$",
    zorder=5,
)

# --- 1σ shaded band --------------------------------------------------------
ax.fill_between(
    log_mstar_grid,
    log_sfr_ms - sigma_speagle,
    log_sfr_ms + sigma_speagle,
    color="crimson",
    alpha=0.15,
    label=r"$\pm 1\sigma$ (0.2 dex)",
    zorder=4,
)

# --- 2σ shaded band --------------------------------------------------------
ax.fill_between(
    log_mstar_grid,
    log_sfr_ms - 2 * sigma_speagle,
    log_sfr_ms + 2 * sigma_speagle,
    color="crimson",
    alpha=0.08,
    label=r"$\pm 2\sigma$ (0.4 dex)",
    zorder=3,
)

# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$",       fontsize=13)
ax.set_ylabel(r"$\log_{10}(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$", fontsize=13)
ax.set_title(r"$M_*$ – SFR Relation (Star-Forming Main Sequence)",        fontsize=13)

ax.set_xlim(9.0, 12.0)
ax.set_ylim(-1.5, 1.5)

ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, ls="--", alpha=0.4)

plt.tight_layout()

plt.show()