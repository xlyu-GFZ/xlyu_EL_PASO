from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import el_paso as ep

ep.setup_logging()

satellites = ["noaa15", "noaa18", "noaa19", "metop1", "metop3"]

data_path = "/home/xlyu/POES/"
mfm = "T89"

start_time = datetime(2024, 5, 11, 0, 0, tzinfo=timezone.utc)
end_time   = datetime(2024, 5, 11, 23, 59, tzinfo=timezone.utc)

ie = 1          # energy channel
r_thr = 0.7     # define "strong precipitation"

all_L = []
all_MLT = []
all_ratio = []

for sat in satellites:
    print("Reading:", sat)

    poes_ds = ep.GFZDataSet(
        ep.recipes.poes.poes_meped_strategy(data_path, mfm, sat),
        start_time,
        end_time,
    )

    flux = np.asarray(poes_ds.FEIU)   # (time, energy, telescope)
    L = np.asarray(poes_ds.R0)        # use Lm, not R0
    MLT = np.asarray(poes_ds.MLT)

    Jpre  = flux[:, ie, 0]
    Jtrap = flux[:, ie, 1]

    valid = (
        np.isfinite(L) &
        np.isfinite(MLT) &
        np.isfinite(Jpre) &
        np.isfinite(Jtrap) &
        (Jpre > 0) &
        (Jtrap > 0)
    )

    all_L.append(L[valid])
    all_MLT.append(MLT[valid])
    all_ratio.append((Jpre / Jtrap)[valid])

L = np.concatenate(all_L)
MLT = np.concatenate(all_MLT)
ratio = np.concatenate(all_ratio)
L_edges = np.arange(2.0, 8.25, 0.25)
MLT_edges = np.arange(0, 25, 1.0)

frac_map = np.full((len(L_edges)-1, len(MLT_edges)-1), np.nan)
count_map = np.zeros((len(L_edges)-1, len(MLT_edges)-1), dtype=int)

min_count = 5   # require at least this many samples in a bin

for i in range(len(L_edges) - 1):
    for j in range(len(MLT_edges) - 1):

        mask = (
            (L >= L_edges[i]) & (L < L_edges[i+1]) &
            (MLT >= MLT_edges[j]) & (MLT < MLT_edges[j+1])
        )

        n_total = np.sum(mask)
        count_map[i, j] = n_total

        if n_total >= min_count:
            n_strong = np.sum(ratio[mask] >= r_thr)
            frac_map[i, j] = n_strong / n_total
theta_edges = MLT_edges / 24.0 * 2.0 * np.pi
Theta, Radius = np.meshgrid(theta_edges, L_edges)

fig, ax = plt.subplots(
    figsize=(9, 9),
    subplot_kw={"projection": "polar"}
)

pcm = ax.pcolormesh(
    Theta,
    Radius,
    frac_map,
    shading="flat",
    cmap="rainbow",
    vmin=0,
    vmax=1
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
ax.set_xticklabels(["00 MLT", "06 MLT", "12 MLT", "18 MLT"])

ax.set_ylim(2, 8)

ax.set_title(
    f"May 11, 2024, 02–12 UT\n"
    f"Fraction of samples with $J_{{pre}}/J_{{trap}} \\geq {r_thr}$",
    pad=25
)

cbar = fig.colorbar(pcm, ax=ax, pad=0.10, shrink=0.8)
cbar.set_label("Occurrence fraction")
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])

plt.tight_layout()
plt.savefig("POES_occurrence_ratio_ge_threshold.png", dpi=300, bbox_inches="tight")
plt.show()