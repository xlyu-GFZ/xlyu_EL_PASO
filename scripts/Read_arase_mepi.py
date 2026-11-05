from datetime import datetime, timedelta, timezone
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

import el_paso as ep
from el_paso.dataset import DataSet, GFZDataSet, PRBEMDataSet
from el_paso.dataset.interp_functions import TargetType
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize,LogNorm

ep.setup_logging()
saving_strategy = ep.saving_strategies.GFZStrategy(
    base_data_path="/home/xlyu/EL_PASO/el_paso/recipes/arase",
    mission="ARASE",
    satellite="arase",
    instrument="mepi",
    mag_field="T89",
    data_standard=ep.data_standards.GFZStandard(),
)
start_time = datetime(2024, 5, 10, tzinfo=timezone.utc)
end_time = datetime(2024, 5, 14, 23, 59, 59, tzinfo=timezone.utc)
gfz_data_set = GFZDataSet(saving_strategy, start_time, end_time)
print(gfz_data_set.possible_variables)


target_mu = [20, 30]
target_K = [0.16, 0.16]
mu_scale=gfz_data_set.B_total/gfz_data_set.B_sat
mu_real=gfz_data_set.InvMu*mu_scale[:, np.newaxis, np.newaxis]
gfz_data_set.InvMu = mu_real

psd_pairs = gfz_data_set.interp_psd(target_K, target_mu=target_mu, target_type=TargetType.TargetPairs, n_threads=16)
psd_pairs.shape  # (n_time, len(target_mu))



# ------------------------------------------
# 1. Convert time from MATLAB datenum
# ------------------------------------------
time_matlab = np.asarray(gfz_data_set.time).ravel()

time_dt = np.array([
    datetime.fromordinal(int(t))
    + timedelta(days=float(t) % 1)
    - timedelta(days=366)
    for t in time_matlab
])

K_all = np.asarray(gfz_data_set.InvK)
Lstar_all = np.asarray(gfz_data_set.Lstar)
Lstar_target = np.full(
    (len(time_dt), len(target_K)),
    np.nan
)
for it in range(len(time_dt)):
    k_row = K_all[it]
    L_row = Lstar_all[it]

    valid = (
        np.isfinite(k_row)
        & np.isfinite(L_row)
        & (k_row > 0)
        & (L_row > 0)
    )
    if np.count_nonzero(valid) < 2:
        continue

    k_valid = k_row[valid]
    L_valid = L_row[valid]

    # Sort by K for interpolation
    order = np.argsort(k_valid)
    k_valid = k_valid[order]
    L_valid = L_valid[order]

    for j, k_target in enumerate(target_K):
        if k_valid[0] <= k_target <= k_valid[-1]:
            Lstar_target[it, j] = np.interp(
                k_target, k_valid, L_valid
            )

psd = np.asarray(psd_pairs, dtype=float)
valid_psd = psd[np.isfinite(psd) & (psd > 0)]
norm = LogNorm(vmin=1e-6, vmax=1e-4)
fig, axes = plt.subplots(
    2, 1, figsize=(12, 7),
    sharex=True, sharey=True,
    constrained_layout=True
)

for j, ax in enumerate(axes):
    valid = (
        np.isfinite(Lstar_target[:, j])
        & np.isfinite(psd[:, j])
        & (psd[:, j] > 0)
    )

    sc = ax.scatter(
        np.asarray(time_dt)[valid],
        Lstar_target[valid, j],
        c=psd[valid, j],
        cmap="turbo",
        norm=norm,
        s=12,
        marker="s",
        linewidths=0
    )

    ax.set_ylabel(r"$L^*$")
    ax.set_title(
        rf"$\mu={target_mu[j]}$ MeV/G, $K={target_K[j]}$"
    )
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Time (UTC)")
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%m/%d")
)

cbar = fig.colorbar(sc, ax=axes, pad=0.02)
cbar.set_label("PSD")

plt.savefig("arase_psd_time_Lstar_diffk_realmu.png", dpi=300)
plt.show()