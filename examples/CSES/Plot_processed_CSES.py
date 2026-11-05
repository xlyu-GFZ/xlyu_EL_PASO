import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone,timedelta
from pathlib import Path
from matplotlib.colors import LogNorm

# ============================================================
# SETTINGS
# ============================================================


def matlab_datenum_to_datetime(dn):
    return (
        datetime.fromordinal(int(dn))
        + timedelta(days=float(dn) % 1)
        - timedelta(days=366)
    )


DATA_DIR = Path(
    "/home/xlyu/CSES_Processed/CSES/cses-01"
)

files = [
    DATA_DIR / f"cses-01_hep-l_202405{day:02d}_T89.nc"
    for day in range(10, 11)
]


# ============================================================
# READ ONE FILE
# ============================================================

def read_one_file(filename):

    with h5py.File(filename, "r") as f:

        time = f["time"][:]


        flux = f["Flux"][:]

        energy = f["energy_channels"][:]

        alpha_local = f["alpha_local"][:]

        alpha_eq = f["alpha_eq_model"][:]

        alpha_lc_eq = f["alpha_lc_eq"][:]

        Lm = f["Lm"][:]

        Lstar = f["Lstar"][:]

        MLT = f["MLT"][:]

        MLT_Eq = f["MLT_Eq"][:]

        geo_lat = f["geo_lat"][:]

        geo_lon = f["geo_lon"][:]

    return {
        "time": time,
        "flux": flux,
        "energy": energy,
        "alpha_local": alpha_local,
        "alpha_eq": alpha_eq,
        "alpha_lc_eq": alpha_lc_eq,
        "Lm": Lm,
        "Lstar": Lstar,
        "MLT": MLT,
        "MLT_Eq":MLT_Eq,
        "geo_lat": geo_lat,
        "geo_lon": geo_lon,
    }


# ============================================================
# LOAD ALL DAYS
# ============================================================

data_list = []

for file in files:

    print("Reading:", file)

    data_list.append(
        read_one_file(file)
    )
# ============================================================
# CONCATENATE ALONG TIME
# ============================================================

time_all = np.concatenate([
    d["time"] for d in data_list
])

flux_all = np.concatenate([
    d["flux"] for d in data_list
], axis=0)

alpha_local_all = np.concatenate([
    d["alpha_local"] for d in data_list
], axis=0)

alpha_eq_all = np.concatenate([
    d["alpha_eq"] for d in data_list
], axis=0)

alpha_lc_eq_all = np.concatenate([
    d["alpha_lc_eq"] for d in data_list
])

Lm_all = np.concatenate([
    d["Lm"] for d in data_list
])

Lstar_all = np.concatenate([
    d["Lstar"] for d in data_list
])

MLT_all = np.concatenate([
    d["MLT"] for d in data_list
])

MLT_Eq_all = np.concatenate([
    d["MLT_Eq"] for d in data_list
])

geo_lat_all = np.concatenate([
    d["geo_lat"] for d in data_list
])

geo_lon_all = np.concatenate([
    d["geo_lon"] for d in data_list
])


# Energy grid should be the same every day
energy_all = np.concatenate([
    d["energy"] for d in data_list
], axis=0)


# ============================================================
# SORT EVERYTHING BY TIME
# ============================================================

sort_index = np.argsort(time_all)

time_all = time_all[sort_index]
flux_all = flux_all[sort_index, :, :]
alpha_local_all = alpha_local_all[sort_index, :]
alpha_eq_all = alpha_eq_all[sort_index, :]
alpha_lc_eq_all = alpha_lc_eq_all[sort_index]

Lm_all = Lm_all[sort_index]
Lstar_all = Lstar_all[sort_index]
MLT_all = MLT_all[sort_index]
MLT_Eq_all = MLT_Eq_all[sort_index]
geo_lat_all = geo_lat_all[sort_index]
geo_lon_all = geo_lon_all[sort_index]

energy_all = energy_all[sort_index, :]


# ============================================================
# TIME FOR PLOTTING
# ============================================================


time_dt = np.array([
    matlab_datenum_to_datetime(t)
    for t in time_all
])

fig, ax = plt.subplots(
     figsize=(14, 6)
)
ax.plot(
    time_dt,
    MLT_all[:],
    ".",
    markersize=2
)

ax.set_ylabel("MLT_local")
ax.set_xlabel("Time (UT)")

ax.set_ylim(0, 24)
ax.set_yticks([0, 6, 12, 18, 24])

ax2 = ax.twinx()
ax2.plot(
    time_dt,
    MLT_Eq_all,
    "x",
    markersize=4,
    label="MLT_eq",
    color="tab:red"
)

ax2.set_ylabel("MLT_eq")

ax2.set_ylim(0, 24)
ax2.set_yticks([0, 6, 12, 18, 24])
plt.tight_layout()
plt.show()
plt.savefig("MLT_Comparison.png")
