from datetime import datetime, timezone

import numpy as np
import el_paso as ep
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ep.setup_logging()

# --------------------------------------------------
# Settings
# --------------------------------------------------

sat = "noaa15"

start_time = datetime(2024, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
end_time   = datetime(2024, 5, 11, 19, 0, 0, tzinfo=timezone.utc)

# L shells
Ltargets = [3.0, 3.5]
dL = 0.05

# Energy channel
ie = 1


# --------------------------------------------------
# Load data
# --------------------------------------------------

poes_ds = ep.GFZDataSet(
    ep.recipes.poes.poes_meped_strategy(
        "/home/xlyu/POES/",
        "T89",
        sat
    ),
    start_time,
    end_time,
)

flux = np.asarray(poes_ds.FEIU)
Lm = np.asarray(poes_ds.R0)
time = np.asarray(poes_ds.datetime)
MLT = np.asarray(poes_ds.MLT)

precip_flux = flux[:, ie, 0]
trapped_flux = flux[:, ie, 1]


# --------------------------------------------------
# Figure
# --------------------------------------------------

# fig, axes = plt.subplots(
#     4, 1,
#     figsize=(10, 9),
#     sharex=True,
#     gridspec_kw={
#         "height_ratios": [2, 1, 2, 1],
#         "hspace": 0.08
#     }
# )

fig = plt.figure(figsize=(10, 9))

# Two groups with a larger gap between them
outer = fig.add_gridspec(
    2, 1,
    hspace=0.2
)

# Group 1: L = 3.0
gs_top = outer[0].subgridspec(
    2, 1,
    height_ratios=[2, 1],
    hspace=0.06
)

# Group 2: L = 3.5
gs_bottom = outer[1].subgridspec(
    2, 1,
    height_ratios=[2, 1],
    hspace=0.06
)

# Panel a
ax0 = fig.add_subplot(gs_top[0])

# Panel b shares x only with panel a
ax1 = fig.add_subplot(gs_top[1], sharex=ax0)

# Panel c: independent x-axis
ax2 = fig.add_subplot(gs_bottom[0])

# Panel d shares x only with panel c
ax3 = fig.add_subplot(gs_bottom[1], sharex=ax2)

axes = [ax0, ax1, ax2, ax3]
# --------------------------------------------------
# L = 3.0
# --------------------------------------------------

Ltarget = Ltargets[0]

mask = (
    (Lm >= Ltarget - dL) &
    (Lm <= Ltarget + dL)
)

axes[0].semilogy(
    time[mask],
    trapped_flux[mask],
    'o--',
    color='black',
    linewidth=1.2,
    markersize=4,
    label='Trapped'
)

axes[0].semilogy(
    time[mask],
    precip_flux[mask],
    'o-',
    color='red',
    linewidth=1.2,
    markersize=4,
    label='Precipitating'
)

axes[0].set_ylabel(
    r'Electron flux' '\n'
    r'[cm$^{-2}$ s$^{-1}$ sr$^{-1}$]'
)

axes[0].set_title(
    rf'> 100 keV electron flux at $L={Ltarget:.1f}\pm{dL:.2f}$ by NOAA-15'
)

axes[0].legend(loc='lower left')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', labelbottom=False)
axes[0].text(
    0.01, 0.82,
    'a)',
    transform=axes[0].transAxes,
    fontsize=16
)


# MLT at L = 3.0

axes[1].plot(
    time[mask],
    MLT[mask],
    'o',
    color='purple',
    markersize=4
)

axes[1].set_ylabel('MLT')
axes[1].set_ylim(0, 24)
axes[1].set_yticks([0, 6, 12, 18, 24])
axes[1].grid(True, alpha=0.3)

axes[1].text(
    0.01, 0.72,
    'b)',
    transform=axes[1].transAxes,
    fontsize=16
)


# --------------------------------------------------
# L = 3.5
# --------------------------------------------------

Ltarget = Ltargets[1]

mask = (
    (Lm >= Ltarget - dL) &
    (Lm <= Ltarget + dL)
)

axes[2].semilogy(
    time[mask],
    trapped_flux[mask],
    'o--',
    color='black',
    linewidth=1.2,
    markersize=4,
    label='Trapped'
)

axes[2].semilogy(
    time[mask],
    precip_flux[mask],
    'o-',
    color='red',
    linewidth=1.2,
    markersize=4,
    label='Precipitating'
)

axes[2].set_ylabel(
    r'Electron flux' '\n'
    r'[cm$^{-2}$ s$^{-1}$ sr$^{-1}$]'
)

axes[2].set_title(
    rf'> 100 keV electron flux at $L={Ltarget:.1f}\pm{dL:.2f}$ by NOAA-15'
)

#axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', labelbottom=False)
axes[2].text(
    0.01, 0.82,
    'c)',
    transform=axes[2].transAxes,
    fontsize=16
)


# MLT at L = 3.5

axes[3].plot(
    time[mask],
    MLT[mask],
    'o',
    color='purple',
    markersize=4
)

axes[3].set_ylabel('MLT')
axes[3].set_xlabel('Time')

axes[3].set_ylim(0, 24)
axes[3].set_yticks([0, 6, 12, 18, 24])
axes[3].grid(True, alpha=0.3)

axes[3].text(
    0.01, 0.72,
    'd)',
    transform=axes[3].transAxes,
    fontsize=16
)


ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax3.set_xlabel('2024-05-11')
# --------------------------------------------------
# Save
# --------------------------------------------------

#plt.tight_layout()

filename = (
    f"POES_{sat}_L3.0_L3.5_"
    f"ie={ie}_TrapvsPrec.png"
)

plt.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()