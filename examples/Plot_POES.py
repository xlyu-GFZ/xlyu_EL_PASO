from datetime import datetime, timezone

import numpy as np
import el_paso as ep
import matplotlib.pyplot as plt

ep.setup_logging()

# --------------------------------------------------
# Settings
# --------------------------------------------------

satellites = ["noaa15", "noaa18", "noaa19", "metop1", "metop3"]

Ltarg = 4.0
dL = 0.05

Lmin = Ltarg - dL
Lmax = Ltarg + dL

# Energy channel
ie = 1

start_time = datetime(2024, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
end_time   = datetime(2024, 5, 11, 18, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------
# Loop over satellites
# --------------------------------------------------

for sat in satellites:

    print(f"Processing {sat}...")

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

    # Select L range
    mask = (Lm >= Lmin) & (Lm <= Lmax)

    # Skip satellite if there are no measurements in this L range
    if not np.any(mask):
        print(f"  No data around L = {Ltarg} for {sat}")
        continue

    # Select energy channel
    precip_flux = flux[:, ie, 0]
    trapped_flux = flux[:, ie, 1]

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel: flux
    axes[0].semilogy(
        time[mask],
        trapped_flux[mask],
        '.-',
        color='black',
        linewidth=1.2,
        markersize=4,
        label='Trapped'
    )

    axes[0].semilogy(
        time[mask],
        precip_flux[mask],
        '.-',
        color='red',
        linewidth=1.2,
        markersize=4,
        label='Precipitating'
    )

    axes[0].set_ylabel('Flux')
    axes[0].set_title(
        rf'{sat.upper()}, $R_0 = {Ltarg:.2f} \pm {dL:.2f}$'
    )
    axes[0].legend()
    axes[0].grid(True)

    # Bottom panel: MLT
    axes[1].plot(
        time[mask],
        MLT[mask],
        '.'
    )

    axes[1].set_ylabel('MLT')
    axes[1].set_xlabel('Time')
    axes[1].set_ylim(0, 24)
    axes[1].set_yticks([0, 6, 12, 18, 24])
    axes[1].grid(True)

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    filename = (
        f"POES_{sat}_L={Ltarg:.2f}_"
        f"dL={dL:.2f}_ie={ie}_TrapvsPrec.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"  Saved: {filename}")