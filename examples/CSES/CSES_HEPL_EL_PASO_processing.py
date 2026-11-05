from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from astropy import units as u

import el_paso as ep
from el_paso import units
# from IRBEM import Coords


# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# CSES HEP-L electron-channel energies.
# Replace these representative values with the exact values from the HEP-L
# product documentation or metadata used to create the MATLAB file.

# The provided MATLAB file contains geographic latitude and longitude, but no
# altitude variable. CSES is a low-Earth-orbit mission, so a constant altitude
# is used here only as a fallback. Replace this with measured altitude whenever
# available.
DEFAULT_ALTITUDE_KM = 507.0

# Values outside these ranges are treated as invalid/fill values.
VALID_LATITUDE_RANGE = (-90.0, 90.0)
VALID_LONGITUDE_RANGE = (-180.0, 180.0)
VALID_L_RANGE = (1.0, 20.0)


def matlab_datenum_to_posix(datenum: np.ndarray) -> np.ndarray:
    """
    Convert MATLAB serial date numbers to POSIX seconds.

    MATLAB datenum 719529 corresponds to 1970-01-01 00:00:00 UTC.
    """
    datenum = np.asarray(datenum, dtype=float).reshape(-1)
    return (datenum - 719529.0) * 86400.0


def geodetic_to_geo_cartesian_re(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_km: np.ndarray | float,
) -> np.ndarray:
    """
    Convert geodetic latitude, longitude, and altitude to geocentric GEO
    Cartesian coordinates in Earth radii.

    A WGS-84 ellipsoid is used. The returned array has shape (time, 3).
    """
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=float).reshape(-1))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=float).reshape(-1))

    if np.isscalar(altitude_km):
        alt = np.full(lat.shape, float(altitude_km))
    else:
        alt = np.asarray(altitude_km, dtype=float).reshape(-1)

    if not (lat.size == lon.size == alt.size):
        raise ValueError("Latitude, longitude, and altitude must have equal lengths.")

    # WGS-84 parameters
    a_km = 6378.137
    e2 = 6.69437999014e-3
    earth_radius_km = 6371.2  # IRBEM Earth-radius convention

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = a_km / np.sqrt(1.0 - e2 * sin_lat**2)

    x_km = (n + alt) * cos_lat * np.cos(lon)
    y_km = (n + alt) * cos_lat * np.sin(lon)
    z_km = (n * (1.0 - e2) + alt) * sin_lat

    return np.column_stack((x_km, y_km, z_km)) / earth_radius_km


def _read_vector(h5: h5py.File, name: str) -> np.ndarray:
    if name not in h5:
        raise KeyError(f"Required MATLAB variable {name!r} was not found.")
    return np.asarray(h5[name], dtype=float).reshape(-1)


def load_cses_merged_h5(
    h5_path
):

    with h5py.File(h5_path, "r") as f:

        time = f["time"][:]

        flux = f["flux"][:]

        energy_mev = f["energy"][:]

        pa = f["pitch_angle"][:]

        L = f["L"][:]

        lon = f["GEO_lon"][:]

        lat = f["GEO_lat"][:]

    energy_kev = (
        energy_mev * 1000.0
    )
    #pa_2d = np.tile(pa[:, np.newaxis],(1, len(time)))
   # Energy_2d = np.tile(energy_kev[:, np.newaxis],(1, len(time)))
    #breakpoint()
    variables = {}

    variables["Epoch"] = ep.Variable(
        original_unit=units.posixtime,
        data=time
    )

    variables["FEDU"] = ep.Variable(
        original_unit=(
            u.cm**2
            * u.s
            * u.sr
            * u.keV
        )**(-1),
        data=flux
    )

    variables["Energy_FEDU"] = ep.Variable(
        original_unit=u.keV,
        data=energy_kev
    )


    variables["Pitch_angle"] = ep.Variable(
        original_unit=u.deg,
        data=pa
    )

    variables["GEO_lat"] = ep.Variable(
        original_unit=u.deg,
        data=lat
    )

    variables["GEO_lon"] = ep.Variable(
        original_unit=u.deg,
        data=lon
    )


    variables["L_file"] = ep.Variable(
        original_unit=u.dimensionless_unscaled,
        data=L
    )
    ALTITUDE_KM = 507.0

    lat = variables[
        "GEO_lat"
    ].get_data()

    lon = variables[
        "GEO_lon"
    ].get_data()


    xGEO = geodetic_to_geo_cartesian_re(
        lat,
        lon,
        ALTITUDE_KM
    )


    variables["xGEO"] = ep.Variable(
        original_unit=units.RE,
        data=xGEO
    )


    return variables


def process_cses_hepl_from_mat(
    save_data_dir: str,
    h5_file_path: str,
    irbem_lib_path: str,
    start_time: datetime,
    end_time: datetime,
    num_cores: int = 24,
    cadence: timedelta = timedelta(seconds=2),
    altitude_km: float = DEFAULT_ALTITUDE_KM,
) -> None:
    """
    Process CSES HEP-L omnidirectional electron flux with EL-PASO.
    """
    irbem_options = [1, 1, 4, 4, 0]
    mag_field = "T89"

    logging.info("Loading CSES HEP-L data from %s", h5_file_path)
    variables = load_cses_merged_h5(h5_path= h5_file_path)

    time_bin_methods = {
        "FEDU":
            ep.TimeBinMethod.NanMedian,
        "Energy_FEDU":
            ep.TimeBinMethod.Repeat,
        "Pitch_angle":
            ep.TimeBinMethod.Repeat,
        "xGEO":
            ep.TimeBinMethod.NanMean,
        "GEO_lat":
            ep.TimeBinMethod.NanMean,
        "GEO_lon":
            ep.TimeBinMethod.NanMean,
        "L_file":
            ep.TimeBinMethod.NanMedian,
    }


    binned_time = ep.processing.bin_by_time(
        variables["Epoch"],
        variables=variables,
        time_bin_method_dict=
            time_bin_methods,
        time_binning_cadence=
            timedelta(seconds=2),
        start_time=start_time,
        end_time=end_time
    )
    ep.processing.fold_pitch_angles_and_flux(variables["FEDU"], variables["Pitch_angle"])
    # For omnidirectional flux there is no measured local pitch-angle dimension,
    # so only position-dependent magnetic quantities are requested.
    variables_to_compute = [
        ("B_Calc", mag_field),
        ("L_star", mag_field),
        ("L_m", mag_field),
        ("R_Eq", mag_field),
        ("Alpha_Eq", mag_field),
        ("MLT", mag_field),
        ("MLT_Eq", mag_field),
        ("B_Eq", mag_field),
        ("Alpha_LC_Eq", mag_field),
        ("Alpha_LC", mag_field)
    ]

    mag_vars = ep.processing.compute_magnetic_field_variables(
        time_var=binned_time,
        xgeo_var=variables["xGEO"],
        variables_to_compute=variables_to_compute,
        pa_local_var=variables["Pitch_angle"],
        energy_var=variables["Energy_FEDU"],
        particle_species="electron",
        irbem_lib_path=irbem_lib_path,
        irbem_options=irbem_options,
        num_cores=num_cores,
    )

    # This calculation assumes that FEDO has the differential-flux unit above
    # and that ELECTRON_ENERGIES_KEV contains the correct representative energy.
    psd_var = ep.processing.compute_phase_space_density(
        variables["FEDU"],
        variables["Energy_FEDU"],
        particle_species="electron",
    )

    variables_to_save = {
        "Epoch": binned_time,
        "FEDU": variables["FEDU"],
        "Energy_FEDU": variables["Energy_FEDU"],
        "Alpha": variables["Pitch_angle"],
        "Position": variables["xGEO"],
        "Position_geo_lon": variables["GEO_lon"],
        "Position_geo_lat": variables["GEO_lat"],
        "L_m": mag_vars[f"L_m_{mag_field}"],
        "MLT": mag_vars[f"MLT_{mag_field}"],
        "MLT_Eq": mag_vars[f"MLT_Eq_{mag_field}"],
        "L_star": mag_vars[f"L_star_{mag_field}"],
        "R_Eq": mag_vars[f"R_Eq_{mag_field}"],
        "B_Calc": mag_vars[f"B_Calc_{mag_field}"],
        "B_Eq": mag_vars[f"B_Eq_{mag_field}"],
        "Alpha_Eq": mag_vars[f"Alpha_Eq_{mag_field}"],
        "Alpha_LC_Eq": mag_vars[f"Alpha_LC_Eq_{mag_field}"],
        "PSD": psd_var,
    }

    saving_strategy = ep.saving_strategies.DailyLEORBStrategy(
        save_data_dir,
        mission="CSES",
        satellite="CSES-01",
        instrument="HEP-L",
        data_standard= ep.data_standards.GFZStandard(),
        mag_field="T89",
    )

    ep.save(
        variables_to_save,
        saving_strategy,
        start_time,
        end_time,
        time_var=binned_time,
    )

    logging.info("CSES HEP-L processing completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

SAVE_DIR = "/home/xlyu/CSES_Processed/"
H5_PATH = "/home/xlyu/CSES_data/CSES_HEPP_L_merged_20240510_15.h5"
IRBEM_LIB = "/home/xlyu/EL_PASO_new/EL_PASO/IRBEM/libirbem.so"

START_DATE = datetime(2024, 5, 14, tzinfo=timezone.utc)
END_DATE   = datetime(2024, 5, 15, tzinfo=timezone.utc)


current_day = START_DATE

while current_day <= END_DATE:

    day_start = current_day

    # End just before next midnight
    day_end = (
        current_day
        + timedelta(days=1)
        - timedelta(seconds=1)
    )

    print("\n======================================")
    print(
        f"Processing {day_start:%Y-%m-%d}"
    )
    print(
        f"{day_start} -> {day_end}"
    )
    print("======================================\n")

    try:

        process_cses_hepl_from_mat(
            save_data_dir=SAVE_DIR,
            h5_file_path=H5_PATH,
            irbem_lib_path=IRBEM_LIB,
            start_time=day_start,
            end_time=day_end,
            num_cores=24,
            cadence=timedelta(seconds=2),
            altitude_km=DEFAULT_ALTITUDE_KM,
        )

        print(
            f"Finished {day_start:%Y-%m-%d}"
        )

    except Exception:

        logging.exception(
            f"CSES HEP-L processing failed "
            f"for {day_start:%Y-%m-%d}"
        )

        # Continue to next day instead of stopping
        # Remove this behavior and use `raise`
        # if you want the script to stop on an error.

    current_day += timedelta(days=1)
