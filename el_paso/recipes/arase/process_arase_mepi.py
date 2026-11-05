# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
import xarray as xr
import numpy as np
import pandas as pd
from astropy import units as u
import cdflib
import el_paso as ep
from el_paso.processing.magnetic_field_utils.irbem import Coords
from el_paso.recipes.arase import (
    get_arase_orbit_level_2_variables,
    get_arase_orbit_level_3_variables,
)


def arase_mepi_gfz_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Legacy GFZ .mat saving strategy for Arase MEP-i."""
    return ep.saving_strategies.GFZStrategy(
        Path(base_data_path), "ARASE", "arase", "mepi", mag_field, data_standard=data_standard
    )


def arase_mepi_h5_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Monthly HDF5 saving strategy for Arase MEP-e."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "Arase",
        "arase",
        "mepi",
        mag_field,
        data_standard=data_standard or ep.data_standards.GFZStandard(),
        file_format="h5",
    )


def arase_mepi_netcdf_strategy(
    base_data_path: str | Path,
    mag_field: ep.typing.MagneticFieldLiteral,
    data_standard: ep.typing.DataStandard[ep.typing.StandardName] | None = None,
) -> ep.SavingStrategy:
    """Monthly NetCDF saving strategy for Arase MEP-i."""
    return ep.saving_strategies.MonthlyRBStrategy(
        Path(base_data_path),
        "Arase",
        "arase",
        "mepi",
        mag_field,
        data_standard=data_standard or ep.data_standards.GFZStandard(),
        file_format="nc",
    )

def process_arase_mepi(
    start_time: datetime,
    end_time: datetime,
    satellite: Literal["arase"] = "arase",
    mag_field: Literal["T89", "TS04", "OP77Q"] = "T89",
    raw_data_path: str | Path = ".",
    processed_data_path: str | Path = ".",
    bin_cadence: timedelta = timedelta(minutes=5),
    num_cores: int = 16,
    save_strategy: Literal["gfz", "h5", "netcdf"] = "gfz",
    data_standard: Literal["gfz", "prbem"] = "gfz",
    *,
    use_level_3_orbit_data: bool = True,
    skip_existing: bool = True,
) -> None:
    """Process Arase MEP-i Level 3 ion flux data and save derived products.

    Downloads the corresponding Arase orbit data (Level 3 if `use_level_3_orbit_data` is True,
    otherwise Level 2) and the daily Arase MEP-e Level 3 pitch-angle resolved flux (FEDU) CDF
    files for the requested time range, then sorts energies into ascending order, time-bins the
    flux, position and pitch-angle variables onto a common cadence, folds the pitch-angle
    distribution, applies a lower flux threshold, and computes the equatorial pitch angle and
    magnetic-field-related quantities (Lm, MLT, R0) either from the Level 3 orbit data directly
    or via IRBEM using the SM position from Level 2 orbit data. The resulting variables are
    saved using the requested saving strategy.

    Args:
        start_time (datetime): Start of the time range to process.
        end_time (datetime): End of the time range to process.
        satellite (Literal["arase"]): Identifier of the satellite to process. Arase is a
                                                    single-satellite mission, so this has only one value.
        mag_field (Literal["T89", "TS04", "OP77Q"]): The magnetic field model used for the
                                                    magnetic-field-related output variables.
        raw_data_path (str | Path): Directory where downloaded raw data files are
                                            stored. Defaults to ".".
        processed_data_path (str | Path): Base directory where the processed output
                                                    data is saved. Defaults to ".".
        num_cores (int): Number of CPU cores used for the IRBEM magnetic field
                                computations (only used when `use_level_3_orbit_data` is
                                False). Defaults to 4.
        bin_cadence (timedelta): Time binning cadence applied to all variables.
        save_strategy (Literal["gfz", "h5", "netcdf"]): The saving strategy used to
                                                                write the processed data.
                                                                Defaults to "gfz".
        data_standard (Literal["gfz", "prbem"]): The data standard used when saving
                                                            the processed data. Defaults to "gfz".
        use_level_3_orbit_data (bool): If True, use Arase Level 3 orbit data (which
                                                already contains precomputed magnetic field
                                                quantities for `mag_field`); if False, use Level 2
                                                orbit data and compute the magnetic field
                                                quantities via IRBEM. Defaults to True.
        skip_existing (bool): If True, skip downloading files that already exist locally.
                                            Defaults to True.
    """
    del satellite

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().setLevel(logging.INFO)

    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    if use_level_3_orbit_data:
        orb_variables = get_arase_orbit_level_3_variables(
            start_time, end_time, mag_field, raw_data_path=raw_data_path, skip_existing=skip_existing
        )
    else:
        orb_variables = get_arase_orbit_level_2_variables(
            start_time, end_time, raw_data_path=raw_data_path, skip_existing=skip_existing
        )

    file_name_stem = "erg_mepi_l3_pa_YYYYMMDD_.{6}.cdf"
    url = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/mepi/l3/pa/YYYY/MM/"

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=url,
        file_name_stem=file_name_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FIDU_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FIDU_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FPDU",
            name_or_column="FPDU",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
    ]
    # -------------------------------------------------------
    # Download Arase MGF L2 8-second magnetic field data
    # -------------------------------------------------------
    mgf_stem = "erg_mgf_l2_8sec_YYYYMMDD_.{6}.cdf"
    mgf_url = (
        "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/"
        "satellite/erg/mgf/l2/8sec/YYYY/MM/"
    )

    ep.download(
        start_time,
        end_time,
        save_path=raw_data_path,
        download_url=mgf_url,
        file_name_stem=mgf_stem,
        file_cadence="daily",
        method="request",
        skip_existing=skip_existing,
    )
    extraction_infos_mgf=[        
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="epoch_8sec",
            unit=ep.units.tt2000,
        ),
        ep.ExtractionInfo(
            result_key="B_total_obs",
            name_or_column="magt_8sec",
            unit=u.nT,
        ),
         ep.ExtractionInfo(
            result_key="Quality",
            name_or_column="quality_8sec",
            unit=u.dimensionless_unscaled,
        )]
    
    mgf_variables = ep.extract_variables_from_files(
            start_time,
            end_time,
            "daily",
            data_path=raw_data_path,
            file_name_stem=mgf_stem,
            extraction_infos=extraction_infos_mgf,
        )
    quality_flags = mgf_variables["Quality"].get_data()
    mask = (
        (quality_flags[:, 0] == 0) &
        (quality_flags[:, 1] == 0) &
        (quality_flags[:, 2] == 0) &
        (quality_flags[:, 4] == 0) &
        (quality_flags[:, 5] == 0) &
        (quality_flags[:, 7] == 0) &
        (quality_flags[:, 15] == 0) &
        (quality_flags[:, 3] < 21)
    )
    B_obs = mgf_variables["B_total_obs"].get_data(u.nT).copy()

    mask &= np.isfinite(B_obs) & (B_obs > 0)

    B_obs[~mask] = np.nan

    mgf_variables["B_total_obs"].set_data(B_obs, "same")

    mgf_variables["B_total_obs"].truncate(
        mgf_variables["Epoch"], start_time, end_time
    )
    mgf_variables["Epoch"].truncate(
        mgf_variables["Epoch"], start_time, end_time
    )

    mepi_variables = ep.extract_variables_from_files(
        start_time,
        end_time,
        "daily",
        data_path=raw_data_path,
        file_name_stem=file_name_stem,
        extraction_infos=extraction_infos,
    )

    mepi_variables["FPDU"].truncate(mepi_variables["Epoch"], start_time, end_time)
    mepi_variables["Epoch"].truncate(mepi_variables["Epoch"], start_time, end_time)

    # sort energies into ascending order
    idx_sorted = np.argsort(mepi_variables["Energy"].get_data())
    mepi_variables["Energy"].set_data(mepi_variables["Energy"].get_data()[idx_sorted[:-1]], "same")
    mepi_variables["FPDU"].set_data(mepi_variables["FPDU"].get_data()[:, idx_sorted[:-1], :], "same")

    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
        "Energy": ep.TimeBinMethod.Repeat,
        "FPDU": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    binned_time_variable = ep.processing.bin_by_time(
        mepi_variables["Epoch"],
        variables=mepi_variables,
        time_bin_method_dict=time_bin_methods,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )
    time_bin_methods_mgf = {

        "B_total_obs": ep.TimeBinMethod.NanMedian,
    }

    binned_time_variable_mgf = ep.processing.bin_by_time(
        mgf_variables["Epoch"],
        variables=mgf_variables,
        time_bin_method_dict=time_bin_methods_mgf,
        time_binning_cadence=bin_cadence,
        start_time=start_time,
        end_time=end_time,
    )

    

    ep.processing.fold_pitch_angles_and_flux(mepi_variables["FPDU"], mepi_variables["Pitch_angle"])

    mepi_variables["FPDU"].apply_thresholds_on_data(lower_threshold=1e-21)
    
    if use_level_3_orbit_data:
        time_bin_methods = {
            "B_local": ep.TimeBinMethod.NanMedian,
            "B_eq": ep.TimeBinMethod.NanMedian,
            "Lm": ep.TimeBinMethod.NanMean,
            "Lstar": ep.TimeBinMethod.NanMean,
            "MLT": ep.TimeBinMethod.NanMean,
            "R0": ep.TimeBinMethod.NanMean,
        }

        binned_time_variable = ep.processing.bin_by_time(
            orb_variables["Epoch"],
            variables=orb_variables,
            time_bin_method_dict=time_bin_methods,
            time_binning_cadence=bin_cadence,
            start_time=start_time,
            end_time=end_time,
        )

        pa_local = mepi_variables["Pitch_angle"].get_data(u.radian)
        pa_eq = np.asin(
            np.sin(pa_local)
            * np.sqrt(orb_variables["B_eq"].get_data(u.nT) / orb_variables["B_local"].get_data(u.nT))[:, np.newaxis]
        )
        mepi_variables["Pa_eq"] = ep.Variable(data=pa_eq, original_unit=u.radian)
    else:
        time_bin_methods = {
            "pos_sm": ep.TimeBinMethod.NanMean,
        }

        binned_time_variable = ep.processing.bin_by_time(
            orb_variables["Epoch"],
            variables=orb_variables,
            time_bin_method_dict=time_bin_methods,
            time_binning_cadence=bin_cadence,
            start_time=start_time,
            end_time=end_time,
        )
 

        datetimes = [
            datetime.fromtimestamp(t, tz=timezone.utc) for t in binned_time_variable.get_data(ep.units.posixtime)
        ]

        geo_data = Coords().transform(
            time=datetimes,
            pos=orb_variables["pos_sm"].get_data().astype(np.float64),
            sysaxes_in=ep.IRBEM_SYSAXIS_SM,
            sysaxes_out=ep.IRBEM_SYSAXIS_GEO,
        )
        pos_geo_var = ep.Variable(data=geo_data, original_unit=ep.units.RE)

        irbem_options = ep.processing.magnetic_field_utils.IrbemOptions()

        variables_to_compute: ep.processing.VariableRequest = [
            ("B_Calc", mag_field),
            ("MLT", mag_field),
            ("B_Eq", mag_field),
            ("R_Eq", mag_field),
            ("Alpha_Eq", mag_field),
            ("L_m", mag_field),
            ("L_star", mag_field),
            ("InvK", mag_field),
            ("InvMu", mag_field),
        ]


        magnetic_field_variables = ep.processing.compute_magnetic_field_variables(
            time_var=binned_time_variable,
            xgeo_var=pos_geo_var,
            variables_to_compute=variables_to_compute,
            irbem_options=irbem_options,
            num_cores=num_cores,
            pa_local_var=mepi_variables["Pitch_angle"],
            energy_var=mepi_variables["Energy"],
            particle_species="proton",
        )

        orb_variables["R0"] = magnetic_field_variables["R_Eq_" + mag_field]
        orb_variables["MLT"] = magnetic_field_variables["MLT_" + mag_field]
        mepi_variables["Pa_eq"] = magnetic_field_variables["Alpha_Eq_" + mag_field]
        orb_variables["Lm"] = magnetic_field_variables["L_m_" + mag_field]
        orb_variables["Lstar"]=magnetic_field_variables["L_star_" + mag_field]
        orb_variables["InvK"] = magnetic_field_variables["InvK_" + mag_field]
        orb_variables["InvMu"] = magnetic_field_variables["InvMu_" + mag_field]
        orb_variables["B_local"] = magnetic_field_variables["B_Calc_" + mag_field]
        orb_variables["B_eq"] = magnetic_field_variables["B_Eq_" + mag_field]
        # mu_scale=orb_variables["B_local"].get_data(u.nT)/mgf_variables["B_total_obs"].get_data(u.nT)
        # mu_real=orb_variables["InvMu"].get_data()*mu_scale[:, np.newaxis, np.newaxis]
        # mgf_variables["Mu_real"] = ep.Variable(data=mu_real, original_unit=u.MeV / u.G)
        #breakpoint()
    match mag_field:
        case "T89":
            mag_field_save = "T89"
        case "TS04":
            mag_field_save = "T04s"
        case "OP77Q":
            mag_field_save = "OP77"

    data_standard_instance = (
        ep.data_standards.GFZStandard() if data_standard == "gfz" else ep.data_standards.PRBEMStandard()
    )
    psd_var = ep.processing.compute_phase_space_density(
        mepi_variables["FPDU"], mepi_variables["Energy"], particle_species="proton"
    )
    variables_to_save: dict[ep.typing.InternalName, ep.Variable] = {
        "Epoch": binned_time_variable,
        "FPDU": mepi_variables["FPDU"],
        "Energy_FPDU": mepi_variables["Energy"],
        "Alpha": mepi_variables["Pitch_angle"],
        "Alpha_Eq": mepi_variables["Pa_eq"],
        "R_Eq": orb_variables["R0"],
        "MLT": orb_variables["MLT"],
        "L_m": orb_variables["Lm"],
        "B_Calc": orb_variables["B_local"],
        "B_Eq": orb_variables["B_eq"],
        "B_total_obs": mgf_variables["B_total_obs"],
        "L_star": orb_variables["Lstar"],
        "InvK": orb_variables["InvK"],
        "InvMu": orb_variables["InvMu"],
        "PSD": psd_var,
    }
    # if not use_level_3_orbit_data:
    #     variables_to_save |= {
    #         "InvK": orb_variables["InvK"],
    #         "InvMu": orb_variables["InvMu"],
    # }


    match save_strategy:
        case "gfz":
            saving_strategy = arase_mepi_gfz_strategy(processed_data_path, mag_field_save, data_standard_instance)

        case "h5":
            saving_strategy = arase_mepi_h5_strategy(processed_data_path, mag_field, data_standard_instance)

        case "netcdf":
            saving_strategy = arase_mepi_netcdf_strategy(processed_data_path, mag_field, data_standard_instance)

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)


if __name__ == "__main__":
    start_time = datetime(2024, 5, 10, tzinfo=timezone.utc)
    end_time = datetime(2024, 5, 14, 23, 59, 59, tzinfo=timezone.utc)
    process_arase_mepi(start_time=start_time, end_time=end_time, mag_field="T89", num_cores=32, use_level_3_orbit_data = False)
