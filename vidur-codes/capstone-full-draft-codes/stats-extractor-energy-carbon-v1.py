import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from vidur.logger import init_logger
from vidur.config_optimizer.analyzer.constants import CPU_MACHINE_COST, GPU_COSTS

logger = init_logger(__name__)

GPU_POWER_VALUES = {
    "a100": {
        "idle": 744.8,
        "10%": 1687.5,
        "50%": 3781.8,
        "100%": 5245.0,
        "manufacturing_emissions": 135.3  # gCO2eq for manufacturing (Scope 3)
    },
    "h100": {
        "idle": 800,  # Placeholder values for H100 GPUs
        "10%": 1700,
        "50%": 4000,
        "100%": 6000,
        "manufacturing_emissions": 150.0 
    },
    "a40": {
        "idle": 700,  # Placeholder values for A40 GPUs
        "10%": 1800,
        "50%": 3000,
        "100%": 5000,
        "manufacturing_emissions": 120.0
    },
}

def interpolate_power(mfu, power_values):
    """
    Interpolate power consumption based on MFU and given power values.

    Args:
        mfu (float): Model FLOP Utilization (0 to 100)
        power_values (dict): Power values for idle, 10%, 50%, and 100% utilization.

    Returns:
        float: Interpolated power consumption in watts.
    """
    if mfu <= 0.1:
        return power_values["idle"] + (power_values["10%"] - power_values["idle"]) * (mfu / 0.1)
    elif mfu <= 0.5:
        return power_values["10%"] + (power_values["50%"] - power_values["10%"]) * ((mfu - 0.1) / 0.4)
    else:
        return power_values["50%"] + (power_values["100%"] - power_values["50%"]) * ((mfu - 0.5) / 0.5)
    

def get_gpu_power(sim_results_dir):
    """
    Retrieve power consumption values for a GPU type from the configuration file.

    Args:
        sim_results_dir (str): Directory containing simulation results and configuration.

    Returns:
        dict: Power consumption values (idle, 10%, 50%, 100%) for the GPU.
    """
    config_file = f"{sim_results_dir}/config.json"
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        gpu_type = config_data['cluster_config']['replica_config']['device']

        # Return the corresponding power values for the GPU type for unknown GPUs
        power_values = GPU_POWER_VALUES.get(gpu_type, {
            "idle": 300,
            "10%": 600,
            "50%": 1500,
            "100%": 3000
        })

        return power_values

    except Exception as e:
        logger.error(f"Error reading config file for GPU power: {e}")
        return {
            "idle": 300,
            "10%": 600,
            "50%": 1500,
            "100%": 3000
        }

def calculate_energy_consumption(gpu_hrs, power_values, mfu):
    """
    Calculate the total energy consumption based on GPU hours, interpolated power, and MFU.

    Args:
        gpu_hrs (float): Number of GPU hours.
        power_values (dict): Power consumption values (idle, 10%, 50%, 100%).
        mfu (float): Model FLOP Utilization (0 to 100).

    Returns:
        float: Energy consumption in kilowatt-hours (kWh).
    """
    effective_power = interpolate_power(mfu, power_values)  # Interpolate power
    energy_kwh = effective_power * gpu_hrs / 1000  # Convert watts to kilowatt-hours
    return energy_kwh

def calculate_total_energy_with_pue(energy_gpu, pue):
    """
    Adjust energy usage with the PUE (Power Usage Effectiveness) factor.
    """
    return energy_gpu * pue  # PUE adjusts for energy overhead in data centers

def calculate_carbon_emissions(total_energy_kwh, carbon_intensity, manufacturing_emissions, gpu_hours):
    """
    Calculate the carbon emissions based on total energy, carbon intensity, and manufacturing emissions.

    Args:
        total_energy_kwh (float): Total energy consumption in kWh.
        carbon_intensity (float): Carbon intensity (gCO2eq/kWh).
        manufacturing_emissions (float): Scope 3 emissions (gCO2eq) per GPU-hour.
        gpu_hours (float): GPU usage in hours.

    Returns:
        float: Total carbon emissions in gCO2eq.
    """
    # Scope 2 emissions
    scope_2_emissions = total_energy_kwh * carbon_intensity
    # Scope 3 emissions (manufacturing)
    scope_3_emissions = manufacturing_emissions * gpu_hours
    # Total emissions
    return scope_2_emissions + scope_3_emissions

def extract_stat_from_request_metrics(
    request_metrics_df: pd.DataFrame,
    stat_name: str,
    stat_short_name: str = None,
):
    if stat_short_name is None:
        stat_short_name = stat_name

    stats = request_metrics_df[stat_name].describe().to_dict()
    # add 95th and 99th percentile
    stats["90%"] = request_metrics_df[stat_name].quantile(0.90)
    stats["95%"] = request_metrics_df[stat_name].quantile(0.95)
    stats["99%"] = request_metrics_df[stat_name].quantile(0.99)

    stats_dict = {f"{stat_short_name}_{k}": v for k, v in stats.items()}
    return stats_dict


def extract_stats_from_cdf_df(
    cdf_df: pd.DataFrame,
    stat_name: str,
    stat_short_name: str = None,
    extract_all: bool = False,
):
    if stat_short_name is None:
        stat_short_name = stat_name

    if extract_all:
        cdf_df["cdf_rounded"] = cdf_df["cdf"].round(2)
        cdf_df = cdf_df.drop_duplicates(subset="cdf_rounded", keep="first")
        return {f"{stat_short_name}_cdf": cdf_df[stat_name].tolist()[1:]}

    percentile_map = {
        "min": 0.0,
        "25%": 0.25,
        "50%": 0.5,
        "75%": 0.75,
        "90%": 0.90,
        "95%": 0.95,
        "99%": 0.99,
        "max": 1.0,
    }
    stats = {
        k: cdf_df[cdf_df["cdf"] == v][stat_name].iloc[0]
        for k, v in percentile_map.items()
    }
    stats_dict = {f"{stat_short_name}_{k}": v for k, v in stats.items()}
    return stats_dict


def extract_utilization_stats(run_dir: str, stat_name: str):
    stat_files = glob.glob(f"{run_dir}/plots/replica_*{stat_name}.json")
    vals = []
    for stat_file in stat_files:
        stat = json.load(open(stat_file))
        for k, v in stat.items():
            if k.endswith("weighted_mean"):
                vals.append(v)

    if len(vals) == 0:
        return {f"{stat_name}_mean": np.nan}

    return {f"{stat_name}_mean": sum(vals) / len(vals)}


def process_run(run_dir: str):
    config_file = f"{run_dir}/config.json"  # Adjusted to json
    request_metrics_file = f"{run_dir}/request_metrics.csv"
    tbt_file = f"{run_dir}/plots/batch_execution_time.csv"
    ttft_file = f"{run_dir}/plots/prefill_e2e_time.csv"
    batch_size_file = f"{run_dir}/plots/batch_size.csv"
    batch_num_tokens_file = f"{run_dir}/plots/batch_num_tokens.csv"
    request_completion_time_series_file = (
        f"{run_dir}/plots/request_completion_time_series.csv"
    )

    try:
        # Load config file
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"Config file {config_file} not found.")

        # Load request metrics
        if os.path.exists(request_metrics_file):
            request_metrics_df = pd.read_csv(request_metrics_file)
        else:
            raise FileNotFoundError(f"Request metrics file {request_metrics_file} not found.")

        # Load tbt_file (optional, may not exist)
        tbt_df = pd.read_csv(tbt_file) if os.path.exists(tbt_file) else None

        # Load ttft_file (optional, may not exist)
        ttft_df = pd.read_csv(ttft_file) if os.path.exists(ttft_file) else None

        # Load batch_size_file (optional, may not exist)
        batch_size_df = pd.read_csv(batch_size_file) if os.path.exists(batch_size_file) else None

        # Load batch_num_tokens_file (optional, may not exist)
        batch_num_tokens_df = pd.read_csv(batch_num_tokens_file) if os.path.exists(batch_num_tokens_file) else None

        # Load request_completion_time_series_file (optional, may not exist)
        request_completion_time_series_df = pd.read_csv(request_completion_time_series_file) if os.path.exists(request_completion_time_series_file) else None

    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

    # Check if replica_scheduler_config exists and access the scheduler name
    if "replica_scheduler_config" in config:
        scheduler_name = config["replica_scheduler_config"].get("name", None)
        if scheduler_name == "sarathi" and config["replica_scheduler_config"].get("chunk_size", None) == 4096:
            config["replica_scheduler_config"]["name"] = "orca+"
    else:
        logger.warning("'replica_scheduler_config' not found in config")

    request_scheduling_delay_stats = extract_stat_from_request_metrics(
        request_metrics_df, "request_scheduling_delay"
    )
    request_e2e_time_normalized_stats = extract_stat_from_request_metrics(
        request_metrics_df, "request_e2e_time_normalized"
    )
    ttft_stats = extract_stat_from_request_metrics(
        request_metrics_df, "prefill_e2e_time", "ttft"
    )
    ttft_cdf = extract_stats_from_cdf_df(
        ttft_df, "prefill_e2e_time", "ttft", extract_all=True
    )
    tbt_stats = extract_stats_from_cdf_df(tbt_df, "batch_execution_time", "tbt")
    tbt_cdf = extract_stats_from_cdf_df(
        tbt_df, "batch_execution_time", "tbt", extract_all=True
    )
    batch_size_stats = extract_stats_from_cdf_df(batch_size_df, "batch_size")
    batch_size_cdf = extract_stats_from_cdf_df(
        batch_size_df, "batch_size", extract_all=True
    )
    batch_num_tokens_cdf = extract_stats_from_cdf_df(
        batch_num_tokens_df, "batch_num_tokens", extract_all=True
    )
    memory_usage_stats = extract_utilization_stats(run_dir, "memory_usage")
    mfu_stats = extract_utilization_stats(run_dir, "mfu")
    busy_time_percent_stats = extract_utilization_stats(run_dir, "busy_time_percent")
    runtime = request_completion_time_series_df["Time (sec)"].max() 

    config.update(
        {
            **request_scheduling_delay_stats,
            **request_e2e_time_normalized_stats,
            **tbt_stats,
            **ttft_stats,
            **memory_usage_stats,
            **mfu_stats,
            **busy_time_percent_stats,
            **ttft_cdf,
            **tbt_cdf,
            **batch_size_stats,
            **batch_size_cdf,
            **batch_num_tokens_cdf,
            "runtime": runtime,
        }
    ) 
    return config

def get_sim_time_from_request_completion(run_dir: str):
    request_completion_file = f"{run_dir}/plots/request_completion_time_series.csv"
    
    if os.path.exists(request_completion_file):
        request_completion_df = pd.read_csv(request_completion_file)
        return request_completion_df["Time (sec)"].max()  # Get the last timestamp in the file
    else:
        logger.warning(f"request_completion_time_series.csv not found in {run_dir}")
        return np.nan 

def process_trace(sim_results_dir: str):
    analysis_dir = f"{sim_results_dir}/analysis"

    if os.path.exists(f"{analysis_dir}/stats.csv") and os.path.exists(
        f"{analysis_dir}/simulation_stats.yml"
    ):
        return

    os.makedirs(analysis_dir, exist_ok=True)

    run_dirs = [sim_results_dir]

    num_cores = os.cpu_count() - 2

    with Pool(num_cores) as p:
        all_results = p.map(process_run, run_dirs)

    all_results = [r for r in all_results if r is not None]
    logger.info(f"Total number of runs: {len(run_dirs)} valid runs: {len(all_results)}")

    df = pd.DataFrame(all_results)

    df["num_gpus"] = (
        df["cluster_config"].apply(lambda x: x["num_replicas"])
    * df["cluster_config"].apply(lambda x: x["replica_config"]["tensor_parallel_size"])
    * df["cluster_config"].apply(lambda x: x["replica_config"]["num_pipeline_stages"])
    )
    df["cost"] = (
        df["runtime"] * df["num_gpus"] * df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS) / 3600
    )

    mfu_stats = extract_utilization_stats(sim_results_dir, "mfu")
    df["mfu_mean"] = mfu_stats["mfu_mean"]

    if "poisson_request_interval_generator_qps" in df.columns:
        df["capacity_per_dollar"] = df["poisson_request_interval_generator_qps"] / (
        df["num_gpus"] * df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS)
    )
    else:
        logger.warning("'poisson_request_interval_generator_qps' not found, skipping capacity_per_dollar calculation.")
        df["capacity_per_dollar"] = np.nan 


    df["gpu_hrs"] = df["runtime"] * df["num_gpus"] / 3600

    df["num_replica_gpus"] = (
        df["cluster_config"].apply(lambda x: x["replica_config"]["tensor_parallel_size"]) * df["cluster_config"].apply(lambda x: x["replica_config"]["num_pipeline_stages"])
    )
    df["hour_cost_per_replica"] = (
       df["cluster_config"].apply(lambda x: x["replica_config"]["device"]).map(GPU_COSTS) * df["num_replica_gpus"]
    )
    if "poisson_request_interval_generator_qps" in df.columns:
        df["capacity_per_replica"] = (
        df["poisson_request_interval_generator_qps"] / df["cluster_config"].apply(lambda x: x["num_replicas"])
    )
    else:
        logger.warning("'poisson_request_interval_generator_qps' not found, skipping capacity_per_replica calculation.")
        df["capacity_per_replica"] = np.nan 

    pue = 1.2  # PUE for California
    carbon_intensity = 350.861  # Carbon intensity for California

    power_values = get_gpu_power(sim_results_dir)
    manufacturing_emissions = power_values["manufacturing_emissions"]

    df["energy_gpu_kwh"] = df.apply(
        lambda row: calculate_energy_consumption(
            row["gpu_hrs"], power_values, row["mfu_mean"]
        ), axis=1
    )

    df["total_energy_kwh"] = df["energy_gpu_kwh"] * pue

    df["carbon_emissions_gco2eq"] = df.apply(
        lambda row: calculate_carbon_emissions(
            row["total_energy_kwh"], carbon_intensity, manufacturing_emissions, row["gpu_hrs"]
        ),
        axis=1
    )
    
    df.to_csv(f"{analysis_dir}/stats_with_energy.csv", index=False)

    gpu_cost = df["cost"].sum()
    total_gpu_hrs = df["gpu_hrs"].sum()
    mfu_mean = df["mfu_mean"].mean()

    sim_time = get_sim_time_from_request_completion(sim_results_dir)
    cpu_hrs = sim_time / 3600
    cpu_cost = cpu_hrs * CPU_MACHINE_COST

    total_gpu_hrs = df["gpu_hrs"].sum()
    total_energy_kwh = df["total_energy_kwh"].sum()
    total_carbon_emissions = df["carbon_emissions_gco2eq"].sum()

    simulation_stats = {
        "gpu_cost": gpu_cost,
        "sim_cpu_cost": cpu_cost,
        "total_gpu_hrs": total_gpu_hrs,
        "sim_time": sim_time,
        "total_runs": len(run_dirs),
        "valid_runs": len(all_results),
        "mfu_mean": mfu_mean,
        "total_energy_kwh": total_energy_kwh,
        "total_carbon_emissions_gco2eq": total_carbon_emissions,
    }

    json.dump(
        simulation_stats, open(f"{analysis_dir}/simulation_stats_with_energy.json", "w"), indent=4
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results-dir", type=str, required=True)
    args = parser.parse_args()

    process_trace(args.sim_results_dir)

if __name__ == "__main__":
    main()
