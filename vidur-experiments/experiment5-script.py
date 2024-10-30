# Experiment 5: Impact of Varying Prefill Tokens on Energy and Carbon Emissions

import os
import subprocess
import json
import shutil  # to rename the directory
import numpy as np

# Define the range for prefill tokens (varying from 64 to 1536)
prefill_tokens_list = np.linspace(64, 3584, 25, dtype=int)

# Set the decode tokens to a constant value (e.g., 512)
decode_tokens = 512

# Paths
simulator_output_base = "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output"
experiment_name = "Experiment-5"
experiment_output_path = os.path.join(simulator_output_base, experiment_name)
stats_file = "simulation_stats_with_energy.json"

# Make the base experiment directory if it doesn't exist
os.makedirs(experiment_output_path, exist_ok=True)

# Function to convert numpy types to native Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar types to native Python types
    else:
        return obj  # Return the object as-is if it's not a numpy type

# Function to run the simulation with specific parameters
def run_simulation(prefill_tokens, run_number):
    # Create directory for this specific run
    run_dir = os.path.join(experiment_output_path, f"Run-{run_number}")
    os.makedirs(run_dir, exist_ok=True)

    # Update the metrics_config_output_dir to point directly to run_dir
    command = [
        "/Users/mirayozcan/Documents/vidur/env/bin/python", "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "2",
        "--request_generator_config_type", "synthetic",
        "--length_generator_config_type", "fixed",  # Use fixed length generator
        "--fixed_request_length_generator_config_prefill_tokens", str(prefill_tokens),  # Varying prefill tokens
        "--fixed_request_length_generator_config_decode_tokens", str(decode_tokens),  # Fixed decode tokens
        "--synthetic_request_generator_config_num_requests", "1000",
        "--replica_scheduler_config_type", "vllm",
        "--vllm_scheduler_config_batch_size_cap", "256",
        "--vllm_scheduler_config_max_tokens_in_batch", "4096",
        "--metrics_config_output_dir", run_dir
    ]

    # Run the simulation
    subprocess.run(command, check=True)

    # Find the time-stamped subdirectory and rename it to "simulation-run"
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    if subdirs:
        time_stamped_dir = os.path.join(run_dir, subdirs[0])
        renamed_dir = os.path.join(run_dir, "simulation-run")
        shutil.move(time_stamped_dir, renamed_dir)  # Rename the directory
    else:
        raise FileNotFoundError(f"No time-stamped subdirectory found in {run_dir}")

    # Create the analysis directory if it doesn't exist
    analysis_dir = os.path.join(renamed_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    if os.path.exists(os.path.join(renamed_dir, "request_metrics.csv")):
        print(f"request_metrics.csv found for run {run_number}")
    else:
        print(f"request_metrics.csv NOT found for run {run_number}")
        return None, None, None, None

    # Run the stats extraction
    stats_extraction_command = [
        "/Users/mirayozcan/Documents/vidur/env/bin/python", "vidur/config_optimizer/analyzer/stats_extractor_energy_carbon.py",
        "--sim-results-dir", renamed_dir  # Now point to the renamed "simulation-run" directory
    ]

    # Run the stats extraction and capture output
    result = subprocess.run(
        stats_extraction_command,
        cwd="/Users/mirayozcan/Documents/vidur_new/vidur_new",
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Print both stdout and stderr for debugging
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        print(f"Error occurred while running stats extraction {run_number}: {result.stderr}")
        return None, None, None, None

    # Extract the results from the stats file
    stats_path = os.path.join(analysis_dir, stats_file)
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
            total_energy_kwh = convert_numpy_types(stats_data.get("total_energy_kwh", "N/A"))
            total_carbon_emissions = convert_numpy_types(stats_data.get("total_carbon_emissions_gco2eq", "N/A"))
            total_gpu_hours = convert_numpy_types(stats_data.get("total_gpu_hrs", "N/A"))
            mfu_mean = convert_numpy_types(stats_data.get("mfu_mean", "N/A"))  # Extract MFU
            return total_energy_kwh, total_carbon_emissions, total_gpu_hours, mfu_mean
    else:
        print(f"Warning: Stats file not found in {analysis_dir}")
        return None, None, None, None

# Run all simulations
run_number = 1
results = []
for prefill_tokens in prefill_tokens_list:
    print(f"Running simulation with prefill tokens {prefill_tokens}...")
    energy, carbon, gpu_hours, mfu = run_simulation(prefill_tokens, run_number)
    results.append({
        "run": run_number,
        "prefill_tokens": int(prefill_tokens),  # Ensure it's a Python int
        "total_energy_kwh": float(energy) if energy is not None else None,  # Ensure it's a Python float
        "total_carbon_emissions_gco2eq": float(carbon) if carbon is not None else None,
        "total_gpu_hours": float(gpu_hours) if gpu_hours is not None else None,
        "mfu_mean": float(mfu) if mfu is not None else None
    })
    run_number += 1

# Save the results to a summary file
results_summary_file = os.path.join(experiment_output_path, "experiment_results_summary.json")
with open(results_summary_file, "w") as summary_f:
    json.dump(results, summary_f, indent=4)

print("Experiment complete! Results saved in experiment_results_summary.json")
