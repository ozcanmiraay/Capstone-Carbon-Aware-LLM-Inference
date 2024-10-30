# Experiment 2: Varying Pipeline Parallelism with Fixed Tensor Parallelism

import os
import subprocess
import json
from itertools import product
import shutil  # to rename the directory

# Define the parameter grid
replicas_list = [1, 2, 4, 8, 16]
pipeline_stages_list = [1, 2, 4, 8, 16]  # Varying number of pipeline stages
tensor_parallel_size = 1  # Fixed tensor parallel size based on previous experiment

# Paths
simulator_output_base = "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output"
experiment_name = "Experiment-2"
experiment_output_path = os.path.join(simulator_output_base, experiment_name)
stats_file = "simulation_stats_with_energy.json"

# Make the base experiment directory if it doesn't exist
os.makedirs(experiment_output_path, exist_ok=True)

# Function to run the simulation with specific parameters
def run_simulation(replicas, pipeline_stages, run_number):
    # Create directory for this specific run
    run_dir = os.path.join(experiment_output_path, f"Run-{run_number}")
    os.makedirs(run_dir, exist_ok=True)

    # Update the metrics_config_output_dir to point directly to run_dir
    simulation_command = [
        "/Users/mirayozcan/Documents/vidur/env/bin/python", "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--cluster_config_num_replicas", str(replicas),
        "--replica_config_tensor_parallel_size", str(tensor_parallel_size),
        "--replica_config_num_pipeline_stages", str(pipeline_stages),
        "--request_generator_config_type", "synthetic",
        "--length_generator_config_type", "trace",
        "--interval_generator_config_type", "static",
        "--trace_request_length_generator_config_max_tokens", "4096",
        "--trace_request_length_generator_config_trace_file", "./data/processed_traces/splitwise_code.csv",
        "--synthetic_request_generator_config_num_requests", "8000",
        "--replica_scheduler_config_type", "vllm",
        "--vllm_scheduler_config_batch_size_cap", "256",
        "--vllm_scheduler_config_max_tokens_in_batch", "4096",
        "--metrics_config_output_dir", run_dir  # Directly pointing to run_dir
    ]
    
    # Run the simulation
    subprocess.run(simulation_command, check=True)

    # Find the time-stamped subdirectory and rename it to "simulation-run"
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    if subdirs:
        time_stamped_dir = os.path.join(run_dir, subdirs[0])
        renamed_dir = os.path.join(run_dir, "simulation-run")
        shutil.move(time_stamped_dir, renamed_dir)  # Rename the directory
    else:
        raise FileNotFoundError(f"No time-stamped subdirectory found in {run_dir}")
    
    # Run the stats extraction
    stats_extraction_command = [
        "/Users/mirayozcan/Documents/vidur/env/bin/python", "vidur/config_optimizer/analyzer/stats_extractor_energy_carbon.py",
        "--sim-results-dir", renamed_dir  # Now point to the renamed "simulation-run" directory
    ]
    
    # Run the stats extraction and capture output
    result = subprocess.run(stats_extraction_command, cwd="/Users/mirayozcan/Documents/vidur_new/vidur_new", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error occurred while running stats extraction {run_number}: {result.stderr}")
        return None, None

    # Extract the results from the stats file
    stats_path = os.path.join(renamed_dir, "analysis", stats_file)
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
            total_energy_kwh = stats_data.get("total_energy_kwh", "N/A")
            total_carbon_emissions = stats_data.get("total_carbon_emissions_gco2eq", "N/A")
            return total_energy_kwh, total_carbon_emissions
    else:
        print(f"Warning: Stats file not found in {renamed_dir}")
        return None, None

# Run all simulations
run_number = 1
results = []
for replicas, pipeline_stages in product(replicas_list, pipeline_stages_list):
    print(f"Running simulation with {replicas} replicas and {pipeline_stages} pipeline stages...")
    energy, carbon = run_simulation(replicas, pipeline_stages, run_number)
    results.append({
        "run": run_number,
        "replicas": replicas,
        "pipeline_stages": pipeline_stages,
        "total_energy_kwh": energy,
        "total_carbon_emissions_gco2eq": carbon
    })
    run_number += 1

# Save the results to a summary file
results_summary_file = os.path.join(experiment_output_path, "experiment_results_summary.json")
with open(results_summary_file, "w") as summary_f:
    json.dump(results, summary_f, indent=4)

print("Experiment complete! Results saved in experiment_results_summary.json")
