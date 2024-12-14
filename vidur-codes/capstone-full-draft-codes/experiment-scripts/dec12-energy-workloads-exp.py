import os
import subprocess
import json
import time
import shutil

# Directory to save outputs
base_output_dir = "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output/dec12-energy-workloads-experiment"
os.makedirs(base_output_dir, exist_ok=True)

# Workload scenarios
workload_scenarios = [
    {"scenario_name": "customer_support", "qps": 10, "num_requests": 2000, "max_tokens": 500},
    {"scenario_name": "content_generation", "qps": 2, "num_requests": 800, "max_tokens": 1500},
    {"scenario_name": "chatbot", "qps": 50, "num_requests": 20000, "max_tokens": 100},
    {"scenario_name": "enterprise_inference", "qps": 100, "num_requests": 10000, "max_tokens": 500},
    {"scenario_name": "real_time_streaming", "qps": 500, "num_requests": 50000, "max_tokens": 50},
]

# Base command template
base_command = (
    "python -m vidur.main "
    "--replica_config_device a100 "
    "--replica_config_model_name meta-llama/Llama-2-7b-hf "
    "--cluster_config_num_replicas 1 "
    "--replica_config_tensor_parallel_size 1 "
    "--replica_config_num_pipeline_stages 1 "
    "--replica_scheduler_config_type vllm "
    "--vllm_scheduler_config_batch_size_cap 64 "
    "--vllm_scheduler_config_max_tokens_in_batch 4096 "
    "--request_generator_config_type synthetic "
    "--synthetic_request_generator_config_num_requests {num_requests} "
    "--length_generator_config_type fixed "
    "--fixed_request_length_generator_config_max_tokens {max_tokens} "
    "--interval_generator_config_type poisson "
    "--poisson_request_interval_generator_config_qps {qps} "
    "--metrics_config_output_dir {output_dir}"
)

# Function to move files from timestamped subdirectory to parent directory
def move_files_to_parent_dir(parent_dir):
    subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if subdirs:
        latest_subdir = max(subdirs, key=os.path.getmtime)  # Most recent subdir
        for file in os.listdir(latest_subdir):
            shutil.move(os.path.join(latest_subdir, file), parent_dir)
        os.rmdir(latest_subdir)

# Initialize results dictionary
experiment_results = []

# Iterate through workload scenarios
for workload in workload_scenarios:
    scenario_name = workload["scenario_name"]
    qps = workload["qps"]
    num_requests = workload["num_requests"]
    max_tokens = workload["max_tokens"]

    # Define output directory for this scenario
    scenario_output_dir = os.path.join(base_output_dir, f"{scenario_name}")
    os.makedirs(scenario_output_dir, exist_ok=True)

    # Generate the command
    command = base_command.format(
        qps=qps,
        num_requests=num_requests,
        max_tokens=max_tokens,
        output_dir=scenario_output_dir,
    )

    print(f"Running simulation for scenario '{scenario_name}' (QPS: {qps}, Requests: {num_requests}, Tokens: {max_tokens})...")
    print(f"Command: {command}")
    subprocess.run(command, shell=True)

    # Move files from timestamped subdirectory to parent directory
    move_files_to_parent_dir(scenario_output_dir)

    # Run stats extraction
    stats_file = os.path.join(scenario_output_dir, "analysis", "simulation_stats_with_energy.json")
    stats_extractor_command = (
        f"python -m vidur.config_optimizer.analyzer.stats_extractor_energy_carbon --sim-results-dir {scenario_output_dir}"
    )
    print(f"Running stats extraction for scenario '{scenario_name}'...")
    subprocess.run(stats_extractor_command, shell=True)

    # Extract energy and carbon emission data
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)
            energy_kwh = stats.get("total_energy_kwh", None)
            carbon_emissions_gco2eq = stats.get("total_carbon_emissions_gco2eq", None)
            experiment_results.append(
                {
                    "scenario_name": scenario_name,
                    "qps": qps,
                    "num_requests": num_requests,
                    "max_tokens": max_tokens,
                    "total_energy_kwh": energy_kwh,
                    "total_carbon_emissions_gco2eq": carbon_emissions_gco2eq,
                }
            )
    else:
        print(f"Stats file missing for scenario '{scenario_name}'. Skipping.")

# Save the results in a summary file
results_file = os.path.join(base_output_dir, "experiment_results_summary.json")
with open(results_file, "w") as f:
    json.dump(experiment_results, f, indent=4)

print(f"Experiment complete. Results saved in {results_file}.")
