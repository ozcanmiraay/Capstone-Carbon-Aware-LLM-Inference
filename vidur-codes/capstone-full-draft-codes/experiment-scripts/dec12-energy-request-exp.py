import os
import subprocess
import json
import time
import shutil

# Directory to save outputs
base_output_dir = "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output/dec12-energy-request-experiment"
os.makedirs(base_output_dir, exist_ok=True)

# Model configurations
configs = [
    {"model_name": "meta-llama/Meta-Llama-3-8B", "tensor_parallel_size": 1, "num_pipeline_stages": 1, "device": "a100"},
    {"model_name": "meta-llama/Llama-2-7b-hf", "tensor_parallel_size": 1, "num_pipeline_stages": 1, "device": "a100"},
    {"model_name": "meta-llama/Meta-Llama-3-70B", "tensor_parallel_size": 2, "num_pipeline_stages": 2, "device": "a100"},
    {"model_name": "meta-llama/Llama-2-70b-hf", "tensor_parallel_size": 2, "num_pipeline_stages": 2, "device": "a100"},
]

# Request sizes to simulate
request_sizes = list(range(1000, 10001, 1000))  # 100 to 10,000 with step size 1000

# Base command template
base_command = (
    "python -m vidur.main "
    "--replica_config_device {device} "
    "--replica_config_model_name {model_name} "
    "--cluster_config_num_replicas 1 "
    "--replica_config_tensor_parallel_size {tensor_parallel_size} "
    "--replica_config_num_pipeline_stages {num_pipeline_stages} "
    "--replica_scheduler_config_type vllm "
    "--vllm_scheduler_config_batch_size_cap 64 "
    "--vllm_scheduler_config_max_tokens_in_batch 4096 "
    "--request_generator_config_type synthetic "
    "--synthetic_request_generator_config_num_requests {request_size} "
    "--length_generator_config_type fixed "
    "--fixed_request_length_generator_config_max_tokens 1024 "
    "--interval_generator_config_type poisson "
    "--poisson_request_interval_generator_config_qps 10"
)

# Function to find the latest output directory
def get_latest_output_dir(base_path):
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

# Initialize results dictionary
experiment_results = []

# Iterate through configurations and run commands
for config in configs:
    model_results = {"model_name": config["model_name"], "results": []}
    
    for request_size in request_sizes:
        # Define the output directory for each simulation run
        model_output_dir = os.path.join(
            base_output_dir, f"{config['model_name'].replace('/', '_')}_req_{request_size}"
        )
        os.makedirs(model_output_dir, exist_ok=True)

        # Generate the command
        command = base_command.format(
            device=config["device"],
            model_name=config["model_name"],
            tensor_parallel_size=config["tensor_parallel_size"],
            num_pipeline_stages=config["num_pipeline_stages"],
            request_size=request_size,
        )

        print(f"Running simulation for {config['model_name']} with request size {request_size}...")
        print(f"Command: {command}")
        subprocess.run(command, shell=True)

        # Wait briefly to ensure the output directory is written
        time.sleep(5)

        # Find the latest output directory
        default_output_dir = get_latest_output_dir(
            "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output"
        )
        if default_output_dir:
            print(f"Moving results from {default_output_dir} to {model_output_dir}...")
            for file in os.listdir(default_output_dir):
                shutil.move(os.path.join(default_output_dir, file), model_output_dir)
            os.rmdir(default_output_dir)  # Remove the empty directory

        # Run the stats extraction script using the correct module path
        stats_extractor_command = (
            f"python -m vidur.config_optimizer.analyzer.stats_extractor_energy_carbon --sim-results-dir {model_output_dir}"
        )
        print(f"Running stats extraction for {config['model_name']} with request size {request_size}...")
        subprocess.run(stats_extractor_command, shell=True)

        # Extract energy and carbon emission data from the stats file
        stats_file = os.path.join(model_output_dir, "analysis", "simulation_stats_with_energy.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                stats = json.load(f)
                energy_kwh = stats.get("total_energy_kwh", None)
                carbon_emissions_gco2eq = stats.get("total_carbon_emissions_gco2eq", None)
                model_results["results"].append(
                    {
                        "request_size": request_size,
                        "total_energy_kwh": energy_kwh,
                        "total_carbon_emissions_gco2eq": carbon_emissions_gco2eq,
                    }
                )
        else:
            print(f"Stats file not found for {config['model_name']} with request size {request_size}")

    experiment_results.append(model_results)

# Save the results in a separate file
results_file = os.path.join(base_output_dir, "experiment_results_summary.json")
with open(results_file, "w") as f:
    json.dump(experiment_results, f, indent=4)

print(f"Experiment complete. Results saved in {results_file}.")
