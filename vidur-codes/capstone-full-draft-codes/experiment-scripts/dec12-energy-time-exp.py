import os
import subprocess
import shutil
import time

# Directory to save outputs
base_output_dir = "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output/dec12-energy-time-experiment"
os.makedirs(base_output_dir, exist_ok=True)

# Model configurations
configs = [
    {"model_name": "meta-llama/Meta-Llama-3-8B", "tensor_parallel_size": 1, "num_pipeline_stages": 1, "device": "a100"},
    {"model_name": "meta-llama/Llama-2-7b-hf", "tensor_parallel_size": 1, "num_pipeline_stages": 1, "device": "a100"},
    {"model_name": "meta-llama/Meta-Llama-3-70B", "tensor_parallel_size": 2, "num_pipeline_stages": 2, "device": "a100"},
    {"model_name": "meta-llama/Llama-2-70b-hf", "tensor_parallel_size": 2, "num_pipeline_stages": 2, "device": "a100"},
]

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
    "--synthetic_request_generator_config_num_requests 1000 "
    "--length_generator_config_type fixed "
    "--fixed_request_length_generator_config_max_tokens 1024 "
    "--interval_generator_config_type poisson "
    "--poisson_request_interval_generator_config_qps 10"
)

# Function to find the latest output directory
def get_latest_output_dir(base_path):
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

# Iterate through configurations and run commands
for config in configs:
    # Define the output directory for each model
    model_output_dir = os.path.join(base_output_dir, config["model_name"].replace("/", "_"))
    os.makedirs(model_output_dir, exist_ok=True)

    # Generate the command
    command = base_command.format(
        device=config["device"],
        model_name=config["model_name"],
        tensor_parallel_size=config["tensor_parallel_size"],
        num_pipeline_stages=config["num_pipeline_stages"],
    )

    print(f"Running simulation for {config['model_name']}...")
    print(f"Command: {command}")

    # Run the command
    subprocess.run(command, shell=True)

    # Wait briefly to ensure the output directory is written
    time.sleep(5)

    # Find the latest output directory
    default_output_dir = get_latest_output_dir("/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output")
    if default_output_dir:
        print(f"Moving results from {default_output_dir} to {model_output_dir}...")
        for file in os.listdir(default_output_dir):
            shutil.move(os.path.join(default_output_dir, file), model_output_dir)
        os.rmdir(default_output_dir)  # Remove the empty directory

    # Run the stats extraction script using the correct module path
    stats_extractor_command = f"python -m vidur.config_optimizer.analyzer.stats_extractor_energy_carbon --sim-results-dir {model_output_dir}"
    print(f"Running stats extraction for {config['model_name']}...")
    print(f"Command: {stats_extractor_command}")
    subprocess.run(stats_extractor_command, shell=True)

print(f"Simulations and stats extraction complete. Results saved in {base_output_dir}.")
