import os
import sys
import subprocess
import json
import shutil
import numpy as np

# Directory for output
base_output_dir = "/Users/mirayozcan/Documents/vidur_new/vidur_new/simulator_output/dec12-max-tokens-experiment"
os.makedirs(base_output_dir, exist_ok=True)

# Define token sizes (30 evenly spaced points from 128 to 4096)
max_tokens_list = np.linspace(128, 4096, 30, dtype=int)

# Path to the trace file
trace_file = "/Users/mirayozcan/Documents/vidur_new/vidur_new/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"

# Stats file path (to be extracted after simulation)
stats_file = "simulation_stats_with_energy.json"

# Ensure all numpy objects are converted to Python-native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Function to run a single simulation
def run_simulation(max_tokens, run_number):
    print(f"Running simulation for max_tokens={max_tokens}, run={run_number}...")

    # Create output directory for this run
    run_dir = os.path.join(base_output_dir, f"Run-{run_number}")
    os.makedirs(run_dir, exist_ok=True)

    # Simulation command
    command = [
        sys.executable, "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "2",
        "--request_generator_config_type", "synthetic",
        "--length_generator_config_type", "trace",
        "--interval_generator_config_type", "static",
        "--trace_request_length_generator_config_max_tokens", str(max_tokens),
        "--trace_request_length_generator_config_trace_file", trace_file,
        "--synthetic_request_generator_config_num_requests", "8000",
        "--replica_scheduler_config_type", "vllm",
        "--vllm_scheduler_config_batch_size_cap", "256",
        "--vllm_scheduler_config_max_tokens_in_batch", "4096",
        "--metrics_config_output_dir", run_dir
    ]

    try:
        # Run the simulation
        subprocess.run(command, check=True)

        # Find and rename time-stamped directory
        subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        if subdirs:
            time_stamped_dir = os.path.join(run_dir, subdirs[0])
            renamed_dir = os.path.join(run_dir, "simulation-run")
            shutil.move(time_stamped_dir, renamed_dir)
        else:
            raise FileNotFoundError(f"No time-stamped subdirectory found in {run_dir}")

        # Run stats extraction
        analysis_dir = os.path.join(renamed_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        stats_extraction_command = [
            sys.executable, "vidur/config_optimizer/analyzer/stats_extractor_energy_carbon.py",
            "--sim-results-dir", renamed_dir
        ]

        result = subprocess.run(
            stats_extraction_command,
            cwd="/Users/mirayozcan/Documents/vidur_new/vidur_new",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Stats extraction STDOUT: {result.stdout}")
        print(f"Stats extraction STDERR: {result.stderr}")

        # Extract stats from the stats file
        stats_path = os.path.join(analysis_dir, stats_file)
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                return {
                    "max_tokens": int(max_tokens),
                    "run": int(run_number),
                    "total_energy_kwh": convert_numpy_types(stats.get("total_energy_kwh", None)),
                    "total_carbon_emissions_gco2eq": convert_numpy_types(stats.get("total_carbon_emissions_gco2eq", None)),
                    "total_gpu_hours": convert_numpy_types(stats.get("total_gpu_hrs", None)),
                    "mfu_mean": convert_numpy_types(stats.get("mfu_mean", None))
                }
        else:
            print(f"Stats file not found in {analysis_dir}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error during simulation for max_tokens={max_tokens}, run={run_number}: {e}")
        return None

# Run the simulation for all token sizes
results = []
for run_number, max_tokens in enumerate(max_tokens_list, start=1):
    result = run_simulation(max_tokens, run_number)
    if result:
        results.append(result)

# Save results to file
results_file = os.path.join(base_output_dir, "experiment_results_summary.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Experiment complete! Results saved to {results_file}.")
