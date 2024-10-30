from vidur.utils.mfu_calculator import MFUCalculator
from vidur.config import ReplicaConfig
from vidur.entities import BatchStage, Request
import sys

print(sys.path)


def test_mfu_over_time():
    # Create a ReplicaConfig for testing (adjust parameters if needed)
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        memory_margin_fraction=0.1,
        num_pipeline_stages=2,
        tensor_parallel_size=1,
        device="a100",
        network_device="a100_pairwise_nvlink"
    )

    # Initialize the MFUCalculator with the config
    mfu_calculator = MFUCalculator(replica_config)
    
    # Simulate a BatchStage with some requests
    batch_stage = BatchStage(
        batch_id=1,
        replica_id=0,
        pipeline_stage=0,
        model_execution_time=0.05,
        requests=[
            Request(arrived_at=0.0, num_prefill_tokens=100, num_decode_tokens=200, num_processed_tokens=300),
            Request(arrived_at=0.0, num_prefill_tokens=100, num_decode_tokens=200, num_processed_tokens=150)
        ],
        num_tokens=[300, 150],
        execution_time=2.0  # BatchStage will run for 2 seconds
    )

    # Define the time interval over which to calculate MFU
    interval = 0.5  # every 500ms

    # Create an empty list to store MFU values over time
    mfu_values = []

    # Simulate the time-stepped MFU calculation
    time_elapsed = 0
    while time_elapsed < batch_stage.execution_time:
        mfu = mfu_calculator.get_mfu(batch_stage, interval)
        mfu_values.append(mfu)
        print(f"MFU at time {time_elapsed} seconds: {mfu:.2f}%")
        time_elapsed += interval

    # After simulation is done, print all MFU values for validation
    print("\nMFU values over time:", mfu_values)

# Run the test function
test_mfu_over_time()
