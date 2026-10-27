from cs336_systems.modal_utils import app
from cs336_systems.benchmarking import BenchmarkConfig, benchmark_modal, benchmark

NUM_WARMUP_STEPS = 5
NUM_MEASUREMENT_STEPS = 1

CONFIGS = [
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=128, benchmark_up_to='forward', mixed_precision='no', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=2048, benchmark_up_to='forward', mixed_precision='no', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=128, benchmark_up_to='optimizer', mixed_precision='no', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=2048, benchmark_up_to='optimizer', mixed_precision='no', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=128, benchmark_up_to='forward', mixed_precision='yes', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=2048, benchmark_up_to='forward', mixed_precision='yes', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=128, benchmark_up_to='optimizer', mixed_precision='yes', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
    BenchmarkConfig(model_size='xl', batch_size=2, context_length=2048, benchmark_up_to='optimizer', mixed_precision='yes', num_warmup_steps = NUM_WARMUP_STEPS, num_measurement_steps=NUM_MEASUREMENT_STEPS, memory_profile='yes'),
]

@app.local_entrypoint()
def modal_main() -> None:
    print("Performing memory profiling sweep on Modal")

    for config in CONFIGS:
        benchmark_modal.remote(config)


if __name__ == "__main__":
    print("Performing memory profiling sweep")

    for config in CONFIGS:
        benchmark(config)