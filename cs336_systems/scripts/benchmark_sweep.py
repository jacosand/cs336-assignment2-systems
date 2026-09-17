from cs336_systems.modal_utils import app
from cs336_systems.benchmarking import BenchmarkConfig, benchmark_modal, benchmark
import pandas as pd

NUM_WARMUP_STEPS = 5

CONFIGS = [
    BenchmarkConfig(model_size='small', num_warmup_steps = NUM_WARMUP_STEPS),
    BenchmarkConfig(model_size='medium', num_warmup_steps = NUM_WARMUP_STEPS),
    BenchmarkConfig(model_size='large', num_warmup_steps = NUM_WARMUP_STEPS),
    BenchmarkConfig(model_size='xl', num_warmup_steps = NUM_WARMUP_STEPS),
    BenchmarkConfig(model_size='10B', num_warmup_steps = NUM_WARMUP_STEPS, benchmark_up_to = 'backward'),
]

RESULTS_COLS = [
    'model_size',
    'num_warmup_steps',
    'num_measurement_steps',
    'forward_mean_ms',
    'forward_std_ms',
    'backward_mean_ms',
    'backward_std_ms',
    'optimizer_mean_ms',
    'optimizer_std_ms',
]


@app.local_entrypoint()
def modal_main() -> None:
    print("Performing benchmarking sweep on Modal")

    results = []
    for config in CONFIGS:
        result = benchmark_modal.remote(config)
        results.append(result)

    df = pd.DataFrame(results)
    print(df[RESULTS_COLS].to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    print("Performing benchmarking sweep")

    results = []
    for config in CONFIGS:
        result = benchmark(config)
        results.append(result)

    df = pd.DataFrame(results)
    print(df[RESULTS_COLS].to_markdown(index=False, floatfmt=".3f"))
