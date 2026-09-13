from cs336_systems.modal_utils import app
from cs336_systems.benchmarking import BenchmarkConfig, benchmark_modal
import pandas as pd


CONFIGS = [
    BenchmarkConfig(model_size='small'),
    BenchmarkConfig(model_size='medium'),
    BenchmarkConfig(model_size='large'),
    BenchmarkConfig(model_size='xl'),
    #BenchmarkConfig(model_size='10B'),
]


@app.local_entrypoint()
def modal_main() -> None:
    print("Performing benchmarking sweep on Modal")

    results = []
    for config in CONFIGS:
        result = benchmark_modal.remote(config)
        results.append(result)

    df = pd.DataFrame(results)
    print(df[['model_size', 'num_warmup_steps', 'num_measurement_steps', 'forward_mean', 'forward_std', 'backward_mean', 'backward_std', 'optimizer_mean', 'optimizer_std']].to_markdown())



if __name__ == "__main__":
    print("Performing benchmarking sweep locally")

    results = []
    for config in CONFIGS:
        result = benchmark_modal.local(config)
        results.append(result)

    df = pd.DataFrame(results)
    print(df[['model_size', 'num_warmup_steps', 'num_measurement_steps', 'forward_mean', 'forward_std', 'backward_mean', 'backward_std', 'optimizer_mean', 'optimizer_std']].to_markdown())
