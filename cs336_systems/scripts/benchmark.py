import argparse
import sys
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, secrets
from cs336_systems.benchmarking import BenchmarkConfig, benchmark


def parse_args(arglist: tuple[str, ...] | list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description = "Benchmark a transformer language model")

    defaults = BenchmarkConfig()

    # Model arguments
    parser.add_argument("--model-size", type=str, choices=["small", "medium", "large", "xl", "10B"], default=defaults.model_size)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--context-length", type=int, default=defaults.context_length)
    parser.add_argument("--rope-theta", type=float, default=defaults.rope_theta)

    # Optimizer arguments
    parser.add_argument("--beta1", type=float, default=defaults.beta1)
    parser.add_argument("--beta2", type=float, default=defaults.beta2)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-l2-norm", type=float, default=defaults.max_l2_norm)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)

    # Benchmarking parameters
    parser.add_argument("--num-warmup-steps", type=int, default=defaults.num_warmup_steps)
    parser.add_argument("--num-measurement-steps", type=int, default=defaults.num_measurement_steps)
    parser.add_argument("--benchmark-up-to", type=str, choices=["forward", "backward", "optimizer"], default=defaults.benchmark_up_to)

    args = parser.parse_args(args = arglist)

    return BenchmarkConfig(**vars(args))


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=45*60)
def benchmark_lm(config: BenchmarkConfig) -> dict[str, str | int | float]:
    return benchmark(config)


@app.local_entrypoint()
def modal_main(*arglist: str) -> None:
    print("Benchmarking LM on Modal")
    config = parse_args(arglist)
    result = benchmark_lm.remote(config)
    print(result)


if __name__ == "__main__":
    print("Benchmarking LM locally")
    config = parse_args(sys.argv[1:])
    result = benchmark_lm.local(config)
    print(result)