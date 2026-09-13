import argparse
import sys
import timeit
import numpy as np
import torch
from dataclasses import dataclass, asdict
from cs336_basics import model, optimizer, nn_utils
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, secrets


VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
NUM_MEASUREMENT_STEPS = 10
NUM_WARMUP_STEPS = 5

MODEL_CONFIGS = {
    "small": {
        'd_model': 768,
        'd_ff': 3_072,
        'num_layers': 12,
        'num_heads': 12,
    },
    "medium": {
        'd_model': 1_024,
        'd_ff': 4_096,
        'num_layers': 24,
        'num_heads': 16,
    },
    "large": {
        'd_model': 1_280,
        'd_ff': 5_120,
        'num_layers': 36,
        'num_heads': 20,
    },
    "xl": {
        'd_model': 2_560,
        'd_ff': 10_240,
        'num_layers': 32,
        'num_heads': 32,
    },
    "10B": {
        'd_model': 4_608,
        'd_ff': 12_288,
        'num_layers': 50,
        'num_heads': 36,
    }
}


@dataclass
class BenchmarkConfig:
    model_size: str = "small"
    batch_size: int = BATCH_SIZE
    vocab_size: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    num_warmup_steps: int = NUM_WARMUP_STEPS
    num_measurement_steps: int = NUM_MEASUREMENT_STEPS
    benchmark_up_to: str = "optimizer"

    rope_theta: float = 10_000
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    learning_rate: float = 4e-3
    max_l2_norm: float = 1.0

    seed: int = 336


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


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(config: BenchmarkConfig) -> dict[str, str | int | float]:

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")

    seed_everything(config.seed)

    model_size = MODEL_CONFIGS[config.model_size]

    transformer_lm = model.BasicsTransformerLM(
        vocab_size = config.vocab_size,
        context_length = config.context_length,
        d_model = model_size['d_model'],
        num_layers = model_size['num_layers'],
        num_heads = model_size['num_heads'],
        d_ff = model_size['d_ff'],
        rope_theta = config.rope_theta,
    ).to(device)

    opt = optimizer.AdamW(
        transformer_lm.parameters(),
        lr = config.learning_rate,
        betas = (config.beta1, config.beta2),
        weight_decay = config.weight_decay,
    )

    transformer_lm.train()

    forward_times = []
    backward_times = []
    optimizer_times = []

    random_tokens = torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.context_length + 1), device=device)
    x = random_tokens[:,:-1]
    y = random_tokens[:, 1:]

    synchronize(device)

    for step in range(config.num_warmup_steps + config.num_measurement_steps):

        t0 = timeit.default_timer()

        logits = transformer_lm(x)
        loss = nn_utils.cross_entropy(logits, y)

        synchronize(device)

        t1 = timeit.default_timer()

        if step >= config.num_warmup_steps:
            forward_times.append(t1-t0)

        if config.benchmark_up_to != "forward":

            t0 = timeit.default_timer()

            loss.backward()

            synchronize(device)

            t1 = timeit.default_timer()

            if step >= config.num_warmup_steps:
                backward_times.append(t1-t0)

        if config.benchmark_up_to == "optimizer":

            t0 = timeit.default_timer()

            nn_utils.clip_gradient(transformer_lm.parameters(), config.max_l2_norm)
            opt.step()

            synchronize(device)

            t1 = timeit.default_timer()

            if step >= config.num_warmup_steps:
                optimizer_times.append(t1-t0)

        if config.benchmark_up_to != "forward":
            opt.zero_grad(set_to_none=True)

    return {
        **asdict(config),
        'd_model': model_size['d_model'],
        'num_layers': model_size['num_layers'],
        'num_heads': model_size['num_heads'],
        'd_ff': model_size['d_ff'],
        'forward_mean': float(np.mean(forward_times)),
        'forward_std': float(np.std(forward_times)),
        'backward_mean': float(np.mean(backward_times)) if backward_times else np.nan,
        'backward_std': float(np.std(backward_times)) if backward_times else np.nan,
        'optimizer_mean': float(np.mean(optimizer_times)) if optimizer_times else np.nan,
        'optimizer_std': float(np.std(optimizer_times)) if optimizer_times else np.nan,
        'device': str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }


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
    config = parse_args(*sys.argv[1:])
    result = benchmark_lm.local(config)
    print(result)