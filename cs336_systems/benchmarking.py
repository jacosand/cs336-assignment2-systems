import timeit
import numpy as np
import torch
from dataclasses import dataclass, asdict
from cs336_basics import model, optimizer, nn_utils
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, secrets
import torch.cuda.nvtx as nvtx
import math
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

from cs336_basics.nn_utils import softmax


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


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    with nvtx.range("applying mask"):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return result


def benchmark(config: BenchmarkConfig) -> dict[str, str | int | float]:

    model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

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

    def run_step() -> tuple[float, float | None, float | None]:

        backward_time = None
        optimizer_time = None

        t0 = timeit.default_timer()

        logits = transformer_lm(x)
        loss = nn_utils.cross_entropy(logits, y)

        synchronize(device)

        t1 = timeit.default_timer()
        forward_time = t1-t0

        if config.benchmark_up_to != "forward":

            t0 = timeit.default_timer()

            loss.backward()

            synchronize(device)

            t1 = timeit.default_timer()
            backward_time = t1-t0

        if config.benchmark_up_to == "optimizer":

            t0 = timeit.default_timer()

            nn_utils.clip_gradient(transformer_lm.parameters(), config.max_l2_norm)
            opt.step()

            synchronize(device)

            t1 = timeit.default_timer()
            optimizer_time = t1-t0

        if config.benchmark_up_to != "forward":
            opt.zero_grad(set_to_none=True)

        return forward_time, backward_time, optimizer_time

    for step in range(config.num_warmup_steps):
        run_step()

    with nvtx.range("measurement"):
        for step in range(config.num_measurement_steps):
            forward_time, backward_time, optimizer_time = run_step()

            forward_times.append(forward_time)
            if config.benchmark_up_to != "forward":
                backward_times.append(backward_time)
            if config.benchmark_up_to == "optimizer":
                optimizer_times.append(optimizer_time)

    return {
        **asdict(config),
        'd_model': model_size['d_model'],
        'num_layers': model_size['num_layers'],
        'num_heads': model_size['num_heads'],
        'd_ff': model_size['d_ff'],
        'forward_mean_ms': 1000 * float(np.mean(forward_times)),
        'forward_std_ms': 1000 * float(np.std(forward_times)),
        'backward_mean_ms': 1000 * float(np.mean(backward_times)) if backward_times else np.nan,
        'backward_std_ms': 1000 * float(np.std(backward_times)) if backward_times else np.nan,
        'optimizer_mean_ms': 1000 * float(np.mean(optimizer_times)) if optimizer_times else np.nan,
        'optimizer_std_ms': 1000 * float(np.std(optimizer_times)) if optimizer_times else np.nan,
        'device': str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=45*60)
def benchmark_modal(config: BenchmarkConfig) -> dict[str, str | int | float]:
    return benchmark(config)