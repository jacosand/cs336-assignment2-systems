import argparse
import sys
import time
import numpy as np
import torch
from cs336_basics import model, optimizer, nn_utils
from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, secrets


VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
NUM_ITERATIONS = 10

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

def parse_args(arglist: tuple[str, ...] | list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train a transformer language model")

    # Model arguments
    parser.add_argument("--model-size", type=str, choices=["small", "medium", "large", "xl", "10B"])
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    #parser.add_argument("--d-model", type=int, default=512)
    #parser.add_argument("--num-layers", type=int, default=4)
    #parser.add_argument("--num-heads", type=int, default=16)
    #parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000)

    # Optimizer arguments
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-learning-rate", type=float, default=4e-3)
    parser.add_argument("--min-learning-rate", type=float, default=4e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10_000)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--seed", type=int, default=336)

    return parser.parse_args(args = arglist)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def benchmark(args: argparse.Namespace) -> None:

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")

    seed_everything(args.seed)

    model_size = MODEL_CONFIGS[args.model_size]

    transformer_lm = model.BasicsTransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = model_size['d_model'],
        num_layers = model_size['num_layers'],
        num_heads = model_size['num_heads'],
        d_ff = model_size['d_ff'],
        rope_theta = args.rope_theta,
    ).to(device)

    opt = optimizer.AdamW(
        transformer_lm.parameters(),
        lr = args.max_learning_rate,
        betas = (args.beta1, args.beta2),
        weight_decay = args.weight_decay,
    )

    transformer_lm.train()

    for step in range(1, args.num_iterations + 1):

        t0 = time.perf_counter()
        lr = optimizer.get_cosine_lr(step, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for group in opt.param_groups:
            group["lr"] = lr
        
        random_tokens = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length + 1), device=device)
        x = random_tokens[:,:-1]
        y = random_tokens[:, 1:]

        opt.zero_grad()
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = transformer_lm(x)
                loss = nn_utils.cross_entropy(logits, y)
        else:
            logits = transformer_lm(x)
            loss = nn_utils.cross_entropy(logits, y)
        loss.backward()
        grad_norm = nn_utils.clip_gradient(transformer_lm.parameters(), args.max_l2_norm)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        dt = time.perf_counter() - t0
        print(dt)


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=45*60)
def benchmark_lm(*arglist: str) -> None:
    args = parse_args(arglist)
    benchmark(args)


@app.local_entrypoint()
def modal_main(*arglist: str) -> None:
    print("Benchmarking LM on Modal")
    benchmark_lm.remote(*arglist)


if __name__ == "__main__":
    print("Benchmarking LM locally")
    benchmark_lm.local(*sys.argv[1:])