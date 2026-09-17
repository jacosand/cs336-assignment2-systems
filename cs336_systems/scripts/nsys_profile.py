import subprocess
import argparse
from dataclasses import dataclass
from cs336_systems.modal_utils import VOLUME_MOUNTS, PROFILE_PATH, app, build_image, secrets


CONTEXT_LENGTH = 512

@dataclass
class ProfileConfig:
    model_size: str = "small"
    context_length: int = CONTEXT_LENGTH


def parse_args(arglist: tuple[str, ...] | list[str] | None = None) -> ProfileConfig:
    parser = argparse.ArgumentParser(description = "Profile a transformer language model")

    defaults = ProfileConfig()

    parser.add_argument("--model-size", type=str, choices=["small", "medium", "large", "xl", "10B"], default=defaults.model_size)
    parser.add_argument("--context-length", type=int, default=defaults.context_length)

    args = parser.parse_args(args = arglist)

    return ProfileConfig(**vars(args))


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=45*60)
def profile_modal(config: ProfileConfig) -> None:
    subprocess.run(
        [
            "uv", "run",
            "nsys", "profile",
            "--trace=cuda-sw,cudnn,cublas,osrt,nvtx",
            "--pytorch=functions-trace,autograd-shapes-nvtx",
            # "--cudabacktrace=all",
            # "--python-backtrace=cuda",
            "--gpu-metrics-devices=0",
            "--force-overwrite=true",
            "--capture-range=nvtx",
            "--capture-range-end=stop",
            "--nvtx-capture=measurement",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
            f"--output=/root/{PROFILE_PATH}/model-size-{config.model_size}_context-length-{config.context_length}",
            "--",
            "python", "-m", "cs336_systems.scripts.benchmark",
            f"--model-size={config.model_size}",
            f"--context-length={config.context_length}",
            "--num-measurement-steps=1",
        ]
    )


@app.local_entrypoint()
def modal_profile_main(*arglist: str) -> None:
    print("Profiling LM on Modal")
    config = parse_args(arglist)
    profile_modal.remote(config)