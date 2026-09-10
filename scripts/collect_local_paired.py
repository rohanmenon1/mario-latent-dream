from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UPSTREAM = ROOT / "third_party" / "dreamerv3"
for path in (str(SRC), str(UPSTREAM), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

ACTION_NAMES = ["NOOP", "right", "right+A", "right+B", "right+A+B"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a frozen DreamerV3 Mario trajectory and align open-loop "
            "predictions with the actual future frames. No training occurs."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "official-backup" / "agent-100k" / "ckpt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "local-paired" / "seed-251",
    )
    parser.add_argument("--seed", type=int, default=251)
    parser.add_argument("--model-size", choices=("size12m",), default="size12m")
    parser.add_argument("--context-steps", type=int, default=15)
    parser.add_argument("--horizon-steps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="Collect until the episode ends instead of stopping at --max-steps.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=1,
        help="Maximum forecast chunks to render; use 0 for every available chunk.",
    )
    return parser.parse_args()


def make_config(upstream, output_dir: Path, args: argparse.Namespace):
    import elements
    import ruamel.yaml as yaml

    configs = yaml.YAML(typ="safe").load(
        elements.Path(upstream.folder / "configs.yaml").read()
    )
    defaults = copy.deepcopy(configs["defaults"])
    defaults["jax"]["precompile"] = False
    defaults["env"]["mario"]["log_render"] = True
    config = elements.Config(defaults)
    config = config.update(configs["mario"])
    config = config.update(configs[args.model_size])
    length = args.context_steps + args.horizon_steps
    return config.update(
        {
            "logdir": str(output_dir / "runtime"),
            "seed": int(args.seed),
            "batch_size": 1,
            "batch_length": length,
            "report_length": length,
            "replay_context": 0,
            "jax.platform": "cpu",
            "jax.expect_devices": 1,
            "jax.prealloc": False,
            "jax.precompile": False,
            "agent.imag_last": 2,
        }
    )


def copy_transition(transition: dict) -> dict:
    return {
        key: np.asarray(value).copy()
        for key, value in transition.items()
        if not key.startswith("finite/")
    }


def write_video(path: Path, frames, fps: int = 15) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def load_local_checkpoint(agent, step, checkpoint_dir: Path) -> None:
    """Load an Elements checkpoint without its Windows path-comparison bug."""

    with (checkpoint_dir / "agent.pkl").open("rb") as file:
        agent.load(pickle.load(file))
    with (checkpoint_dir / "step.pkl").open("rb") as file:
        step.load(pickle.load(file))


def main() -> None:
    args = parse_args()
    if args.context_steps <= 0 or args.horizon_steps <= 0:
        raise ValueError("context and horizon steps must be positive")
    if args.max_steps < args.context_steps + args.horizon_steps:
        raise ValueError("max-steps must fit at least one context-plus-horizon window")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import elements
    import embodied
    import jax
    from dreamerv3 import main as upstream

    from mario_dreamer.official.evaluation import resolve_elements_checkpoint
    from mario_dreamer.official.paired_diagnostic import (
        build_report_batch,
        compose_paired_frame,
        extract_openloop_predictions,
        forecast_window_starts,
        frame_metadata,
    )
    from official_dreamerv3_entry import install_mario_env_factory

    started = time.time()
    resolved_checkpoint = resolve_elements_checkpoint(args.checkpoint.resolve())
    install_mario_env_factory(upstream)
    config = make_config(upstream, args.output_dir, args)
    print(f"device={jax.devices()[0]}", flush=True)
    print(f"checkpoint={resolved_checkpoint}", flush=True)
    print("network_updates=0", flush=True)

    agent = upstream.make_agent(config)
    checkpoint_step = elements.Counter()
    load_local_checkpoint(agent, checkpoint_step, resolved_checkpoint)
    checkpoint_steps = int(checkpoint_step)
    print(f"checkpoint_agent_steps={checkpoint_steps}", flush=True)

    transitions: list[dict] = []
    driver = embodied.Driver(
        [lambda: upstream.make_env(config, 0)],
        parallel=False,
    )
    driver.on_step(lambda transition, _: transitions.append(copy_transition(transition)))

    def policy(*policy_args):
        return agent.policy(*policy_args, mode="eval")

    try:
        driver.reset(agent.init_policy)
        if args.full_episode:
            driver(policy, episodes=1)
        else:
            driver(policy, steps=args.max_steps)
    finally:
        driver.close()

    raw_frames = [transition["log/render"] for transition in transitions]
    write_video(args.output_dir / "gameplay.mp4", raw_frames)
    starts = forecast_window_starts(
        transitions,
        context_steps=args.context_steps,
        horizon_steps=args.horizon_steps,
    )
    if args.max_windows:
        starts = starts[: args.max_windows]
    if not starts:
        raise RuntimeError("trajectory did not contain a boundary-safe forecast window")

    paired_frames = []
    metadata_rows = []
    length = args.context_steps + args.horizon_steps
    report_batches = [
        build_report_batch(
            transitions,
            start=start,
            length=length,
            spaces=agent.spaces,
        )
        for start in starts
    ]

    def report_source():
        yield from report_batches
        while True:
            yield report_batches[-1]

    report_stream = iter(agent.stream(report_source()))
    for number, start in enumerate(starts, 1):
        print(f"rendering_window={number}/{len(starts)} start={start}", flush=True)
        report_batch = next(report_stream)
        carry = agent.init_report(1)
        _, metrics = agent.report(carry, report_batch)
        predictions = extract_openloop_predictions(
            metrics["openloop/image"],
            context_steps=args.context_steps,
            horizon_steps=args.horizon_steps,
            image_size=64,
        )
        actual_start = start + args.context_steps
        actual = raw_frames[actual_start : actual_start + args.horizon_steps]
        if len(predictions) != len(actual):
            raise RuntimeError(
                f"prediction/actual length mismatch: {len(predictions)} != {len(actual)}"
            )
        paired_frames.extend(
            compose_paired_frame(prediction, frame)
            for prediction, frame in zip(predictions, actual)
        )
        metadata_rows.extend(
            frame_metadata(
                transitions,
                window_start=start,
                context_steps=args.context_steps,
                horizon_steps=args.horizon_steps,
                output_offset=len(metadata_rows),
            )
        )

    write_video(args.output_dir / "paired.mp4", paired_frames)
    with (args.output_dir / "frames.jsonl").open("w", encoding="utf-8") as file:
        for row in metadata_rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    completed = any(bool(transition["log/flag_get"]) for transition in transitions)
    first_flag_step = next(
        (
            index
            for index, transition in enumerate(transitions)
            if bool(transition["log/flag_get"])
        ),
        None,
    )
    manifest = {
        "status": "ok",
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_agent_steps": checkpoint_steps,
        "seed": args.seed,
        "device": str(jax.devices()[0]),
        "network_updates": 0,
        "policy_mode": "eval_stochastic_categorical",
        "trajectory_steps": len(transitions),
        "episode_ended": bool(transitions[-1]["is_last"]),
        "completed_level": completed,
        "first_flag_step": first_flag_step,
        "max_x": max(float(step["log/x_pos"]) for step in transitions),
        "return": sum(float(step["reward"]) for step in transitions),
        "context_steps": args.context_steps,
        "forecast_horizon_steps": args.horizon_steps,
        "action_repeat": 4,
        "fps": 15,
        "forecast_seconds": args.horizon_steps / 15,
        "forecast_windows": len(starts),
        "paired_frames": len(paired_frames),
        "pairing": "same_trajectory_recorded_action_conditioned_open_loop",
        "reanchor_every_steps": args.horizon_steps,
        "panels": {"left": "predicted_64x64", "right": "actual_raw_256x240"},
        "gameplay_path": str(args.output_dir / "gameplay.mp4"),
        "paired_path": str(args.output_dir / "paired.mp4"),
        "frames_path": str(args.output_dir / "frames.jsonl"),
        "wall_seconds": time.time() - started,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    for key in sorted(manifest):
        print(f"{key}={manifest[key]}")


if __name__ == "__main__":
    main()
