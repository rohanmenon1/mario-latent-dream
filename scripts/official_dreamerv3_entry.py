from __future__ import annotations

import sys


def install_mario_env_factory(upstream) -> None:
    """Install the Mario factory while leaving non-Mario suites unchanged."""

    if getattr(upstream.make_env, "_dreamer_mario_factory", False):
        return
    original_make_env = upstream.make_env

    def make_env(config, index, **overrides):
        suite, task = config.task.split("_", 1)
        if suite != "mario":
            return original_make_env(config, index, **overrides)
        kwargs = dict(config.env.get("mario", {}))
        kwargs.update(overrides)
        kwargs["seed"] = hash((int(config.seed), int(index))) % (2**32 - 1)
        return upstream.wrap_env(MarioEmbodiedEnv(task, **kwargs), config)

    from mario_dreamer.official import MarioEmbodiedEnv

    make_env._dreamer_mario_factory = True
    upstream.make_env = make_env


def run_official_dreamerv3(argv: list[str]) -> None:
    """Run upstream DreamerV3 with only the Mario environment factory replaced."""

    from dreamerv3 import main as upstream

    install_mario_env_factory(upstream)
    upstream.main(argv)


if __name__ == "__main__":
    run_official_dreamerv3(sys.argv[1:])
