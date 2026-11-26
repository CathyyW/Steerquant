"""Fixed, method-independent RoboCasa rebuttal protocol (720 episodes)."""

from collections import Counter
import hashlib
import json
import statistics

TASKS = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnStove",
    "TurnOffStove",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
)
SEEDS = (195, 196, 197)
EPISODE_IDS = (0, 1, 10, 11, 20, 21, 30, 31, 40, 41)
SCENES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize_metadata(value):
    """Object membership tags are sets; their Python hash order has no meaning.

    Preserve all physical values, language and all other list orders exactly.
    """
    if isinstance(value, dict):
        return {
            k: sorted(v) if k == "groups_containing_sampled_obj" else normalize_metadata(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize_metadata(v) for v in value]
    return value


def make_manifest():
    episodes = [
        dict(
            key=f"{task}__seed{seed}__ep{episode:02d}",
            task=task,
            policy_seed=seed,
            episode_id=episode,
            scene_index=episode // 10,
            layout_style=list(SCENES[episode // 10]),
            env_seed=seed * episode * 256,
        )
        for seed in SEEDS
        for task in TASKS
        for episode in EPISODE_IDS
    ]
    return dict(
        version=2,
        python_hash_seed=0,
        tasks=list(TASKS),
        seeds=list(SEEDS),
        episode_ids=list(EPISODE_IDS),
        scenes=[list(s) for s in SCENES],
        episodes=episodes,
        total_episodes=720,
        initialization="Reset global RNG to env_seed before construction and again before the single reset; "
        "env_seed = policy_seed * original_episode_id * 256. "
        "Compare exact model/state/observation/metadata fingerprints across methods.",
        inference=dict(
            chunk_size=32,
            num_open_loop_steps=16,
            num_denoising_steps_action=5,
            ar_future_prediction=False,
            ar_value_prediction=False,
            num_queries_best_of_n=1,
            search_depth=1,
            obj_instance_split="B",
            randomize_cameras=False,
            env_img_res=224,
            deterministic=True,
            use_jpeg_compression=True,
            flip_images=True,
            dit_cuda_graph=True,
            task_horizons="upstream TASK_MAX_STEPS",
            stabilization_steps=10,
        ),
        note="Reduced evaluation: 24 tasks x 5 scenes x 2 trials x 3 seeds; "
        "not the official 3600-episode benchmark. Smoke episodes are excluded.",
    )


def summarize(manifest, results):
    expected = {e["key"]: e for e in manifest["episodes"]}
    if len({r["key"] for r in results}) != len(results):
        raise ValueError("Duplicate result keys")
    for r in results:
        if r["key"] not in expected or any(r[k] != expected[r["key"]][k] for k in expected[r["key"]]):
            raise ValueError("Result does not match fixed episode manifest")
        if type(r["success"]) is not bool:
            raise ValueError("Success must be a boolean")
    counts = Counter(r["policy_seed"] for r in results)
    per_seed = [
        dict(
            seed=s, episodes=counts[s], successes=sum(r["success"] for r in results if r["policy_seed"] == s)
        )
        for s in SEEDS
    ]
    for r in per_seed:
        r["success_rate"] = r["successes"] / r["episodes"] if r["episodes"] else None
    complete = len(results) == len(expected)
    rates = [r["success_rate"] for r in per_seed]
    return dict(
        complete=complete,
        episodes=len(results),
        expected_episodes=len(expected),
        successes=sum(r["success"] for r in results),
        per_seed=per_seed,
        mean_success_rate=statistics.mean(rates) if complete else None,
        seed_sample_sd=statistics.stdev(rates) if complete else None,
        missing=[key for key in expected if key not in {r["key"] for r in results}],
    )
