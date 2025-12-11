import json
import itertools
from pathlib import Path

# Where to save all generated JSON configs
OUTPUT_DIR = Path("configs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Base config (values that do NOT change)
base_config = {
    "dataset_path": "../Dataset_Split",
    "model": "ResNet18",
    "batch_size": 64,
    "num_workers": 4,
    "use_wandb": True,
    "wandb_project": "age",
    "wandb_entity": "IAU_2025",
    "wandb_log_frequency": 500,
    "weight_decay": 0.001,
    "label_smoothing": 0.1,
    "early_stop_patience": 5,
}

# Parameters with multiple possible values
param_grid = {
    "epochs": [30, 50, 100],
    "lr": [0.001, 0.0005, 0.0001],
}

param_names = list(param_grid.keys())
param_values = list(param_grid.values())

# Generate all combinations
combinations = list(itertools.product(*param_values))

print(f"Generating {len(combinations)} configuration files...")

for i, combo in enumerate(combinations, start=1):
    config = base_config.copy()

    # --- Insert parameter values ---
    for name, value in zip(param_names, combo):
        config[name] = value

    # --- Unique run name ---
    run_id = f"{i:03d}"
    config["wandb_run_name"] = f"resnet18_run_{run_id}"

    # --- Unique output directory ---
    config["out_dir"] = f"../runs/resnet18_run_{run_id}"

    # Save JSON
    file_path = OUTPUT_DIR / f"config_{run_id}.json"
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)

print("Done!")