"""
Run the full MobilePoser pipeline end to end:

    1. train all modules from scratch        -> checkpoints/<N>/{poser,joints,foot_contact,velocity}
    2. finetune (joints + poser) on DIP       -> checkpoints/<N>/finetuned_dip
    3. finetune (joints + poser) on IMUPoser  -> checkpoints/<N>/finetuned_imuposer
    4. combine weights (base, dip, imuposer)  -> checkpoints/<N>/{base_model,model_finetuned_*}.pth
    5. evaluate each combined model

Each step shells out to the existing scripts (so wandb logging behaves exactly as
when run by hand). Edit the CONFIG block below, then run:  python run_pipeline.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------
# Existing checkpoint number to reuse. Set to None to train from scratch (a new
# numbered checkpoint dir is created and detected automatically).
CHECKPOINT = None

RUN_FINETUNE = True                 # run the DIP + IMUPoser finetuning stages
RUN_EVAL = True                     # run the evaluation stage

EVAL_DATASETS = ["dip"]             # any of: dip, totalcapture, imuposer
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
PY = sys.executable


def run(cmd):
    """Run a command from the repo root, streaming output; abort the pipeline on failure."""
    print("\n" + "=" * 70)
    print("$ " + " ".join(str(c) for c in cmd))
    print("=" * 70, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=os.environ, check=True)


def latest_checkpoint_number():
    """Highest numbered directory under checkpoints/ (matches train.py's get_dir_number scheme)."""
    nums = [int(d.name) for d in CHECKPOINTS.iterdir() if d.is_dir() and d.name.isdigit()]
    if not nums:
        raise RuntimeError(f"No numbered checkpoint dir found under {CHECKPOINTS}")
    return max(nums)


def train_base():
    # no --module trains all four modules; creates checkpoints/<next N>/
    run([PY, "mobileposer/train.py"])


def finetune(dataset, checkpoint, init_subdir):
    """Replicates finetune.sh: removes any existing finetuned dir, then finetunes joints + poser."""
    finetuned_dir = CHECKPOINTS / str(checkpoint) / f"finetuned_{dataset}"
    if finetuned_dir.exists():
        print(f"Removing existing {finetuned_dir}")
        shutil.rmtree(finetuned_dir)
    for module in ("joints", "poser"):
        init_from = f"checkpoints/{checkpoint}/{init_subdir}/{module}" if init_subdir \
                    else f"checkpoints/{checkpoint}/{module}"
        run([PY, "mobileposer/train.py", "--module", module,
             "--init-from", init_from, "--finetune", dataset])


def combine(checkpoint, finetune_dataset=None):
    cmd = [PY, "mobileposer/combine_weights.py", "--checkpoint", str(checkpoint)]
    if finetune_dataset:
        cmd += ["--finetune", finetune_dataset]
    run(cmd)


def evaluate(checkpoint, model_name, dataset):
    model_path = CHECKPOINTS / str(checkpoint) / model_name
    if not model_path.exists():
        print(f"Skipping eval: {model_path} does not exist.")
        return
    run([PY, "mobileposer/evaluate.py", "--model", str(model_path), "--dataset", dataset])


# 1. base training (or reuse an existing checkpoint)
if CHECKPOINT is None:
    train_base()
    checkpoint = latest_checkpoint_number()
else:
    checkpoint = CHECKPOINT
print(f"\nUsing checkpoint directory: checkpoints/{checkpoint}")

# 2 + 3. finetuning: DIP first, then IMUPoser (which inits from the DIP-finetuned weights)
if RUN_FINETUNE:
    finetune("dip", checkpoint, init_subdir=None)
    finetune("imuposer", checkpoint, init_subdir="finetuned_dip")

# 4. combine weights into the deployable models
combine(checkpoint)                       # base_model.pth
if RUN_FINETUNE:
    combine(checkpoint, "dip")            # model_finetuned_dip.pth
    combine(checkpoint, "imuposer")       # model_finetuned_imuposer.pth

# 5. evaluate each produced model on the requested datasets
if RUN_EVAL:
    models = ["base_model.pth"]
    if RUN_FINETUNE:
        models += ["model_finetuned_dip.pth", "model_finetuned_imuposer.pth"]
    for model_name in models:
        for dataset in EVAL_DATASETS:
            evaluate(checkpoint, model_name, dataset)

print(f"\nPipeline complete. Artifacts in checkpoints/{checkpoint}/")
