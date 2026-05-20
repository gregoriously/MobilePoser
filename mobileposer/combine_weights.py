"""
Combine network weights into a single weight file. 
"""

import os
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import wandb

from mobileposer.models import MobilePoserNet, Poser, Joints, Velocity, FootContact
from mobileposer.constants import MODULES
from mobileposer.config import wandb_config
from mobileposer.utils.file_utils import get_file_number, get_best_checkpoint, link_and_verify_artifact


def load_module_weights(module_name, weight_path):
    try:
        return MODULES[module_name].load_from_checkpoint(weight_path)
    except Exception as e:
        print(f"Error loading {module_name} weights from {weight_path}: {e}")
        return None


def get_module_path(module_name, checkpoint, finetune=None):
    module_path = Path("checkpoints") / str(checkpoint)
    if args.finetune and module_name in ["poser", "joints"]:
        module_path = module_path / f"finetuned_{finetune}" / module_name
    else:
        module_path = module_path / module_name
    return module_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", nargs="+", help="List of weight paths.")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number.", default=1) 
    args = parser.parse_args()

    # start a wandb run for the combine step
    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        group=f"exp-{args.checkpoint}",
        job_type="combine",
        name=f"combine-{args.finetune}" if args.finetune else "combine-base",
        tags=["combine"] + ([args.finetune] if args.finetune else []),
        config={"checkpoint": args.checkpoint, "finetune": args.finetune},
    )

    checkpoints = {}
    for module_name in MODULES.keys():
        module_path = get_module_path(module_name, args.checkpoint, args.finetune)
        best_ckpt = get_best_checkpoint(module_path)
        if best_ckpt:
            ckpt_file = module_path / best_ckpt
            checkpoints[module_name] = load_module_weights(module_name, ckpt_file)
            print(f"Module: {module_name.ljust(15)} | Best Checkpoint: {best_ckpt}")
            # establish lineage from the module training runs, if enabled.
            # verifies the artifact is byte-identical to the local best checkpoint
            # we actually load above; raises if they differ.
            if wandb_config.use_artifacts:
                stage = f"finetune_{args.finetune}" if args.finetune and module_name in ["poser", "joints"] else "train"
                link_and_verify_artifact(run, f"model-{module_name}-{stage}", ckpt_file,
                                         aliases=("best", "latest"), file_glob="*.ckpt")
        else:
            print(f"No checkpoint found for {module_name} in {module_path}")

    # load combined model and save
    #model_name = "base_model.pth" if not args.finetune else "model_finetuned.pth"
    model_name = "base_model.pth" if not args.finetune else f"model_finetuned_{args.finetune}.pth"
    model = MobilePoserNet(**checkpoints)
    model_path = Path("checkpoints") / str(args.checkpoint) / model_name
    torch.save(model.state_dict(), model_path)
    print(f"Model written to {model_path}.")

    # log the combined model as a new artifact
    if wandb_config.log_model:
        artifact = wandb.Artifact(
            name=Path(model_name).stem,
            type="model",
            metadata={"checkpoint": args.checkpoint, "finetune": args.finetune,
                      "modules": list(checkpoints.keys())},
        )
        artifact.add_file(str(model_path))
        run.log_artifact(artifact)

    wandb.finish()
