import os
import math
import numpy as np
import torch

torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from tqdm import tqdm
import wandb

from mobileposer.constants import MODULES
from mobileposer.data import PoseDataModule
from mobileposer.utils.file_utils import (
    get_datestring,
    make_dir,
    get_dir_number,
    get_best_checkpoint,
    link_and_verify_artifact,
)
from mobileposer.config import (
    paths,
    train_hypers,
    finetune_hypers,
    wandb_config,
    poser_hypers,
    joints_hypers,
    velocity_hypers,
    footcontact_hypers,
)

# per-module hyperparameter configs, for logging the divergent knobs to wandb
MODULE_HYPERS = {
    "poser": poser_hypers,
    "joints": joints_hypers,
    "velocity": velocity_hypers,
    "foot_contact": footcontact_hypers,
}


def _hyper_dict(cls):
    """Public (non-dunder) attributes of a hypers class as a plain dict."""
    return {k: getattr(cls, k) for k in vars(cls) if not k.startswith("__")}


# set precision for Tensor cores
torch.set_float32_matmul_precision("medium")


class TrainingManager:
    """Manage training of MobilePoser modules."""

    def __init__(self, finetune: str = None, fast_dev_run: bool = False):
        self.finetune = finetune
        self.fast_dev_run = fast_dev_run
        self.hypers = finetune_hypers if finetune else train_hypers
        self.job_type = f"finetune_{finetune}" if finetune else "train"

    def _experiment_group(self, module_path: Path) -> str:
        """Experiment id shared by all modules of one pipeline run (e.g. 'exp-3').

        The <N> is the base checkpoint number, so finetune runs co-locate with the
        base run that produced them.
          base:     checkpoints/<N>/<module>                 -> <N> = parent
          finetune: checkpoints/<N>/finetuned_<ds>/<module>  -> <N> = parent.parent
        """
        n = module_path.parent.parent.name if self.finetune else module_path.parent.name
        return f"exp-{n}"

    def _setup_wandb_logger(self, module_path: Path, module_name: str):
        group = self._experiment_group(module_path)
        wandb_logger = WandbLogger(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=f"{module_name}-{self.job_type}",
            group=group,
            job_type=self.job_type,
            tags=[module_name, self.job_type],
            save_dir=module_path,
            log_model=wandb_config.log_model,
            checkpoint_name=f"model-{module_name}-{self.job_type}",
        )
        # record hyperparameters and run context in the wandb config.
        # this captures the knobs that diverge from the paper so changes are tracked per run.
        config = {
            "module": module_name,
            "finetune": self.finetune,
            "batch_size": self.hypers.batch_size,
            "num_epochs": self.hypers.num_epochs,
            "lr": self.hypers.lr,
            "grad_clip_val": self.hypers.grad_clip_val,
            "early_stopping": self.hypers.early_stopping,
        }
        if module_name in MODULE_HYPERS:
            config.update(
                {
                    f"module/{k}": v
                    for k, v in _hyper_dict(MODULE_HYPERS[module_name]).items()
                }
            )
        wandb_logger.experiment.config.update(config, allow_val_change=True)
        return wandb_logger

    def _setup_callbacks(self, save_path):
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_step_loss",
            save_top_k=3,
            mode="min",
            verbose=False,
            dirpath=save_path,
            save_weights_only=True,
            filename="{epoch}-{validation_step_loss:.4f}",
        )
        callbacks = [checkpoint_callback]
        if self.hypers.early_stopping:
            # NOTE: this is currently a no-op. The trainer below sets
            # min_epochs == max_epochs == num_epochs, so Lightning always runs the
            # full epoch count and EarlyStopping can never trigger early. The paper
            # specifies a fixed 80 epochs (no early stopping), so this is intentional;
            # to actually use early stopping, lower min_epochs in the hypers/trainer.

            # 21 May - have created control flow for trainer if finetuning for the paper-faithful branch.
            callbacks.append(
                EarlyStopping(
                    monitor="validation_step_loss",
                    mode="min",
                    patience=self.hypers.early_stopping_patience,
                )
            )
        return callbacks

    def _setup_trainer(self, module_path: Path, module_name: str):
        print("Module Path: ", module_path.name, module_path)
        logger = self._setup_wandb_logger(module_path, module_name)
        callbacks = self._setup_callbacks(module_path)
        if self.finetune is None:
            trainer = L.Trainer(
                fast_dev_run=self.fast_dev_run,
                #min_epochs=self.hypers.num_epochs,
                #max_epochs=self.hypers.num_epochs,
                devices=[self.hypers.device],
                accelerator=self.hypers.accelerator,
                gradient_clip_val=self.hypers.grad_clip_val,
                logger=logger,
                callbacks=callbacks,
                deterministic=True,
            )
        else:  # if finetune is true, min epochs not set so that we get early stopping, max so it definitely stops
            trainer = L.Trainer(
                fast_dev_run=self.fast_dev_run,
                #min_epochs=self.hypers.num_epochs,
                #max_epochs=self.hypers.num_epochs,
                devices=[self.hypers.device],
                accelerator=self.hypers.accelerator,
                gradient_clip_val=self.hypers.grad_clip_val,
                logger=logger,
                callbacks=callbacks,
                deterministic=True,
            )

        return trainer

    def _parent_artifact_name(self, module_name: str, init_from: Path) -> str:
        """Name of the training artifact this finetune run is initialized from.
        Mirrors the checkpoint_name scheme: model-<module>-<parent_stage>."""
        init_str = str(init_from)
        if "finetuned_dip" in init_str:
            parent_stage = "finetune_dip"
        elif "finetuned_imuposer" in init_str:
            parent_stage = "finetune_imuposer"
        else:
            parent_stage = "train"
        return f"model-{module_name}-{parent_stage}"

    def train_module(
        self,
        model: L.LightningModule,
        module_name: str,
        checkpoint_path: Path,
        init_from: Path = None,
        init_ckpt: str = None,
    ):
        # set the appropriate hyperparameters
        model.hypers = self.hypers

        # create directory for module
        module_path = checkpoint_path / module_name
        make_dir(module_path)
        datamodule = PoseDataModule(finetune=self.finetune)
        trainer = self._setup_trainer(module_path, module_name)

        # link lineage from the checkpoint this finetune was initialized from.
        # repo-faithful: match the exact version of the local init checkpoint
        # (init_ckpt) that from_pretrained loaded; verifies byte-identity.
        if (
            self.finetune
            and wandb_config.use_artifacts
            and init_from is not None
            and init_ckpt is not None
        ):
            link_and_verify_artifact(
                trainer.logger.experiment,
                self._parent_artifact_name(module_name, init_from),
                os.path.join(str(init_from), init_ckpt),
                match_filename=init_ckpt,
                file_glob="*.ckpt",
            )

        print()
        print("-" * 50)
        print(f"Training Module: {module_name}")
        print("-" * 50)
        print()

        try:
            trainer.fit(model, datamodule=datamodule)
        finally:
            wandb.finish()
            del model
            torch.cuda.empty_cache()


def get_checkpoint_path(finetune: str, init_from: str):
    if finetune:
        # finetune from a checkpoint
        parts = init_from.split(os.path.sep)
        checkpoint_path = Path(os.path.join(parts[0], parts[1]))
        finetune_dir = f"finetuned_{finetune}"
        checkpoint_path = checkpoint_path / finetune_dir
    else:
        # make directory for trained models
        dir_name = get_dir_number(paths.checkpoint)
        checkpoint_path = paths.checkpoint / str(dir_name)

    make_dir(checkpoint_path)
    return Path(checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--init-from", nargs="?", default="scratch", type=str)
    args = parser.parse_args()

    # set seed for reproducible results
    seed_everything(42, workers=True)

    # create checkpoint directory, if missing
    paths.checkpoint.mkdir(exist_ok=True)

    # initialize training manager
    checkpoint_path = get_checkpoint_path(args.finetune, args.init_from)
    training_manager = TrainingManager(
        finetune=args.finetune, fast_dev_run=args.fast_dev_run
    )

    # train single module
    if args.module:
        if args.module not in MODULES.keys():
            raise ValueError(f"Module {args.module} not found.")

        model_dir = Path(args.init_from)
        module = MODULES[args.module]
        model = module()  # init model from scratch

        model_path = None
        if args.finetune:
            model_path = get_best_checkpoint(model_dir)
            model = module.from_pretrained(
                model_path=os.path.join(model_dir, model_path)
            )  # load pre-trained model

        training_manager.train_module(
            model,
            args.module,
            checkpoint_path,
            init_from=model_dir if args.finetune else None,
            init_ckpt=model_path,
        )
    else:
        # train all modules
        for module_name, module in MODULES.items():
            training_manager.train_module(module(), module_name, checkpoint_path)
