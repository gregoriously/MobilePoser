import os
import re
import glob
import hashlib
import datetime
from pathlib import Path
from typing import Any, Optional, Iterable

from mobileposer.config import paths, amass, datasets


def file_md5(path: str) -> str:
    """MD5 of a file's contents, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_version_by_filename(run, artifact_name: str, original_filename: str,
                              type_name: str = "model"):
    """Return the artifact version whose metadata['original_filename'] matches
    `original_filename` (the local checkpoint the repo code selected), or None.

    Lightning rounds the checkpoint loss in the filename, so multiple epochs can
    print the same loss and wandb's ':best' (full-precision) may point at a
    different epoch than the repo's filename-based picker. To stay repo-faithful
    we link the exact version corresponding to the locally-selected file.
    """
    import wandb
    api = wandb.Api()
    full = f"{run.entity}/{run.project}/{artifact_name}"
    try:
        versions = api.artifact_versions(type_name, full)
    except Exception as e:
        print(f"Could not list versions of '{artifact_name}' ({e}).")
        return None
    for v in versions:
        if (v.metadata or {}).get("original_filename") == original_filename:
            return v.version
    return None


def link_and_verify_artifact(run, artifact_name: str, local_path: str,
                             match_filename: str = None,
                             aliases=("best", "latest"), file_glob="*.ckpt"):
    """Link a wandb artifact for lineage and verify it matches the local file
    the original (non-wandb) code path loads.

    Selection:
      - If `match_filename` is given (repo-faithful mode), link the artifact
        VERSION whose original_filename equals it -- i.e. the exact checkpoint the
        repo's filename-based picker chose, not necessarily wandb ':best'.
      - Otherwise link by `aliases` in order (e.g. ':best' then ':latest'), for
        non-repo-faithful use (combined-model eval).

    If no matching artifact is found, falls back to local-only with a warning (no
    lineage link). If one IS found, raises RuntimeError when its payload is not
    byte-identical to `local_path` -- guarding against linking a model different
    from what the code actually loads.
    """
    artifact = None
    if match_filename is not None:
        version = _find_version_by_filename(run, artifact_name, match_filename)
        if version is None:
            print(f"No artifact version of '{artifact_name}' matches local "
                  f"'{match_filename}'; using local file without lineage link.")
            return
        artifact = run.use_artifact(f"{artifact_name}:{version}")
        print(f"Linked artifact {artifact_name}:{version} ({match_filename}) for lineage.")
    else:
        for alias in aliases:
            try:
                artifact = run.use_artifact(f"{artifact_name}:{alias}")
                print(f"Linked artifact {artifact_name}:{alias} for lineage.")
                break
            except Exception:
                continue
        if artifact is None:
            print(f"No wandb artifact '{artifact_name}' found; using local file without lineage link.")
            return

    art_dir = artifact.download()
    art_files = list(Path(art_dir).glob(file_glob))
    if not art_files:
        raise RuntimeError(
            f"Artifact '{artifact_name}' contains no file matching '{file_glob}'; "
            f"cannot verify it against local '{local_path}'."
        )
    if file_md5(art_files[0]) != file_md5(local_path):
        raise RuntimeError(
            f"wandb artifact '{artifact_name}' ({art_files[0].name}) differs from the local "
            f"file '{Path(local_path).name}'. Refusing to proceed: the lineage-linked artifact "
            f"would not match the model the original code produces."
        )


def make_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def get_datestring():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def get_dir_number(path: str):
    return max([int(d) for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))] + [0]) + 1

def get_file_number(path: str):
    return len(glob.glob(f"{path}/*"))

def get_best_checkpoint(path: str):
    pattern = re.compile(r"epoch=\d+-validation_step_loss=([0-9.]+).ckpt")
    files = [f for f in os.listdir(path) if pattern.search(f)]
    best_ckpt = min(files, key=lambda x: float(pattern.search(x).group(1))) if files else None
    return best_ckpt
