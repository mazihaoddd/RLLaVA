# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import json
import random
import shutil
import tempfile
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed
from filelock import FileLock
from transformers import PreTrainedTokenizer, ProcessorMixin
from rllava.utils.device import is_cuda_available, is_npu_available


CHECKPOINT_TRACKER = "checkpoint_tracker.json"


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_contents = checkpoint_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def local_mkdir(path):
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):  # Add timeout
                # make a new dir
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            # Even if the lock is not acquired, try to create the directory
            os.makedirs(path, exist_ok=True)

        return path

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if is_cuda_available:
            rng_state["cuda"] = torch.cuda.get_rng_state()
        elif is_npu_available:
            rng_state["npu"] = torch.npu.get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if is_cuda_available:
            torch.cuda.set_rng_state(rng_state["cuda"])
        elif is_npu_available:
            torch.npu.set_rng_state(rng_state["npu"])


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None

    with open(tracker_file, "rb") as f:
        checkpointer_tracker_info = json.load(f)

    ckpt_path = os.path.join(path, directory_format.format(checkpointer_tracker_info["last_global_step"]))
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint does not exist: {ckpt_path}")
        return None

    print(f"Found latest checkpoint: {ckpt_path}, will resume from it. Turn off `find_last_checkpoint` to disable it.")
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, CHECKPOINT_TRACKER)


def remove_obsolete_ckpt(
    path: str, global_step: int, best_global_step: int, save_limit: int = -1, directory_format: str = "global_step_{}"
):
    """
    Remove the obsolete checkpoints that exceed the save limit.
    """
    if save_limit <= 0 or not os.path.exists(path):
        return

    num_ckpt_to_keep = save_limit - 1  # exclude the current ckpt
    pattern = re.escape(directory_format).replace(r"\{\}", r"(\d+)")
    ckpt_global_steps = []
    for folder in os.listdir(path):
        if match := re.match(pattern, folder):
            step = int(match.group(1))
            if step < global_step:
                ckpt_global_steps.append(step)

    ckpt_global_steps.sort(reverse=True)
    if best_global_step in ckpt_global_steps:  # do not remove the best ckpt
        ckpt_global_steps.remove(best_global_step)
        num_ckpt_to_keep = max(num_ckpt_to_keep - 1, 0)

    for step in ckpt_global_steps[num_ckpt_to_keep:]:
        folder_path = os.path.join(path, directory_format.format(step))
        try:
            shutil.rmtree(folder_path, ignore_errors=True)
            print(f"Removed obsolete checkpoint: {folder_path}")
        except Exception as e:
            print(f"Failed to remove {folder_path}: {e}")