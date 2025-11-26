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

from typing import Optional

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import RLHFDataset
from .data_utils import collate_fn
from .config import DataConfig
from rllava.utils.dist_utils import is_rank0



def create_dataloader(config: DataConfig, 
                      tokenizer: PreTrainedTokenizer, 
                      processor: Optional[ProcessorMixin], 
                      dist_sampler: bool = False) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.train_image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    # Sampler setup: use DistributedSampler in distributed; otherwise Random/Sequential
    if dist_sampler:
        sampler = DistributedSampler(
            dataset=train_dataset,
            shuffle=config.shuffle,
            seed=config.seed,
            drop_last=True,
        )
    else:
        if config.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(config.seed)
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        image_dir=config.val_image_dir,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = min(config.val_batch_size, len(val_dataset))

    val_sampler = None
    if dist_sampler:
        val_sampler = DistributedSampler(
            dataset=val_dataset,
            shuffle=False,
            drop_last=False,
        )

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    # Only print in main process to avoid duplicate output in multi-GPU training
    if is_rank0():
        print(f"Size of train dataloader: {len(train_dataloader)}")
        print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
