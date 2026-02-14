from typing import Optional

import numpy as np
import torch

import rllava.utils.torch_functional as VF
from rllava.data.dataset import RLHFDataset


class RLHFDatasetWithTarget(RLHFDataset):
    def __init__(
        self,
        *args,
        target_key: Optional[str] = None,
        max_target_length: int = 8192,
        sample_target_ratio: float = 1.0,
        target_list_key: Optional[str] = None,
        target_probs_key: Optional[str] = None,
        max_num_targets: int = 5,
        strip_target_think_tag: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_key = target_key
        self.max_target_length = max_target_length
        self.sample_target_ratio = sample_target_ratio
        self.target_list_key = target_list_key
        self.target_probs_key = target_probs_key
        self.max_num_targets = max_num_targets
        self.strip_target_think_tag = strip_target_think_tag

    def _normalize_target_text(self, tgt, prompt_text: Optional[str]):
        if tgt is None:
            return None
        if isinstance(tgt, list):
            tgt = tgt[0]
        if isinstance(tgt, dict) and "content" in tgt:
            tgt = tgt["content"]
        if not isinstance(tgt, str):
            return None
        if self.strip_target_think_tag and prompt_text and prompt_text.endswith("<think>\n") and tgt.startswith("<think>\n"):
            tgt = tgt[len("<think>\n"):]
        return tgt

    def __getitem__(self, index):
        example = super().__getitem__(index)

        if not self.target_key:
            return example

        if self.target_key not in self.dataset.features:
            # no target field, return original example
            example["tgt_input_ids"] = torch.zeros((self.max_target_length,), dtype=torch.long).fill_(self.tokenizer.pad_token_id)
            return example

        raw = self.dataset[index]
        tgt = raw.get(self.target_key, None)

        # sample_target_ratio
        if not (np.random.rand() < self.sample_target_ratio):
            tgt = None

        prompt_text = None
        if "raw_prompt_ids" in example:
            prompt_text = self.tokenizer.decode(example["raw_prompt_ids"], skip_special_tokens=False)

        tgt = self._normalize_target_text(tgt, prompt_text)
        if tgt is None or tgt == "":
            tgt_input_ids = torch.tensor([], dtype=torch.long)
        else:
            tgt_input_ids = self.tokenizer(
                tgt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].reshape(-1)

        # pad/truncate
        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        if tgt_input_ids.size(-1) > self.max_target_length:
            tgt_input_ids = tgt_input_ids[:, : self.max_target_length]
        tgt_input_ids = VF.pad_sequence_to_length(
            tgt_input_ids,
            max_seq_len=self.max_target_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
        ).squeeze(0)

        example["tgt_input_ids"] = tgt_input_ids
        return example


class RLHFDatasetWithTargetClevr(RLHFDatasetWithTarget):
    
    def __getitem__(self, index):
        example = super().__getitem__(index)

        if not self.target_key:
            return example

        if self.target_key not in self.dataset.features:
            # no target field, return original example
            example["tgt_input_ids"] = torch.zeros((self.max_target_length,), dtype=torch.long).fill_(self.tokenizer.pad_token_id)
            return example

        raw = self.dataset[index]
        tgt = raw.get(self.target_key, None)
        answer = raw.get(self.answer_key, None)

        tgt = tgt + "\n" + answer

        # sample_target_ratio
        if not (np.random.rand() < self.sample_target_ratio):
            tgt = None

        prompt_text = None
        if "raw_prompt_ids" in example:
            prompt_text = self.tokenizer.decode(example["raw_prompt_ids"], skip_special_tokens=False)

        tgt = self._normalize_target_text(tgt, prompt_text)
        if tgt is None or tgt == "":
            tgt_input_ids = torch.tensor([], dtype=torch.long)
        else:
            tgt_input_ids = self.tokenizer(
                tgt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].reshape(-1)

        # pad/truncate
        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        if tgt_input_ids.size(-1) > self.max_target_length:
            tgt_input_ids = tgt_input_ids[:, : self.max_target_length]
        tgt_input_ids = VF.pad_sequence_to_length(
            tgt_input_ids,
            max_seq_len=self.max_target_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
        ).squeeze(0)

        example["tgt_input_ids"] = tgt_input_ids
        return example
