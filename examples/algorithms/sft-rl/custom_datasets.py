import json
import re
from typing import Optional, Dict, Any, List

import numpy as np
import torch

import rllava.utils.torch_functional as VF
from rllava.data.dataset import RLHFDataset
from mathruler.grader import grade_answer


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


class RLHFDatasetWithTargetWemath(RLHFDatasetWithTarget):
    """Dataset for We-Math.

    Handles two data formats transparently:
    - Training set: question, answer (numerical), image — no option field
    - Test set (We-Math HF): question, option, answer (letter), image_path, ID, key

    Automatically:
    1. Merges option into prompt when the option field exists (for MCQ)
    2. Encodes ID/key metadata into ground_truth for We-Math four-dimensional evaluation
    3. Normalizes answer format for clean SFT targets (Luffy etc.)
    """

    @staticmethod
    def _normalize_answer_for_sft(answer: str, max_answer_len: int = 300) -> str:
        """Normalize answer for clean SFT target in <answer> tags.

        The goal is to teach the model a clean, consistent output format via SFT,
        while the RL reward function uses the original answer with flexible matching.

        Returns empty string to signal "skip SFT for this sample" (it still
        participates in RL with the original answer for reward matching).

        Rules:
        1. Skip overly long answers (proofs / full explanations belong in <think>)
        2. Normalize irregular multi-space/tab separation to comma format
        """
        if not answer or not answer.strip():
            return ""

        answer = answer.strip()

        # 1. Skip overly long answers — proofs, paragraphs, full explanations.
        #    These should live in <think>, not <answer>.
        if len(answer) > max_answer_len:
            return ""

        # 2. Normalize multi-space / tab separation to clean comma format.
        #    "9     4     3     2"  -> "9, 4, 3, 2"
        #    "circle     triangle" -> "circle, triangle"
        #    "35     Obtuse"       -> "35, Obtuse"
        #    "Cone     78.5"       -> "Cone, 78.5"
        if re.search(r'\S\s{2,}\S', answer):
            parts = re.split(r'\s{2,}', answer)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                answer = ', '.join(parts)

        return answer

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Normalize image fields across mixed datasets.
        # Common variants:
        # - GeoQA: "images" (list)
        # - some datasets: "image" (single image)
        # - We-Math: "image_path" (single image object/path)
        if self.image_key not in example:
            for candidate_key in ("images", "image", "image_path"):
                if candidate_key in example and example.get(candidate_key) is not None:
                    image_value = example.pop(candidate_key)
                    if candidate_key in ("image", "image_path") and not isinstance(image_value, list):
                        image_value = [image_value]
                    example[self.image_key] = image_value
                    break

        # Support mixed datasets with different prompt field names:
        # - GeoQA uses "problem"
        # - We-Math uses "question"
        prompt_field = self.prompt_key
        if prompt_field not in example:
            if "question" in example:
                prompt_field = "question"
            elif "problem" in example:
                prompt_field = "problem"
            else:
                raise KeyError(
                    f"Prompt field '{self.prompt_key}' not found in sample keys: {list(example.keys())}"
                )

        question = example[prompt_field]
        if "option" in example and example.get("option"):
            # MCQ (test set): official We-Math prompt structure with raw options
            example[self.prompt_key] = (
                "Now, we require you to solve a multiple-choice math question. "
                "Please briefly describe your thought process and provide the final answer(option).\n"
                f"Question: {question}\n"
                f"Option: {example['option']}"
            )
        else:
            # Non-MCQ (training set): plain Question: prefix
            example[self.prompt_key] = f"Question: {question}"

        return super()._build_messages(example)

    def __getitem__(self, index):
        example = super().__getitem__(index)

        # Encode We-Math metadata (ID, key) into ground_truth for four-dim evaluation.
        # Training set doesn't have these fields, so ground_truth stays as plain answer.
        raw = self.dataset[index]
        has_metadata = (
            "ID" in self.dataset.features
            and "key" in self.dataset.features
        )
        if has_metadata:
            example["ground_truth"] = json.dumps({
                "answer": example["ground_truth"],
                "ID": raw.get("ID", ""),
                "key": raw.get("key", ""),
            }, ensure_ascii=False)

        # --- Target (SFT) processing below ---
        if not self.target_key:
            return example

        if self.target_key not in self.dataset.features:
            example["tgt_input_ids"] = torch.zeros(
                (self.max_target_length,), dtype=torch.long
            ).fill_(self.tokenizer.pad_token_id)
            return example

        tgt = raw.get(self.target_key, None)
        answer = raw.get(self.answer_key, None)

        # Normalize answer for clean SFT target.
        # Original answer is preserved in ground_truth for RL reward matching.
        clean_answer = self._normalize_answer_for_sft(answer) if answer else ""
        if tgt and clean_answer:
            if "<think>" not in tgt and "</think>" not in tgt and "<answer>" not in tgt and "</answer>" not in tgt:
                tgt = "<think>" + tgt + "</think>\n" + "<answer>" + clean_answer + "</answer>"
            elif "<answer>" in tgt and "</answer>" in tgt:
                # Validate: if the answer embedded in tgt is wrong, skip SFT
                m = re.search(r'<answer>(.*?)</answer>', tgt, re.DOTALL)
                if m and answer:
                    tgt_answer = m.group(1).strip()
                    gt = answer.strip()
                    if not grade_answer(tgt_answer, gt):
                        tgt = None
        else:
            tgt = None

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
