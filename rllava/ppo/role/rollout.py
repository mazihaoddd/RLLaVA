import torch
import numpy as np
from ..config import RolloutConfig
from rllava.engine import EngineFactory
from rllava.data.protocol import DataProto
from rllava.utils.model_utils import print_gpu_memory_usage
from rllava.utils import torch_functional as VF
from typing import Callable
from transformers import GenerationConfig
from copy import deepcopy



__all__ = ["Rollout"]


class Rollout():
    def __init__(self, config: RolloutConfig, reward, tokenizer, processor, workflow=None, rollout_processor=None):
        self.config = config
        self.n = config.n
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward = reward
        if workflow is not None:
            self.workflow = workflow(self.reward, self.tokenizer, self.processor, self.config.max_turns, self.config.discount, self.config.tool_config_path, self.config.env_config_path)
        else:
            self.workflow = None
        if rollout_processor is not None:
            self.rollout_processor = rollout_processor
        else:
            self.rollout_processor = None

    def initialize(self, model_path):
        self.model_path = model_path
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        """Initialize the rollout engine for sequence generation."""
        print_gpu_memory_usage(f"Before {self.config.name} rollout engine init")
        engine_class = EngineFactory(self.config.name)
        self.rollout_engine = engine_class(
            model_name_or_path=self.model_path,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        print_gpu_memory_usage(f"After {self.config.name} rollout engine init")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the rollout engine.
        
        Args:
            prompts: Input prompts for generation
            
        Returns:
            Generated sequences
        """
        if self.rollout_engine is None:
            raise RuntimeError("Rollout engine not initialized. Call init_model first.")
        
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        output = self.rollout_engine.generate(prompts=prompts)
        output = output.to("cpu")
        return output

    def generate_off_batch(self, prompts: DataProto) -> DataProto:
        """Generate sequences using off-policy targets (no sampling)."""
        if "tgt_input_ids" not in prompts.batch.keys():
            return prompts

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        tgt_input_ids = prompts.batch["tgt_input_ids"]

        pad_id = self.tokenizer.pad_token_id
        eos_id = prompts.meta_info["eos_token_id"]
        batch_size = tgt_input_ids.size(0)

        tgt_list = [trim_right_pad(tgt_input_ids[i], pad_id) for i in range(batch_size)]
        tgt_list = [t + [eos_id] if len(t) > 0 else t for t in tgt_list]
        response_length = self.config.response_length
        tgt_list = [t[:response_length] for t in tgt_list]

        response_ids = VF.pad_2d_list_to_length(
            tgt_list, pad_id, max_length=self.config.response_length
        ).to(input_ids.device)

        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        prefix_mask = torch.zeros(
            [batch_size, self.config.response_length],
            dtype=torch.bool,
            device=input_ids.device,
        )
        for i, tgt in enumerate(tgt_list):
            prefix_len = min(len(tgt), self.config.response_length)
            if prefix_len > 0:
                prefix_mask[i, :prefix_len] = True

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        batch = {
            "prompts": input_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "tgt_input_ids": tgt_input_ids,
            "prefix_mask": prefix_mask,
        }

        return DataProto.from_dict(
            tensors=batch,
            non_tensors=prompts.non_tensor_batch,
            meta_info=prompts.meta_info,
        )

    def generate_one_batch(self, data: DataProto, filter: Callable = lambda sample: sample, val=False, global_step: int | None = None) -> DataProto:  
        # uid
        import uuid
        data.non_tensor_batch["uid"] = np.array([
            str(uuid.uuid4()) for _ in range(len(data.batch))
        ], dtype=object)

        # pop keys for generation
        batch_keys = ["input_ids", "attention_mask", "position_ids"]
        if "tgt_input_ids" in data.batch.keys():
            batch_keys.append("tgt_input_ids")

        gen_batch = data.pop(
            batch_keys=batch_keys,
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        if val:
            gen_batch.meta_info = dict(self.config.val_override_config)
        else:
            gen_batch.meta_info = dict()
        if global_step is not None:
            gen_batch.meta_info["global_step"] = int(global_step)

        gen_batch.meta_info["min_pixels"] = self.config.min_pixels
        gen_batch.meta_info["max_pixels"] = self.config.max_pixels
        gen_batch.meta_info["video_fps"] = self.config.video_fps

        if self.rollout_processor is not None:
            gen_batch = self.rollout_processor.pre_process(gen_batch, self.config, self.tokenizer)

        if self.workflow is None:
        # DP handled by Accelerate's sharded dataloaders; generate normally on each rank's batch
            gen_batch_output = self.generate_sequences(gen_batch)
        else:
            # MultiTurnWorkflow: process each sample individually
            results = []
            for i in range(len(gen_batch.batch['input_ids'])):
                single_data = gen_batch[[i]]
                result = self.workflow.arun_single(self, single_data)
                results.append(result)
            gen_batch_output = DataProto.concat(results)

        if self.rollout_processor is not None:
            gen_batch_output = self.rollout_processor.post_process(gen_batch_output, gen_batch, self.config, self.tokenizer)

        if val:
            data = data.repeat(repeat_times=self.config.val_override_config.get("n", 1), interleave=True)
            new_batch = data.union(gen_batch_output)
            return new_batch
        
        if self.config.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.generate_sequences(gen_baseline_batch)
            new_batch = data.union(gen_baseline_output)
            reward_baseline_tensor, _ = self.reward.compute_rewards(new_batch)
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output
        
        # repeat to align with repeated responses in rollout
        data = data.repeat(repeat_times=self.config.n, interleave=True)
        new_batch = data.union(gen_batch_output)

        # compute rewards first
        reward_tensor, reward_metrics = self.reward.compute_rewards(new_batch)
        new_batch.batch["token_level_scores"] = reward_tensor
        # store per-sample reward metrics into batch for later slicing/aggregation
        for k, v in reward_metrics.items():
            new_batch.batch[f"reward_{k}"] = torch.tensor(v, dtype=torch.float32, device=reward_tensor.device)

        new_batch = filter(new_batch)
        return new_batch

def trim_right_pad(tokens: torch.Tensor, pad_token_id: int) -> list[int]:
    non_pad = torch.nonzero(tokens != pad_token_id, as_tuple=False)
    if len(non_pad) == 0:
        return []
    last = non_pad[-1].item()
    return tokens[: last + 1].tolist()

