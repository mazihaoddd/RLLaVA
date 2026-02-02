import math
import random
import torch

from rllava.data.protocol import DataProto
from rllava.utils import torch_functional as VF


class RolloutProcessor:

    def pre_process(self, prompts: DataProto, config, tokenizer) -> DataProto:
        return prompts

    def post_process(self, generated: DataProto, prompts: DataProto, config, tokenizer) -> DataProto:
        return generated


class PrefixRolloutProcessor(RolloutProcessor):
    def _trim_right_pad(self, tokens: torch.Tensor, pad_token_id: int) -> list[int]:
        non_pad = torch.nonzero(tokens != pad_token_id, as_tuple=False)
        if len(non_pad) == 0:
            return []
        last = non_pad[-1].item()
        return tokens[: last + 1].tolist()

    def _expand_to_match(self, base_list: list, target_len: int) -> list | None:
        if len(base_list) == target_len:
            return base_list
        if len(base_list) == 0 or target_len % len(base_list) != 0:
            return None
        repeat_factor = target_len // len(base_list)
        return [item for item in base_list for _ in range(repeat_factor)]

    def _sample_prefix_ratio(self, config, prompts) -> float:
        if config.prefix_strategy == "hint":
            hint_steps = max(int(config.hint_steps), 0)
            min_ratio = config.min_prefix_ratio
            max_ratio_base = config.max_prefix_ratio
            if hint_steps <= 0:
                max_ratio = max_ratio_base
            else:
                step = int(prompts.meta_info.get("global_step", 0))
                step = min(max(step, 0), hint_steps)
                max_ratio = 0.5 * (1 + math.cos(math.pi * step / hint_steps)) * max_ratio_base
            max_ratio = max(max_ratio, min_ratio)
            return random.uniform(min_ratio, max_ratio)
        return random.uniform(config.min_prefix_ratio, config.max_prefix_ratio)

    def _build_prefix_ratios(self, prefix_tokens: list, config, prompts) -> list:
        cached = prompts.meta_info.get("prefix_ratios")
        if cached is not None:
            return list(cached)

        total_samples = len(prefix_tokens)
        group_n = int(prompts.meta_info.get("orig_n", config.n))
        if group_n <= 0:
            group_n = 1

        if group_n > 1 and total_samples % group_n == 0:
            prompt_count = total_samples // group_n
        else:
            prompt_count = None

        if config.n_prefix is None:
            n_prefix = -1
        else:
            n_prefix = int(config.n_prefix)

        if prompt_count is None:
            # Fallback: sample per token list entry
            return [self._sample_prefix_ratio(config, prompts) for _ in range(total_samples)]

        prefix_ratios = []
        for _ in range(prompt_count):
            share_across = config.prefix_share_across_samples
            shared_ratio = self._sample_prefix_ratio(config, prompts) if share_across else None

            def _sample(count: int) -> list[float]:
                if count <= 0:
                    return []
                if share_across:
                    return [shared_ratio] * count
                return [self._sample_prefix_ratio(config, prompts) for _ in range(count)]

            if n_prefix >= 0:
                prefix_ratios.extend(_sample(n_prefix))
                prefix_ratios.extend([0.0] * (group_n - n_prefix))
            else:
                prefix_ratios.extend(_sample(group_n))

        return prefix_ratios

    def pre_process(self, prompts: DataProto, config, tokenizer) -> DataProto:
        if config.prefix_strategy == "none":
            return prompts
        if "tgt_input_ids" not in prompts.batch.keys():
            return prompts
        if "raw_prompt_ids" not in prompts.non_tensor_batch:
            return prompts

        if config.n > 1:
            prompts = prompts.repeat(repeat_times=config.n, interleave=True)
            prompts.meta_info["orig_n"] = config.n
            prompts.meta_info["n"] = 1

        pad_token_id = tokenizer.pad_token_id
        eos_token_id = prompts.meta_info.get("eos_token_id", tokenizer.eos_token_id)
        target_ids = prompts.batch["tgt_input_ids"]
        prefix_tokens = [self._trim_right_pad(target_ids[i], pad_token_id) for i in range(target_ids.size(0))]
        prefix_tokens = [tokens + [eos_token_id] if len(tokens) > 0 else tokens for tokens in prefix_tokens]
        prefix_ratios = self._build_prefix_ratios(prefix_tokens, config, prompts)
        prompts.meta_info["prefix_ratios"] = prefix_ratios

        raw_prompt_ids = prompts.non_tensor_batch["raw_prompt_ids"]
        # sequences_str = tokenizer.decode(raw_prompt_ids[0])
        merged_prompt_ids = []
        for base_ids, prefix_ids, ratio in zip(raw_prompt_ids, prefix_tokens, prefix_ratios):
            prefix_len = int(len(prefix_ids) * ratio)
            merged_prompt_ids.append(list(base_ids) + prefix_ids[:prefix_len])
        prompts.non_tensor_batch["raw_prompt_ids"] = merged_prompt_ids
        return prompts

    def post_process(self, generated: DataProto, prompts: DataProto, config, tokenizer) -> DataProto:
        if config.prefix_strategy == "none":
            return generated
        if "tgt_input_ids" not in prompts.batch.keys():
            return generated

        response_ids = generated.batch["responses"]
        prompt_ids = generated.batch["prompts"]
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = prompts.meta_info.get("eos_token_id", tokenizer.eos_token_id)


        # path='/data/mazihao/zhaolei/code/rl/sft-rl/Unify-Post-Training/hpt/verl/response_ids_20260130_132937_698091_idx0.npy'
        # import numpy as np
        # response_ids_1 = np.load(path)
        # path='/data/mazihao/zhaolei/code/rl/sft-rl/Unify-Post-Training/hpt/verl/response_ids_20260130_132937_712626_idx1.npy'
        # response_ids_2 = np.load(path)
        # response_ids = torch.tensor(np.stack([response_ids_1, response_ids_2], axis=0))

        response_tokens = [self._trim_right_pad(resp, pad_token_id) for resp in response_ids]
        target_ids = prompts.batch["tgt_input_ids"]
        prefix_tokens = [self._trim_right_pad(target_ids[i], pad_token_id) for i in range(target_ids.size(0))]
        prefix_tokens = [tokens + [eos_token_id] if len(tokens) > 0 else tokens for tokens in prefix_tokens]

        prefix_tokens = self._expand_to_match(prefix_tokens, len(response_tokens))
        if prefix_tokens is None:
            return generated

        prefix_ratios = self._build_prefix_ratios(prefix_tokens, config, prompts)
        prefix_ratios = self._expand_to_match(prefix_ratios, len(prefix_tokens))
        if prefix_ratios is None:
            return generated

        prefix_mask = torch.zeros(
            [len(prefix_tokens), config.response_length],
            dtype=torch.bool,
            device=response_ids.device,
        )

        merged_responses = []
        for i, (prefix_ids, response_ids_list) in enumerate(zip(prefix_tokens, response_tokens)):
            ratio = prefix_ratios[i]
            prefix_len = min(int(len(prefix_ids) * ratio), config.response_length)
            response_len = min(int(len(response_ids_list) * (1 - ratio)), config.response_length)
            concat = prefix_ids[:prefix_len] + response_ids_list[:response_len]
            merged_responses.append(concat)
            if prefix_len > 0:
                prefix_mask[i, :prefix_len] = True

        resp_max_len = max([len(resp) for resp in merged_responses])
        # new_response_ids = VF.pad_2d_list_to_length(
        #     merged_responses, pad_token_id, max_length=resp_max_len
        # ).to(response_ids.device)[:, :config.response_length]
        new_response_ids = VF.pad_2d_list_to_length(
            merged_responses, pad_token_id, max_length=config.response_length
        ).to(response_ids.device)[:, :config.response_length]

        response_length = new_response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=prompt_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(prompt_ids.size(0), -1)
        prompt_position_ids = generated.batch["position_ids"][..., : prompt_ids.size(1)]
        # prompt_position_ids = prompts.batch["position_ids"][..., : prompt_ids.size(1)]
        if prompt_position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(prompt_ids.size(0), 1, -1).expand(
                prompt_ids.size(0), prompt_position_ids.size(1), -1
            )
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        # response_length = response_ids.size(1)
        # delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        # delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        # if position_ids.dim() == 3:  # qwen2vl mrope
        #     delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # response_position_ids = position_ids[..., -1:] + delta_position_id
        # position_ids = torch.cat([position_ids, response_position_ids], dim=-1)


        eos_token_id = prompts.meta_info.get("eos_token_id", tokenizer.eos_token_id)
        response_mask = VF.get_response_mask(
            response_ids=new_response_ids, eos_token_id=eos_token_id, dtype=generated.batch["attention_mask"].dtype
        )
        # response_mask_l = response_mask.sum(dim=-1)
        prompt_attention_mask = generated.batch["attention_mask"][:, : prompt_ids.size(1)]
        attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        generated.batch["responses"] = new_response_ids
        generated.batch["input_ids"] = torch.cat([prompt_ids, new_response_ids], dim=-1)
        generated.batch["response_mask"] = response_mask
        generated.batch["attention_mask"] = attention_mask
        generated.batch["position_ids"] = position_ids
        generated.batch["prefix_mask"] = prefix_mask
        generated.meta_info["prefix_ratios"] = prefix_ratios
        return generated
