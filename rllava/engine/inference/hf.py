import inspect
import torch
from typing import Optional, List, Iterable, Tuple, TYPE_CHECKING
from transformers import PreTrainedTokenizer, ProcessorMixin
from contextlib import contextmanager
from .base import InferenceEngine
from .. import register_engine
from rllava.utils import torch_functional as VF
from tensordict import TensorDict
from transformers import GenerationConfig
from rllava.data.protocol import DataProto
from .base import _process_multi_modal_data, _repeat_interleave
from tqdm import tqdm


if TYPE_CHECKING:
    from rllava.ppo.config import RolloutConfig



@register_engine("hf")
class HFEngine(InferenceEngine):
    def __init__(
        self,
        model_name_or_path: str,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__(model_name_or_path, config, tokenizer, processor)
        self.model = None

    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", getattr(self.config, "top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", getattr(self.config, "top_k", 0)))  
        num_return_sequences = prompts.meta_info.get("n", getattr(self.config, "n", 1))

        input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = input_ids.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        
        # When temperature is 0.0, use greedy decoding (do_sample=False)
        # Otherwise use sampling (do_sample=True)
        do_sample = temperature > 0.0
        
        self.generation_config = GenerationConfig(
            max_new_tokens=response_length,
            do_sample=do_sample,
            num_beams=1,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            temperature=temperature if do_sample else None, 
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        self.model.eval()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                generation_config=self.generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        
        # Handle Qwen2-VL MRoPE 3D position_ids
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(generated_batch_size, 1, -1).expand(generated_batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = VF.get_response_mask(
            response_ids=response, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "response_mask": response_attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()
        self.model.train()

        # batch = TensorDict(
        #     {
        #         "prompts": input_ids,
        #         "responses": response_ids,
        #         "input_ids": sequence_ids,  # here input_ids become the whole sentences
        #         "attention_mask": attention_mask,
        #         "response_mask": response_mask,
        #         "position_ids": position_ids,
        #     },
        #     batch_size=batch_size,
        # )
        # if batch_multi_modal_data is not None:
        #     non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        # else:
        #     non_tensor_batch = {}

        # return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

        return DataProto(batch=batch)

    def generate(self, prompts: DataProto) -> DataProto:
        prompts = prompts.to(torch.cuda.current_device())
        batch_size = prompts.batch.batch_size[0]
        num_chunks = min(batch_size, max(batch_size // getattr(self.config, "micro_batch_size", batch_size), 1))
        batch_prompts = prompts.chunk(chunks=num_chunks)
        
        # Add progress bar for batch processing
        batch_prompts_iter = tqdm(
            batch_prompts, 
            desc="Generating batches", 
            disable=False,
            total=len(batch_prompts)
        )
        
        output = [self._generate_minibatch(p) for p in batch_prompts_iter]
        output = DataProto.concat(output)
        return output

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("VLLM input preprocessing size mismatch across TP ranks.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        with self.update_sampling_params(**prompts.meta_info):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def update_weights(self, model):
        pass

    def load(self, model):
        self.model = model

    def offload(self):
        self.model = None
        

    