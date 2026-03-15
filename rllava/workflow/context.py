"""Context builder for multi-turn token-first state management.

Maintains the exact prompt token chain used by vLLM generation and by the
training-side forward pass. Assistant turns are appended by exact sampled token
IDs, while external turns (tool / environment) are appended as template deltas.

The only remaining full re-encode path is turn-0 initial observation injection,
which uses a one-shot dataset-style re-encode from preserved
``initial_prompt_text`` to compute multimodal expansion and mRoPE position IDs.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rllava.data.data_utils import process_image
from rllava.data.protocol import DataProto
import rllava.utils.torch_functional as VF


class ContextBuilder:
    """Manage per-session message/image state and materialize model inputs."""

    def __init__(self, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        self._messages: List[Dict[str, Any]] = []
        self._images: list = []
        self._meta_info: dict = {}
        self._engine_prompt_ids: List[int] = []
        self._train_input_ids: List[int] = []
        self._engine_multi_modal_data: Dict[str, list] = {}
        self._train_position_ids: Optional[torch.Tensor] = None
        self._derived_dirty: bool = False

    # ================================================================
    # Session initialisation
    # ================================================================

    def start_from_data(self, data: DataProto) -> DataProto:
        """Initialise context from exact dataset-provided token IDs.

        Unlike the previous message-first initialisation, this method does
        **not** decode and re-encode ``raw_prompt_ids``. The dataset-provided
        IDs become the session's initial token truth, completely bypassing the
        tokeniser encode / decode asymmetry at turn 0.

        Call :meth:`inject_initial_observation` afterwards when the
        environment provides an initial screenshot at reset time.
        """
        self._meta_info = dict(data.meta_info) if data.meta_info else {}

        # ---- exact engine-side IDs ----
        raw = data.non_tensor_batch.get("raw_prompt_ids")
        self._engine_prompt_ids = (
            list(raw[0]) if raw is not None and len(raw) > 0 else []
        )

        # ---- exact train-side IDs (strip left-padding) ----
        input_ids_t = data.batch["input_ids"][0]
        attn_mask_t = data.batch["attention_mask"][0]
        valid = attn_mask_t.bool()
        self._train_input_ids = input_ids_t[valid].tolist()

        # ---- multimodal state ----
        existing_mmd = data.non_tensor_batch.get("multi_modal_data")
        if (
            existing_mmd is not None
            and len(existing_mmd) > 0
            and isinstance(existing_mmd[0], dict)
        ):
            self._engine_multi_modal_data = {
                k: list(v) for k, v in existing_mmd[0].items()
            }
        else:
            self._engine_multi_modal_data = {}
        # ---- position state (strip left-padding like input_ids) ----
        pos = data.batch["position_ids"]
        if pos.dim() == 3:
            self._train_position_ids = pos[:, :, valid].clone()
        else:
            self._train_position_ids = pos[:, valid].clone()

        self._derived_dirty = False

        # ---- semantic seed for turn-0 multimodal injection ----
        self._images = list(
            self._engine_multi_modal_data.get("images", [])
        )
        self._messages = []
        initial_prompt_text = str(data.non_tensor_batch.get("initial_prompt_text")[0])
        self._messages.append(
            {"role": "user", "content": initial_prompt_text.strip()}
        )

        return data

    def inject_initial_observation(
        self, data: DataProto, image,
    ) -> DataProto:
        """Append an env-reset screenshot to the initial context.

        Processes the image, adds an observation user message to the
        semantic history, and re-encodes the full initial prompt through
        the processor so that image tokens and mRoPE position IDs are
        computed correctly.

        Must be called right after :meth:`start_from_data`.
        """
        initial_prompt_text = str(data.non_tensor_batch.pop("initial_prompt_text", None)[0])
        
        return self._encode_initial_prompt_with_image(
            data, initial_prompt_text, image,
        )

    def _encode_initial_prompt_with_image(
        self, data: DataProto, prompt_text: str, image,
    ) -> DataProto:
        """Dataset-style one-shot re-encode for turn-0 prompt + reset image."""
        processed_image = process_image(
            image,
            self._meta_info.get("min_pixels"),
            self._meta_info.get("max_pixels"),
            self.processor,
        )

        content: List[Dict[str, Any]] = []
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})
        content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        model_inputs = self.processor(
            images=[processed_image],
            text=[prompt],
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

        if (
            hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from rllava.model.patch.qwen3_vl import get_rope_index
            else:
                from rllava.model.patch.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat(
                (text_position_ids, vision_position_ids), dim=0,
            )
        else:
            position_ids = torch.clip(
                attention_mask.cumsum(dim=0) - 1, min=0, max=None,
            )

        max_prompt_length = int(data.batch["input_ids"].size(-1))
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[:max_prompt_length]

        batch_input_ids = input_ids.unsqueeze(0)
        batch_attention_mask = attention_mask.unsqueeze(0)
        batch_position_ids = position_ids.unsqueeze(0)

        data.batch["input_ids"] = batch_input_ids
        data.batch["attention_mask"] = batch_attention_mask
        data.batch["position_ids"] = batch_position_ids
        data.non_tensor_batch["raw_prompt_ids"] = np.array(
            [raw_prompt_ids], dtype=object,
        )
        data.non_tensor_batch["multi_modal_data"] = [
            {"images": [processed_image]}
        ]

        self._engine_prompt_ids = list(raw_prompt_ids)
        self._engine_multi_modal_data = {"images": [processed_image]}

        valid = batch_attention_mask[0].bool()
        self._train_input_ids = batch_input_ids[0][valid].tolist()
        if batch_position_ids.dim() == 3:
            self._train_position_ids = batch_position_ids[:, :, valid].clone()
        else:
            self._train_position_ids = batch_position_ids[:, valid].clone()

        self._messages = [{"role": "user", "content": content}]
        self._images = [processed_image]
        self._derived_dirty = False
        return data

    # ================================================================
    # Turn-level state management
    # ================================================================

    def _detect_assistant_close_ids(self) -> List[int]:
        """Return the token IDs that close an assistant turn in the template.

        Compares two template expansions with different assistant content
        and isolates their common suffix (= close tokens such as
        ``<|im_end|>\\n``).  Cached after first call.
        """
        if hasattr(self, "_cached_assistant_close_ids"):
            return self._cached_assistant_close_ids

        template_fn = (
            self.processor.apply_chat_template
            if self.processor is not None
            else self.tokenizer.apply_chat_template
        )
        close: List[int] = []
        try:
            ids_a = template_fn(
                [{"role": "user", "content": "x"},
                 {"role": "assistant", "content": "a"}],
                add_generation_prompt=False, tokenize=True,
            )
            ids_ab = template_fn(
                [{"role": "user", "content": "x"},
                 {"role": "assistant", "content": "ab"}],
                add_generation_prompt=False, tokenize=True,
            )
            i, j = len(ids_a) - 1, len(ids_ab) - 1
            suffix: List[int] = []
            while i >= 0 and j >= 0 and ids_a[i] == ids_ab[j]:
                suffix.append(ids_a[i])
                i -= 1
                j -= 1
            close = suffix[::-1]
        except Exception:
            pass

        self._cached_assistant_close_ids = close
        return close

    def append_assistant_generation(
        self,
        gen_output: DataProto,
        decoded_text: Optional[str] = None,
    ) -> str:
        """Exact-append assistant response token IDs to the token chains.

        Uses the raw token IDs from ``gen_output.batch["responses"]``
        instead of decoding to text and re-encoding, so the sampled
        token sequence is preserved bit-exactly.

        If the response does not already contain the full assistant
        close suffix (e.g. the model hit ``max_tokens``), the missing
        tokens are appended as context-only suffix tokens.

        Returns:
            Decoded assistant text (for parser / logging).
        """
        responses = gen_output.batch.get("responses")
        if responses is None or responses.size(0) == 0:
            return ""

        response_ids_t = responses[0]
        if "response_mask" in gen_output.batch:
            resp_len = int(
                gen_output.batch["response_mask"][0].sum().item()
            )
            response_ids_t = response_ids_t[:resp_len]

        valid_ids = response_ids_t.tolist()
        if not valid_ids:
            return ""

        # ---- exact append of model-sampled tokens ----
        self._engine_prompt_ids.extend(valid_ids)
        self._train_input_ids.extend(valid_ids)

        # ---- append missing close suffix ----
        close_ids = self._detect_assistant_close_ids()
        remaining: List[int] = []
        if close_ids:
            overlap = 0
            for n in range(len(close_ids), 0, -1):
                if len(valid_ids) >= n and valid_ids[-n:] == close_ids[:n]:
                    overlap = n
                    break
            remaining = close_ids[overlap:]
            if remaining:
                self._engine_prompt_ids.extend(remaining)
                self._train_input_ids.extend(remaining)

        # ---- extend position_ids for the new tokens ----
        n_new = len(valid_ids) + len(remaining)
        if self._train_position_ids is not None and n_new > 0:
            if self._train_position_ids.dim() == 3:
                last = self._train_position_ids[:, :, -1:]
                delta = torch.arange(1, n_new + 1, dtype=torch.long)
                delta = delta.view(1, 1, -1).expand(
                    1, self._train_position_ids.size(1), -1,
                )
                self._train_position_ids = torch.cat(
                    [self._train_position_ids, last + delta], dim=-1,
                )
            else:
                last_val = int(self._train_position_ids[0, -1].item())
                ext = torch.arange(
                    last_val + 1, last_val + 1 + n_new, dtype=torch.long,
                )
                self._train_position_ids = torch.cat(
                    [self._train_position_ids[0], ext],
                ).unsqueeze(0)

        # ---- decode for parser / logging ----
        if decoded_text is None:
            decoded_text = self.tokenizer.decode(
                valid_ids, skip_special_tokens=True,
            ).strip()

        # ---- keep semantic history aligned with token-first state ----
        self._messages.append({"role": "assistant", "content": decoded_text})
        self._derived_dirty = True

        return decoded_text

    def materialize_generation_inputs(self, data: DataProto) -> None:
        """Write current state into *data* for the next generation call.

        Writes the exact token chains directly — no re-encode.
        """
        if not self._derived_dirty:
            return

        seq_len = len(self._train_input_ids)
        data.batch["input_ids"] = torch.tensor(
            [self._train_input_ids], dtype=torch.long,
        )
        data.batch["attention_mask"] = torch.ones(
            1, seq_len, dtype=torch.long,
        )
        if self._train_position_ids is not None:
            data.batch["position_ids"] = self._train_position_ids
        else:
            data.batch["position_ids"] = torch.arange(
                seq_len, dtype=torch.long,
            ).unsqueeze(0)

        data.non_tensor_batch["raw_prompt_ids"] = np.array(
            [self._engine_prompt_ids], dtype=object,
        )
        if self._engine_multi_modal_data:
            data.non_tensor_batch["multi_modal_data"] = [
                self._engine_multi_modal_data
            ]
        else:
            data.non_tensor_batch.pop("multi_modal_data", None)

        self._derived_dirty = False

    def append_external_delta(
        self,
        content: str,
        role: str = "user",
        image=None,
        tool_name: Optional[str] = None,
    ) -> None:
        """Delta-append an external turn (tool / env observation) to the token chains.

        Encodes only the template boundaries + new content as a delta,
        completely avoiding a full re-encode of the conversation history.

        For text-only observations the delta is computed purely from
        template token arithmetic.  For multimodal observations a single
        processor call is made on a minimal synthetic conversation to
        expand image tokens and obtain correct mRoPE position IDs; the
        delta is then extracted and offset to match the current state.
        """
        # ---- prepare template content (role mapping + tool prefix) ----
        display = content
        if role == "tool" and tool_name:
            display = f"[Tool: {tool_name}]\n{content}"

        processed_image = None
        if image is not None:
            processed_image = process_image(
                image,
                self._meta_info.get("min_pixels"),
                self._meta_info.get("max_pixels"),
                self.processor,
            )

        has_image = (
            processed_image is not None
            and self.processor is not None
            and "<image>" in display
        )

        template_fn = (
            self.processor.apply_chat_template
            if self.processor is not None
            else self.tokenizer.apply_chat_template
        )

        # ---- synthetic base: completed [user, assistant] ----
        base_msgs = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
        base_ids = template_fn(
            base_msgs, add_generation_prompt=False, tokenize=True,
        )
        base_len = len(base_ids)

        # ---- build the external message for the template ----
        if has_image:
            parts: list = []
            for idx, seg in enumerate(display.split("<image>")):
                if idx > 0:
                    parts.append({"type": "image"})
                if seg:
                    parts.append({"type": "text", "text": seg})
            ext_content: Any = parts or [{"type": "image"}]
        else:
            ext_content = display
        full_msgs = base_msgs + [{"role": "user", "content": ext_content}]

        # ---- engine-side delta (unexpanded image markers) ----
        engine_full = template_fn(
            full_msgs, add_generation_prompt=True, tokenize=True,
        )
        engine_delta = engine_full[base_len:]
        self._engine_prompt_ids.extend(engine_delta)

        if processed_image is not None:
            if "images" not in self._engine_multi_modal_data:
                self._engine_multi_modal_data["images"] = []
            self._engine_multi_modal_data["images"].append(processed_image)

        # ---- train-side delta ----
        delta_pos: Optional[torch.Tensor] = None
        if has_image:
            full_text = template_fn(
                full_msgs, add_generation_prompt=True, tokenize=False,
            )
            full_inputs = self.processor(
                text=[full_text],
                images=[processed_image],
                add_special_tokens=False,
                return_tensors="pt",
            )
            full_ids = full_inputs["input_ids"][0]
            train_delta = full_ids[base_len:].tolist()

            full_pos = full_inputs.get("position_ids")
            if full_pos is not None:
                if (
                    full_pos.dim() >= 3
                    and full_pos.size(1) == 1
                    and full_pos.size(0) in (3, 4)
                ):
                    full_pos = full_pos.transpose(0, 1).contiguous()
                raw_delta_pos = full_pos[..., base_len:]
                if self._train_position_ids is not None:
                    offset = (
                        self._train_position_ids[..., -1:] - (base_len - 1)
                    )
                    delta_pos = raw_delta_pos + offset
                else:
                    delta_pos = raw_delta_pos

        else:
            train_delta = list(engine_delta)

        self._train_input_ids.extend(train_delta)

        # ---- extend position_ids ----
        n_new = len(train_delta)
        if delta_pos is not None:
            self._train_position_ids = torch.cat(
                [self._train_position_ids, delta_pos], dim=-1,
            )
        elif self._train_position_ids is not None and n_new > 0:
            if self._train_position_ids.dim() == 3:
                last = self._train_position_ids[:, :, -1:]
                d = torch.arange(1, n_new + 1, dtype=torch.long)
                d = d.view(1, 1, -1).expand(
                    1, self._train_position_ids.size(1), -1,
                )
                self._train_position_ids = torch.cat(
                    [self._train_position_ids, last + d], dim=-1,
                )
            else:
                lv = int(self._train_position_ids[0, -1].item())
                ext = torch.arange(
                    lv + 1, lv + 1 + n_new, dtype=torch.long,
                )
                self._train_position_ids = torch.cat(
                    [self._train_position_ids[0], ext],
                ).unsqueeze(0)

        # ---- keep semantic history aligned with token-first state ----
        if processed_image is not None:
            self._images.append(processed_image)
        if role == "tool":
            self._messages.append(
                {"role": "tool", "name": tool_name or "tool", "content": content}
            )
        elif role == "environment":
            self._messages.append({"role": "environment", "content": content})
        else:
            self._messages.append({"role": "user", "content": content})

        self._derived_dirty = True

