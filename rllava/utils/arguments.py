from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING, Union, List, Any
import transformers



@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default=None)
    
    model_name_or_path: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default='')
    vision_tower2: Optional[str] = field(default='')
    connector_type: str = field(default='linear')
    
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    resampler_hidden_size: Optional[int] = field(default=768)
    num_queries: Optional[int] = field(default=128)
    num_resampler_layers: Optional[int] = field(default=3)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tokenizer_use_fast: bool = field(default=False)
    tokenizer_padding_side: str = field(default='right')

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    conv_version: str = 'pretrain'
    system_prompt_template: Optional[str] = field(
        default="reasoning",
        metadata={"help": "System prompt template. Possible values: 'llava', 'qwen2', 'reasoning', 'grounding', 'ocr'"},
    )
    question_template: Optional[str] = field(
        default="default",
        metadata={"help": "Question template. Possible values: 'default', 'llava', 'qwen2', 'reasoning'"},
    )
    answer_template: Optional[str] = field(
        default="default",
        metadata={"help": "Answer template. Possible values: 'default', 'llava', 'qwen2', 'reasoning'"},
    )
    problem_key: str = field(
        default="problem",
        metadata={"help": "the key of problem in the dataset"}
    )
    answer_key: str = field(
        default="solution",
        metadata={"help": "the key of answer in the dataset"}
    )
    image_key: str = field(
        default="image",
        metadata={"help": "the key of image in the dataset"}
    )
    image_dir: str = field(
        default="",
        metadata={"help": "the directory of image in the dataset"}
    )
    image_size: Optional[str] = field(
        default=None,
        metadata={"help": "the size of image in the dataset"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    train_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Train sample size. If None, use all samples."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default='common')
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    tune_vision_tower_from_layer: Optional[int] = field(default=10)
    tune_type_connector: str = field(default="full") # support only: frozen, full
    tune_embed_tokens: bool = field(default=False)
    
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_task_type: str = field(default="CAUSAL_LM")
    lora_r: int = field(default=128)
    lora_target_modules: Union[List[str], str, None] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    lora_use_rslora: bool = field(default=False)
    lora_modules_to_save: Optional[List[str]] = field(default=None)
    lora_weight_path: str = ""
    lora_bias: str = "none"
    use_rslora: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = field(default=None)
   
@dataclass
class TrainingRLArguments(TrainingArguments):
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "whether to skip special tokens, use when rec task."}
    )
    temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: Optional[float] = field(
        default=0.9,
        metadata={"help": "Top-p for sampling."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "Top-k for sampling."},
    )
    temperature_func: str = field(
        default="constant",
        metadata={"help":"which temperature function to use while training. Unlike reward_funcs, you can only use one temperature function."}
    )
    temperature_begin: float = field(
        default=0.1, 
        metadata={"help": "the beginning temperature for training(optional for linear temperature)"}
    )
    temperature_end: float = field(
        default=1.0, 
        metadata={"help": "the ending temperature for training(optional for linear temperature)"}
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={"help": "Number of generations to sample."},
    )
    entropy_reg : bool = field(
        default=False, 
        metadata={"help": "whether to use entropy regularization while training. For discriminative tasks like grounding, ocr and counting, we expect entropy to decrease. For literary creation task, we expect entropy to increase. this can be controlled by entropy_weight."}
    )
    entropy_weight: float = field(
        default=0.01, 
        metadata={"help": "the weight for entropy loss. It's only valid when entropy_reg is true. If it's positive, the entropy is to increase. If it's negetive, the entropy is to decrease."}
    )
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    reward_scale: float = field(
        default=1, 
        metadata={"help": "reward scale of all rewards"}
    )
    no_mean_for_same_reward: bool = field(
        default=False,
        metadata={"help": "whether to not minus reward mean if same reward"}
    )
    use_kl: bool = field(
        default=True, 
        metadata={"help":"whether to use kl in loss. If false, no kl will be included into loss. But you can also view kl change trends in pandb"}
    )
    kl_approximator: str = field(
        default="k3", 
        metadata={"help": "which type kl to use for computing loss.you can use k1(not good), k3(official in grpo, unbias, lowest variance), kimikl(only the kl used in kimi1.5), kimifull(the same setting as the core idea of kimi1.5, your value of sync_ref_model, ref_model_mixup_alpha and ref_model_sync_steps will be invalid, they are all set the same as kimi1.5)"}
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.8,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    model_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )