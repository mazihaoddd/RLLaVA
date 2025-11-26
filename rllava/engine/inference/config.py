from dataclasses import dataclass
from typing import Optional, List, Dict
from rllava.utils.config import conf_as_dict
from rllava.utils import pkg_version



@dataclass
class SGLangConfig:
    """Configuration for SGLang runtime. Refer to:
    https://github.com/sgl-project/sglang for detailed documentation.
    """

    model_path: str = ""
    random_seed: int = 1
    skip_tokenizer_init: bool = False
    disable_cuda_graph: bool = False
    disable_radix_cache: bool = False
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    attention_backend: Optional[str] = "fa3"
    sampling_backend: Optional[str] = None
    context_length: Optional[int] = 32768
    mem_fraction_static: Optional[float] = 0.9
    max_running_requests: Optional[int] = None
    # NOTE: chunked_prefill_size is by default 8192 on GPUs with 80GB mem in SGLang,
    # but we disable it to avoid precision issues
    chunked_prefill_size: Optional[int] = -1
    max_prefill_tokens: int = 32768
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    dp_size: int = 1  # only used for dp attention
    ep_size: int = 1
    # logging
    log_level: str = "warning"
    log_level_http: Optional[str] = "warning"
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = True  # Exports Prometheus-like metrics
    # The interval (in decoding iterations) to log throughput
    # and update prometheus metrics
    decode_log_interval: int = 1

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host,
        port,
        dist_init_addr: Optional[str] = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args = SGLangConfig.build_args(
            sglang_config=sglang_config,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id,
            host=host,
            port=port,
            dist_init_addr=dist_init_addr,
            n_nodes=n_nodes,
            node_rank=node_rank,
        )

        # convert to flags
        flags = []
        for k, v in args.items():
            if v is None or v is False or v == "":
                continue
            if v is True:
                flags.append(f"--{k.replace('_','-')}")
            elif isinstance(v, list):
                flags.append(f"--{k.replace('_','-')} {' '.join(map(str, v))}")
            else:
                flags.append(f"--{k.replace('_','-')} {v}")
        return f"python3 -m sglang.launch_server {' '.join(flags)}"

    @staticmethod
    def build_args(
        sglang_config: "SGLangConfig",
        tp_size,
        base_gpu_id,
        host,
        port,
        dist_init_addr: Optional[str] = None,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):

        args: Dict = conf_as_dict(sglang_config)
        args = dict(
            host=host,
            port=port,
            # Model and tokenizer
            tokenizer_path=sglang_config.model_path,
            tokenizer_mode="auto",
            load_format="auto",
            trust_remote_code=True,
            device="cuda",
            is_embedding=False,
            # Other runtime options
            tp_size=tp_size,
            # Because we have set CUDA_VISIBLE_DEVICES to a single GPU in each process
            base_gpu_id=base_gpu_id,
            nnodes=n_nodes,
            node_rank=node_rank,
            # initialization addresses and ports
            dist_init_addr=dist_init_addr,
            **args,
        )
        if not pkg_version.is_version_greater_or_equal("sglang", "0.4.9.post2"):
            raise RuntimeError("Needs sglang>=0.4.9.post2 to run the code.")
        return args


@dataclass
class VLLMConfig:
    """Configuration for vLLM runtime. Refer to:
    https://github.com/vllm-project/vllm for detailed documentation.
    """

    model: str = ""
    seed: int = 1
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    dtype: str = "bfloat16"
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    distributed_executor_backend: Optional[str] = None
    tokenizer: Optional[str] = None
    chat_template: Optional[str] = None

    # Use staticmethod to make OmegaConf happy.
    @staticmethod
    def build_cmd(
        vllm_config: "VLLMConfig",
        tp_size,
        host,
        port,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):
        args = VLLMConfig.build_args(
            vllm_config=vllm_config,
            tp_size=tp_size,
            host=host,
            port=port,
            n_nodes=n_nodes,
            node_rank=node_rank,
        )

        # convert to flags
        flags = []
        for k, v in args.items():
            if v is None or v is False or v == "":
                continue
            if v is True:
                flags.append(f"--{k.replace('_','-')}")
            elif isinstance(v, list):
                flags.append(f"--{k.replace('_','-')} {' '.join(map(str, v))}")
            else:
                flags.append(f"--{k.replace('_','-')} {v}")
        return f"python3 -m vllm.entrypoints.openai.api_server {' '.join(flags)}"

    @staticmethod
    def build_args(
        vllm_config: "VLLMConfig",
        tp_size,
        host,
        port,
        n_nodes: int = 1,
        node_rank: int = 0,
    ):

        # ensure vllm is installed
        if not pkg_version.is_available("vllm"):
            raise RuntimeError("vllm is not installed in the current environment.")

        cfg: Dict = conf_as_dict(vllm_config)

        args = dict(
            host=host,
            port=port,
            tensor_parallel_size=tp_size,
            # multi-node placeholders (not all backends require explicit flags)
            # Expose distributed backend if provided via config
            **cfg,
        )
        return args