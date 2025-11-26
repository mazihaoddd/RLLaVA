DATA_PATH=/mnt/ve_share/zhaolei/data/llava/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/mnt/ve_share/zhaolei/data/llava/text_files/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/mnt/ve_share/zhaolei/data/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/mnt/ve_share/zhaolei/data/llava #finetune image dir

LLM_VERSION=/mnt/ve_share/zhaolei/.cache/huggingface/hub/Qwen/Qwen3-0.6B-Base # llm path in huggingface
VT_VERSION=/mnt/ve_share/zhaolei/.cache/huggingface/hub/google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen3-0.6b-base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm


bash scripts/train/qwen3/pretrain_qwen3.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/qwen3/finetune_qwen3.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
