#!/bin/bash

# Common argument parsing library for TinyLLaVA training scripts
# Usage: source this file and call parse_training_args "$@"

# Default values for training parameters
DEFAULT_DATA_PATH="/mnt/ve_share/zhaolei/data/llava_v1_5_mix665k"
DEFAULT_IMAGE_PATH="/mnt/ve_share/zhaolei/data/llava_v1_5_mix665k/images"
DEFAULT_LLM_VERSION="Qwen/Qwen2.5-0.5B"
DEFAULT_VT_VERSION="openai/clip-vit-large-patch14-336"
DEFAULT_VT_VERSION2="openai/clip-vit-large-patch14-336"
DEFAULT_CN_VERSION="qwen2.5"
DEFAULT_CONV_VERSION="qwen2.5"
DEFAULT_VERSION="qwen2.5-0.5b"
DEFAULT_TRAIN_RECIPE="llava_v1.5"
DEFAULT_MODEL_MAX_LENGTH="2048"

# Initialize variables with defaults
DATA_PATH="$DEFAULT_DATA_PATH"
IMAGE_PATH="$DEFAULT_IMAGE_PATH"
LLM_VERSION="$DEFAULT_LLM_VERSION"
VT_VERSION="$DEFAULT_VT_VERSION"
VT_VERSION2="$DEFAULT_VT_VERSION2"
CN_VERSION="$DEFAULT_CN_VERSION"
CONV_VERSION="$DEFAULT_CONV_VERSION"
VERSION="$DEFAULT_VERSION"
TRAIN_RECIPE="$DEFAULT_TRAIN_RECIPE"
MODEL_MAX_LENGTH="$DEFAULT_MODEL_MAX_LENGTH"

# Function to show usage
show_training_usage() {
    local script_name="$1"
    echo "Usage: $script_name [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --data-path PATH              Data path (default: $DEFAULT_DATA_PATH)"
    echo "  --image-path PATH             Image path (default: $DEFAULT_IMAGE_PATH)"
    echo "  --llm-version VERSION         LLM version (default: $DEFAULT_LLM_VERSION)"
    echo "  --vt-version VERSION          Vision tower version (default: $DEFAULT_VT_VERSION)"
    echo "  --vt-version2 VERSION         Vision tower2 version (default: $DEFAULT_VT_VERSION2)"
    echo "  --cn-version VERSION          Connector version (default: $DEFAULT_CN_VERSION)"
    echo "  --conv-version VERSION        Conversation version (default: $DEFAULT_CONV_VERSION)"
    echo "  --version VERSION             Model version (default: $DEFAULT_VERSION)"
    echo "  --train-recipe RECIPE         Training recipe (default: $DEFAULT_TRAIN_RECIPE)"
    echo "  --model-max-length LENGTH     Model max length (default: $DEFAULT_MODEL_MAX_LENGTH)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $script_name                                    # Use all defaults"
    echo "  $script_name --data-path /custom/path           # Use custom data path"
    echo "  $script_name --llm-version Qwen/Qwen2.5-1.5B   # Use custom LLM version"
    echo "  $script_name --data-path /custom/path --llm-version Qwen/Qwen2.5-1.5B  # Multiple custom params"
    echo ""
    echo "Legacy positional arguments (still supported):"
    echo "  $script_name [DATA_PATH] [IMAGE_PATH] [LLM_VERSION] [VT_VERSION] [VT_VERSION2] [CN_VERSION] [CONV_VERSION] [VERSION] [TRAIN_RECIPE] [MODEL_MAX_LENGTH]"
}

# Function to display current configuration
show_config() {
    echo "=== Configuration ==="
    echo "DATA_PATH: $DATA_PATH"
    echo "IMAGE_PATH: $IMAGE_PATH"
    echo "LLM_VERSION: $LLM_VERSION"
    echo "VT_VERSION: $VT_VERSION"
    echo "VT_VERSION2: $VT_VERSION2"
    echo "CN_VERSION: $CN_VERSION"
    echo "CONV_VERSION: $CONV_VERSION"
    echo "VERSION: $VERSION"
    echo "TRAIN_RECIPE: $TRAIN_RECIPE"
    echo "MODEL_MAX_LENGTH: $MODEL_MAX_LENGTH"
    echo "===================="
    echo ""
}

# Main argument parsing function
parse_training_args() {
    local script_name="$1"
    shift
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data-path)
                DATA_PATH="$2"
                shift 2
                ;;
            --image-path)
                IMAGE_PATH="$2"
                shift 2
                ;;
            --llm-version)
                LLM_VERSION="$2"
                shift 2
                ;;
            --vt-version)
                VT_VERSION="$2"
                shift 2
                ;;
            --vt-version2)
                VT_VERSION2="$2"
                shift 2
                ;;
            --cn-version)
                CN_VERSION="$2"
                shift 2
                ;;
            --conv-version)
                CONV_VERSION="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --train-recipe)
                TRAIN_RECIPE="$2"
                shift 2
                ;;
            --model-max-length)
                MODEL_MAX_LENGTH="$2"
                shift 2
                ;;
            -h|--help)
                show_training_usage "$script_name"
                exit 0
                ;;
            -*)
                echo "Unknown option $1"
                show_training_usage "$script_name"
                exit 1
                ;;
            *)
                # Handle legacy positional arguments
                if [ -z "$POSITIONAL_COUNT" ]; then
                    POSITIONAL_COUNT=0
                fi
                POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
                
                case $POSITIONAL_COUNT in
                    1) DATA_PATH="$1" ;;
                    2) IMAGE_PATH="$1" ;;
                    3) LLM_VERSION="$1" ;;
                    4) VT_VERSION="$1" ;;
                    5) VT_VERSION2="$1" ;;
                    6) CN_VERSION="$1" ;;
                    7) CONV_VERSION="$1" ;;
                    8) VERSION="$1" ;;
                    9) TRAIN_RECIPE="$1" ;;
                    10) MODEL_MAX_LENGTH="$1" ;;
                    *)
                        echo "Too many positional arguments. Use --help for usage information."
                        exit 1
                        ;;
                esac
                shift
                ;;
        esac
    done
}
