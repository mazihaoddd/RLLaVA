from dataclasses import dataclass
from typing import Optional
import os



@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    train_image_dir: Optional[str] = None
    val_image_dir: Optional[str] = None
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    train_batch_size: int = 512
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    num_workers: int = 8
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None