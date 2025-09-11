import os
import cv2
import math
import yaml
from PIL import Image
from typing import List, Union
import transformers
import torch
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor

from qwen2_5_vl_date import (
    date_processing_qwen2_5_vl__call__,
    date_get_rope_index,
)

def get_frame_indices(total_frames, max_num_frames, sample_fps, extraction_fps):
    # Get number of sampled frames
    sample_frames = float(total_frames / extraction_fps) * sample_fps
    sample_frames = min(total_frames, max_num_frames, sample_frames)
    sample_frames = math.floor(sample_frames)
    sample_frames = int(sample_frames / 2) * 2
    # Get sampled frame indices
    frame_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
    return frame_indices


def load_specific_frames(cap, frame_indices):
    # List to store the frames
    frames = []
    # Read frames from the video
    for frame_index in frame_indices:
        # Set the video position to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Read the frame
        ret, frame = cap.read()
        # If the frame was read successfully, append it to the list
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create a PIL Image from the frame
            frame = Image.fromarray(frame_rgb)
            frames.append(frame)
        else:
            ValueError(f"Warning: Could not read frame at index {frame_index}. It may be out of range.")
    return frames


def load_video(video_path: str, max_num_frames: int, fps: Union[int, float]=None, frame_extraction_fps: Union[int, float]=None, timestamps=None):
    """Load video frames at fps. If total frames larger than `max_num_frames`, do downsample.
       If 'fps' is `None`, load uniformly sample `max_num_frames` frames.

       video_path: Should either be a videofile or a directory of extracted frames.

       # NOTE: The extract frames must have name pattern of `%06d.(ext)`, or the loaded frame order will be wrong.
    """
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    if os.path.isdir(video_path): # directory extracted frames
        assert frame_extraction_fps is not None
        frame_files = [
            os.path.join(video_path, file) for file in list(sorted(os.listdir(video_path)))
        ]
        num_total_frames = len(frame_files)
        # Get indices of sampled frame
        frame_indices = get_frame_indices(num_total_frames, max_num_frames, fps, frame_extraction_fps)

        frames = []
        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx in frame_indices:
                image = Image.open(frame_file)
                frames.append(image)
    else: # filename of a video
        if timestamps is None:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error: Could not open video.")
            num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_extraction_fps = cap.get(cv2.CAP_PROP_FPS)
            # Get indices of sampled frame
            frame_indices = get_frame_indices(num_total_frames, max_num_frames, fps, frame_extraction_fps)
            # Get frames
            frames = load_specific_frames(cap, frame_indices)
            # Release the video capture object
            cap.release()
        else:
            cap = cv2.VideoCapture(video_path)
            sampling_fps = None
            frame_indices = []
            frame_extraction_fps = cap.get(cv2.CAP_PROP_FPS)
            for timestamp in timestamps:
                frame_indices.append(math.floor(timestamp * frame_extraction_fps))
            # Get frames
            frames = load_specific_frames(cap, frame_indices)
            # Release the video capture object
            frames = [
                frame.convert("RGB") if frame.mode != "RGB" else frame
                for frame in frames
            ]
            cap.release()
            return frames, sampling_fps, timestamps

    timestamps = [idx / frame_extraction_fps for idx in frame_indices]  # ← 添加此行，计算每帧的时间戳

    # Convert into RGB format
    frames = [
        frame.convert("RGB") if frame.mode != "RGB" else frame
        for frame in frames
    ]

    # Calculate the final sampling fps
    duration = num_total_frames / frame_extraction_fps
    sampling_fps = len(frames) / duration
    print("Sampling config: max_num_frames-%d, fps-%d, frame_extraction_fps-%d, final sampleing fps: %.1f" % (
        max_num_frames, fps, frame_extraction_fps, sampling_fps)
    )

    return frames, sampling_fps, timestamps


def resize_image_longside(image, image_resolution):
    r"""
    Pre-processes a single image.
    """
    if max(image.width, image.height) > image_resolution:
        resize_factor = image_resolution / max(image.width, image.height)
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.NEAREST)

    return image


def resize_video_longside(frames: List, video_resolution):
    """
    frames: list of PIL images.
    """
    frames = [
        resize_image_longside(frame, video_resolution)
        for frame in frames
    ]
    return frames


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def fetch_video(video_info, max_num_frames, sample_fps, longsize_resolution, timestamps=None):
    frames, sampling_fps, timestamps = load_video(video_info['video'], max_num_frames, sample_fps, video_info.get('frame_extraction_fps', None), timestamps)
    frames = resize_video_longside(frames, longsize_resolution)
    frames = [pil_to_tensor(frame) for frame in frames]
    return frames, sampling_fps, timestamps


def load_and_patch_model(model_name, hf_model_path, exp_configs, device):
    model_name = model_name if model_name is not None else exp_configs['model_name']
    model_name = model_name.lower().replace('-', '').replace('_', '')
    if model_name == 'qwen25vl': # QWen2_5VL
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        qwen2_5_vl_config = Qwen2_5_VLConfig.from_pretrained(hf_model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model_path,
            config=qwen2_5_vl_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=exp_configs.get('attn_implementation', None),
            device_map=device # "auto"
        ).eval()
        # ！！！！！Timestamp Injection Mechanism ！！！！！
        transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor.__call__ = date_processing_qwen2_5_vl__call__
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_rope_index = date_get_rope_index

        processor = AutoProcessor.from_pretrained(hf_model_path)
        
    else:
        raise NotImplementedError
    return model, processor




# if __name__ == "__main__":
#     #------------------- Modify the following configs ------------------#
#     hf_model_path = 'Qwen/Qwen2.5-VL-7B-Instruct'
#     model_name = 'qwen2_5_vl'
#     #------------------- Modify the following settings ------------------#
#     DEMO_VIDEO = 'asserts/demo.mp4'
#     question = "When did his the straps of his pants slip off when he turned his back to dance?"
#     max_frames = 24 
#     Q2C = True # Query to Caption, if set to False, it will use the question directly.
#     SAMPLE = True # Sample the video frames with TASS, if set to False, it will use the original method.
    

#     config_path = 'configs/demo.yaml'
#     # NOTE: for 7B models in Nvidia GPUs
#     device = 'cuda:0'
#     # NOTE: for 72B models in Nvidia GPUs
#     # device = 'auto'
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B-Instruct Demo")

    # 添加命令行参数
    parser.add_argument("--hf_model_path", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help="Huggingface model path")
    parser.add_argument("--model_name", type=str, default='qwen2_5_vl', help="Model name")
    parser.add_argument("--video", type=str, default='asserts/demo.mp4', help="Input demo video path")
    parser.add_argument("--question", type=str, default="When did the straps of his pants slip off when he turned his back to dance?", help="Question to ask")
    parser.add_argument("--max_frames", type=int, default=24, help="Max frames to use")
    parser.add_argument("--q2c", action='store_true', help="Enable Query to Caption")
    parser.add_argument("--tass_sample", action='store_true', help="Enable TASS sampling")
    parser.add_argument("--config_path", type=str, default='configs/demo.yaml', help="Path to config YAML")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to use. Use 'auto' for automatic device selection (e.g., for 72B models in Nvidia GPUs)")

    args = parser.parse_args()
    

    #------------------------ No need to change ------------------------#
    video_info = {"type": "video", 
                  "video": args.video, 
                  "fps": 2.0}

    exp_configs = load_yaml(args.config_path)
    model, processor = load_and_patch_model(args.model_name, args.hf_model_path, args.config_path, args.device)
    caption = args.question
    
    # query->caption
    if args.q2c:
        from utils.query import chatcompletions
        caption = chatcompletions(question = args.question)
  
    if args.tass_sample:
        # sample with fps, and calculate the similarity score between video frames and the question
        from utils.clip_sim import process_video_question
        scores, timestamps, ids, embedding = process_video_question(video_info["video"], caption, fps=video_info['fps'])

        # sample with TASS
        from utils.sampling import tass_sampling
        final_timestamps, final_indices = tass_sampling(timestamps, scores, topk=None, max_frames=args.max_frames)
    else:
        final_timestamps = None


    # Video
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Here is a video, you need to answer the question in the video.",
        },
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": args.question},
            ],
        }
    ]

    # If final_timestamps is provided, it will sample the corresponding frames; if not provided or set to None, it will sample uniformly.
    video, sampling_fps, timestamps = fetch_video(video_info, exp_configs['max_num_frames'], exp_configs['sample_fps'], exp_configs['longsize_resolution'], timestamps = final_timestamps)

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    videos_kwargs = dict(fps=sampling_fps)
    
    # If `timestamps` is provided, it will embed the timestamps; if not provided or set to None, it will use the original method.
    inputs = processor(text=[text_prompt], videos=[video], padding=True, return_tensors="pt", timestamps=timestamps, **videos_kwargs)

    if args.device == 'auto':
        inputs = inputs.to('cuda')
    else:
        inputs = inputs.to(args.device)
    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    output_text = output_text
    print('Output text:\n', output_text)
