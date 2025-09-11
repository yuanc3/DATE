import torch
import os
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from decord import VideoReader
import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

batch_size = 2048


def extract_frames(video_path, fps):
    if os.path.isdir(video_path):
        all_frames = [
            os.path.join(video_path, file) for file in list(sorted(os.listdir(video_path)))
        ]
        num_frames = len(all_frames)
        frame_rate = 25
    
        frames, timestamps, ids = [], [], []
        # idx = 0
        total = int(num_frames * fps / frame_rate)
        indices = np.linspace(0, num_frames - 1, total, dtype=int) 

        for idx in indices:
            image = Image.open(all_frames[idx])
            frames.append(image)
            timestamps.append(idx / frame_rate)
            ids.append(float(idx))

        return frames, timestamps, ids
    elif video_path.endswith(".mp4"):
        cap = VideoReader(video_path, num_threads=2)
        frame_rate = cap.get_avg_fps()
        num_frames = len(cap)

        frames, timestamps = [], []
        idx = 0
        total = int(num_frames * fps / frame_rate)
        indices = np.linspace(0, num_frames - 1, total, dtype=int) 
       
        for idx in indices:
            idx = int(idx)
            frame = cap[idx]
            image = Image.fromarray(frame.asnumpy())
            frames.append(image)
            timestamps.append(idx / frame_rate)
    return frames, timestamps, None


def process_video_question(video_path, question, topk=512, fps=4, max_frames=256):
    frames, timestamps, ids = extract_frames(video_path, fps=fps)

    inputs_text = processor(text=question, return_tensors="pt", padding=True,truncation=True).to(device)
    text_features = model.get_text_features(**inputs_text)

    images = frames
    # Process images in batches
    score = []
    embedding = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # Process batch of images
        inputs_image = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            batch_image_features = model.get_image_features(**inputs_image) 

        embedding.append(batch_image_features)
        
        # Compute scores for the batch
        batch_scores = torch.nn.CosineSimilarity(dim=-1)(
            text_features.repeat(len(batch_images), 1),
            batch_image_features
        )
        
        # Store results
        score.extend(batch_scores.cpu().tolist())


    embedding = torch.cat(embedding,0)
  
    return score, timestamps, ids, embedding

