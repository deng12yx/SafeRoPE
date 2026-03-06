import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import random

class CocoDataset(Dataset):
    def __init__(self, image_dir, caption_path, transform=None, tokenizer=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        with open(caption_path, 'r') as f:
            data = json.load(f)
        self.annotations = data['annotations']
        self.id2filename = {img['id']: img['file_name'] for img in data['images']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']
        img_path = os.path.join(self.image_dir, self.id2filename[image_id])
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if self.tokenizer:
            text = self.tokenizer(caption, truncation=True, padding="max_length", return_tensors="pt")
        else:
            text = caption
        
        return {
            "image": image,
            "text": text,
            "is_safe": True
        }


class UnsafePromptDataset(Dataset):
    def __init__(self, txt_path, tokenizer=None):
        with open(txt_path, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()]
            # Randomly shuffle the prompts
            random.shuffle(self.prompts)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        if self.tokenizer:
            text = self.tokenizer(prompt, truncation=True, padding="max_length", return_tensors="pt")
        else:
            text = prompt
        return {
            "image": "",  # No image for unsafe prompts
            "text": text,
            "is_safe": False
        }


class SafeUnlearningDataset(Dataset):
    def __init__(self, coco_dataset, unsafe_dataset, unsafe_ratio=0.2):
        self.coco_dataset = coco_dataset
        self.unsafe_dataset = unsafe_dataset
        self.unsafe_ratio = unsafe_ratio
        
        self.safe_len = len(coco_dataset)
        self.unsafe_len = len(unsafe_dataset)

    def __len__(self):
        # Use the number of safe samples as the primary length
        return self.safe_len

    def __getitem__(self, idx):
        if random.random() < self.unsafe_ratio:
            # Randomly sample an unsafe example
            return self.unsafe_dataset[random.randint(0, self.unsafe_len - 1)]
        else:
            return self.coco_dataset[idx]
