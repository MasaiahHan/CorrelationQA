import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from PIL import Image
import argparse
import os
import json
from datasets import load_dataset
import numpy as np
import cv2
from PIL import Image

def load_model(model_name):
# Load the model in half-precision
    model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").to('cuda')
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def get_image(ds, index):
    img_bytes = ds[index]["image"]
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    img = Image.fromarray(img[..., ::-1])
    return img


def inference(args):
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset_df = load_dataset("ScottHan/CorrelationQA")['train']

    model, processor = load_model(args.model_path)

    with open(args.output_path, 'w')as f:
        for i in tqdm(range(len(dataset_df))):

            image = get_image(dataset_df, i)
            question = dataset_df['question'][i]

            ########################
            # custom different MLLMs
            ########################
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, torch.float16)

            # Generate
            generate_ids = model.generate(**inputs, max_new_tokens=30)
            answer = processor.batch_decode(generate_ids, skip_special_tokens=True)

            # write file
            f.write(answer[0].split('ASSISTANT: ')[-1]+'\n')
            f.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default= "llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--output_path", type=str, default= "output.txt")



    args = parser.parse_args()
    inference(args)