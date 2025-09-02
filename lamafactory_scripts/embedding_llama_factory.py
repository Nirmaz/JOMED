import argparse
from src.util import load_jsonl_file
from datasets import Dataset
import tqdm
from PIL import Image
from transformers import AutoModel
import numpy as np
import os

def main(path_save, dataset_file, dataset_name, float_16=False):
    """
    Generate embeddings for text and images using jina-clip-v1 model.
    
    Args:
        path_save (str): Directory to save the processed dataset
        dataset_file (str): Path to the JSONL dataset file
        dataset_name (str): Name for the output dataset
        float_16 (bool): Whether to use float16 precision
    """
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

    if float_16:
        model = model.half().eval()
    else:
        model = model.eval()

    model = model.cuda()
    pmc_dataset = load_jsonl_file(dataset_file)
    
    dataset_dict = {
        "caption": [],
        "image": [],
        "text_embeddings": [],
        "image_embeddings": []
    }
    
    for i, obj in tqdm.tqdm(enumerate(pmc_dataset), desc="Processing dataset"):
        dataset_dict["caption"].append(obj["text"])
        dataset_dict["image"].append(obj["img_path"])
        
        try:
            with Image.open(obj["img_path"]) as img:
                img_embeddings = model.encode_image(img)
                if float_16:
                    img_embeddings = img_embeddings.astype(np.float16)

            text_embeddings = model.encode_text([obj["text"]])[0]
            if float_16:
                text_embeddings = text_embeddings.astype(np.float16)

            dataset_dict["text_embeddings"].append(text_embeddings)
            dataset_dict["image_embeddings"].append(img_embeddings)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    ds = Dataset.from_dict(dataset_dict)

    path_embed = os.path.join(path_save, f"{dataset_name}_dataset")
    if float_16:
        path_embed += '_float16'
    
    os.makedirs(os.path.dirname(path_embed), exist_ok=True)
    ds.save_to_disk(path_embed)
    print(f"Dataset saved to: {path_embed}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings for text and images using jina-clip-v1')
    parser.add_argument('--dataset_file', type=str, required=True, 
                       help='Path to the JSONL dataset file')
    parser.add_argument('--path_save', type=str, required=True,
                       help='Directory to save the processed dataset')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name for the output dataset')
    parser.add_argument('--float_16', action='store_true',
                       help='Use float16 precision for embeddings')
    
    args = parser.parse_args()
    
    main(
        path_save=args.path_save,
        dataset_file=args.dataset_file, 
        dataset_name=args.dataset_name,
        float_16=args.float_16
    )







