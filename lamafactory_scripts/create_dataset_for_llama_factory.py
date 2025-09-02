import os
import copy
import json
import argparse
from PIL import Image
from datasets import Dataset
import tqdm
import numpy as np
from transformers import AutoModel

def load_jsonl_file(file_path):
    """Load JSONL file and return list of dictionaries."""
    files = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                files.append(json.loads(line))
    return files

def dump_jsonl_file(list_dict, path_json):
    """Save list of dictionaries to JSONL file."""
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, 'w') as f:
        for entry in list_dict:
            json.dump(entry, f)
            f.write('\n')

def build_dataset_for_gina_qa(instructions, index_dataset_path, question, path_to_save, name_data, path_to_data, float_16=False):
    """
    Build dataset for GINA QA task with retrieval-augmented generation.
    
    Args:
        instructions (str): Task instructions
        index_dataset_path (str): Path to the indexed dataset with embeddings
        question (str): Question template
        path_to_save (str): Output directory
        name_data (str): Dataset name prefix
        path_to_data (list): List of dataset file paths
        float_16 (bool): Use float16 precision
    """
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
    ds_with_embeddings = Dataset.load_from_disk(index_dataset_path)
    ds_with_embeddings = ds_with_embeddings.add_faiss_index(column="text_embeddings")
    ds_with_embeddings = ds_with_embeddings.add_faiss_index(column="image_embeddings")
    
    if float_16:
        model = model.half().eval()
    else:
        model = model.eval()

    model = model.cuda()

    template_format = {
        "messages": [
            {
                "content": "",
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant"
            },
        ],
        "images": []
    }

    for path in path_to_data:
        if not os.path.exists(path):
            print(f"Warning: File {path} does not exist, skipping...")
            continue
            
        print(f"Processing: {path}")
        training_samples = load_jsonl_file(path)
        counter = 0
        training_data = []
        
        for sample in tqdm.tqdm(training_samples, desc=f"Processing {os.path.basename(path)}"):
            counter += 1
            
            try:
                with Image.open(sample["image_path"]) as img:
                    img_embeddings = model.encode_image(img)
                    if float_16:
                        img_embeddings = img_embeddings.astype(np.float16)
            except Exception as e:
                print(f"Error processing image {sample['image_path']}: {e}")
                continue

            try:
                scores_im, retrieved_examples_im = ds_with_embeddings.get_nearest_examples(
                    "image_embeddings", img_embeddings.astype(np.float32), k=2)
                scores_t, retrieved_examples_t = ds_with_embeddings.get_nearest_examples(
                    "text_embeddings", img_embeddings.astype(np.float32), k=2)

                retrieved_examples = {
                    'image': retrieved_examples_im['image'] + retrieved_examples_t['image'],
                    'caption': retrieved_examples_im['caption'] + retrieved_examples_t['caption']
                }
            except Exception as e:
                print(f"Error retrieving examples for sample {counter}: {e}")
                continue

            for index_sample, exm in enumerate(retrieved_examples["image"]):
                temp = copy.deepcopy(template_format)
                text = retrieved_examples["caption"][index_sample]
                temp['messages'][0]['content'] = instructions + f"<image>" + f'background: {text}\n\n' + f"<image>" + question
                temp['messages'][1]['content'] = sample["answers"]
                temp['images'].append(exm)
                temp['images'].append(sample["image_path"])
                training_data.append(temp)

        # Determine dataset type from filename
        filename = os.path.basename(path)
        if 'val' in filename.lower():
            full_name_data = name_data + '_val_.json'
        elif 'train' in filename.lower():
            full_name_data = name_data + '_train_.json'
        elif 'test' in filename.lower():
            full_name_data = name_data + '_test_.json'
        else:
            full_name_data = name_data + '_unknown_.json'

        print(f"Dataset type: {full_name_data}")
        print(f"Training samples: {len(training_data)}")
        if training_data:
            print(f"Sample format: {training_data[0]}")
        
        output_path = os.path.join(path_to_save, full_name_data)
        dump_jsonl_file(training_data, output_path)
        print(f"Saved to: {output_path}")

def build_dataset_for_vqa_rad(instructions, index_dataset_path, path_to_save, name_data, path_to_data, float_16=False):
    """
    Build dataset for VQA-RAD task with retrieval-augmented generation.
    
    Args:
        instructions (str): Task instructions
        index_dataset_path (str): Path to the indexed dataset with embeddings
        path_to_save (str): Output directory
        name_data (str): Dataset name prefix
        path_to_data (list): List of dataset file paths
        float_16 (bool): Use float16 precision
    """
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
    ds_with_embeddings = Dataset.load_from_disk(index_dataset_path)
    ds_with_embeddings = ds_with_embeddings.add_faiss_index(column="text_embeddings")
    ds_with_embeddings = ds_with_embeddings.add_faiss_index(column="image_embeddings")
    
    if float_16:
        model = model.half().eval()
    else:
        model = model.eval()

    model = model.cuda()

    template_format = {
        "messages": [
            {
                "content": "",
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant"
            },
        ],
        "images": []
    }

    for path in path_to_data:
        if not os.path.exists(path):
            print(f"Warning: File {path} does not exist, skipping...")
            continue
            
        print(f"Processing: {path}")
        training_samples = load_jsonl_file(path)
        counter = 0
        training_data = []
        
        for sample in tqdm.tqdm(training_samples, desc=f"Processing {os.path.basename(path)}"):
            counter += 1
            
            try:
                with Image.open(sample["image_path"]) as img:
                    img_embeddings = model.encode_image(img)
                    if float_16:
                        img_embeddings = img_embeddings.astype(np.float16)
            except Exception as e:
                print(f"Error processing image {sample['image_path']}: {e}")
                continue

            try:
                scores_im, retrieved_examples_im = ds_with_embeddings.get_nearest_examples(
                    "image_embeddings", img_embeddings.astype(np.float32), k=2)
                scores_t, retrieved_examples_t = ds_with_embeddings.get_nearest_examples(
                    "text_embeddings", img_embeddings.astype(np.float32), k=2)

                retrieved_examples = {
                    'image': retrieved_examples_im['image'] + retrieved_examples_t['image'],
                    'caption': retrieved_examples_im['caption'] + retrieved_examples_t['caption']
                }
            except Exception as e:
                print(f"Error retrieving examples for sample {counter}: {e}")
                continue

            for index_sample, exm in enumerate(retrieved_examples["image"]):
                temp = copy.deepcopy(template_format)
                text = retrieved_examples["caption"][index_sample]
                temp['messages'][0]['content'] = instructions + f"<image>" + f'background: {text}\n\n' + f"<image>" + sample["question"]
                temp['messages'][1]['content'] = sample["answer"]
                temp['images'].append(exm)
                temp['images'].append(sample["image_path"])
                training_data.append(temp)

        # Determine dataset type from filename
        filename = os.path.basename(path)
        if 'val' in filename.lower():
            full_name_data = name_data + '_val_.json'
        elif 'train' in filename.lower():
            full_name_data = name_data + '_train_.json'
        elif 'test' in filename.lower():
            full_name_data = name_data + '_test_.json'
        else:
            full_name_data = name_data + '_unknown_.json'

        print(f"Dataset type: {full_name_data}")
        print(f"Training samples: {len(training_data)}")
        if training_data:
            print(f"Sample format: {training_data[0]}")
        
        output_path = os.path.join(path_to_save, full_name_data)
        dump_jsonl_file(training_data, output_path)
        print(f"Saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create LlamaFactory training dataset with retrieval augmentation')
    parser.add_argument('--task_type', type=str, choices=['classification', 'vqa_rad'], required=True,
                       help='Type of task (gina or vqa_rad)')
    parser.add_argument('--instructions', type=str, default="",
                       help='Task instructions for the model')
    parser.add_argument('--question', type=str, default="",
                       help='Question template (only for gina task)')
    parser.add_argument('--index_dataset_path', type=str, required=True,
                       help='Path to the indexed dataset with embeddings')
    parser.add_argument('--path_to_save', type=str, required=True,
                       help='Output directory for training data')
    parser.add_argument('--name_data', type=str, required=True,
                       help='Dataset name prefix')
    parser.add_argument('--path_to_data', type=str, nargs='+', required=True,
                       help='List of input dataset file paths')
    parser.add_argument('--float_16', action='store_true',
                       help='Use float16 precision')
    
    args = parser.parse_args()
    
    if args.task_type == 'classification':
        if not args.question:
            parser.error("--question is required for classification task")
        build_dataset_for_gina_qa(
            instructions=args.instructions,
            index_dataset_path=args.index_dataset_path,
            question=args.question,
            path_to_save=args.path_to_save,
            name_data=args.name_data,
            path_to_data=args.path_to_data,
            float_16=args.float_16
        )
    elif args.task_type == 'vqa_rad':
        build_dataset_for_vqa_rad(
            instructions=args.instructions,
            index_dataset_path=args.index_dataset_path,
            path_to_save=args.path_to_save,
            name_data=args.name_data,
            path_to_data=args.path_to_data,
            float_16=args.float_16
        )