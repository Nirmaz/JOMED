# Training the Reader on the LlamaFactory Platform


## Constructing the Index

To generate text and image embeddings for your dataset, use the embedding script:

```bash
python3 embedding_llama_factory.py \
  --dataset_file /path/to/your/dataset.jsonl \
  --path_save /path/to/save/embeddings \
  --dataset_name your_dataset_name \
  --float_16
```

## Construct Training Data for Model

As explained in the Llama-Factory documentation (<https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md>), to create
a valid dataset you need to create a custom dataset. We supply the script to create such a dataset.

### For classification tasks:
```bash
python3 create_dataset_for_llama_factory.py \
  --task_type gina \
  --instructions "Answer the medical question based on the provided context." \
  --question "What is the severity of diabetic retinopathy?" \
  --index_dataset_path /path/to/indexed/embeddings \
  --path_to_save /path/to/save/training/data \
  --name_data dataset_name \
  --path_to_data /path/to/train.jsonl /path/to/val.jsonl /path/to/test.jsonl \
  --float_16
```

### For VQA-RAD tasks:
```bash
python3 create_dataset_for_llama_factory.py \
  --task_type vqa_rad \
  --instructions "Answer the medical question about the image." \
  --index_dataset_path /path/to/indexed/embeddings \
  --path_to_save /path/to/save/training/data \
  --name_data dataset_name \
  --path_to_data /path/to/train.jsonl /path/to/val.jsonl /path/to/test.jsonl \
  --float_16
```

## Train the Reader

We supply the training configuration used for training the models in `lamafactory_scripts/training_configuration` and then
train as detailed in https://github.com/hiyouga/LLaMA-Factory/tree/main.


