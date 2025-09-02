#!/bin/bash
#SBATCH --cpus-per-task 5
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-user=nir.mazor@mail.huji.ac.il
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --time=1-12:00:00
#SBATCH --job-name=re_ge_embeddings




python  evaluate.py \
    --name "[ENTER NAME]" \
   --reader_model_type flamingo \
    --retriever_format "{img_path}" \
    --retriever_text_format "{text}" \
    --text_maxlength 512 \
    --eval_data "dsfdk.json" \
    --n_context 5 --retriever_n_context 5 \
    --main_port 15000 \
    --index_mode "flat" \
    --task "vqa_rad" \
    --save_index_path "[ENTER PATH]" \
    --write_results \
    --ret_acc_text \
    --multi-modal \
    --save_index_n_shards 128 \
    --retrieve_only \
    --per_gpu_embedder_batch_size 512 \
    --passages "[ENTER PATH]" \
    --retriever_multimodal_model_type "jina"

