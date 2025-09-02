import os
from typing import List
import argparse
import random
import torch.cuda
import sys
from src.torchrun_utils import init_distributed_mode_torchrun
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, load_or_initialize_index_text
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from train import train
import torch.distributed as dist
from datasets import Dataset
import warnings
from src.util import zip_project_files, save_script_to_checkpoint
os.environ["TOKENIZERS_PARALLELISM"] = "true"
NCONTEXT: str = "40"
PBSZ: str = "1"
PRECISION: str = "bf16"
GOLD_SCORE_MODE: str = "ppmean"
GPU_MAX_LENGTH: str = "384"
GEN_MAX_LENGTH: str = "32"
EPSILON: str = "1.0"
SMALL_EPSILON: str = "1e-5"
DROPOUT: str = "0.1"
WARMUP_STEPS: str = "20"
EVAL_FREQ: str = "30"
TEST_FREQ: str = "30"
LOG_FREQ: str = "30"
NO_REFRESH: str = "-1"
CHECK_FREQS: List[str] = ["--warmup_steps", "--save_freq", "--eval_freq"]
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
print("CUDA GPU", os.system("echo $CUDA_VISIBLE_DEVICES"))
PORT: str = str(random.randrange(15000, 16000))


def get_argument_value(all_args: List[str], argument_name: str) -> int:

    argument_idx = all_args.index(argument_name)
    return int(all_args[argument_idx + 1])


def check_valid_input_params(all_args: List[str], total_steps: int) -> None:

    for freq in CHECK_FREQS:
        try:
            arg_val = get_argument_value(all_args, freq)
        except ValueError:
            print(f"List does not contain value {freq}")

        assert arg_val < total_steps, f"The {freq} cannot be higher than the total steps {total_steps}. "


def set_parser_options(parser: argparse.Namespace, passed_args: List[str]) -> argparse.ArgumentParser:
    """
    Sets the default options for finetuning an Atlas model for a q&a task.
    """

    total_steps = get_argument_value(passed_args, "--total_steps")

    all_args = [
        "--write_results",
        "--shard_optim",
        "--shard_grads",
        "--temperature_gold",
        EPSILON,
        "--temperature_score",
        EPSILON,
        "--lr",
        SMALL_EPSILON,
        "--lr_retriever",
        SMALL_EPSILON,
        "--scheduler",
        "linear",
        "--weight_decay",
        EPSILON,
        "--generation_max_length",
        GEN_MAX_LENGTH,
        "--target_maxlength",
        GEN_MAX_LENGTH,
        "--gold_score_mode",
        GOLD_SCORE_MODE,
        "--precision",
        PRECISION,
        "--text_maxlength",
        GPU_MAX_LENGTH,
        "--per_gpu_batch_size",
        PBSZ,
        "--n_context",
        NCONTEXT,
        "--retriever_n_context",
        NCONTEXT,
        "--warmup_steps",
        WARMUP_STEPS,
        "--save_freq",
        str(total_steps - 1),
        "--eval_freq",
        EVAL_FREQ,
       "--test_freq",
       TEST_FREQ,
        "--log_freq",
        LOG_FREQ,
        "--main_port",
        PORT,
    ] + passed_args

    check_valid_input_params(all_args, total_steps)
    return parser.parse_args(all_args)


if __name__ == "__main__":

    options = get_options()
    opt = set_parser_options(options.parser, sys.argv[1:])
    opt.use_gradient_checkpoint_reader = False
    # run_distributed = True
    print("seed-2")
    if not opt.distributed_training:
        warnings.warn("Not distributed training")
        if "SLURM_JOB_ID" in os.environ.keys():
            os.environ.pop("SLURM_JOB_ID")
        opt.local_rank = -1

    torch.manual_seed(opt.seed)
    print("seed-1")
    if "TORCHELASTIC_RUN_ID" in os.environ:
        init_distributed_mode_torchrun(opt)
        print(opt.is_distributed, "is_distrubted")
        torch.cuda.set_device(dist.get_rank())
        print(dist.get_rank(), "rank")
    else:
        warnings.warn("START Init dis")
        slurm.init_distributed_mode(opt)
        slurm.init_signal_handler()
    print(dist_utils.get_world_size(), " world size")
    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    print("seed0")
    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    print("After seed 0 is main ", opt.is_main)
    zip_project_files(os.path.join(checkpoint_path, "python_code.zip"))
    save_script_to_checkpoint(checkpoint_path, opt.running_script_path)
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")
    logger.info(f"Seed 4")
    if opt.retriever_from_hf_index:
        passages = None
        ds_with_embeddings = Dataset.load_from_disk(opt.load_index_path)
        # ds_with_embeddings = ds_with_embeddings.add_faiss_index(column="text_embeddings")
        index = ds_with_embeddings.add_faiss_index(column="image_embeddings", device=0)
    else:
        index, passages = load_or_initialize_index(opt)
        if opt.text_retrieval:
            index_text, passages_text = load_or_initialize_index_text(opt)
        else:
            index_text = None

    model, optimizer, scheduler, retr_optimizer, retr_scheduler, text_retr_optimizer, text_retr_scheduler = load_or_initialize_atlas_model(opt)

    if opt.disable_gradient_reader and not opt.skip_model:
        # if False:
        if opt.qwen_model or opt.pixtral_model:
            for name, param in model.reader.named_parameters():
                param.requires_grad = False

        elif opt.reader_model_type == 'flamingo':
            for name, param in model.reader.named_parameters():
                if name in model.params_to_optimize_reader:
                    # print("enable gradients")
                    param.requires_grad = False



    # if opt.is_distributed:
    if opt.is_distributed:
        logger.info(f"distributed training")
        # if opt.shard_grads:
        if False:
            logger.info(f"distributed training opt.shard_grads")
            import fairscale.nn.data_parallel

            model.reader = fairscale.nn.data_parallel.ShardedDataParallel(
                model.reader, optimizer, auto_refresh_trainable=False
            )
            if opt.train_retriever:
                model.retriever = fairscale.nn.data_parallel.ShardedDataParallel(
                    model.retriever, retr_optimizer, auto_refresh_trainable=False
                )
        else:
            logger.info(f"distributed training regular")
            logger.info(f"local rank {str(opt.local_rank)}")
            if 'SLURM_NTASKS' in os.environ.keys():
                logger.info(f"slurm ntasks {os.environ['SLURM_NTASKS']}")

            if 'SLURM_PROCID' in os.environ.keys():
                logger.info(f"slurm rank {os.environ['SLURM_PROCID']}")
            model = model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=True,
            )
            model._set_static_graph()

    logger.info("Start finetuning")
    dist_utils.barrier()
    step = 0
    train(
        model,
        index,
        index_text,
        passages,
        optimizer,
        scheduler,
        retr_optimizer,
        retr_scheduler,
        text_retr_optimizer,
        text_retr_scheduler,
        step,
        opt,
        checkpoint_path,
    )
