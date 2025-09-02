
import json
import logging
import math
import sys
import os
import zipfile
from pathlib import Path
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import shutil
import numpy as np
import torch
import re

from src import dist_utils

Number = Union[float, int]

logger = logging.getLogger(__name__)


def f1_macro(y_true, y_pred):
    """
    Calculate F1 macro score using NumPy.

    Parameters:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels

    Returns:
    float: F1 macro score (average F1 score across all classes)
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    f1_scores = []

    for class_label in classes:
        # Calculate True Positives, False Positives, False Negatives for this class
        tp = np.sum((y_true == class_label) & (y_pred == class_label))
        fp = np.sum((y_true != class_label) & (y_pred == class_label))
        fn = np.sum((y_true == class_label) & (y_pred != class_label))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 score for this class
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # Return macro average (mean of all class F1 scores)
    return np.mean(f1_scores)

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy between two NumPy arrays.

    Parameters:
    y_true (numpy array): Ground truth labels.
    y_pred (numpy array): Predicted labels.

    Returns:
    float: The accuracy as a percentage.
    """
    # Ensure both arrays have the same shape
    assert y_true.shape == y_pred.shape, "Shape of ground truth and prediction arrays must be the same"

    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Calculate the total number of predictions
    total_predictions = y_true.shape[0]

    # Calculate accuracy as a percentage
    accuracy = correct_predictions / total_predictions

    return accuracy

class AbstractProcessor(ABC):
    """
    Abstract class for processors to show what methods they need to implement.
    Processors handle text encoding and image preprocessing.
    """

    @abstractmethod
    def encode_text(self, prompt):
        pass

    @abstractmethod
    def preprocess_images(self, images: list):
        pass

def clean_substring_from_nonascii(string):
    return re.sub(r'[^\x00-\x7F]+', '', string)

def load_jsonl_file(file_path):
    files = []
    for line in open(file_path):
        if line.strip() != "":
            item = json.loads(line)
            files.append(item)


    return files

def dump_jsonl_file(list_dict, path_json):
    # path_json_save = os.path.join(path_json, 'mimic.jsonl.all')
    with open(path_json, 'w') as f:
        for entry in list_dict:
            json.dump(entry, f)
            f.write('\n')

def init_logger(is_main=True, is_distributed=False, filename=None):

    print("waiting second and is main", is_main)
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return logger


class LlavaProcessor(AbstractProcessor):
    """
    Processor class for Flamingo.
    """

    def __init__(self, tokenizer, vision_processor, build_chat_template = None):
        """
        OF does not use same vision processor, image_processor only transforms single image
        """

        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.builder = build_chat_template
        self.tokenizer.padding_side = 'left'
        self.builder.padding_side = 'left'


    def encode_text(self, prompt):
        pass


    def preprocess_images(self, images: list):
        pass



def init_tb_logger(dirname, is_main):

    tb_logger = None
    if is_main:
        try:
            from torch.utils import tensorboard

            tb_logger = tensorboard.SummaryWriter(dirname)
        except:
            logger.warning("Tensorboard is not available.")
    return tb_logger


def cast_to_precision(model, precision):
    if precision == "fp32":
        return model
    elif precision == "fp16":
        model.to(torch.float16)
    elif precision == "bf16":
        model.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported precision {precision}, must be one of fp32, fp16, bf16")
    return model


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup)) + self.ratio

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        return 1.0


class IndexRefreshScheduler(object):
    def __init__(self, format_str: str, freeze_retriever_steps: int, train_retriever: bool):
        """Build an index refresh scheduler

        format_str: string that specifies the schedule.
            has the format: startstep-endstep:refreshrate,startstep-endstep:refreshrate
            e.g. format_str="0-100:10,100-1000000:500" will refresh the index every 10 steps for the first 100 steps
            and then every 500 steps from step 100 to 1M.

            Syntactic Sugar for a fixed schedule: can just pass in a single number
            e.g. format_str="100" will refresh the index every 100 steps

            -1 to never refresh
        )
        """
        self.format_str = format_str
        self.train_retriever = train_retriever
        self.freeze_retriever_steps = freeze_retriever_steps
        self.steps2rates = IndexRefreshScheduler.parse_index_refresh_schedule_string(format_str)

    @classmethod
    def parse_index_refresh_schedule_string(cls, format_str):
        parsed = []
        if format_str == "-1":
            parsed = [(0, 2**32, 2**32)]
        elif format_str.isdigit():
            parsed = [(0, 2**32, int(format_str))]
        else:
            for piece in format_str.split(","):
                startend, rate = piece.split(":")
                start, end = startend.split("-")
                parsed.append((int(start), int(end), int(rate)))
        return parsed

    def is_time_to_refresh(self, step):
        if not (self.train_retriever or step == 0):  # if retriever is not trained only refresh at step 0
            return False
        if not step == 0 and step < self.freeze_retriever_steps:  # freeze first steps
            return False
        for st, en, rate in self.steps2rates:
            if st <= step < en:
                steps_since_refresh_schedule_change = step - st
                return (steps_since_refresh_schedule_change % rate) == 0
        logger.warn(
            "cant calculate refresh rate for this step, I dont have data here"
            " its likely training step is higher than the specificed refresh rate see --index_refresh_rate for help."
        )
        return False


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim_open_flamingo(opt, model):
    retr_optimizer = None
    text_retr_optimizer = None

    if not opt.qwen_model and not opt.pixtral_model:

        params_to_optimize_reader = model.reader.named_parameters()
        params_to_optimize_reader = list(
            filter(
                lambda x: x[1].requires_grad
                          and not getattr(x[1], "exclude_from_optimizer", False),
                params_to_optimize_reader,
            )
        )
        params_to_optimize_reader_names = [name for name, params in model.reader.named_parameters() if
                                              params.requires_grad]

        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize_reader:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": opt.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize_reader), lr=opt.lr
        )
    params_to_optimize_retriever_names = None
    params_to_optimize_text_retriever_names = None
    if opt.train_retriever_img:
        if opt.retriever_multimodal_model_type == 'biomedclip' or opt.retriever_multimodal_model_type == 'jina':
            params_to_optimize_retriever = [p for p in model.retriever.parameters() if p.requires_grad]
            retr_optimizer = torch.optim.AdamW(params_to_optimize_retriever, lr=opt.lr_retriever, weight_decay=opt.weight_decay)
            params_to_optimize_retriever_names = [name for name, params in model.retriever.named_parameters() if params.requires_grad]

    if opt.train_retriever_text:
        params_to_optimize_text_retriever = [p for p in model.retriever_for_q_to_text.parameters() if p.requires_grad]
        text_retr_optimizer = torch.optim.AdamW(params_to_optimize_text_retriever, lr=opt.lr_retriever,
                                           weight_decay=opt.weight_decay)
        params_to_optimize_text_retriever_names = [name for name, params in model.retriever_for_q_to_text.named_parameters() if
                                              params.requires_grad]

        # retr_optimizer = optim_class(params=model.retriever.parameters(), lr=opt.lr_retriever, **optim_args)

    retr_scheduler = None
    text_retr_scheduler = None
    if not opt.qwen_model and not opt.pixtral_model:
        if opt.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=opt.warmup_steps,
                num_training_steps=opt.total_steps,
            )
        elif opt.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=opt.warmup_steps,
                num_training_steps=opt.total_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=opt.warmup_steps
            )

    if opt.train_retriever_img:

        retr_scheduler = get_cosine_schedule_with_warmup(
            retr_optimizer,
            num_warmup_steps=opt.warmup_steps,
            num_training_steps=opt.total_steps,
        )

    if opt.train_retriever_text:
        text_retr_scheduler = get_cosine_schedule_with_warmup(
            text_retr_optimizer,
            num_warmup_steps=opt.warmup_steps,
            num_training_steps=opt.total_steps,
        )

    if not opt.qwen_model and not opt.pixtral_model:
        return optimizer, scheduler, retr_optimizer, retr_scheduler, params_to_optimize_reader_names, params_to_optimize_retriever_names, params_to_optimize_text_retriever_names, text_retr_optimizer, text_retr_scheduler
    else:
        return None, None, retr_optimizer, retr_scheduler, None, params_to_optimize_retriever_names, params_to_optimize_text_retriever_names, text_retr_optimizer, text_retr_scheduler



def interleave_tensors(A: torch.Tensor, B: torch.Tensor, chunk_size: int = 2) -> torch.Tensor:

    # Interleave chunks from A and B
    interleaved_chunks = []
    for i in range(0, A.shape[0], chunk_size):
        interleaved_chunks.append(A[i: i + chunk_size])
        interleaved_chunks.append(B[i: i + chunk_size])

    # Concatenate all chunks into one tensor
    return torch.cat(interleaved_chunks, dim=0)
def normalize_sublists(lists):
    normalized = []
    for sub in lists:
        if not sub:  # if the sublist is empty, keep it as is
            normalized.append(sub)
        else:
            max_val = max(sub)
            # Avoid division by zero if the maximum value is 0
            if max_val == 0:
                normalized.append(sub)
            else:
                normalized.append([x / max_val for x in sub])
    return normalized
def set_optim(opt, model):
    from src_before_txt.AdamWFP32Copy import AdamWFP32Copy

    retr_optimizer = None
    optim_class = AdamWFP32Copy
    optim_args = {"weight_decay": opt.weight_decay, "betas": (0.9, opt.beta2), "eps": opt.epsilon}
    if opt.is_distributed and opt.shard_optim:
        from fairscale.optim.oss import OSS

        optim_args["optim"] = optim_class
        optim_args["force_broadcast_object"] = True
        optim_class = OSS
    optimizer = optim_class(params=model.reader.parameters(), lr=opt.lr, **optim_args)
    if opt.train_retriever:
        retr_optimizer = optim_class(params=model.retriever.parameters(), lr=opt.lr_retriever, **optim_args)

    retr_scheduler = None
    scheduler_args = {"warmup": opt.warmup_steps, "total": opt.total_steps, "ratio": 0.1}
    if opt.scheduler == "linear":
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == "cosine":
        scheduler_class = CosineScheduler
    elif opt.scheduler == "fixed":
        scheduler_class = FixedScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    if opt.train_retriever:
        retr_scheduler = scheduler_class(retr_optimizer, **scheduler_args)

    return optimizer, scheduler, retr_optimizer, retr_scheduler


def compute_grad_stats(model):
    with torch.no_grad():
        stats = []
        for name, p in get_unwrapped_model_if_wrapped(model).reader.named_parameters():
            if p.grad is not None:
                s1 = torch.min(torch.abs(p.grad)).item()
                s2 = torch.max(torch.abs(p.grad)).item()
                s3 = torch.mean(torch.abs(p.grad)).item()
                s4 = torch.linalg.norm(p.grad).item()
                stats += [s1, s2, s3, s4]
            else:
                stats += [0.0, 0.0, 0.0, 0.0]
        stats = torch.Tensor(stats).cuda()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        stats = stats.view(-1, 4)

        res = {}
        res["skip_example"] = (torch.any(torch.isinf(stats)) or torch.any(torch.isnan(stats))).item()
        res["min"] = stats.min(0)[0][0].item()
        res["max"] = stats.max(0)[0][1].item()
        res["mean"] = stats.mean(0)[2].item()
        return res


def write_output(glob_path, output_path):
    files = list(glob_path.glob("*.txt"))
    files.sort()
    with open(output_path, "w") as outfile:
        for path in files:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, dataset_name, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / "tmp_dir"
    write_path.mkdir(exist_ok=True)

    # Write the current rank's data to a temporary file
    tmp_path = write_path / f"{opt.global_rank}.json"
    with open(tmp_path, "w") as fw:
        json.dump(data, fw)

    # Synchronize processes if distributed
    if opt.is_distributed:
        torch.distributed.barrier()

    # If this is the main process, gather and consolidate all data
    if opt.is_main:
        final_path = dir_path / f"{dataset_name}.jsonl"
        logger.info(f"Writing dataset with scores at {final_path}")

        results_path = list(write_path.glob("*.json"))
        results_path.sort()  # Ensure that the sorting is appropriate for your filenames

        alldata = []
        for path in results_path:
            try:
                # Check if the file is empty
                if path.stat().st_size == 0:
                    logger.error(f"File is empty: {path}")
                    continue

                # Attempt to load the JSON data
                with open(path, "r") as f:
                    first_char = f.read(1)  # Read the first character
                    if not first_char:  # If the file is empty
                        logger.error(f"File is empty: {path}")
                        continue
                    f.seek(0)  # Reset file pointer to the start
                    data = json.load(f)

            except FileNotFoundError:
                logger.error(f"File not found: {path}")
                continue
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON in file: {path}")
                continue

            # Extend the final data list with the content of the current file
            alldata.extend(data)

            # Remove the temporary file after processing
            path.unlink()

        # Write the consolidated data to the final file
        with open(final_path, "w") as fout:
            for ex in alldata:
                json.dump(ex, fout, ensure_ascii=True)
                fout.write("\n")

        # Remove the temporary directory if it's empty
        if not any(write_path.iterdir()):
            write_path.rmdir()
# def save_distributed_dataset(data, dataset_name, opt):
#     dir_path = Path(opt.checkpoint_dir) / opt.name
#     write_path = dir_path / "tmp_dir"
#     write_path.mkdir(exist_ok=True)
#     tmp_path = write_path / f"{opt.global_rank}.json"
#     with open(tmp_path, "w") as fw:
#         json.dump(data, fw)
#     if opt.is_distributed:
#         torch.distributed.barrier()
#     if opt.is_main:
#         final_path = dir_path / f"{dataset_name}.jsonl"
#         logger.info(f"Writing dataset with scores at {final_path}")
#         results_path = list(write_path.glob("*.json"))
#         results_path.sort()
#
#         alldata = []
#         for path in results_path:
#             with open(path, "r") as f:
#                 data = json.load(f)
#             alldata.extend(data)
#             path.unlink()
#         with open(final_path, "w") as fout:
#             for ex in alldata:
#                 json.dump(ex, fout, ensure_ascii=True)
#                 fout.write("\n")
#         write_path.rmdir()


def avg_dist_dict(keys, dictionary):
    avg = {}
    for m in keys:
        v = dictionary[m]
        if len(v) > 0:
            avg[m] = np.mean(v)
        else:
            avg[m] = 0.0
        avg[m] = dist_utils.weighted_average(avg[m], len(v))[0]
    return avg


class WeightedAvgStats:
    """provides an average over a bunch of stats"""

    def __init__(self):
        self.raw_stats: Dict[str, float] = defaultdict(float)
        self.total_weights: Dict[str, float] = defaultdict(float)

    def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
        for key, (value, weight) in vals.items():
            self.raw_stats[key] += value * weight
            self.total_weights[key] += weight

    @property
    def stats(self) -> Dict[str, float]:
        return {x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()}

    @property
    def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
        return {x: (self.raw_stats[x] / self.total_weights[x], self.total_weights[x]) for x in self.raw_stats.keys()}

    def reset(self) -> None:
        self.raw_stats = defaultdict(float)
        self.total_weights = defaultdict(float)

    @property
    def average_stats(self) -> Dict[str, float]:
        keys = sorted(self.raw_stats.keys())
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(keys, src=0)
        global_dict = {}
        for k in keys:
            if not k in self.total_weights:
                v = 0.0
            else:
                v = self.raw_stats[k] / self.total_weights[k]
            v, _ = dist_utils.weighted_average(v, self.total_weights[k])
            global_dict[k] = v
        return global_dict


def get_unwrapped_model_if_wrapped(model):
    if hasattr(model, "module"):
        return model.module
    return model


class AbstractProcessor(ABC):
    """
    Abstract class for processors to show what methods they need to implement.
    Processors handle text encoding and image preprocessing.
    """

    @abstractmethod
    def encode_text(self, prompt):
        pass

    @abstractmethod
    def preprocess_images(self, images: list):
        pass


class FlamingoProcessor(AbstractProcessor):
    """
    Processor class for Flamingo.
    """

    def __init__(self, tokenizer, vision_processor, opt):
        """
        OF does not use same vision processor, image_processor only transforms single image
        """
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.opt = opt
    def encode_text(self, prompt, generation_mode = False):
        if generation_mode:
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.padding_side = "right"
        # For generation padding tokens should be on the left
        return self.tokenizer(prompt,
                    max_length=self.opt.text_maxlength,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                              )

    def preprocess_images(self, images: list):
        vision_x = [self.vision_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        return vision_x

def save_script_to_checkpoint(checkpoint_dir, script_path):
    try:
        # Get the absolute path of the current running script
        # script_path = os.path.abspath(__file__)
        script_name = os.path.basename(script_path)  # Get script file name

        # Create the checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Define the path where the script will be saved in the checkpoint directory
        checkpoint_script_path = os.path.join(checkpoint_dir, script_name)

        # Copy the script to the checkpoint directory
        shutil.copy(script_path, checkpoint_script_path)

        print(f"Script successfully saved to {checkpoint_script_path}")

    except:
        # Handle the case where __file__ is not available (e.g., Jupyter notebook)
        print("Cannot save the script as __file__ is not available (interactive session).")


def zip_project_files(output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, start='.'))


def load_jsonl_file(file_path):

    files = []
    counter = 0
    for line in open(file_path):
        if line.strip() != "":
            item = json.loads(line)
            files.append(item)
        counter = counter + 1
    return files