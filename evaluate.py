import os
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src.torchrun_utils import init_distributed_mode_torchrun
from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task
from src.util import clean_substring_from_nonascii
from src.util import normalize_sublists, interleave_tensors
from src.util import zip_project_files, load_jsonl_file
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator


@torch.no_grad()
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)

        if not opt.multi_modal:
            retrieved_passages, _ = unwrapped_model.retrieve(
                index,
                opt.n_context,
                query,
                query_enc["input_ids"].cuda(),
                query_enc["attention_mask"].cuda(),
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
        else:

            if opt.retriever_multimodal_model_type == "clipmd":
                if query_enc is not None:
                    image_encode = query_enc["pixel_values"].cuda()
                retrieved_passages, _ = unwrapped_model.retrieve(
                    index,
                    opt.n_context,
                    query,
                    image_encode,
                    batch_metadata=batch_metadata,
                    filtering_fun=task.filter,
                )
            elif opt.retriever_multimodal_model_type == "biomedclip":
                if query_enc is not None:
                    query_enc = query_enc.cuda()
                retrieved_passages, _ = unwrapped_model.retrieve(
                    index,
                    opt.n_context,
                    query,
                    query_enc,
                    batch_metadata=batch_metadata,
                    filtering_fun=task.filter,
                )

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {"query": query[k], "answers": gold, "passages": retrieved_passages[k]}
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


def extract_model_answer_from_pred(pred, task):

    if task == 'cr_multiple_choice':
        return pred[pred.rindex("answer") + 8:pred.rindex("answer") + 9]
    elif task == 'cr_qa':
        return pred[pred.rindex("answer") + 8:].split()[0][:-1]
    elif task == 'cr_qa_open':
        return pred[pred.rindex("answer") + 8:]
    else:
        return pred

@torch.no_grad()
def fuse_model_predicitions(generation, scores, logits,  batch_size, n_context):
    batch_generation = generation.view(batch_size, n_context, generation.shape[1])
    batch_socores = torch.tensor(scores).cuda()
    fused_gen = []
    for bbb in range(batch_size):
        bgT = torch.transpose(batch_generation[bbb], 0, 1)
        bsc = batch_socores[bbb].repeat(bgT.shape[0], 1)


        unique_numbers = torch.unique(bgT)
        num_unique = len(unique_numbers)
        num_rows = bgT.size(0)

        # Create an array to store aggregated weights for each unique number in each row
        aggregated_weights = torch.zeros((num_rows, num_unique))

        # Compute the aggregated weights for each unique number in each row
        for i, number in enumerate(unique_numbers):
            mask = (bgT == number).float()
            aggregated_weights[:, i] = torch.sum(mask * bsc, dim=1)

        # Find the index of the maximum weight for each row
        max_indices = torch.argmax(aggregated_weights, dim=1)

        # Get the unique numbers corresponding to the max indices
        chosen_numbers = unique_numbers[max_indices]
        fused_gen.append(chosen_numbers)

    fused_gen = torch.stack(fused_gen, dim = 0)
    return fused_gen

@torch.no_grad()
def fuse_model_predicitions_u( scores, logits,  batch_size, n_context, max_new_tokens):
    logits_r = torch.stack(logits)
    logits_reformated = logits_r.view(len(logits_r), batch_size, n_context, -1)
    logits_reformated = logits_reformated.permute(1, 2, 0, 3)

    lambda_weights = torch.softmax(torch.tensor(np.array(scores)), dim=-1).unsqueeze(-1).unsqueeze(-1).cuda()

    weighted_logits = (logits_reformated * lambda_weights).sum(dim=1)
    probabilities = torch.softmax(weighted_logits, dim=-1)

    predicted_token_ids = torch.argmax(probabilities, dim=-1)
    return predicted_token_ids



@torch.no_grad()
def process_results(generation, opt, reader_tokenizer, retrieved_passages,answers, task, batch, query_text, batch_metadata, logits,dataset_wpred, append_only_one_passage_to_query, metrics, save_logits = False):
    if save_logits:
        if not retrieved_passages[0][0]["img_path"] == "":
            logits_r = torch.stack(logits)
            logits_reformated = logits_r.view(len(logits_r), len(retrieved_passages), len(retrieved_passages[0]), -1)
            num_tokens = [reader_tokenizer.tokenizer(answ)["input_ids"][0] for answ in task.answers]
            logits_reformated_answer_tokens = logits_reformated[:, :, :, num_tokens]
            logits_reformated_answer_tokens = logits_reformated_answer_tokens.sum(dim=0)
        else:
            logits_r = torch.stack(logits)
            logits_reformated = logits_r.view(len(logits_r), len(retrieved_passages), -1)
            num_tokens = [reader_tokenizer.tokenizer(answ)["input_ids"][0] for answ in task.answers]
            logits_reformated_answer_tokens = logits_reformated[:, :, num_tokens]
            logits_reformated_answer_tokens = logits_reformated_answer_tokens.sum(dim=0)




    for k, g in enumerate(generation):
        if opt.decoder_prompt_format is not None:
            query_ids = reader_tokenizer.encode(
                opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
            )
            g = g[513 :]
        if opt.reader_model_type != 'flamingo':
            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
        else:
            if len(g) > opt.max_new_tokens + 1:
                g = g[-opt.max_new_tokens:]
            pred = reader_tokenizer.tokenizer.decode(g)

        if append_only_one_passage_to_query:
            index_pass = int(np.floor((k) / len(retrieved_passages[0])))
            gold = [answers[index_pass]] if not "answers" in batch else batch["answers"][index_pass]
        else:
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
        sample_metrics = task.evaluation(pred, gold)
        for key, value in sample_metrics.items():
            metrics[key].append(value)


        if opt.write_results:
            clean_pred = clean_substring_from_nonascii(pred)
            if append_only_one_passage_to_query:
                index_pass = int(np.floor((k) / len(retrieved_passages[0])))
            else:
                index_pass = k
            if save_logits:
                if not retrieved_passages[0][0]["img_path"] == "":
                    logits_batch = [list(log.detach().cpu().numpy().astype(float)) for log in logits_reformated_answer_tokens[index_pass]]
                else:
                    logits_batch =  list(logits_reformated_answer_tokens[index_pass].detach().cpu().numpy().astype(float))

                ex = {"query": query_text[index_pass], "answers": gold, "generation": clean_pred, "logits": logits_batch}
            else:
                ex = {"query": query_text[index_pass], "answers": gold, "generation": clean_pred}
            if not opt.dont_write_passages:
                if append_only_one_passage_to_query:
                    ex["passages"] = [retrieved_passages[index_pass][int(k - len(retrieved_passages[0])*index_pass)]]
                else:
                    ex["passages"] = retrieved_passages[index_pass]
            if batch_metadata is not None:
                ex["metadata"] = batch_metadata[index_pass]
            if opt.task == "multiple_choice" and False:
                ex["choice_logits"] = task.get_choice_logits(logits[index_pass])
            if "id" in batch:
                ex["id"] = batch["id"][index_pass]
            dataset_wpred.append(ex)


def split_tensor_in_T_chunks(input_tensor: torch.Tensor, T: int):
    """
    Splits the input tensor into three parts:
      1) The original tensor.
      2) A tensor composed of every "even" T-chunk along the batch dimension.
      3) A tensor composed of every "odd" T-chunk along the batch dimension.

    Args:
        input_tensor (torch.Tensor): A tensor of shape [B, X, Y]
        T (int): The chunk size in the batch dimension.
                 Assumes that B is a multiple of 2*T.

    Returns:
        original (torch.Tensor): The original input tensor [B, X, Y].
        even_tensor (torch.Tensor): The tensor formed by stacking chunks
                                    at indices 0, 2, 4, ... (each chunk is size T).
                                    Shape: [B/2, X, Y].
        odd_tensor (torch.Tensor):  The tensor formed by stacking chunks
                                    at indices 1, 3, 5, ... (each chunk is size T).
                                    Shape: [B/2, X, Y].
    """
    # 1) The original
    original = input_tensor

    B, X = input_tensor.shape

    # 2) Reshape to group every T slices into a block
    #    Now shape is [B//T, T, X, Y].
    chunked = input_tensor.reshape(B // T, T, X)

    # 3) Split into even and odd blocks
    #    even_blocks has shape [B/(2T), T, X, Y] after slicing
    #    odd_blocks has the same shape
    even_blocks = chunked[::2]
    odd_blocks = chunked[1::2]

    # 4) Reshape each set of blocks back to [B/2, X, Y]
    even_tensor = even_blocks.reshape(-1, X)
    odd_tensor = odd_blocks.reshape(-1, X)

    return original, even_tensor, odd_tensor



@torch.no_grad()
def evaluate_multi_modal(model, index,index_text,  opt, data_path, step=None):
    # torch.cuda.empty_cache()
    model.eval()
    metrics = defaultdict(lambda: [])
    metrics_text = defaultdict(lambda: [])
    metrics_img = defaultdict(lambda: [])
    metrics_fuse = defaultdict(lambda: [])
    metrics_fuse_img = defaultdict(lambda: [])
    metrics_fuse_text = defaultdict(lambda: [])

    dataset_wpred = []
    dataset_wpred_text = []
    dataset_wpred_img = []

    dataset_wpred_fused = []
    dataset_wpred_fused_text = []
    dataset_wpred_fused_img = []

    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    if hasattr(task, 'set_tokenizer'):
        task.set_tokenizer(unwrapped_model.reader_tokenizer.tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):

        query_image = batch.get("query", [""])
        query_text = batch.get("query_text", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")

        if opt.closed_book:
            retrieved_passages = []
            for i in range(len(query_image)):
                retrieved_passages.append(task.passages_to_use)
        else:

            query_enc, labels, decoder_input_ids = unwrapped_model.multi_modal_tokenize(query_image,query_text, answers, target_tokens=target_tokens)
            query_enc = query_enc.cuda()
            retrieved_passages, retrieved_passages_text, retrieved_passages_img, scores, scores_text, scores_img  = unwrapped_model.multi_modal_retrieve(index, index_text, opt.n_context, query_image, query_enc,
                                                    batch_metadata, task)
            if scores_img is not None and scores_text is not None:
                scores_text_norm = normalize_sublists(scores_text)
                scores_img_norm = normalize_sublists(scores_img)

                scores = [
                    sub1 + sub2
                    for sub1, sub2 in zip(scores_img_norm, scores_text_norm)
                ]
                # scores = scores_img_norm + scores_text_norm


        if (len(query_image) == 0) or (len(query_image[0]) == 0):
            continue
        if opt.qwen_model:
            reader_tokens_img, reader_tokens_text = unwrapped_model.preprocess_input_for_qwen( query_image, query_text, task, retrieved_passages)
            if opt.img_retrieval:
                reader_tokens_img_img, reader_tokens_text_img = unwrapped_model.preprocess_input_for_qwen(query_image,
                                                                                                     query_text, task,
                                                                                                     retrieved_passages_img)
            if opt.text_retrieval:
                reader_tokens_img_txt, reader_tokens_text_txt = unwrapped_model.preprocess_input_for_qwen(query_image,
                                                                                                             query_text,
                                                                                                             task,
                                                                                                             retrieved_passages_text)
        elif opt.pixtral_model:
            reader_tokens_img, reader_tokens_text = unwrapped_model.preprocess_input_for_pixtral( query_image, query_text, task, retrieved_passages)
            if opt.img_retrieval:
                reader_tokens_img_img, reader_tokens_text_img = unwrapped_model.preprocess_input_for_pixtral(query_image,
                                                                                                     query_text, task,
                                                                                                     retrieved_passages_img)
            if opt.text_retrieval:
                reader_tokens_img_txt, reader_tokens_text_txt = unwrapped_model.preprocess_input_for_pixtral(query_image,
                                                                                                             query_text,
                                                                                                             task,
                                                                                                             retrieved_passages_text)
        else:
            reader_tokens_img, reader_tokens_text, _, _, _, _ = unwrapped_model.multi_modal_preprocess_input_to_reader(query_image, query_text, retrieved_passages, None, task,
                                                   generation_mode=True, append_only_one_passage_to_query = opt.append_only_one_passage_to_query)

        if opt.qwen_model or opt.pixtral_model:
            generation, logits, _, _ = unwrapped_model.generate_qwen(
                reader_tokens_text, reader_tokens_img, choices=batch["choices"] if "choices" in batch else None
            )
            if opt.img_retrieval:
                generation_img2, logits_img2, _, _ = unwrapped_model.generate_qwen(
                    reader_tokens_text_img, reader_tokens_img_img, choices=batch["choices"] if "choices" in batch else None
                )
            if opt.text_retrieval:
                generation_txt2, logits_txt2, _, _ = unwrapped_model.generate_qwen(
                    reader_tokens_text_txt, reader_tokens_img_txt, choices=batch["choices"] if "choices" in batch else None
                )

        else:
            generation, logits, _, _ = unwrapped_model.generate_flamingo(
                reader_tokens_text, reader_tokens_img,  choices=batch["choices"] if "choices" in batch else None
            )

        if opt.text_retrieval and opt.img_retrieval:
            generation_txt22 = generation_txt2[:, -(opt.max_new_tokens + 2): ]
            generation_img22 = generation_img2[:, -(opt.max_new_tokens + 2):]
            generation_img2_txt2 = interleave_tensors(generation_img22, generation_txt22, chunk_size= len(retrieved_passages_text[0]))
            logits_img2_txt2 = tuple(
                interleave_tensors(a, b, len(retrieved_passages_text[0])) for a, b in zip(logits_img2, logits_txt2))

            process_results(generation_img2_txt2, opt, reader_tokenizer, retrieved_passages, answers, task, batch,
                            query_text,
                            batch_metadata, logits_img2_txt2, dataset_wpred, opt.append_only_one_passage_to_query,
                            metrics, save_logits=True)

            process_results(generation_img2, opt, reader_tokenizer, retrieved_passages_img, answers, task, batch,
                            query_text,
                            batch_metadata, logits, dataset_wpred_img, opt.append_only_one_passage_to_query,
                            metrics_img)

            process_results(generation_txt2, opt, reader_tokenizer, retrieved_passages_text, answers, task, batch,
                            query_text,
                            batch_metadata, logits, dataset_wpred_text, opt.append_only_one_passage_to_query,
                            metrics_text)
        else:
            process_results(generation, opt, reader_tokenizer, retrieved_passages, answers, task, batch, query_text,
                        batch_metadata, logits, dataset_wpred, opt.append_only_one_passage_to_query, metrics, save_logits=True)



        if opt.fused_eval and opt.append_only_one_passage_to_query:
            batch_size = len(query_image)
            n_context = len(retrieved_passages[0])

            if opt.text_retrieval and opt.img_retrieval:
                n_context_img_text = len(retrieved_passages_text[0])

                logits__img2_txt2 = tuple(
                    interleave_tensors(a, b, len(retrieved_passages_text[0])) for a, b in zip(logits_img2, logits_txt2))

                fused_generation_img2_txt2 = fuse_model_predicitions_u(scores, logits__img2_txt2, batch_size, n_context,
                                                                   max_new_tokens=opt.max_new_tokens,)

                fused_generation_img2 = fuse_model_predicitions_u(scores_img, logits_img2, batch_size,
                                                                 n_context_img_text,
                                                                 max_new_tokens=opt.max_new_tokens)
                fused_generation_txt2 = fuse_model_predicitions_u(scores_text, logits_txt2, batch_size, n_context_img_text,
                                                                  max_new_tokens=opt.max_new_tokens)

                process_results(fused_generation_img2_txt2.detach().cpu().numpy(), opt, reader_tokenizer,
                                retrieved_passages,
                                answers, task, batch, query_text,
                                batch_metadata, logits_img2_txt2, dataset_wpred_fused, False, metrics_fuse, save_logits=True)

                process_results(fused_generation_img2.detach().cpu().numpy(), opt, reader_tokenizer,
                                retrieved_passages_img,
                                answers, task, batch, query_text,
                                batch_metadata, logits, dataset_wpred_fused_img, False, metrics_fuse_img)

                process_results(fused_generation_txt2.detach().cpu().numpy(), opt, reader_tokenizer,
                                retrieved_passages_text,
                                answers, task, batch, query_text,
                                batch_metadata, logits, dataset_wpred_fused_text, False, metrics_fuse_text)

            else:
                batch_size = len(query_image)
                n_context = len(retrieved_passages[0])
                fused_generation = fuse_model_predicitions_u(scores,logits, batch_size, n_context, max_new_tokens = opt.max_new_tokens)
                process_results(fused_generation.detach().cpu().numpy(), opt, reader_tokenizer, retrieved_passages, answers, task, batch, query_text,
                                batch_metadata, logits, dataset_wpred_fused, False, metrics_fuse)


    if opt.is_distributed:
        torch.distributed.barrier()


    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    if opt.text_retrieval and opt.img_retrieval:

        metrics_img, dataset_wpred_img = task.evaluation_postprocessing(metrics_img, dataset_wpred_img)
        metrics_text, dataset_wpred_text= task.evaluation_postprocessing(metrics_text, dataset_wpred_text)


    acc_img_fused = 0
    acc_text_fused = 0

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"

        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)
        if opt.text_retrieval and opt.img_retrieval:
            dataset_name_orig, _ = os.path.splitext(os.path.basename(data_path))
            dataset_name_img = f"{dataset_name_orig}-img-step-{step}"
            dataset_name_text = f"{dataset_name_orig}-text-step-{step}"

            util.save_distributed_dataset(dataset_wpred_img, dataset_name_img, opt)
            util.save_distributed_dataset(dataset_wpred_text, dataset_name_text, opt)

        if hasattr(task, "cal_acc"):
            if opt.is_distributed:
                torch.distributed.barrier()

            if opt.is_main:
                path_to_pred = Path(opt.checkpoint_dir) / opt.name / f"{dataset_name}.jsonl"
                pred_files = load_jsonl_file(path_to_pred)
                acc = task.cal_acc(pred_files, dataset_name)

                if opt.text_retrieval and opt.img_retrieval:

                    path_to_pred_img = Path(opt.checkpoint_dir) / opt.name / f"{dataset_name_img}.jsonl"
                    pred_files_img = load_jsonl_file(path_to_pred_img)
                    acc_img = task.cal_acc(pred_files_img, dataset_name_img)

                    path_to_pred_text = Path(opt.checkpoint_dir) / opt.name / f"{dataset_name_text}.jsonl"
                    pred_files_text = load_jsonl_file(path_to_pred_text)
                    acc_text = task.cal_acc(pred_files_text, dataset_name_text)


            if opt.is_distributed:
                torch.distributed.barrier()

    if opt.fused_eval and opt.append_only_one_passage_to_query:

        metrics_fused, dataset_wpred_fused = task.evaluation_postprocessing(metrics_fuse, dataset_wpred_fused)
        if opt.write_results:
            dataset_name, _ = os.path.splitext(os.path.basename(data_path))
            dataset_name = f"{dataset_name}-step-{step}-fused"
            util.save_distributed_dataset(dataset_wpred_fused, dataset_name, opt)
            if opt.text_retrieval and opt.img_retrieval:


                dataset_name_orig, _ = os.path.splitext(os.path.basename(data_path))

                dataset_name_fused= f"{dataset_name_orig}-step-{step}-fused"

                dataset_name_img_fused = f"{dataset_name_orig}-img-step-{step}-fused"
                dataset_name_text_fused = f"{dataset_name_orig}-text-step-{step}-fused"

                util.save_distributed_dataset(dataset_wpred_fused_img, dataset_name_img_fused, opt)
                util.save_distributed_dataset(dataset_wpred_fused_text, dataset_name_text_fused, opt)

                util.save_distributed_dataset(dataset_wpred_fused, dataset_name_fused, opt)
                util.save_distributed_dataset(dataset_wpred_fused_img, dataset_name_img_fused, opt)
                util.save_distributed_dataset(dataset_wpred_fused_text, dataset_name_text_fused, opt)

            if hasattr(task, "cal_acc"):

                if opt.is_distributed:
                    torch.distributed.barrier()

                if opt.is_main:
                    path_to_pred = Path(opt.checkpoint_dir) / opt.name / f"{dataset_name}.jsonl"
                    pred_files = load_jsonl_file(path_to_pred)
                    acc_fused = task.cal_acc(pred_files, dataset_name)

                    if opt.text_retrieval and opt.img_retrieval:


                        path_to_pred_img_fused = Path(opt.checkpoint_dir) / opt.name / f"{dataset_name_img_fused}.jsonl"
                        pred_files_img_fused = load_jsonl_file(path_to_pred_img_fused)
                        acc_img_fused = task.cal_acc(pred_files_img_fused, dataset_name_img_fused)


                        path_to_pred_text_fused = Path(opt.checkpoint_dir) / opt.name / f"{dataset_name_text_fused}.jsonl"
                        pred_files_text_fused = load_jsonl_file(path_to_pred_text_fused)
                        acc_text_fused = task.cal_acc(pred_files_text_fused, dataset_name_text_fused)



                if opt.is_distributed:
                    torch.distributed.barrier()

    if opt.is_distributed:
        torch.distributed.barrier()

    if opt.is_main:

        return (acc, acc_img, acc_text, acc_fused, acc_img_fused, acc_text_fused)
    else:
        return None



if __name__ == "__main__":

    options = get_options()
    opt = options.parse()

    run_distributed = True
    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)
    zip_project_files(os.path.join(checkpoint_path, "python_code.zip"))

    if not run_distributed:
        os.environ.pop("SLURM_JOB_ID")
        opt.local_rank = -1

    torch.manual_seed(opt.seed)

    if "TORCHELASTIC_RUN_ID" in os.environ:
        init_distributed_mode_torchrun(opt)
        torch.cuda.set_device(dist.get_rank())
    else:
        warnings.warn("START Init dis")
        slurm.init_distributed_mode(opt)
        slurm.init_signal_handler()

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    model, _, _, _, _, _, _ = load_or_initialize_atlas_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    dist_utils.barrier()

    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        if not opt.multi_modal:
            model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)
        else:
            model.build_index_multi_modal(index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")
        metrics = evaluate_multi_modal(model, index, opt, data_path, 0)
        log_message = f"Dataset: {dataset_name}"
        for k, v in metrics.items():
            log_message += f" | {v:.3f} {k}"
        logger.info(log_message)
