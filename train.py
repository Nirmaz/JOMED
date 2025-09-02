import os
import time
from collections import defaultdict
import numpy as np
import torch
import torch.cuda
import logging
from evaluate import evaluate_multi_modal
from src import  util
from src.index_io import  save_embeddings_and_index
from src.model_io import  save_atlas_model
from src.tasks import get_task
import warnings
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "true"
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100

logger = logging.getLogger(__name__)



def train(
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
):
    print("start train")

    tb_logger = util.init_tb_logger(os.path.join(opt.checkpoint_dir, opt.name), is_main=opt.is_main)
    run_stats = util.WeightedAvgStats()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)

    # different seed for different sampling depending on global_rank
    torch.manual_seed(opt.global_rank + opt.seed)
    print("define seed")
    scale = 0.001
    scale = 1000
    grad_stats = defaultdict(lambda: [])
    task = get_task(opt, unwrapped_model.reader_tokenizer)
    if hasattr(task, 'set_tokeinzer_create_labels'):
        task.set_tokeinzer_create_labels(unwrapped_model.reader_tokenizer.tokenizer)

    if hasattr(task, 'set_tokenizer'):
        task.set_tokenizer(unwrapped_model.reader_tokenizer.tokenizer)


    print("start loop")
    while step < opt.total_steps:
        print(f"step {step}")
        data_iterator = task.data_iterator(
            opt.train_data, opt.global_rank, opt.world_size, repeat_if_less_than_world_size=True, opt=opt
        )
        data_iterator = filter(None, map(task.process, data_iterator))
        data_iterator = task.batch_iterator(data_iterator, opt.per_gpu_batch_size, drop_last=True, shuffle=opt.shuffle)
        for i, batch in enumerate(data_iterator):
            try:
                for q in batch["query"]:
                    task.query_unique_id_set.add(q)
            except:
                pass

            iter_stats = {}
            model.train()

            train_step_start = time.time()

            if not opt.disable_gradient_reader:
                optimizer.zero_grad(set_to_none=True)

            if opt.train_retriever_img:
                retr_optimizer.zero_grad(set_to_none=True)
            if opt.train_retriever_text:
                text_retr_optimizer.zero_grad(set_to_none=True)




            if step == opt.eval_in_specific_step:

                for data_path in opt.eval_data:
                    print(data_path)
                    dataset_name = os.path.basename(data_path)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                    metrics_tuple = evaluate_multi_modal(model, index, index_text, opt, data_path, step)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                        # If this is the main process, gather and consolidate all data
                    if opt.is_main:

                        log_message = f"Dataset: {dataset_name} | acc {str(metrics_tuple[0])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img {str(metrics_tuple[1])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text {str(metrics_tuple[2])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc fused {str(metrics_tuple[3])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img fused{str(metrics_tuple[4])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text fused {str(metrics_tuple[5])} | step {step}"
                        logger.info(log_message)
                        logger.info(log_message)

                    if opt.is_distributed:
                        torch.distributed.barrier()




            if step == opt.eval_in_specific_step:
                for data_path in opt.test_data:
                    print(data_path)
                    dataset_name = os.path.basename(data_path)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                    metrics_tuple = evaluate_multi_modal(model, index, index_text, opt, data_path, step)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                    if opt.is_main:

                        log_message = f"Dataset: {dataset_name} | acc {str(metrics_tuple[0])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img {str(metrics_tuple[1])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text {str(metrics_tuple[2])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc fused {str(metrics_tuple[3])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img fused{str(metrics_tuple[4])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text fused {str(metrics_tuple[5])} | step {step}"
                        logger.info(log_message)

                    if opt.is_distributed:
                        torch.distributed.barrier()


            step += 1


            reader_loss, retriever_loss, retriever_loss_text = model(
                index=index,
                index_text = index_text,
                query=batch["query"],
                query_text = batch["query_text"],
                target=batch["target"],
                target_tokens=batch.get("target_tokens"),
                passages=batch["passages"] if (opt.use_file_passages or opt.closed_book) else None,
                batch_metadata=batch.get("metadata"),
                filtering_fun=task.filter,
                train_retriever=opt.train_retriever and step > opt.freeze_retriever_steps,
                iter_stats=iter_stats,
                task = task,
            )


            if retriever_loss is not None and opt.train_retriever and not opt.disable_gradient_reader:
                train_loss = reader_loss.float() + retriever_loss

            elif not opt.disable_gradient_reader:
                train_loss = reader_loss
            elif opt.train_retriever and opt.train_retriever_text and opt.train_retriever_img:
                train_loss = retriever_loss + retriever_loss_text
            elif opt.train_retriever and opt.train_retriever_text:
                train_loss = retriever_loss_text
            elif opt.train_retriever and opt.train_retriever_img:
                train_loss = retriever_loss



            iter_stats["loss/train_loss"] = (train_loss.item(), len(batch["query"]))

            backward_start = time.time()
            train_loss = scale * train_loss
            iter_stats["runtime/backward"] = (time.time() - backward_start, 1)

            model_update_start = time.time()

            if opt.reader_model_type == 'flamingo':

                stats = {}
                stats["skip_example"] = True
                stats["max"] = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if not opt.disable_gradient_reader:
                    reader_loss.backward()
                    optimizer.step()
                    scheduler.step()

                if opt.train_retriever_img:
                    retriever_loss = retriever_loss * scale

                    retriever_loss.backward()
                    retr_optimizer.step()
                    retr_scheduler.step()
                if opt.train_retriever_text:
                    retriever_loss_text = retriever_loss_text * scale
                    retriever_loss_text.backward()
                    text_retr_optimizer.step()
                    text_retr_scheduler.step()


            if stats["skip_example"]:
                if opt.reader_model_type != 'flamingo':
                    pass
            else:
                for k, v in stats.items():
                    grad_stats[k].append(v)

            if len(grad_stats["max"]) >= THRESHOLD_GRAD_STATS:
                if np.mean(grad_stats["max"]) > GRAD_SCALE_UPPER_BOUND_MEAN:
                    scale /= 2
                elif np.mean(grad_stats["mean"]) < GRAD_SCALE_LOWER_BOUND_MEAN:
                    scale *= 2
                grad_stats.clear()

            if step % opt.accumulation_steps == 0 and not stats["skip_example"]:
                if opt.is_distributed and opt.shard_optim:
                    optimizer.clip_grad_norm(scale * opt.clip)
                    if opt.train_retriever:
                        retr_optimizer.clip_grad_norm(scale * opt.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), scale * opt.clip)


            iter_stats["runtime/model_update"] = (time.time() - model_update_start, 1)
            iter_stats["runtime/train_step"] = (time.time() - train_step_start, 1)
            run_stats.update(iter_stats)

            if step % opt.update_base_retr_freq == 0 and opt.train_retriever and opt.use_base_retriever:
                if opt.train_retriever_img:
                    unwrapped_model.base_retriever = copy.deepcopy(unwrapped_model.retriever)
                elif opt.train_retriever_text:
                    unwrapped_model.base_retriever = copy.deepcopy(unwrapped_model.retriever_for_q_to_text)

                unwrapped_model.base_retriever = unwrapped_model.base_retriever.cuda()
                unwrapped_model.base_retriever.eval()
                for name, param in unwrapped_model.base_retriever.named_parameters():
                    param.requires_grad = False

            if step % opt.log_freq == 0:


                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3g}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                try:
                    log += f" | lr: {scheduler.get_last_lr()[0]:0.2g}"
                except:
                    pass
                if opt.train_retriever:
                    try:
                        log += f" | lr: {retr_scheduler.get_last_lr()[0]:0.2g}"
                    except:
                        pass
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                if tb_logger:
                    try:
                        tb_logger.add_scalar("lr", scheduler.get_last_lr()[0], step)
                    except:
                        pass

                logger.info(log)
                run_stats.reset()



            if step % opt.eval_freq == 0 or step == opt.eval_in_specific_step:
                if opt.weighting_kl:
                    pass

                for data_path in opt.eval_data:
                    dataset_name = os.path.basename(data_path)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                    eval_acc_tuple = evaluate_multi_modal(model, index,index_text, opt, data_path, step)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                    if opt.is_main:

                        log_message = f"Dataset: {dataset_name} | acc {str(eval_acc_tuple[0])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img {str(eval_acc_tuple[1])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text {str(eval_acc_tuple[2])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc fused {str(eval_acc_tuple[3])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img fused{str(eval_acc_tuple[4])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text fused {str(eval_acc_tuple[5])} | step {step}"
                        logger.info(log_message)


                    if opt.is_distributed:
                        torch.distributed.barrier()


            if step % opt.test_freq == 0 or step == opt.eval_in_specific_step:
                for data_path in opt.test_data:
                    dataset_name = os.path.basename(data_path)

                    if opt.is_distributed:
                        torch.distributed.barrier()


                        test_acc_tuple = evaluate_multi_modal(model, index,index_text, opt, data_path, step)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                        # If this is the main process, gather and consolidate all data

                        log_message = f"Dataset: {dataset_name} | acc {str(test_acc_tuple[0])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img {str(test_acc_tuple[1])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text {str(test_acc_tuple[2])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc fused {str(test_acc_tuple[3])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc img fused{str(test_acc_tuple[4])} | step {step}"
                        logger.info(log_message)
                        log_message = f"Dataset: {dataset_name} | acc text fused {str(test_acc_tuple[5])} | step {step}"
                        logger.info(log_message)


                    if opt.is_distributed:
                        torch.distributed.barrier()

            if opt.is_distributed:
                torch.distributed.barrier()

            if opt.is_main:

                if (step % opt.save_freq == 0):
                    # TODO Fix bug here
                    save_atlas_model(
                        unwrapped_model,
                        optimizer,
                        scheduler,
                        retr_optimizer,
                        retr_scheduler,
                        step,
                        opt,
                        checkpoint_path,
                        f"step-{step}",
                    )
            if opt.is_distributed:
                torch.distributed.barrier()

            if step > opt.total_steps:

                for data_path in opt.test_data:

                    dataset_name = os.path.basename(data_path)
                    test_acc_end = evaluate_multi_modal(model, index, index_text, opt, data_path, step)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                    if opt.is_main:

                        log_message = f"Dataset: {dataset_name} | acc {str(test_acc_end)} | step {step}"
                        logger.info(log_message)

                    if opt.is_distributed:
                        torch.distributed.barrier()

                exit()


