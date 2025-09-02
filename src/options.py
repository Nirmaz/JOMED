
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of the experiment - also used as directory name "
        )

        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoint/",
            help="models are saved here",
        )
        self.parser.add_argument(
            "--reader_checkpoint_path",
            type=str,
            default="none",
            help="load specific checkpoint to reader",
        )
        self.parser.add_argument(
            "--running_script_path",
            type=str,
            default="",
            help="If the command is running from bash file, add it here if you want to save it in the checkpoint dir",
        )
        self.parser.add_argument(
            "--model_path",
            type=str,
            default="none",
            help="Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)",
        )
        self.parser.add_argument(
            "--model_checkpoint_path",
            type=str,
            default="none",
            help="Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)",
        )


        self.parser.add_argument(
            "--path_retriever_separate_wights",
            type=str,
            default="none",
            help="Path to a pretrained retirever model to initialize from",
        )
        self.parser.add_argument(
            "--per_gpu_batch_size",
            default=1,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )
        self.parser.add_argument(
            "--max_new_tokens",
            default=8,
            type=int,
            help="Number of new token the model generate",
        )

        self.parser.add_argument(
            "--per_gpu_embedder_batch_size",
            default=512,
            type=int,
            help="Embedder's batch size per GPU.",
        )

        self.parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank",
        )
        self.parser.add_argument(
            "--main_port",
            type=int,
            default=-1,
            help="Main port (for multi-node jobs)",
        )
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        self.parser.add_argument(
            "--log_freq",
            type=int,
            default=100,
            help="log train stats <log_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--update_base_retr_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_in_specific_step",
            type=int,
            default=-1,
            help="evaluate model in specific step while training",
        )
        self.parser.add_argument(
            "--save_in_specific_step",
            type=int,
            default=-1,
            help="evaluate model in specific step while training",
        )

        self.parser.add_argument(
            "--test_freq",
            type=int,
            default=500,
            help="evaluate model every <test_freq> steps during training",
        )
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=5000,
            help="save model every <save_freq> steps during training",
        )
        self.parser.add_argument(
            "--train_data", nargs="+", default=[], help="list of space-separated paths to jsonl-formatted train sets"
        )
        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="list of space-separated paths to jsonl-formatted evaluation sets",
        )

        # eval_embed_data
        self.parser.add_argument(
            "--test_data",
            nargs="+",
            default=[],
            help="list of space-separated paths to jsonl-formatted evaluation sets",
        )

        self.parser.add_argument("--write_results", action="store_true", help="save evaluation results to file")
        self.parser.add_argument(
            "--dont_write_passages",
            action="store_true",
            help="if writing results, passages can take up a lot of space, pass this flag not to write passages as part of dumped results",
        )

    def add_optim_options(self):
        self.parser.add_argument("--warmup_steps", type=int, default=1000, help="number of learning rate warmup steps")
        self.parser.add_argument("--total_steps", type=int, default=1000, help="total number of training steps")
        self.parser.add_argument(
            "--scheduler_steps",
            type=int,
            default=None,
            help="total number of step for the scheduler, if None then scheduler_total_step = total_step",
        )
        self.parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_retriever", type=float, default=1e-5, help="learning rate for retriever")
        self.parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            choices=["linear", "cosine", "fixed"],
            help="learning rate schedule to use",
        )
        self.parser.add_argument(
            "--weight_decay", type=float, default=0.01, help="amount of weight decay to apply in training"
        )
        self.parser.add_argument(
            "--save_optimizer", action="store_true", help="Pass flag to save optimizer state in saved checkpoints"
        )
        self.parser.add_argument("--epsilon", type=float, default=1e-6, help="adamw epsilon value")
        self.parser.add_argument("--alpha", type=float, default=1.0, help="adamw alpha value")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="adamw beta2 value")
        self.parser.add_argument("--shuffle", action="store_true", help="shuffle data for training")
        self.parser.add_argument("--multi-modal", action="store_true", help="Apply multi modal model")

        # memory optimizations:
        self.parser.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp16", "fp32", "bf16"],
            help="numerical precision - recommend bf16 if available, fp16 likely to be unstable for training",
        )
        self.parser.add_argument(
            "--shard_optim",
            action="store_true",
            help="train-time memory optimization: shards optimizer state over available GPUs using sharded data parallel, recommended for larger models",
        )
        self.parser.add_argument(
            "--shard_grads",
            action="store_true",
            help="train-time memory optimization: shards gradients over available GPUs using sharded data parallel, recommended for larger models",
        )
        self.parser.add_argument(
            "--use_gradient_checkpoint_reader",
            action="store_true",
            help="use gradient checkpointing in the reader",
        )
        self.parser.add_argument(
            "--disable_gradient_reader",
            action="store_true",
            help="Dont calculate gradients to the reader ",
        )
        self.parser.add_argument(
            "--use_gradient_checkpoint_retriever",
            action="store_true",
            help="use gradient checkpointing for retriever",
        )

        self.parser.add_argument(
            "--qwen_model",
            action="store_true",
            help="use gradient checkpointing for retriever",
        )
        self.parser.add_argument(
            "--ret_acc_text",
            action="store_true",
            help="`Add retriever that bring thing according text",
        )

        self.parser.add_argument(
            "--pixtral_model",
            action="store_true",
            help="use gradient checkpointing for retriever",
        )
        self.parser.add_argument(
            "--skip_model",
            action="store_true",
            help="use gradient checkpointing for retriever",
        )
    def add_modeling_options(self):
        self.parser.add_argument(
            "--reader_model_type",
            required=True,
            type=str,
            help="t5 Architecture for reader FID model, e.g. google/t5-xl-lm-adapt",
            choices=[
                "flamingo"
            ],
        )

        self.parser.add_argument(
            "--load_small_model",
            action="store_true",
            help="load small model that is easy to run",
        )

        self.parser.add_argument(
            "--retriever_multimodal_model_type",
            required=True,
            type=str,
            help="Choose the type of multi modal retriever you would like to use",
            choices=[
                "clipmd",
                "biomedclip",
                "jina"
            ],
        )

        self.parser.add_argument(
            "--retrieve_passage_for_train_retriever",
            action="store_true",
            help="To retrieve for the training the retriever a custom number of samples",
        )

        self.parser.add_argument(
            "--random_query",
            action="store_true",
            help="to replace the query with randominze ",
        )

        self.parser.add_argument(
            "--partial_learning_query_embeddings",
            action="store_true",
            help="Partial learning of the query. Change that the query only partially changed",
        )
        self.parser.add_argument(
            "--text_retrieval",
            action="store_true",
            help="Partial learning of the query. Change that the query only partially changed",
        )
        self.parser.add_argument(
            "--img_retrieval",
            action="store_true",
            help="Partial learning of the query. Change that the query only partially changed",
        )

        self.parser.add_argument(
            "--use_base_retriever",
            action="store_true",
            help="Use base retriever that doesnt update until ceration freq",
        )
        self.parser.add_argument(
            "--fused_eval",
            action="store_true",
            help="When evaluate the model to fuse the reprsentation of the token for different retrieve stuff",
        )

        self.parser.add_argument(
            "--only_prob_retrieve_loss",
            action="store_true",
            help="For the loss of the retriever build the GT not based on the loss that just make things complicated",
        )


        self.parser.add_argument(
            "--use_targets",
            action="store_true",
            help="When evaluate the model to fuse the reprsentation of the token for different retrieve stuff",
        )


        self.parser.add_argument(
            "--append_only_one_passage_to_query",
            action="store_true",
            help="The input to the model is only one query and passage",
        )


        self.parser.add_argument(
            "--text_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--target_maxlength",
            type=int,
            default=None,
            help="Maximum length of target outputs in tokens when training the model. Targets longer than this will be truncated. No truncation if -1",
        )
        self.parser.add_argument("--n_context", type=int, default=1, help="number of top k passages to pass to reader")
        self.parser.add_argument("--n_context_for_training", type=int, default=128, help="number of top k passages to retriever training")

        # Retriever modelling options
        self.parser.add_argument(
            "--passages",
            nargs="+",
            help="list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path",
        )
        self.parser.add_argument(
            "--max_passages",
            type=int,
            default=-1,
            help="maximum number of passages to index. -1 to read all passages in passage files",
        )
        self.parser.add_argument(
            "--retriever_model_path",
            type=str,
            default="facebook/contriever",
            help="path to contriever model to init from (overridden if passing a value to --model_path ",
        )
        self.parser.add_argument(
            "--retrieve_only",
            action="store_true",
            help="Pass this to prevent loading a reader, and only run retrieval evaluation",
        )
        self.parser.add_argument(
            "--distributed_training",
            action="store_true",
            help="Pass this to prevent loading a reader, and only run retrieval evaluation",
        )


        self.parser.add_argument(
            "--train_retriever", action="store_true", help="Pass to train retriever as well as reader"
        )

        self.parser.add_argument(
            "--train_retriever_text", action="store_true", help="Pass to train retriever for query of text"
        )

        self.parser.add_argument(
            "--train_retriever_img", action="store_true", help="Pass to train retriever for query of img"
        )

        self.parser.add_argument(
            "--add_aug_to_query", action="store_true", help="Add augmentation to query"
        )
        self.parser.add_argument(
            "--retriever_from_hf_index", action="store_true", help="replace the index with hf index"
        )


        self.parser.add_argument(
            "--aug_query_before_retrieve", action="store_true", help="Add augmentation to query"
        )

        # add_aug_to_query
        self.parser.add_argument(
            "--supervised_gold_score", action="store_true", help="Simulate the gold score "
        )

        self.parser.add_argument(
            "--add_prior_to_gold_score", action="store_true", help="Simulate the gold score "
        )

        self.parser.add_argument(
            "--specific_tokens_for_retriever_loss", action="store_true", help="calculate the tokens loss only  for specific tokens "
        )
        self.parser.add_argument(
            "--weighting_kl", action="store_true",
            help="calculate the tokens loss only  for specific tokens "
        )
        self.parser.add_argument(
            "--symmetric_kl", action="store_true",
            help="Switch between the gold score and scores for creating a symatric "
        )
        self.parser.add_argument(
            "--retriever_training_logits_from_generation", action="store_true",
            help="use the logits for generation instead of regular loss "
        )

        self.parser.add_argument(
            "--add_prior_to_gold_score_with_ref", action="store_true", help="Simulate the gold score "
        )

        self.parser.add_argument(
            "--use_file_passages",
            action="store_true",
            help='uses passages in "passages" field in train or eval jsonl files rather than retrieving passages',
        )
        self.parser.add_argument(
            "--retriever_n_context",
            type=int,
            default=5,
            help="number of top k passages to use to train the retriever with",
        )
        self.parser.add_argument(
            "--gold_score_mode",
            type=str,
            choices=["evalnormsum", "loop", "ppmean", "emdr", "pdist", "adist"],
            default="ppmean",
            help="retriever training method. `pdist` is the name used in the paper for `ppmean`. `adist` is the name used in the paper for `evalnormsum`",
        )
        self.parser.add_argument(
            "--remove_ret_image",
            action="store_true",
            help="Dont give the model the retrieved image",
        )
        self.parser.add_argument(
            "--remove_input_image",
            action="store_true",
            help="Dont give the model the input image",
        )
        self.parser.add_argument(
            "--ret_from_training_set",
            action="store_true",
            help="Dont give the model the input image",
        )
        self.parser.add_argument(
            "--closed_book",
            action="store_true",
            help="Dont use retrieval - reduces to T5. Overrides n_context, n_context_retriever and encoder_format if they are set",
        )

        self.parser.add_argument(
            "--temperature_score", type=float, default=1.0, help="softmax temperature for retriever"
        )
        self.parser.add_argument(
            "--normalize_factor_loss", type=float, default=0.5, help="For loss of the retriever"
        )
        self.parser.add_argument(
            "--temperature_gold",
            type=float,
            default=1.0,
            help="softmax temperature for target distribution for retriever distillation",
        )
        self.parser.add_argument("--compute_crossattention_stats", action="store_true")
        self.parser.add_argument(
            "--filtering_overretrieve_ratio",
            type=int,
            default=1,
            help="if filtering, over-retrieve the topK by this factor, and then filter out undesirable results. Useful, Set to 1 only if using a task that doesn't filter retrieved results",
        )
        self.parser.add_argument("--freeze_retriever_steps", type=int, default=-1, help="freezes retriever for n steps")
        self.parser.add_argument(
            "--query_side_retriever_training",
            action="store_true",
            help="pass to enable query-side finetuning of retriever (unties the parameters of the contriever encoder's passage and query encoders, and freezes the passage encoder. Useful to avoid index refreshes.",
        )
        self.parser.add_argument(
            "--freeze_query_encoder",
            action="store_true",
            help="pass to disable query-side finetuning of retriever (unties the parameters of the contriever encoder's passage and query encoders, and freezes the passage encoder. Useful to avoid index refreshes.",
        )
        self.parser.add_argument(
            "--freeze_passages_encoder_for_train",
            action="store_true",
            help="pass to disable passage update while finetunning by the retriever",
        )
        # freeze_passages_encoder_for_train
        self.parser.add_argument(
            "--retrieve_with_rerank",
            action="store_true",
            help="pass this to enable reranking with fresh passage encoder for retriever",
        )
        self.parser.add_argument(
            "--n_to_rerank_with_retrieve_with_rerank",
            type=int,
            default=128,
            help="n passages to rerank when passing --retrieve_with_rerank. Higher is slower but more accurate. Recommend 64-128",
        )

        # input and output formatting options:
        self.parser.add_argument(
            "--decoder_format",  # TODO: decide whether to remove functionality
            type=str,
            default=None,
            help="format for decoder, model will be train on the format and evaluation will be performed with the format contrary to the decoder_prompt_format option",
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--decoder_prompt_format",
            type=str,
            default=None,
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(
            "--encoder_format",
            type=str,
            default="{query} title: {title} context: {text}",
            help="format string for reader's encoder preprocessing",
        )
        #retriever_text_format
        self.parser.add_argument(
            "--retriever_format",
            type=str,
            default="{title} {text}",
            help="format string for retriever's encoder preprocessing",
        )

        self.parser.add_argument(
            "--retriever_text_format",
            type=str,
            default="{title} {text}",
            help="format string for retriever's encoder preprocessing",
        )

        # Generation options
        self.parser.add_argument("--generation_max_length", type=int, default=128)
        self.parser.add_argument("--generation_min_length", type=int, default=None)
        self.parser.add_argument("--generation_length_penalty", type=float, default=1.0)
        self.parser.add_argument("--generation_num_beams", type=int, default=1)

        # Task-specific options:
        self.parser.add_argument(
            "--task",
            type=str,
            default=None,
            choices=["base","multi_label", "vqa_rad", "cr_qa_breast", "cr_qa_retina", "cr_qa_derma"],
            help="Task performed by the model. Used to setup preprocessing, retrieval filtering, evaluations, etc.",
        )
        # when add the retrieve passages to the prompt to add also the retrieve text associated to the por
        self.parser.add_argument(
            "--add_retrieve_text_to_prompt",
            action="store_true",
            help="add the text of the retrieved passages to the prompt",
        )



        # Open-domain task options:
        self.parser.add_argument(
            "--qa_prompt_format",
            type=str,
            default="question: {question} answer: <extra_id_0>",
            help="How to format question as input prompts when using --task qa",
        )





    def add_index_options(self):
        self.parser.add_argument(
            "--load_index_path",
            default=None,
            type=str,
            help="path for loading the index, passage embeddings and passages",
        )
        self.parser.add_argument(
            "--load_index_text_path",
            default=None,
            type=str,
            help="path for loading the index, passage embeddings and passages",
        )
        self.parser.add_argument(
            "--save_index_path",
            default=None,
            type=str,
            help="path for saving the index and/or embeddings",
        )
        self.parser.add_argument(
            "--save_index_n_shards",
            default=128,
            type=int,
            help="how many shards to save an index to file with. Must be an integer multiple of the number of workers.",
        )
        self.parser.add_argument(
            "--index_mode",
            type=str,
            default="flat",
            help="Use flat torch index or a faiss index for retrieving the k nearest neighbors",
            choices=["flat", "faiss"],
        )
        # faiss options:

    def print_options(self, opt):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f"\t(default: {default_value})"
            message += f"{k:>30}: {str(v):<40}{comment}\n"

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        with open(expr_dir / "opt.log", "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        if opt.closed_book:  # override flags to enable closed book mode
            opt.n_context = 1
            opt.retriever_n_context = 1
            opt.encoder_format = "{query}"
            opt.use_file_passages = True
        if opt.gold_score_mode == "pdist":  # allow paper name of retriever losses
            opt.gold_score_mode = "ppmean"
        if opt.gold_score_mode == "adist":  # allow paper name of retriever losses
            opt.gold_score_mode = "evalnormsum"
        if (
            opt.use_file_passages
        ):  # if passing use_file_passges, the following should be false (There is no retreiver loaded in this case)
            opt.train_retriever = False
            opt.query_side_retriever_training = False
            opt.use_gradient_checkpoint_retriever = False
        return opt


def get_options():
    options = Options()
    options.add_index_options()
    options.add_modeling_options()
    options.add_optim_options()
    return options
