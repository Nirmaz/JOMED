import errno
import logging
import os
from pathlib import Path
from typing import  Union
import open_clip
import torch
import transformers
from transformers import AutoModel
from src.util import  LlavaProcessor
import sys
from src import dist_utils
import copy
from src.jomed import JOMED
from src.retrievers import DualEncoderRetriever, UntiedDualEncoderRetriever,  VisionRetrieverOpenClip, TextRetrieverOpenClip, VisionRetrieverJina, TextRetrieverJina
from src.util import cast_to_precision, set_optim, set_optim_open_flamingo

Number = Union[float, int]

logger = logging.getLogger(__name__)


def total_size(o, handlers={}):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and their subclasses:  tuple, list, deque, dict, set and frozenset.
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)   # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    return checkpoint_path


def create_checkpoint_directories(opt):
    checkpoint_path = get_checkpoint_path(opt)
    os.makedirs(checkpoint_path, exist_ok=True)
    if opt.save_index_path:
        os.makedirs(opt.save_index_path, exist_ok=True)
    print("waiting")
    dist_utils.barrier()
    print("finish waiting")
    return checkpoint_path, opt.save_index_path


def load_retriever(opt, opt_checkpoint=None):
    if opt.use_file_passages:
        return None, None, None

    if not opt.multi_modal:
        contriever_encoder = Contriever.from_pretrained(opt.retriever_model_path)
        retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(opt.retriever_model_path)
    else:
        custom_cache_path = ''
        if opt.retriever_multimodal_model_type == 'biomedclip':
            biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            ret_tokenizer_text = open_clip.get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', cache_dir=custom_cache_path)
            # biomedclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            processor = preprocess_val
            contriever_encoder = VisionRetrieverOpenClip(biomedclip)
            if opt.ret_acc_text:
                contriever_encoder_text = TextRetrieverOpenClip(copy.deepcopy(biomedclip))

        if opt.retriever_multimodal_model_type == 'jina':
            model_jina = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True,
                                              cache_dir='/cs/labs/tomhope/nirm/.cache4')
            model_jina.cuda()
            ret_tokenizer_text = model_jina.get_tokenizer()
            processor = model_jina.get_preprocess()
            contriever_encoder = VisionRetrieverJina(model_jina)
            if opt.ret_acc_text:
                contriever_encoder_text = TextRetrieverJina(copy.deepcopy(model_jina))



            if opt.add_aug_to_query:
                retriever_tokenizer_train = preprocess_train
            else:
                retriever_tokenizer_train = None
        else:
            retriever_tokenizer_train = None





        retriever_tokenizer = processor


    # once you have done query side training you cannot go back to a parameter-tied retriever
    if opt_checkpoint is not None:
        retriever_is_untied = opt_checkpoint.query_side_retriever_training or opt.query_side_retriever_training
    else:
        retriever_is_untied = opt.query_side_retriever_training

    if retriever_is_untied:
        retriever = UntiedDualEncoderRetriever(opt, contriever_encoder)
    else:
        retriever = DualEncoderRetriever(opt, contriever_encoder)
        if opt.ret_acc_text:
            retriever_text = DualEncoderRetriever(opt, contriever_encoder_text)
        else:
            retriever_text = None

    return retriever, retriever_text, retriever_tokenizer, retriever_tokenizer_train, ret_tokenizer_text

def _cast_and_set_attrs_and_send_to_device(model, opt):
    try:
        model = model.to(opt.device)
    except:
        model = model.cuda()

    return model

def _convert_state_dict_from_dual_encoder_retriever(state_dict):
    """handles when we want to load an UntiedDualEncoderRetriever from a DualEncoderRetriever state dict"""
    new_state_dict = {}
    for k, tensor in state_dict.items():
        if k.startswith("retriever"):
            new_state_dict[k.replace("retriever.contriever", "retriever.passage_contriever")] = tensor
            new_state_dict[k.replace("retriever.contriever", "retriever.query_contriever")] = tensor
        else:
            new_state_dict[k] = tensor
    return new_state_dict




def _set_reader_encoder_cfg(model, opt):
    if model.reader is not None:
        cfg = model.reader.encoder.config
        cfg.n_context = opt.n_context
        cfg.bsz = opt.per_gpu_batch_size


def _cast_atlas_to_precision(atlas_model, precision):
    if atlas_model.reader is not None:
        atlas_model.reader = cast_to_precision(atlas_model.reader, precision)
    if atlas_model.retriever is not None and precision == "bf16":
        atlas_model.retriever = cast_to_precision(atlas_model.retriever, precision)



def _load_atlas_model_state(opt, opt_checkpoint, model, model_dict):
    model_dict = {
        k.replace("retriever.module", "retriever").replace("reader.module", "reader"): v for k, v in model_dict.items()
    }
    if opt.query_side_retriever_training and not opt_checkpoint.query_side_retriever_training:
        model_dict = _convert_state_dict_from_dual_encoder_retriever(model_dict)

    if opt.retrieve_only:  # dont load reader if in retrieve only mode
        model_dict = {k: v for k, v in model_dict.items() if not k.startswith("reader")}

    if opt.use_file_passages:  # dont load retriever if in use_file_passages mode
        model_dict = {k: v for k, v in model_dict.items() if not k.startswith("retriever")}

    # model.load_state_dict(model_dict)
    model.load_state_dict(model_dict,   strict= False)



    model = _cast_and_set_attrs_and_send_to_device(model, opt)


    return model

def load_reader(opt):
    reader = None
    reader_tokenizer = None
    return reader, reader_tokenizer

def load_atlas_model(dir_path, opt, reset_params=False, eval_only=False):
    epoch_path = os.path.realpath(dir_path)
    save_path = os.path.join(epoch_path, "model.pth.tar")

    logger.info(f"Loading {epoch_path}")
    logger.info(f"loading checkpoint {save_path}")
    checkpoint = torch.load(save_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    model_dict = checkpoint["model"]

    reader, reader_tokenizer = load_reader(opt)
    retriever, retriever_text, retriever_tokenizer, retriever_tokenizer_train, retriever_tokenizer_text = load_retriever(opt, opt_checkpoint)

    model = Atlas(opt, reader, retriever, reader_tokenizer, retriever_tokenizer)
    # model = _load_atlas_model_state(opt, opt_checkpoint, model, model_dict)

    if eval_only:
        return model, None, None, None, None, opt_checkpoint, step

    if not reset_params:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, step

def get_cast_dtype( precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype
def load_atlas_model_w_flamingo(dir_path, opt, reset_params=False, eval_only=False):
    epoch_path = os.path.realpath(dir_path)
    retriever, retriever_text, retriever_tokenizer, retriever_tokenizer_train, retriever_tokenizer_text = load_retriever(opt, None)

    if retriever_text is not None:
        retriever_text.eval()
        for name, param in retriever_text.named_parameters():
            param.requires_grad = False
    save_path = os.path.join(epoch_path, "model.pt")

    if opt.precision == 'fp16':
        pass

    if not opt.retrieve_only:
        if opt.pixtral_model:
            from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor
            model_id = "mistral-community/pixtral-12b"
            if not opt.skip_model:

                if opt.reader_checkpoint_path == 'none':

                    reader = LlavaForConditionalGeneration.from_pretrained(model_id,  cache_dir='/cs/labs/tomhope/nirm/.cache')

                else:
                    if opt.is_distributed:
                        reader = LlavaForConditionalGeneration.from_pretrained(opt.reader_checkpoint_path, torch_dtype="auto", cache_dir='/cs/labs/tomhope/nirm/.cache')
                    else:
                        reader = LlavaForConditionalGeneration.from_pretrained(opt.reader_checkpoint_path, torch_dtype="auto", device_map="auto",
                                                                               cache_dir='/cs/labs/tomhope/nirm/.cache')
            else:
                reader = None
                model_id = "mistral-community/pixtral-12b"

            reader_text_tokenizer = AutoProcessor.from_pretrained(model_id)
            reader_text_tokenizer.tokenizer.pad_token = reader_text_tokenizer.tokenizer.eos_token
            reader_tokenizer = LlavaProcessor(reader_text_tokenizer.tokenizer, None,
                                              build_chat_template=reader_text_tokenizer)

        elif opt.qwen_model:
            from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from qwen_vl_utils import process_vision_info

            if not opt.skip_model:
                if opt.reader_checkpoint_path == 'none':
                    reader = Qwen2VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto",
                        cache_dir='/cs/labs/tomhope/nirm/.cache'
                    )
                else:
                    if opt.is_distributed:
                        reader = Qwen2VLForConditionalGeneration.from_pretrained(
                            opt.reader_checkpoint_path, torch_dtype="auto",
                            cache_dir='/cs/labs/tomhope/nirm/.cache'
                        )
                    else:
                        reader = Qwen2VLForConditionalGeneration.from_pretrained(
                            opt.reader_checkpoint_path, torch_dtype="auto", device_map="auto",
                            cache_dir='/cs/labs/tomhope/nirm/.cache'
                        )

            else:
                reader = None

            reader_text_tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            reader_tokenizer = LlavaProcessor(reader_text_tokenizer.tokenizer, process_vision_info,
                                              build_chat_template=reader_text_tokenizer)



        logger.info(f"loading checkpoint {save_path}")
        if not opt.load_small_model and not opt.qwen_model and not opt.pixtral_model:
            logger.info(f"loading checkpoint {save_path}")
            reader.load_state_dict(torch.load(save_path, map_location="cpu"), strict=False)

        if opt.precision == "fp16" or opt.precision == "bf16" and not opt.qwen_model and not opt.pixtral_model:
            print(f" load {opt.precision} ")
            cast_dtype = get_cast_dtype(opt.precision)
            reader = reader.to('cuda', dtype=cast_dtype, non_blocking=True)

        elif not opt.qwen_model and not opt.pixtral_model:
            reader = reader.cuda()


    else:
        reader = None
        reader_tokenizer = None




    if opt.freeze_passages_encoder_for_train:
        passage_encoder_for_train = copy.deepcopy(retriever)
        passage_encoder_for_train = passage_encoder_for_train.cuda()
        passage_encoder_for_train.eval()
        for name, param in passage_encoder_for_train.named_parameters():
                param.requires_grad = False

    else:
        passage_encoder_for_train = None



    if opt.text_retrieval:
        retriever_for_q_to_text = copy.deepcopy(retriever)
        retriever_for_q_to_text = retriever_for_q_to_text.cuda()
    else:
        retriever_for_q_to_text = None

    if opt.disable_gradient_reader and not opt.skip_model:
        for name, param in reader.named_parameters():
                param.requires_grad = False


    if opt.freeze_query_encoder:
        query_encoder = copy.deepcopy(retriever)
        for name, param in query_encoder.named_parameters():
                param.requires_grad = False

    else:
        query_encoder = None
    if opt.use_base_retriever:
        if opt.train_retriever_img:
            base_retriever = copy.deepcopy(retriever)
        elif opt.train_retriever_text:
            base_retriever = copy.deepcopy(retriever_for_q_to_text)

        base_retriever = base_retriever.cuda()
        base_retriever.eval()
        for name, param in base_retriever.named_parameters():
                param.requires_grad = False

    else:
        base_retriever = None

    model = JOMED(opt, reader, retriever, retriever_text,retriever_for_q_to_text, reader_tokenizer, retriever_tokenizer, retriever_tokenizer_train = retriever_tokenizer_train,  query_encoder = query_encoder, passage_encoder_for_train = passage_encoder_for_train, base_retriever = base_retriever, retriever_tokenizer_text = retriever_tokenizer_text)




    if opt.model_checkpoint_path != 'none':
        checkpoint = torch.load(opt.model_checkpoint_path, map_location="cpu")
        opt_checkpoint = checkpoint["opt"]
        step = checkpoint["step"]
        model_dict = checkpoint["model"]

        model = _load_atlas_model_state(opt, opt_checkpoint, model, model_dict)

    if opt.path_retriever_separate_wights != "none":
        checkpoint_ret = torch.load(opt.path_retriever_separate_wights , map_location="cpu")
        model.retriever.contriever.model.load_state_dict(checkpoint_ret)
        if model.retriever_for_q_to_text != None:
            model.retriever_for_q_to_text.contriever.model.load_state_dict(checkpoint_ret)
        if model.base_retriever != None:
            model.base_retriever.contriever.model.load_state_dict(checkpoint_ret)
        if model.passage_encoder_for_train != None:
            model.passage_encoder_for_train.contriever.model.load_state_dict(checkpoint_ret)


        # opt_checkpoint = checkpoint["opt"]
        # # step = checkpoint["step"]
        # model_dict = checkpoint["model"]

        # model.retriever.contriever.model.load_state_dict(model_dict, strict=False)

    # if opt.freeze_passages_encoder_for_train:
    #     model.passage_encoder_for_train.contriever.model.load_state_dict(model_dict, strict=False)

    if eval_only:
        return model, None, None, None, None, None, None

    # if not reset_params:
    if not True:
        optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        pass

        # for name, param in model.retriever.contriever.model.named_parameters():
            # print(f" only Atlas retriever: {name} has gradients {'enabled' if param.requires_grad else 'disabled'}")

        optimizer, scheduler, retr_optimizer, retr_scheduler, params_to_optimize_reader,params_to_optimize_retriever, params_to_optimize_text_retriever, text_retr_optimizer, text_retr_scheduler = set_optim_open_flamingo(opt, model)
        model.set_optimizer_params_reader(params_to_optimize_reader)
        model.set_optimizer_params_retriever(params_to_optimize_retriever)

    freeze_visual_layers = True
    # for k, param_group in enumerate(retr_optimizer.param_groups):
    #     for t, param in enumerate(param_group['params']):
    #         print("t before", t)
    #         # if param.requires_grad is not None:
    #         #     print("retriver params before freezing: ", param.grad[0])


    if freeze_visual_layers:
        if opt.retriever_multimodal_model_type == 'biomedclip':
            if opt.train_retriever_img:
                model.retriever.contriever.model.visual.lock(unlocked_groups=4)
            else:
                for name, param in model.retriever.contriever.model.named_parameters():
                    param.requires_grad = False

            if opt.text_retrieval:
                if opt.train_retriever_text:
                    model.retriever_for_q_to_text.contriever.model.visual.lock(unlocked_groups=4)
                else:
                    for name, param in model.retriever_for_q_to_text.contriever.model.named_parameters():
                        param.requires_grad = False


    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, text_retr_optimizer, text_retr_scheduler


def init_atlas_model(opt, eval_only):
    reader, reader_tokenizer = load_reader(opt)
    retriever, retriever_tokenizer = load_retriever(opt)

    model = Atlas(opt, reader, retriever, reader_tokenizer, retriever_tokenizer)
    model = _cast_and_set_attrs_and_send_to_device(model, opt)

    if eval_only:
        return model, None, None, None, None, opt, 0

    optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)
    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, 0


def load_or_initialize_atlas_model(opt, eval_only=False):
    """
    Either initializes a Atlas from t5 and contriever or loads one from disk.

    if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} doesn't exist, it will init a Atlas

    or, if opt.model_path is "none" and {opt.checkpoint_dir/opt.name} does exist, it will load the Atlas at opt.checkpoint_dir/opt.name/latest

    or, if opt.model_path is not "none" it will load the saved Atlas in opt.model_path
    """
    checkpoint_path = get_checkpoint_path(opt)
    latest_checkpoint_path = os.path.join(checkpoint_path, "checkpoint", "latest")

    if opt.reader_model_type == 'flamingo':
        model, optimizer, scheduler, retr_optimizer, retr_scheduler, text_retr_optimizer, text_retr_scheduler = load_atlas_model_w_flamingo(
            "None", opt, reset_params=None, eval_only=eval_only
        )
    else:
        model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt_checkpoint, loaded_step = load_atlas_model(
            "None", opt, reset_params=None, eval_only=eval_only
        )

    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, text_retr_optimizer, text_retr_scheduler


def save_atlas_model(model, optimizer, scheduler, retr_optimizer, retr_scheduler, step, opt, dir_path, name):

    if opt.save_optimizer and opt.shard_optim:
        optimizer.consolidate_state_dict()
        if retr_optimizer:
            retr_optimizer.consolidate_state_dict()

    if not opt.is_main:
        return 0

    def symlink_force(target, link_name):
        try:
            os.symlink(target, link_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(target, link_name)
            else:
                raise e

    # print(total_size(my_dict), ' total size before filtering: bytes')

    model_to_save = model.module if hasattr(model, "module") else model

    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "model.pth.tar")

    optim_state = optimizer.state_dict() if opt.save_optimizer else None
    if retr_optimizer and opt.save_optimizer:
        retr_optim_state = retr_optimizer.state_dict()
    else:
        retr_optim_state = None


    if scheduler == None:
        checkpoint = {
            "step": step,
            "model": model_to_save.state_dict(),
            "optimizer": optim_state,
            "retr_optimizer": retr_optim_state,
            "scheduler": None,
            "retr_scheduler": retr_scheduler.state_dict() if retr_scheduler else None,
            "opt": opt,
        }
    else:
        checkpoint = {
            "step": step,
            "model": model_to_save.state_dict(),
            "optimizer": optim_state,
            "retr_optimizer": retr_optim_state,
            "scheduler": scheduler.state_dict(),
            "retr_scheduler": retr_scheduler.state_dict() if retr_scheduler else None,
            "opt": opt,
        }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)
    if opt.save_optimizer and opt.shard_optim:
        optimizer._all_states = []


