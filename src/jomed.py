import copy
import logging
from contextlib import suppress
from einops import repeat
from einops import rearrange
import math
import time
from functools import reduce
from typing import List, Optional, Union
from PIL import Image
# import pydicom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from src import dist_utils
from src.retrievers import EMBEDDINGS_DIM
from torchvision import transforms
# import faiss.contrib.torch_utils
# import Random

logger = logging.getLogger(__name__)
IGNORE_INDEX: int = -100
BERT_MAX_SEQ_LENGTH: int = 512


class CustomMultiplicationLoss(nn.Module):
    def __init__(self):
        super(CustomMultiplicationLoss, self).__init__()

    def forward(self, predictions, y):
        # Custom loss: Multiply each element in predictions by 10 and return the sum
        loss = torch.sum(predictions * y, dim=-1)
        loss = torch.sum(loss)
        return loss

def encode_passages(batch, tokenizer, max_length):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    tokens = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


class JOMED(nn.Module):
    def __init__(self, opt, reader, retriever, retriever_text, retriever_for_q_to_text, reader_tokenizer, retriever_tokenizer,retriever_tokenizer_train = None, query_encoder = None, passage_encoder_for_train = None, base_retriever = None, retriever_tokenizer_text = None):
        super(JOMED, self).__init__()
        self.passage_encoder_for_train = passage_encoder_for_train
        self.query_encoder = query_encoder
        self.retriever_tokenizer_text = retriever_tokenizer_text
        self.retriever_for_q_to_text = retriever_for_q_to_text
        self.set_old_params = None
        self.reader = reader
        self.retriever = retriever
        self.retriever_text = retriever_text
        if self.retriever_text != None:
            self.retriever_text.cuda()
            self.retriever_text.eval()
        self.reader_tokenizer = reader_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.retriever_tokenizer_train = retriever_tokenizer_train
        self.opt = opt
        self.model_query_answer_diversity = dict()
        self.params_to_optimize_reader = None
        self.params_to_optimize_retriever = None
        self.base_retriever = base_retriever
        self.bce_loss = nn.BCELoss()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
        ])
        self.dict_q_passage = {}
        self.dict_q_passage_text = {}


    def set_optimizer_params_reader(self, params):
        self.params_to_optimize_reader = params

    def set_optimizer_params_retriever(self, params):
        self.params_to_optimize_retriever = params

    def set_optimizer_params_text_retriever(self, params):
        self.params_to_optimize_text_retriever = params
    def _get_fp16_retriever_copy(self):
        if hasattr(self.retriever, "module"):
            retriever_to_copy = self.retriever.module
        else:
            retriever_to_copy = self.retriever
        return copy.deepcopy(retriever_to_copy).half().eval()

    def _get_fp16_retriever_text_copy(self):
        if hasattr(self.retriever_text, "module"):
            retriever_to_copy = self.retriever_text.module
        else:
            retriever_to_copy = self.retriever_text
        return copy.deepcopy(retriever_to_copy).half().eval()

    def _get_fp16_passage_encoder_for_train_copy(self):
        if hasattr(self.passage_encoder_for_train, "module"):
            passage_to_copy = self.passage_encoder_for_train.module
        else:
            passage_to_copy = self.passage_encoder_for_train
        return copy.deepcopy(passage_to_copy).half().eval()


    @torch.no_grad()
    def build_index(self, index, passages, gpu_embedder_batch_size, logger=None):
        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
        retrieverfp16 = self._get_fp16_retriever_copy()

        total = 0
        for i in range(n_batch):
            batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            batch = [self.opt.retriever_format.format(**example) for example in batch]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, gpu_embedder_batch_size),
                truncation=True,
            )

            embeddings = retrieverfp16(**_to_cuda(batch_enc), is_passages=True)
            index.embeddings[:, total : total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            if i % 500 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        dist_utils.barrier()
        logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()}")

        if not index.is_index_trained():
            logger.info(f"Building faiss indices")
            index.train_index()

    @torch.no_grad()
    def build_index_multi_modal(self, index, passages, gpu_embedder_batch_size, logger=None):
        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)

        if self.opt.ret_acc_text:
            retrieverfp16 = self._get_fp16_retriever_text_copy().cuda()
        else:
            retrieverfp16 = self._get_fp16_retriever_copy().cuda()

        total = 0
        for i in range(n_batch):
            batch = passages[i * gpu_embedder_batch_size: (i + 1) * gpu_embedder_batch_size]
            embeddings = None
            if self.opt.retriever_multimodal_model_type == "clipmd":
                batch = [Image.open(self.opt.retriever_format.format(**example)) for example in batch]
                batch_enc = self.retriever_tokenizer(
                    images=batch,
                    return_tensors="pt", padding=True
                )
                embeddings = retrieverfp16(**_to_cuda(batch_enc)).pooler_output
            elif self.opt.retriever_multimodal_model_type == "biomedclip":
                if self.opt.ret_acc_text:
                    batch_enc = torch.stack(
                        [self.retriever_tokenizer_text(self.opt.retriever_text_format.format(**example), context_length=256) for example in
                         batch]).cuda()
                else:
                    batch_enc = torch.stack([self.retriever_tokenizer(Image.open(self.opt.retriever_format.format(**example))) for example in batch]).half().cuda()

                embeddings = retrieverfp16(batch_enc)

            elif self.opt.retriever_multimodal_model_type == "jina":
                if self.opt.ret_acc_text:
                    batch_enc = torch.stack(
                        [self.retriever_tokenizer_text(self.opt.retriever_text_format.format(**example),
                                                                    max_length=256, padding="max_length",truncation=True, return_tensors="pt")["input_ids"]
                         for example in
                         batch]).cuda()
                else:
                    batch_enc = torch.stack([self.retriever_tokenizer(Image.open(self.opt.retriever_format.format(**example)), return_tensors="pt").pixel_values.squeeze(dim=0) for example in batch]).half().cuda()

                embeddings = retrieverfp16(batch_enc)



            # embeddings = retrieverfp16(**_to_cuda(batch_enc), is_passages=True)

            index.embeddings[:, total: total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            if i % 500 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        dist_utils.barrier()
        logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()}")

        if not index.is_index_trained():
            logger.info(f"Building faiss indices")
            index.train_index()


    def embed_query(self, query_embedr, query):
        if self.opt.retriever_multimodal_model_type == "clipmd":
            query_emb = query_embedr(query).pooler_output
        elif self.opt.retriever_multimodal_model_type == "biomedclip":
            query_emb = query_embedr(query)
        elif self.opt.retriever_multimodal_model_type == "jina":
            query_emb = query_embedr(query)
        return query_emb

    def batch_normalize_tensor(self, tensor, new_min=0.0, new_max=0.025, dim=-1):
        """
        Normalize a batch of tensors along a specified dimension to be between new_min and new_max.
        """
        # Find the min and max along the specified dimension
        current_min = tensor.min(dim=dim, keepdim=True).values
        current_max = tensor.max(dim=dim, keepdim=True).values

        # Handle the case where the maximum is 0 by setting it to new_max
        current_max[current_max == 0] = new_max

        # Avoid division by zero by setting any zero range to 1 (will be scaled to new_max later)
        current_range = current_max - current_min
        current_range[current_range == 0] = 1

        # Scale and shift the tensor elements to the range [new_min, new_max]
        normalized_tensor = (tensor - current_min) / current_range * (new_max - new_min) + new_min

        return normalized_tensor

    @torch.no_grad()
    def _retrieve(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
    ):
        self.retriever.eval()
        if len(query) > 0:
            query_emb = self.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        else:
            query_emb = torch.empty((0, EMBEDDINGS_DIM)).cuda()  # TODO: broken
        if self.training:
            self.retriever.train()

        search_start = time.time()
        if filtering_fun is not None:
            passages, scores = index.search_knn(query_emb, topk * self.opt.filtering_overretrieve_ratio)
            passages, scores = filtering_fun(batch_metadata, passages, scores, topk, training=self.training)
        else:
            passages, scores = index.search_knn(query_emb, topk)
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        return passages, scores, query_emb


    def partial_encoding(self, data, learned_retriever, fixed_retriever, apply_dropout = False):
        query_emb_learned = self.embed_query(learned_retriever, data)
        query_emb_fixed = self.embed_query(fixed_retriever, data)
        if apply_dropout:
            query_emb_learned = F.dropout(query_emb_learned, p=0.20, training=True)

        return query_emb_learned * 0.15 + query_emb_fixed * 0.85

    @torch.no_grad()
    def _image_retrieve(
        self,
        index,
        topk,
        query,
        query_pixel_values,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
        retriever = None,
    ):

        if len(query) > 0:
            retriever = retriever.cuda()

            if self.opt.partial_learning_query_embeddings and self.passage_encoder_for_train != None:
                retriever.eval()
                self.passage_encoder_for_train.eval()
                self.passage_encoder_for_train = self.passage_encoder_for_train.cuda()
                if self.base_retriever == None or not ((self.opt.train_retriever_img and retriever == self.retriever) or (self.opt.train_retriever_text and retriever == self.retriever_for_q_to_text)):
                    query_emb = self.partial_encoding(query_pixel_values, retriever, self.passage_encoder_for_train)
                else:
                    self.base_retriever.eval()
                    query_emb = self.partial_encoding(query_pixel_values, self.base_retriever,
                                                      self.passage_encoder_for_train)

            else:
                if self.query_encoder == None and self.base_retriever == None:
                    retriever.eval()
                    query_emb = self.embed_query(retriever, query_pixel_values)
                elif self.base_retriever != None:
                    self.base_retriever.eval()
                    query_emb = self.embed_query(self.base_retriever, query_pixel_values)
                else:
                    self.query_encoder.eval()
                    self.query_encoder = self.query_encoder.cuda()
                    retriever = retriever.cuda()
                    query_emb = self.embed_query(self.query_encoder, query_pixel_values)
        else:
            query_emb = torch.empty((0, EMBEDDINGS_DIM)).cuda()  # TODO: broken
        # if self.training:
        #     self.retriever.train()

        search_start = time.time()
        if filtering_fun is not None:
            passages, scores = index.search_knn(query_emb, topk + self.opt.filtering_overretrieve_ratio)
            passages, scores = filtering_fun(batch_metadata, passages, scores, topk, training=self.training)
        else:
            if self.opt.retriever_from_hf_index:
                scores, passages_hf = index.get_nearest_examples_batch("image_embeddings",
                                      query_emb.cpu().numpy().astype(np.float32),
                                      k=topk)
                passages = []
                for ex_batch in passages_hf:
                    passages_sample = []
                    for ex_num in range(len(ex_batch["caption"])):
                        ex = {'img_path': ex_batch['image'][ex_num], "text": ex_batch['caption'][ex_num]}
                        passages_sample.append(ex)
                    passages.append(passages_sample)
            else:
                if self.opt.random_query:
                    query_emb = torch.randn_like(query_emb)
                passages, scores = index.search_knn(query_emb, topk)
                # print(len(passages), "len_passages")
                # print(len(passages[0]), "len_tat_passages")
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        return passages, scores, query_emb

    @torch.no_grad()
    def update_text_to_fit_max_len(self,text_tokens, tokenizer, depth_update = 1):

        # Add take the text until certain point
        tokens = text_tokens[:512 - depth_update - 2]
        token_end = tokenizer("</s>", add_special_tokens=False)["input_ids"][-1]
        # answer_token_id = tokenizer("Answer:", add_special_tokens=False)["input_ids"][-1]
        token_eoc = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        if tokens[-1] != token_eoc:
            tokens = torch.cat((tokens, torch.tensor([token_eoc])), dim=0)
        tokens = torch.cat((tokens, torch.tensor([token_end])), dim=0)
        updated_text = tokenizer.decode(tokens)
        updated_text = updated_text.replace('<s>', '')
        updated_text = updated_text.replace('<image> ', '<image>')
        updated_text = updated_text.replace('<PAD>', '')
        updated_text = updated_text.replace(' <|endofchunk|>', '<|endofchunk|>')
        updated_text = updated_text.replace('<|endofchunk|> ', '<|endofchunk|>')
        return updated_text

    @torch.no_grad()
    def check_maxlen_seq_for_reader_tokenizer(self, text, text_tokens, tokenizer):

        token_end = tokenizer("</s>", add_special_tokens=False)["input_ids"][-1]
        token_eoc = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        end_token_id = tokenizer("</s>", add_special_tokens=False)["input_ids"][-1]
        last_answer_idx = torch.where(text_tokens == end_token_id)

        if len(last_answer_idx[-1]) > 0:
            return text
        for i in range(10, 100, 5):
            updated_text = self.update_text_to_fit_max_len(text_tokens, tokenizer, depth_update=i)
            tokens_updated = tokenizer([updated_text])
            last_answer_idx_updated = torch.where(torch.tensor(tokens_updated["input_ids"][0]) == end_token_id)
            if len(last_answer_idx_updated[-1]) > 0:
                return updated_text




    @torch.no_grad()
    def retrieve_with_rerank(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
    ):
        bsz = len(query)
        to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

        # first, do the retrieval
        passages, _, query_emb = self._retrieve(
            index,
            to_rerank,
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata,
            filtering_fun,
            iter_stats,
        )

        retrieverfp16 = self._get_fp16_retriever_copy()
        fstr = self.opt.retriever_format
        flat_passage_strings = [fstr.format(**p) for ps in passages for p in ps]
        encoder_batch_size = min(len(flat_passage_strings), self.opt.per_gpu_embedder_batch_size)
        passage_emb, output_passages, output_scores = (
            query_emb.new_zeros(len(flat_passage_strings), query_emb.shape[-1]),
            [],
            [],
        )

        for b in range(0, len(flat_passage_strings), encoder_batch_size):
            batch = flat_passage_strings[b : b + encoder_batch_size]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                truncation=True,
            )
            batch_emb = retrieverfp16(**_to_cuda(batch_enc), is_passages=True).to(query_emb)
            passage_emb[b : b + encoder_batch_size] = batch_emb

        passage_emb = passage_emb.view(bsz, to_rerank, -1)
        retriever_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
        top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, topk, dim=1)

        for i in range(bsz):
            output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
            output_scores.append(top_retriever_scores[i].tolist())
        return output_passages, output_scores

    @torch.no_grad()
    def image_retrieve_with_rerank(
            self,
            index,
            topk,
            query,
            query_pixel_values,
            batch_metadata=None,
            filtering_fun=None,
            iter_stats={},
    ):
        bsz = len(query)
        to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

        # first, do the retrieval
        passages, _, query_emb = self._image_retrieve(
            index,
            to_rerank,
            query,
            query_pixel_values,
            batch_metadata,
            filtering_fun,
            iter_stats,
        )
        # print("passages shape", len(passages))
        # print("passages", passages)
        retrieverfp16 = self._get_fp16_retriever_copy()

        if self.passage_encoder_for_train != None:
            passage_encoder_for_train_fp16 = self._get_fp16_passage_encoder_for_train_copy()


        fstr = self.opt.retriever_format
        # print("ret format", fstr)
        flat_passage_strings = [fstr.format(**p) for ps in passages for p in ps]
        encoder_batch_size = min(len(flat_passage_strings), self.opt.per_gpu_embedder_batch_size)
        passage_emb, output_passages, output_scores = (
            query_emb.new_zeros(len(flat_passage_strings), query_emb.shape[-1]),
            [],
            [],
        )
        passage_emb = passage_emb.half()

        for b in range(0, len(flat_passage_strings), encoder_batch_size):
            batch = flat_passage_strings[b : b + encoder_batch_size]
            embeddings = None
            if self.opt.retriever_multimodal_model_type == "clipmd":
                batch = [Image.open(self.opt.retriever_format.format(**example)) for example in batch]
                batch_enc = self.retriever_tokenizer(
                    images=batch,
                    return_tensors="pt", padding=True
                )
                batch_emb = retrieverfp16(**_to_cuda(batch_enc)).pooler_output
            elif self.opt.retriever_multimodal_model_type == "biomedclip":

                # [print("example:", example) for example in batch]
                # print("example:", example)
                batch_enc = torch.stack(
                    [self.retriever_tokenizer(Image.open(example)) for example in
                     batch]).half().cuda()

                if self.opt.partial_learning_query_embeddings and self.passage_encoder_for_train != None:

                    batch_emb = self.partial_encoding(batch_enc, retrieverfp16, passage_encoder_for_train_fp16)
                else:
                    batch_emb = retrieverfp16(batch_enc)

            passage_emb[b: b + encoder_batch_size] = batch_emb

        passage_emb = passage_emb.view(bsz, to_rerank, -1)
        retriever_scores = torch.einsum("id, ijd->ij", [query_emb.half(), passage_emb])
        top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, topk, dim=1)

        for i in range(bsz):
            output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
            output_scores.append(top_retriever_scores[i].tolist())
        return output_passages, output_scores

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        if not self.opt.multi_modal:
            retrieve_func = self.retrieve_with_rerank if self.opt.retrieve_with_rerank else self._retrieve
        else:
            retrieve_func = self.image_retrieve_with_rerank if self.opt.retrieve_with_rerank else self._image_retrieve
        passages, scores = retrieve_func(*args, **kwargs)[:2]
        return passages, scores

    @torch.no_grad()
    def enable_gradients_flamingo(self):
        for name, param in self.reader.named_parameters():
                if name in self.params_to_optimize_reader:
                    # print("enable gradients")
                    param.requires_grad = True

    @torch.no_grad()
    def enable_gradients_biomedclip(self):
        for name, param in self.reader.named_parameters():
            if name in self.params_to_optimize_reader:
                # print("enable gradients")
                param.requires_grad = True

    @torch.no_grad()
    def disable_gradients_flamingo(self):
        for name, param in self.reader.named_parameters():
            if name in self.params_to_optimize_reader:
                # print("enable gradients")
                param.requires_grad = False

    @torch.no_grad()
    def disable_gradients_biomedclip(self):
        for name, param in self.reader.named_parameters():
            if name in self.params_to_optimize_reader:
                # print("enable gradients")
                param.requires_grad = False

    @torch.no_grad()
    def disable_gradients_flamingo(self):
        for name, param in self.reader.named_parameters():
            if name in self.params_to_optimize_reader:
                # print("enable gradients")
                param.requires_grad = False

    def retrieve_for_training(self, *args, **kwargs):
        if not self.opt.multi_modal:
            retrieve_func =  self._retrieve
        else:
            retrieve_func =  self._image_retrieve
        passages, scores = retrieve_func(*args, **kwargs)[:2]
        return passages, scores

    def append_query(self, query, passages):
        return [self.opt.encoder_format.format(query=query, **p) for p in passages]


    def augment_image(self, image):
        width, height = image.size

        # Randomly rotate the image between 0 and 360 degrees
        rotation_angle = random.randint(0, 360)
        rotated_image = image.rotate(rotation_angle)

        # Randomly decide whether to flip the image horizontally
        if random.choice([True, False]):
            flipped_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            flipped_image = rotated_image

        # Randomly crop the image
        # Ensure the crop box is smaller than the image and randomly positioned
        # max_crop_size = min(width, height) // 8
        crop_width = random.randint(30, 50)
        crop_height = random.randint(30, 50)
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        crop_box = (left, top, left + crop_width, top + crop_height)
        cropped_image = flipped_image.crop(crop_box)

        return cropped_image
    def image_query_process(self, img_path):
        if '.dcm' in img_path:
            # dcm_data = pydicom.read_file(img_path)
            dcm_data = pydicom.dcmread(img_path)
            img = dcm_data.pixel_array
            img = Image.fromarray(img).convert('RGB')
        else:
            with Image.open(img_path) as img:
                img.load()
        return img


    def retriever_tokenize(self, query):
        if self.retriever_tokenizer:

            if not self.opt.multi_modal:
                query_enc = self.retriever_tokenizer(
                    query,
                    max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                query_enc = _to_cuda(query_enc)
            else:
                if self.opt.retriever_multimodal_model_type == "clipmd":
                    batch = [ self.image_query_process(q) for q in query]
                    query_enc = self.retriever_tokenizer(
                        images=batch,
                        return_tensors="pt", padding=True
                    )
                    query_enc = _to_cuda(query_enc)
                elif self.opt.retriever_multimodal_model_type == "biomedclip":
                    query_enc = torch.stack(
                        [self.retriever_tokenizer( self.image_query_process(q)) for q
                         in query]).cuda()
                elif self.opt.retriever_multimodal_model_type == "jina":
                    query_enc = torch.stack(
                        [self.retriever_tokenizer(self.image_query_process(q)).pixel_values.squeeze(dim=0) for q
                         in query]).cuda()

        else:
            query_enc = None
        return query_enc



    def reader_tokenize(self, query, target, target_tokens, generation_mode = False):
        if target_tokens is None:
            if self.opt.decoder_prompt_format is not None:
                modified_query = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
                target = [q + t for (q, t) in zip(modified_query, target)]

                query_mask = self.reader_tokenizer(
                    modified_query,
                    max_length=self.opt.target_maxlength,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"]

            if self.opt.decoder_format is not None:
                target = [self.opt.decoder_format.format(target=t) for t in target]

            if self.opt.multi_modal:
                target_app = [q + ' ' + t for q, t in zip(query, target)]

            target_app = [t + "</s>" if not t.endswith("</s>") else t for t in target_app]
            if not self.opt.multi_modal:
                target_tokens = self.reader_tokenizer(
                    target_app,
                    max_length=self.opt.target_maxlength,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
            elif self.opt.reader_model_type == 'flamingo' and not self.opt.qwen_model and not self.opt.pixtral_model:
                target_tokens =self.reader_tokenizer.encode_text(target_app, generation_mode = generation_mode)
            elif self.opt.qwen_model:
                target_tokens = self.reader_tokenizer.encode_text(target_app)


        if not self.opt.multi_modal:
            decoder_input_ids = self.reader._shift_right(target_tokens["input_ids"])
        else:
            decoder_input_ids = target_tokens["input_ids"]
        labels = target_tokens["input_ids"].masked_fill(~target_tokens["attention_mask"].bool(), IGNORE_INDEX)

        # If decoder prompt is not None mask labels such that the model is not trained to predict the prompt
        if self.opt.decoder_prompt_format is not None:
            query_mask = self.reader_tokenizer(
                modified_query,
                max_length=self.opt.target_maxlength,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"]

            padding = torch.zeros((query_mask.size(0), target_tokens["input_ids"].size(-1) - query_mask.size(-1)))
            query_mask = torch.cat([query_mask, padding], dim=1)
            labels = labels.masked_fill(query_mask.bool(), IGNORE_INDEX)

        return labels.cuda(), decoder_input_ids.cuda()

    def tokenize(self, query, target, target_tokens):
        if query is None and target is None:
            return None, None, None

        assert (
            target_tokens is None or self.opt.decoder_prompt_format is None
        ), "decoder_prompt_format not compatible with target tokenized in iterator"

        query_enc = self.retriever_tokenize(query) if not self.opt.use_file_passages else None
        # labels, decoder_input_ids = self.reader_tokenize(query, target, target_tokens)
        labels, decoder_input_ids = None, None
        return query_enc, labels, decoder_input_ids

    def multi_modal_create_labels_to_text(self, reader_tok_text, task):
        label_tok_text = reader_tok_text["input_ids"].clone().cuda()
        label_tok_text_m = torch.tensor([], dtype=torch.int64).cuda()
        for ltt in label_tok_text:
            # warnings.warn(f"labels before, {ltt} \n")
            label_tok_text_m = torch.cat(
                (label_tok_text_m, task.create_labels_to_flamingo(torch.stack([ltt]), self.reader_tokenizer.tokenizer)),
                dim=0)


        label_tok_text_m = label_tok_text_m.cuda()

        return label_tok_text_m

    def process_tensors_with_grad(self, tensor_dict):
        """
        Takes a dictionary of tensors, detaches and clones each tensor, sets them to require gradients,
        and returns a new dictionary with these processed tensors.

        Args:
        - tensor_dict (dict): A dictionary where keys are tensor names and values are PyTorch tensors.

        Returns:
        - new_tensor_dict (dict): A new dictionary with each tensor detached, cloned, and set to require gradients.
        """
        new_tensor_dict = {}
        for key, tensor in tensor_dict.items():
            # Detach, clone, and set to require gradients
            processed_tensor = tensor.clone()
            # Store the processed tensor in the new dictionary
            new_tensor_dict[key] = processed_tensor
        return new_tensor_dict

    def multi_modal_preprocess_input_to_reader(self, query_image, query_text, passages, target,  task, generation_mode = False, train_retriever = False, append_only_one_passage_to_query = False):
        if len(query_image) == 0:
            return None, None, None

        if self.opt.reader_model_type == "flamingo":
            query_passages_image = []

            for i in range(0, len(query_image)):

                qpi = [ppp['img_path'] for ppp in passages[i]] + [query_image[i]]
                query_passages_image.append(qpi)



            query_passages_image_opened = []

            for i in range(0, len(query_image)):
                qpiopend = []
                for q in query_passages_image[i]:
                    if not q == '':
                        qpiopend.append(self.image_query_process(q))


                query_passages_image_opened.append(qpiopend)

            if self.opt.reader_model_type == 'flamingo':

                # prepapre images for model

                reader_tok_img_no_aug = [self.reader_tokenizer.preprocess_images(qpiopened) for qpiopened in query_passages_image_opened ]
                # reader_tok_img = reader_tok_img_no_aug
                if not generation_mode:
                    reader_tok_img = [self.transform(re_to_img) for re_to_img in reader_tok_img_no_aug]
                else:
                    reader_tok_img = reader_tok_img_no_aug

                reader_tok_img = torch.stack([repeat(rtimg, 'N c h w -> b N T c h w', b=1, T=1)  for rtimg in reader_tok_img], dim=0).squeeze(1)

                reader_tok_img_no_aug = torch.stack([repeat(rtimg, 'N c h w -> b N T c h w', b=1, T=1)  for rtimg in reader_tok_img_no_aug], dim=0).squeeze(1)

                if (train_retriever and not generation_mode) or append_only_one_passage_to_query:
                    # Extract the last channel
                    query_tensor = reader_tok_img[:, -1:, :, :, :, :].clone()  # This will have a shape of (2, 1, 255, 255)

                    # Initialize a list to hold the concatenated tensors
                    training_retrieve_reader_tok_image = []

                    # Concatenate the last channel with each of the other channels
                    for i in range(reader_tok_img.shape[0]):  # Loop through the batch
                        for j in range(reader_tok_img.shape[1] - 1):  # Loop through the first two channels
                            training_retrieve_reader_tok_image.append(
                                torch.cat((reader_tok_img[i:i + 1, j:j + 1, :, :, :, :], query_tensor[i:i + 1, :, :, :, :, :]), dim=1))


                    training_retrieve_reader_tok_image = torch.cat(training_retrieve_reader_tok_image, dim=0)


                    # One to make sure that they are without aug the training
                    query_tensor_no_aug = reader_tok_img_no_aug[:, -1:, :, :, :,
                                   :].clone()  # This will have a shape of (2, 1, 255, 255)

                    # Initialize a list to hold the concatenated tensors
                    training_retrieve_reader_tok_image_no_aug = []

                    # Concatenate the last channel with each of the other channels
                    for i in range(reader_tok_img.shape[0]):  # Loop through the batch
                        for j in range(reader_tok_img.shape[1] - 1):  # Loop through the first two channels
                            training_retrieve_reader_tok_image_no_aug.append(
                                torch.cat((reader_tok_img_no_aug[i:i + 1, j:j + 1, :, :, :, :],
                                           query_tensor_no_aug[i:i + 1, :, :, :, :, :]), dim=1))

                    training_retrieve_reader_tok_image_no_aug = torch.cat(training_retrieve_reader_tok_image_no_aug, dim=0)

                    reader_tok_img = reader_tok_img.cuda()
                    # reader_tok_img_no_aug = reader_tok_img.cuda()



                # prepare text for model
                # input_model, input_model_training_retriever = task.create_input_flamingo_model(passages, query_text, target)
                input_model, input_model_training_retriever = task.create_input_flamingo_model(passages, query_text, target)
                # input_model_training_retriever = task.create_input_flamingo_model_for_training_retriever(passages, query_text, target)
                if not generation_mode:
                    input_model = [t + "</s>" if not t.endswith("</s>") else t for t in input_model]
                    if train_retriever or append_only_one_passage_to_query:
                        input_model_training_retriever = [t + "</s>" if not t.endswith("</s>") else t for t in input_model_training_retriever]


                if (train_retriever and not generation_mode) or append_only_one_passage_to_query:
                    reader_tok_text_training_retriever = self.reader_tokenizer.encode_text(input_model_training_retriever,
                                                                                           generation_mode=generation_mode)

                reader_tok_text = self.reader_tokenizer.encode_text(input_model, generation_mode = generation_mode)

                # make sure the last doest truncated by the tokenizer
                if not generation_mode:
                    reader_updated_text = [self.check_maxlen_seq_for_reader_tokenizer(text, reader_tok_text["input_ids"][i], self.reader_tokenizer.tokenizer) for i, text in enumerate(input_model)]
                    reader_tok_text = self.reader_tokenizer.encode_text(reader_updated_text, generation_mode=generation_mode)

                    if train_retriever or append_only_one_passage_to_query:
                        reader_updated_text_training_retriever = [
                            self.check_maxlen_seq_for_reader_tokenizer(text, reader_tok_text_training_retriever["input_ids"][i],
                                                                       self.reader_tokenizer.tokenizer) for i, text in
                            enumerate(input_model_training_retriever)]
                        reader_tok_text_training_retriever = self.reader_tokenizer.encode_text(reader_updated_text_training_retriever,
                                                                            generation_mode=generation_mode)


                # prepare label for model
                if not generation_mode:
                    label_tok_text = reader_tok_text["input_ids"].clone().cuda()
                    label_tok_text_m = torch.tensor([], dtype = torch.int64).cuda()

                    if train_retriever or append_only_one_passage_to_query:
                        label_tok_text_training_retriever = reader_tok_text_training_retriever["input_ids"].clone()
                        label_tok_text_m_training_retriever = torch.tensor([], dtype=torch.int64)

                        for ltt in label_tok_text_training_retriever:
                            label_tok_text_m_training_retriever = torch.cat((label_tok_text_m_training_retriever,
                                                                             task.create_labels_to_flamingo(
                                                                                 torch.stack([ltt]),
                                                                                 self.reader_tokenizer.tokenizer)),
                                                                            dim=0)

                    for ltt in label_tok_text:
                        # warnings.warn(f"labels before, {ltt} \n")
                        # if self.opt.closed_book:
                        label_tok_text_m = torch.cat((label_tok_text_m, task.create_labels_to_flamingo(torch.stack([ltt]), self.reader_tokenizer.tokenizer)), dim = 0)


                    reader_tok_text = _to_cuda(reader_tok_text)
                    label_tok_text_m = label_tok_text_m.cuda()

                else:
                    label_tok_text_m = None

                if append_only_one_passage_to_query and not generation_mode:
                    reader_tok_img, reader_tok_text, label_tok_text_m  = training_retrieve_reader_tok_image.clone(), self.process_tensors_with_grad(reader_tok_text_training_retriever), label_tok_text_m_training_retriever.clone()
                elif append_only_one_passage_to_query and generation_mode:
                    reader_tok_img, reader_tok_text = training_retrieve_reader_tok_image.clone(), self.process_tensors_with_grad(
                        reader_tok_text_training_retriever)


                if not generation_mode and train_retriever:
                    return reader_tok_img, reader_tok_text, label_tok_text_m, training_retrieve_reader_tok_image_no_aug, reader_tok_text_training_retriever, label_tok_text_m_training_retriever
                else:
                    return reader_tok_img, reader_tok_text, label_tok_text_m, None, None, None
        else:
            return None, None, None





    def multi_modal_tokenize(self, query, query_text, target, target_tokens, generation_mode = False):
        if query is None and target is None:
            return None, None, None

        assert (
            target_tokens is None or self.opt.decoder_prompt_format is None
        ), "decoder_prompt_format not compatible with target tokenized in iterator"

        if not self.opt.use_file_passages and not self.opt.closed_book:
           query_enc = self.retriever_tokenize(query)
        else:
            query_enc = None

        if not self.opt.qwen_model and not self.opt.pixtral_model:
            labels, decoder_input_ids = self.reader_tokenize(query_text, target, target_tokens, generation_mode = generation_mode)
            return query_enc, labels, decoder_input_ids
        else:
            return query_enc, None, None


    def multi_modal_tokenize_passages(self, query_text,query_image, passages, task, generation_mode = False):
        if len(query_image) == 0:
            return None, None
        if self.opt.reader_model_type == "flamingo":
            query_passages_text = task.create_query_passages_flamingo(passages, query_text)
            query_passages_image = [ppp['img_path'] for ppp in passages[0]] + query_image
        else:
            query_passages = [self.append_query(q, p) for q, p in zip(query, passages)]

        # fstr = self.opt.retriever_format
        # retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        # if self.retriever_tokenizer:
        #     retriever_tok = encode_passages(
        #         retriever_passages,
        #         self.retriever_tokenizer,
        #         min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
        #     )
        #     retriever_tok = _to_cuda(retriever_tok)
        # else:
        #     retriever_tok = None

        query_passages_image_opened = [self.image_query_process(q) for q in query_passages_image]
        if self.opt.reader_model_type == 'flamingo':
            reader_tok_img = self.reader_tokenizer.preprocess_images(query_passages_image_opened)
            reader_tok_img = repeat(reader_tok_img, 'N c h w -> b N T c h w', b=1, T=1)
            reader_tok_text = self.reader_tokenizer.encode_text(query_passages_text, generation_mode = generation_mode)
            reader_tok_text = _to_cuda(reader_tok_text)
            reader_tok_img = reader_tok_img.cuda()
        else:
            reader_tok_text = encode_passages(query_passages_text, self.reader_tokenizer, self.opt.text_maxlength)
            reader_tok_text = _to_cuda(reader_tok_text)
            reader_tok_img = _to_cuda(reader_tok_img)

        return reader_tok_text, reader_tok_img, None, query_passages_text
        # return reader_tok_text, reader_tok_img, retriever_tok

    def tokenize_passages(self, query, passages):
        if len(query) == 0:
            return None, None

        query_passages = [self.append_query(q, p) for q, p in zip(query, passages)]

        fstr = self.opt.retriever_format
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_tokenizer:
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_tokenizer,
                min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None
        reader_tok = encode_passages(query_passages, self.reader_tokenizer, self.opt.text_maxlength)
        reader_tok = _to_cuda(reader_tok)
        return reader_tok, retriever_tok

    def perplexity_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            self.reader.eval()
            total_context = reader_ids.size(1)
            cfg.n_context = 1
            cfg.bsz = bsz * total_context
            reader_ids_score = reader_ids.view(bsz * total_context, -1)
            reader_mask_score = reader_mask.view(bsz * total_context, -1)
            repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, total_context, dim=0)
            repeated_labels = torch.repeat_interleave(labels, total_context, dim=0)
            reader_output = self.reader(
                input_ids=reader_ids_score.cuda(),
                attention_mask=reader_mask_score.cuda(),
                decoder_input_ids=repeated_decoder_input_ids,
                labels=repeated_labels,
                use_cache=False,
            )
            token_loss = nn.functional.cross_entropy(
                reader_output.logits.view(-1, reader_output.logits.size(-1)),
                repeated_labels.flatten(),
                reduction="none",
            )
            gold_score = token_loss.view(bsz, total_context, -1)
            z = (repeated_labels.view(bsz, total_context, -1) > -1).sum(dim=-1)
            gold_score = -gold_score.sum(dim=-1) / z

            return gold_score

    def perplexity_score_multi_modal(self, reader_tokens_img, reader_tokens_text, labels, n_context, bsz, iter_stats = None, specific_tokens = None):
        with torch.no_grad():
            self.reader.eval()
            _to_cuda(reader_tokens_text)
            reader_ids = reader_tokens_text["input_ids"]  # FIXME
            reader_mask = reader_tokens_text["attention_mask"].bool()
            labels = labels.cuda()
            if self.opt.reader_model_type == 'flamingo' and self.opt.multi_modal:
                autocast = self.get_autocast(
                    self.opt.precision, cache_enabled=False)
                if self.opt.precision == 'fp16' or self.opt.precision == 'bf16':
                    # reader_tokens_img = reader_tokens_img.half()
                    # print(f"load {self.opt.precision} ")
                    cast_dtype = self.get_cast_dtype(self.opt.precision)
                    reader_tokens_img = reader_tokens_img.to('cuda', dtype=cast_dtype, non_blocking=True)
                with autocast():
                    reader_output = self.reader(
                        vision_x=reader_tokens_img,
                        lang_x=reader_ids.cuda(),
                        attention_mask=reader_mask.cuda(),
                        labels=labels,
                    )
            if iter_stats is not None:
                iter_stats["train_retriever_reader_loss"] = (reader_output[0], 1)

            # print("reader loss while training the ret", reader_output[0])
            if specific_tokens != None:
                reader_logit = reader_output.logits.view(-1, reader_output.logits.size(-1))
                reader_logit = reader_logit[(labels.flatten() == specific_tokens[0]) | (labels.flatten() == specific_tokens[1])]
                reader_logit = reader_logit[:, specific_tokens]
                labels_spe = labels.flatten()[(labels.flatten() == specific_tokens[0]) | (labels.flatten() == specific_tokens[1])]

                for i, token in enumerate(specific_tokens):
                    labels_spe[labels_spe == token] = i

                reader_logit = torch.softmax(reader_logit, dim=-1)
            else:
                reader_logit = reader_output.logits.view(-1, reader_output.logits.size(-1))
                labels_spe = labels.flatten()



            token_loss = nn.functional.cross_entropy(
                reader_logit,
                labels_spe,
                reduction="none",
            )
            gold_score = token_loss.view(bsz, n_context, -1)
            z = (labels.view(bsz, n_context, -1) > -1).sum(dim=-1)
            gold_score = -gold_score.sum(dim=-1) / z

            return gold_score


    def loss_score_multi_modal(self, reader_tokens_img, reader_tokens_text, labels, n_context, bsz,
                                     iter_stats=None):
        with torch.no_grad():
            self.reader.eval()
            _to_cuda(reader_tokens_text)
            reader_ids = reader_tokens_text["input_ids"]  # FIXME
            reader_mask = reader_tokens_text["attention_mask"].bool()
            labels = labels.cuda()
            if self.opt.reader_model_type == 'flamingo' and self.opt.multi_modal:
                autocast = self.get_autocast(
                    self.opt.precision, cache_enabled=False)
                if self.opt.precision == 'fp16' or self.opt.precision == 'bf16':
                    # reader_tokens_img = reader_tokens_img.half()
                    # print(f"load {self.opt.precision} ")
                    cast_dtype = self.get_cast_dtype(self.opt.precision)
                    reader_tokens_img = reader_tokens_img.to('cuda', dtype=cast_dtype, non_blocking=True)
                with autocast():
                    reader_output = self.reader(
                        vision_x=reader_tokens_img,
                        lang_x=reader_ids.cuda(),
                        attention_mask=reader_mask.cuda(),
                        labels=labels,
                    )
            if iter_stats is not None:
                iter_stats["train_retriever_reader_loss"] = (reader_output[0], 1)

            return reader_output[0]

    def eval_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz, mask_query):
        self.reader.eval()
        self.reader.reset_score_storage()
        cfg.bsz = reader_ids.size(0)
        cfg.n_context = reader_ids.size(1)
        reader_ids_score = reader_ids.view(reader_ids.size(0), -1)
        reader_mask_score = reader_mask.view(reader_mask.size(0), -1)
        with torch.no_grad():
            reader_output = self.reader(
                input_ids=reader_ids_score,
                attention_mask=reader_mask_score,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            crossattention_scores = self.reader.get_crossattention_scores(
                cfg.n_context,
                reader_mask_score,
                labels=labels,
                ids=reader_ids,
                mode=self.opt.gold_score_mode,
                mask_query=mask_query,
            )
            gold_score = select_crossattention_scores(crossattention_scores, self.opt.gold_score_mode)

            if self.training:
                self.reader.train()
            return gold_score

    def loop_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            total_context = reader_ids.size(1)
            doc_len = reader_ids.size(-1)
            self.reader.eval()
            cfg.bsz = bsz
            cfg.n_context = total_context
            reader_ids_score_eval = reader_ids.view(reader_ids.size(0), -1)
            reader_mask_score_eval = reader_mask.view(reader_mask.size(0), -1)

            # forward pass for calculating and caching the encoder states:
            reader_output_eval = self.reader(
                input_ids=reader_ids_score_eval,
                attention_mask=reader_mask_score_eval,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            eval_hidden_state = reader_output_eval.encoder_last_hidden_state

            # run n_docs - 1 forward passes to calculate pp when leaving a doc out
            gold_scores = []
            for loo_index in range(total_context):
                reader_mask_loo = reader_mask.clone()
                reader_mask_loo[:, loo_index] = False  # mask out this doc
                loo_output_eval = self.reader(
                    encoder_outputs=[eval_hidden_state],
                    attention_mask=reader_mask_loo.view(bsz, (total_context) * doc_len),
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    use_cache=False,
                )
                token_loss = nn.functional.cross_entropy(
                    loo_output_eval.logits.view(-1, loo_output_eval.logits.size(-1)), labels.view(-1), reduction="none"
                )
                mean_loss = token_loss.view(bsz, labels.shape[-1]).sum(dim=-1) / (labels > -1).sum(-1)
                gold_scores.append(mean_loss)

            gold_score = torch.stack(gold_scores, dim=1)

            return gold_score

    @torch.no_grad()
    def emdr_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        self.reader.eval()
        cfg.n_context = 1
        cfg.bsz = bsz * self.opt.retriever_n_context
        reader_ids_score = reader_ids.view(bsz * self.opt.retriever_n_context, -1)
        reader_mask_score = reader_mask.view(bsz * self.opt.retriever_n_context, -1)
        repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, self.opt.retriever_n_context, dim=0)
        repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
        reader_output = self.reader(
            input_ids=reader_ids_score.cuda(),
            attention_mask=reader_mask_score.cuda(),
            labels=repeated_labels,
            use_cache=False,
        )
        gold_score = reader_output.logits
        return gold_score

    @torch.no_grad()
    def emdr_score_multi_modal(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        self.reader.eval()
        cfg.n_context = 1
        cfg.bsz = bsz * self.opt.retriever_n_context
        reader_ids_score = reader_ids.view(bsz * self.opt.retriever_n_context, -1)
        reader_mask_score = reader_mask.view(bsz * self.opt.retriever_n_context, -1)
        repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, self.opt.retriever_n_context, dim=0)
        repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
        reader_output = self.reader(
            input_ids=reader_ids_score.cuda(),
            attention_mask=reader_mask_score.cuda(),
            labels=repeated_labels,
            use_cache=False,
        )
        gold_score = reader_output.logits
        return gold_score

    @torch.no_grad()
    def clean_generation(self, response):
        """
        for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
        """
        response = response.replace('<unk> ', '').strip()
        response = response.replace('<PAD>', '').strip()
        return response
    @torch.no_grad()
    def get_autocast(self, precision, cache_enabled=True):
        if precision == "amp":
            return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
        elif precision == "amp_bfloat16" or precision == "amp_bf16":
            # amp_bfloat16 is more stable than amp float16 for clip training
            return lambda: torch.cuda.amp.autocast(
                dtype=torch.bfloat16, cache_enabled=cache_enabled
            )
        else:
            return suppress

    @torch.no_grad()
    def get_cast_dtype(self, precision: str):
        cast_dtype = None
        if precision == "bf16":
            cast_dtype = torch.bfloat16
        elif precision == "fp16":
            cast_dtype = torch.float16
        return cast_dtype

    def cal_gold_score_for_ret_training(self,cfg, query, query_enc, passages_train,  reader_tokens_img_training_retriever,reader_token_text_training_retriever, label_training_retriever, iter_stats , specific_tokens, task, target, ppp):
        encode_images = False
        if self.opt.train_retriever_text and self.opt.train_retriever_img:
            if ppp > self.retriever_n_context:
                encode_images = False
                retriever = self.retriever_for_q_to_text
            else:
                encode_images = True
                retriever = self.retriever
        elif self.opt.train_retriever_text:
            encode_images = False
            retriever = self.retriever_for_q_to_text
        elif self.opt.train_retriever_img:
            encode_images = True
            retriever = self.retriever



        if self.opt.use_gradient_checkpoint_retriever:
            # if False:
            if self.opt.retriever_multimodal_model_type == "biomedclip" or self.opt.retriever_multimodal_model_type == "jina":
                for name, param in self.retriever.named_parameters():
                    if name in self.params_to_optimize_retriever:
                        param.requires_grad = True

        if self.retriever_tokenizer_train != None:
            query_enc_train = torch.stack(
                [self.retriever_tokenizer_train(self.image_query_process(q)) for q
                 in query]).cuda()
        else:
            query_enc_train = query_enc

        if self.opt.partial_learning_query_embeddings and self.passage_encoder_for_train != None:

            self.passage_encoder_for_train.eval()
            self.passage_encoder_for_train = self.passage_encoder_for_train.cuda()
            query_emb_fixed = self.embed_query(self.passage_encoder_for_train, query_enc_train)
            query_emb = self.partial_encoding(query_enc_train, retriever, self.passage_encoder_for_train,
                                              apply_dropout=False)


        elif self.query_encoder == None:
            # query_emb = self.embed_query(self.retriever, query_enc_train)
            query_emb = self.embed_query(retriever, query_enc_train)
        else:
            # query_emb = self.embed_query(self.query_encoder, query_enc_train)
            query_emb = self.embed_query(retriever, query_enc_train)
            query_emb = query_emb.detach()

        # if self.opt.use_gradient_checkpoint_retriever:
        #     self.retriever.gradient_checkpointing_enable()
        if encode_images:
            passages_images_only_tokenized = []
            for i in range(0, len(query_enc)):
                po = [Image.open(self.opt.retriever_format.format(**example)) for example in passages_train[i]]
                if self.opt.retriever_multimodal_model_type == "biomedclip":
                    if self.retriever_tokenizer_train != None:
                        bpassages = [self.retriever_tokenizer_train(passage) for passage in po]
                    else:
                        bpassages = [self.retriever_tokenizer(passage) for passage in po]
                    passages_images_only_tokenized.append(torch.stack(bpassages))


                elif self.opt.retriever_multimodal_model_type == "jina":
                    if self.retriever_tokenizer_train != None:
                        bpassages = [self.retriever_tokenizer_train(passage).pixel_values.squeeze(dim=0) for passage in po]
                    else:
                        bpassages = [self.retriever_tokenizer(passage).pixel_values.squeeze(dim=0) for passage in po]
                    passages_images_only_tokenized.append(torch.stack(bpassages))

            torch_passages_images_only_tokenized = torch.stack(passages_images_only_tokenized).cuda()
            bsz, n_retrive = cfg["bsz"], cfg["retriever_n_context"]
            torch_passages_images_only_tokenized = rearrange(torch_passages_images_only_tokenized,
                                                             "b n c h w -> (b n) c h w")


            if self.opt.partial_learning_query_embeddings and self.passage_encoder_for_train != None and self.opt.retrieve_with_rerank:
                passage_emb = self.partial_encoding(torch_passages_images_only_tokenized, self.retriever,
                                                    self.passage_encoder_for_train)
            elif self.passage_encoder_for_train == None:
                if self.opt.retriever_multimodal_model_type == "biomedclip" or self.opt.retriever_multimodal_model_type == "jina":
                    passage_emb = self.retriever(torch_passages_images_only_tokenized)
            else:
                if self.opt.retriever_multimodal_model_type == "biomedclip"or self.opt.retriever_multimodal_model_type == "jina":
                    passage_emb = self.passage_encoder_for_train(torch_passages_images_only_tokenized).detach()

        else:
            passages_images_only_tokenized = []
            for i in range(0, len(query_enc)):
                if self.opt.retriever_multimodal_model_type == "jina":
                    bpassages = [self.retriever_tokenizer_text(self.opt.retriever_text_format.format(**example),
                                                       max_length=256, padding="max_length", truncation=True,
                                                       return_tensors="pt")["input_ids"]
                         for example in
                         passages_train[i]]
                else:
                    bpassages = [self.retriever_tokenizer_text(self.opt.retriever_text_format.format(**example), context_length=256) for example in passages_train[i]]

                passages_images_only_tokenized.append(torch.stack(bpassages))

            torch_passages_images_only_tokenized = torch.stack(passages_images_only_tokenized).squeeze().cuda()
            bsz, n_retrive = cfg["bsz"], cfg["retriever_n_context"]
            torch_passages_images_only_tokenized = rearrange(torch_passages_images_only_tokenized,
                                                             "b n c -> (b n) c")
            passage_emb = self.retriever_text(torch_passages_images_only_tokenized.cuda()).detach()



        passage_emb = passage_emb.view(bsz, n_retrive, -1)
        retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
        retriever_score_copied = torch.einsum("id, ijd->ij", [query_emb_fixed, passage_emb]).detach().clone()
        retriever_score_copied = retriever_score.detach().clone()

        if self.opt.use_gradient_checkpoint_retriever:
            # if False:
            if self.opt.retriever_multimodal_model_type == "biomedclip":
                for name, param in self.retriever.named_parameters():
                    if name in self.params_to_optimize_retriever:
                        param.requires_grad = False

        if self.opt.retriever_training_logits_from_generation:

            if self.opt.qwen_model or self.opt.pixtral_model:
                generation, inf_logits, gold_score, weights = self.generate_qwen(
                    reader_token_text_training_retriever, reader_tokens_img_training_retriever,
                    specific_tokens=specific_tokens, labels=label_training_retriever,
                    bsz=cfg["bsz"], n_context=cfg["retriever_n_context"], task=task, targets=target,
                )
            else:
                generation, inf_logits, gold_score, weights = self.generate_flamingo(
                    reader_token_text_training_retriever, reader_tokens_img_training_retriever, specific_tokens=specific_tokens, labels=label_training_retriever,
                    bsz=cfg["bsz"], n_context=cfg["retriever_n_context"],task=task, targets= target,
                )
        else:
            gold_score = self.perplexity_score_multi_modal(reader_tokens_img_training_retriever,
                                                           reader_token_text_training_retriever,
                                                           label_training_retriever, cfg["retriever_n_context"],
                                                           cfg["bsz"], iter_stats, specific_tokens)
            weights = None


        return gold_score, retriever_score, retriever_score_copied, query_emb, weights

    @torch.no_grad()
    def generate_qwen(self, tokens_text, token_img, specific_tokens = None, labels= None,bsz = None, n_context = None,task = None, targets= None, choices=None):
        # print("hereeeeee")
        if self.reader != None:
            self.reader.eval()

        weights = None
        self.reader_tokenizer.builder.padding_side = "left"
        self.reader_tokenizer.tokenizer.padding_side = "left"
        if self.opt.qwen_model:
            inputs = self.reader_tokenizer.builder(
                text=tokens_text,
                images=token_img,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
        else:
            self.reader_tokenizer.tokenizer.pad_token = '</s>'
            inputs = self.reader_tokenizer.builder(text=tokens_text, images=token_img,padding=True, return_tensors="pt")
            inputs = inputs.to(self.reader.device)



        if not self.opt.skip_model:
            with torch.no_grad():
                # torch.cuda.empty_cache()
                output = self.reader.generate(
                    **inputs,
                    num_beams=1,
                    max_new_tokens=self.opt.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=False)
        else:
            output = dict()
            output["scores"] = torch.rand(self.opt.max_new_tokens, n_context * bsz, self.reader_tokenizer.tokenizer.vocab_size)
            output["sequences"] = torch.rand(self.opt.max_new_tokens, n_context * bsz, self.reader_tokenizer.tokenizer.vocab_size)

        if specific_tokens != None and labels != None and bsz != None and n_context != None:
            list_labels_value = getattr(task, 'label', None)
            pad_token = \
            self.reader_tokenizer.tokenizer(self.reader_tokenizer.tokenizer.pad_token, add_special_tokens=False)[
                "input_ids"][-1]
            if list_labels_value is not None:
                new_labels = torch.zeros((output['scores'][0].shape[0], 1))
                for index_output_to_label in range(output['scores'][0].shape[0]):
                    target_index = int(np.floor((index_output_to_label) / n_context))
                    new_labels[index_output_to_label, :] = list_labels_value.index(targets[target_index])

                logits_all = torch.zeros((output['scores'][0].shape[0], len(list_labels_value)))
                for num_label, label in enumerate(list_labels_value):
                    label_tokens = specific_tokens[num_label]
                    logits_label = None
                    counter_vals = 0
                    for num_token, token in enumerate(label_tokens):
                        if token == pad_token:
                            continue
                        counter_vals = counter_vals + 1
                        logits_first_word = output['scores'][num_token]
                        value = logits_first_word[:, [token]]
                        if logits_label is None:
                            logits_label = value
                        else:
                            logits_label = logits_label + value
                    logits_label = logits_label / counter_vals
                    # print(logits_label.shape)
                    # print(logits_all.shape)
                    logits_all[:, num_label] = logits_label.squeeze()

                reader_logit = torch.softmax(logits_all, dim=-1)
                labels_spe = new_labels.clone().flatten().long()
            elif self.opt.only_prob_retrieve_loss:
                size_labels = min(torch.stack(output['scores']).shape[0], labels.shape[-1])
                labels = labels[:, : size_labels]
                raw_logits = torch.stack(output['scores'])[:size_labels, :, :]
                logits = raw_logits.permute(1, 0, 2)
                reader_logit = logits.reshape(-1, logits.size(-1))

                labels_spe = labels.reshape(-1)
                labels_spe[labels_spe == pad_token] = -100
                # logits_all = torch.zeros((output['scores'][0].shape[0], 1))
                # for num_label in range(bsz * n_context):
                #     index_token = int(np.floor((num_label) / n_context))
                #     label_tokens = specific_tokens[index_token]
                #     counter_vals = 0
                #     logits_label = None
                #     for num_token, token in enumerate(label_tokens):
                #         if token == pad_token or num_token > (self.opt.max_new_tokens - 1):
                #             continue
                #         counter_vals = counter_vals + 1
                #         if num_token < len(output['scores']):
                #             logits_first_word = output['scores'][num_token]
                #             value = logits_first_word[num_label, [token]]
                #         else:
                #             value = 0
                #         if logits_label is None:
                #             logits_label = value
                #         else:
                #             logits_label = logits_label + value
                #
                #     logits_label = logits_label / counter_vals
                #     logits_all[num_label, 0] = logits_label.squeeze()
            else:
                logits_first_word = output['scores'][0]
                logits_first_word = logits_first_word[:, specific_tokens]
                labels_spe = labels.clone()

                for i, token in enumerate(specific_tokens):
                    labels_spe[labels_spe == token] = i

                # max_vals, _ = logits_first_word.max(dim=-1, keepdim=True)
                # logits_first_word = logits_first_word / (max_vals + 1e-8)
                reader_logit = torch.softmax(logits_first_word, dim=-1)

            if self.opt.weighting_kl:
                _, predicted_classes = torch.max(reader_logit.view(bsz, n_context, -1), 2)
                uniformity_check = (predicted_classes == predicted_classes[:, 0:1]).all(dim=1)
                weights = torch.where(uniformity_check, torch.tensor(0.0), torch.tensor(1.0))

            # if self.opt.only_prob_retrieve_loss:
            #     token_loss = logits_all.view(bsz, n_context, -1).cuda()
            # else:
            token_loss = nn.functional.cross_entropy(
                reader_logit.cuda(),
                labels_spe.cuda(),
                reduction="none",
            )


            gold_score = token_loss.view(bsz, n_context, -1)
            z = (labels.view(bsz, n_context, -1) > -1).sum(dim=-1)
            if not self.opt.only_prob_retrieve_loss:
                gold_score = -gold_score.sum(dim=-1)
            else:
                gold_score = -gold_score.sum(dim=-1)
        else:
            gold_score = None
        # torch.cuda.empty_cache()
        return output['sequences'], output['scores'], gold_score, weights

    def resize_image(self, image):

        if image.size[0] > 256 or image.size[1] > 256:
            resize_factor = 256 / max(image.width, image.height)
            width, height = max(int(image.width * resize_factor), 28), max(int(image.height * resize_factor), 28)
            image = image.resize((width, height), resample=Image.NEAREST)
            # image = image.resize((256, 256))
        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    def preprocess_input_for_qwen(self, query, query_text, task, passages):
        input_model, input_model_training_retriever = task.create_input_qwen_model(passages,
                                                                                         query_text
                                                                                         )
        reader_token_text_training_retriever = []
        reader_tokens_img_training_retriever = []

        if self.opt.closed_book:
            text_encode = input_model
        else:
            text_encode = input_model_training_retriever

        for i in range(len(text_encode)):
            index = int(np.floor((i) / len(passages[0])))
            # print(index)
            if self.opt.closed_book:
                img_path = query[index]

                val = {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": text_encode[i]},
                    ],
                }
            else:
                index_im = i % len(passages[0])
                # print(index_im)
                img_path = query[index]
                ret_image = passages[index][index_im]['img_path']
                text = passages[index][index_im]['text']
                if self.opt.add_retrieve_text_to_prompt and self.opt.ret_from_training_set:
                    text_background_image = f'background: The reference answer of the most similar case is {text}\n\n'
                elif self.opt.add_retrieve_text_to_prompt:
                    text_background_image = f'background: {text}\n\n'
                else:
                    text_background_image = "background"
                if not self.opt.remove_ret_image and not self.opt.remove_input_image:
                    val = {
                        "role": "user",
                        "content": [
                                {"type": "image", "image": ret_image},
                                {"type": "text", "text": text_background_image},
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": text_encode[i]},
                        ],
                    }
                elif self.opt.remove_input_image:
                    val = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ret_image},
                            {"type": "text", "text": text_background_image},
                            {"type": "text", "text": text_encode[i]},
                        ],
                    }
                else:
                    val = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_background_image},
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": text_encode[i]},
                        ],
                    }

            reader_token_text_training_retriever.append(self.reader_tokenizer.builder.apply_chat_template(
            [val], tokenize=False, add_generation_prompt=True
            ))
            images, _ = self.reader_tokenizer.vision_processor([val])
            images_resized = []
            for image in images:
                image = self.resize_image(image)
                # if image.size[0] > 256 or image.size[1] > 256:
                #     resize_factor = 256 / max(image.width, image.height)
                #     width, height = max(int(image.width * resize_factor), 28), max(int(image.height * resize_factor), 28)
                #     image = image.resize((width, height), resample=Image.NEAREST)
                #     # image = image.resize((256, 256))
                # if image.width / image.height > 200:
                #     width, height = image.height * 180, image.height
                #     image = image.resize((width, height), resample=Image.NEAREST)
                #
                # if image.height / image.width > 200:
                #     width, height = image.width, image.width * 180
                #     image = image.resize((width, height), resample=Image.NEAREST)

                images_resized.append(image)

            reader_tokens_img_training_retriever.append(images_resized)


        return reader_tokens_img_training_retriever, reader_token_text_training_retriever
    def preprocess_input_for_pixtral(self, query, query_text, task, passages):
        input_model, input_model_training_retriever = task.create_input_qwen_model(passages,
                                                                                         query_text
                                                                                         )
        reader_token_text_training_retriever = []
        reader_tokens_img_training_retriever = []

        if self.opt.closed_book:
            text_encode = input_model
        else:
            text_encode = input_model_training_retriever

        for i in range(len(text_encode)):
            index = int(np.floor((i) / len(passages[0])))
            # print(index)
            if self.opt.closed_book:
                img_path = query[index]
                images = [img_path]
                val = {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "content": text_encode[i]},
                    ],
                }
            else:
                index_im = i % len(passages[0])
                # print(index_im)
                img_path = query[index]
                ret_image = passages[index][index_im]['img_path']
                if self.opt.remove_input_image:
                    images = [ret_image]
                else:
                    images = [ret_image, img_path]
                text = passages[index][index_im]['text']
                if self.opt.add_retrieve_text_to_prompt and self.opt.ret_from_training_set:
                    text_background_image = f'background: The reference answer of the most similar case is {text}\n\n'
                elif self.opt.add_retrieve_text_to_prompt:
                    text_background_image = f'background: {text}\n\n'
                else:
                    text_background_image = "background"
                if self.opt.remove_input_image:
                    val = {
                        "role": "user",
                        "content": [
                                {"type": "image"},
                                {"type": "text", "content": text_background_image},
                                {"type": "text", "content": text_encode[i]},
                        ],
                    }
                else:
                    val = {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "content": text_background_image},
                            {"type": "image"},
                            {"type": "text", "content": text_encode[i]},
                        ],
                    }

            reader_token_text_training_retriever.append(self.reader_tokenizer.builder.apply_chat_template(
            [val]
            ))

            images_resized = []
            for image in images:
                img = Image.open(image)
                img = self.resize_image(img)
                img = np.array(img)
                if img.shape[-1] != 3:
                    img = np.stack((img, img, img), axis=-1)
                img = Image.fromarray(img)
                images_resized.append(img)

            # img_torch = self.reader_tokenizer.builder.image_processor(images_resized,  return_tensors = "pt").pixel_values
            reader_tokens_img_training_retriever.append(images_resized)


        return reader_tokens_img_training_retriever, reader_token_text_training_retriever

    def multi_modal_retrieve(self, index, index_text, n_context, query, query_enc, batch_metadata, task, for_train = False):


        if (self.opt.load_index_path is not None and query_enc is not None and self.opt.img_retrieval) and ((for_train and self.opt.train_retriever_img) or not for_train ):
            query_enc = query_enc.cuda()


            passages_img, scores_img = self.retrieve(
                index,
                n_context,
                query,
                query_enc,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
                retriever = self.retriever
            )
            # for rr, qqq in enumerate(query):
            #     if qqq in self.dict_q_passage.keys():
            #         for ppp in passages_img[rr]:
            #             self.dict_q_passage[qqq].add(ppp["img_path"])
            #     else:
            #         self.dict_q_passage[qqq] = set()
            #         for ppp in passages_img[rr]:
            #             self.dict_q_passage[qqq].add(ppp["img_path"])



        else:
            passages_img, scores_img = None, None

        if (self.opt.load_index_text_path is not None and query_enc is not None and self.opt.text_retrieval) and ((for_train and self.opt.train_retriever_text) or not for_train):

            passages_text, scores_text = self.retrieve(
                index_text,
                n_context,
                query,
                query_enc,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
                retriever=self.retriever_for_q_to_text
            )
            # for rr, qqq in enumerate(query):
            #     if qqq in self.dict_q_passage_text.keys():
            #         for ppp in passages_text[rr]:
            #             self.dict_q_passage_text[qqq].add(ppp["img_path"])
            #     else:
            #         self.dict_q_passage_text[qqq] = set()
            #         for ppp in passages_text[rr]:
            #             self.dict_q_passage_text[qqq].add(ppp["img_path"])
            # for i_batch in range(len(passages_text)):
            #     for i_passage in len(passages_text[0]):
            #         passages_text[i_batch][i_passage]["text_source"] = True
        else:
            passages_text, scores_text = None, None



        if passages_text is not None and passages_img is not None:
            passages = [
                sub1 + sub2
                for sub1, sub2 in zip(passages_img, passages_text)
            ]
            scores = [
                sub1 + sub2
                for sub1, sub2 in zip(scores_img, scores_text)
            ]

        elif passages_text is not None:
            passages = passages_text
            scores = scores_text
        else:
            passages = passages_img
            scores = scores_img
        return passages, passages_text, passages_img, scores, scores_text, scores_img



    def forward(
        self,
        index,
        index_text,
        query,
        query_text,
        target,
        target_tokens=None,
        passages=None,
        batch_metadata=None,
        filtering_fun=None,
        use_cache=False,
        train_retriever=False,
        iter_stats={},
        task = None
    ):
        forward_start = time.time()
        bsz = len(query)

        if not self.opt.multi_modal:
            query_enc, labels, decoder_input_ids = self.tokenize(query, target, target_tokens)
        else:
            query_enc, labels, decoder_input_ids = self.multi_modal_tokenize(query, query_text, target, target_tokens)

        if not self.opt.use_file_passages and not self.opt.closed_book:
            retrieve_start = time.time()
            passages, _,_,_,_,_ = self.multi_modal_retrieve(index, index_text,self.opt.n_context, query, query_enc, batch_metadata, task)
            iter_stats["runtime/retrieve"] = (time.time() - retrieve_start, 1)

        if not self.opt.multi_modal:
            reader_tokens, retriever_tokens = self.tokenize_passages(query, passages)
            reader_ids = reader_tokens["input_ids"]  # FIXME
            reader_mask = reader_tokens["attention_mask"].bool()
        else:
            # warnings.warn("start multi modal preprocessing \n")
            if (not self.opt.qwen_model) and (not self.opt.pixtral_model):
                reader_tokens_img, reader_tokens_text, labels, reader_tokens_img_training_retriever, reader_token_text_training_retriever, label_training_retriever = self.multi_modal_preprocess_input_to_reader(query, query_text, passages, target, task,
                                                       generation_mode=False, train_retriever = train_retriever, append_only_one_passage_to_query=self.opt.append_only_one_passage_to_query)
                decoder_input_ids = reader_tokens_text
                # reader_tokens_text, reader_tokens_img, _, query_passages_text = self.multi_modal_tokenize_passages(
                #     query_text, query, passages, task)
                # _, labels, decoder_input_ids = self.multi_modal_tokenize(query, query_passages_text, target, target_tokens)
                reader_ids = reader_tokens_text["input_ids"]  # FIXME
                reader_mask = reader_tokens_text["attention_mask"].bool()


                n_context_training = min(self.opt.n_context, reader_ids.size(1))

        # cfg = self.reader.encoder.config
        cfg = {"bsz": len(target), "n_context": self.opt.n_context,"retriever_n_context":self.opt.n_context}

        retriever_loss = None
        retriever_loss_text = None
        if train_retriever:
            self.retriever.train()
            if self.opt.retrieve_passage_for_train_retriever:

                if self.opt.aug_query_before_retrieve:
                    query_enc_training = []  # This will be the new list containing 10 images as specified
                    query_for_training = []

                    for bbb, img in enumerate(query_enc):
                        # Append the original image first (without augmentation)
                        query_enc_training.append(img)
                        query_for_training.append(query[bbb])
                        # Now, apply the transformation 4 times and append each result
                        for _ in range((self.opt.retriever_n_context // 2) - 1):
                            transformed_img = self.transform(img)
                            query_enc_training.append(transformed_img)
                            query_for_training.append(query[bbb])

                    query_enc_training = torch.stack(query_enc_training, dim=0).cuda()
                    number_to_retrieve = 2
                else:
                    number_to_retrieve = self.opt.retriever_n_context
                    query_for_training = query
                    query_enc_training = query_enc

                passages_train, _,_,_,_,_ = self.multi_modal_retrieve(index, index_text, number_to_retrieve, query_for_training, query_enc_training,
                                                        batch_metadata, task, for_train = True)


                if self.opt.aug_query_before_retrieve:
                    flat_list = [item for sublist in passages_train for item in sublist]
                    passages_train = list(np.array(flat_list).reshape(len(query_enc), self.opt.retriever_n_context))

                if self.opt.retriever_training_logits_from_generation:
                    if self.opt.qwen_model:
                        reader_tokens_img_training_retriever, reader_token_text_training_retriever = self.preprocess_input_for_qwen(
                            query, query_text, task, passages_train)
                    elif self.opt.pixtral_model:
                        reader_tokens_img_training_retriever, reader_token_text_training_retriever = self.preprocess_input_for_pixtral(
                            query, query_text, task, passages_train)
                    else:
                        reader_tokens_img_training_retriever, reader_token_text_training_retriever, _, _, _, _ = self.multi_modal_preprocess_input_to_reader(
                            query, query_text, passages_train, None, task, generation_mode=True,
                            train_retriever=train_retriever,
                            append_only_one_passage_to_query=self.opt.append_only_one_passage_to_query)
                else:
                    _, _, _, reader_tokens_img_training_retriever, reader_token_text_training_retriever, label_training_retriever = self.multi_modal_preprocess_input_to_reader(
                        query, query_text, passages_train, target, task, generation_mode=False,
                        train_retriever=train_retriever,
                        append_only_one_passage_to_query=self.opt.append_only_one_passage_to_query)


                cfg["retriever_n_context"] = self.opt.retriever_n_context
            else:
                passages_train = passages
                if self.opt.retriever_training_logits_from_generation:
                    reader_tokens_img_training_retriever, reader_token_text_training_retriever, _, _, _, _, = self.multi_modal_preprocess_input_to_reader(
                        query, query_text, passages_train, None, task, generation_mode=True,
                        train_retriever=train_retriever,
                        append_only_one_passage_to_query=self.opt.append_only_one_passage_to_query)

            # encode passage and query
            gold_score = torch.tensor([]).cuda()
            gold_score_text = torch.tensor([]).cuda()
            final_weights = None
            retriever_score = torch.tensor([], requires_grad=True).cuda()
            retriever_score_text = torch.tensor([], requires_grad=True).cuda()
            retriever_score_copied = torch.tensor([]).cuda()
            retriever_score_copied_text = torch.tensor([]).cuda()
            self.retriever.train()



            jumps = 5
            for ppp in range(0, len(passages_train[0]), jumps):
                # print("jumps",ppp)

                if self.opt.qwen_model or self.opt.pixtral_model:
                    batch_reader_tokens_img_training_retriever = [reader_tokens_img_training_retriever[
                                                                  (mmm * self.opt.retriever_n_context) + ppp:(
                                                                                                                         mmm * self.opt.retriever_n_context) + ppp + jumps]
                                                                  for mmm in range(len(passages))]
                    batch_reader_tokens_img_training_retriever = [item for sublist in
                                                                  batch_reader_tokens_img_training_retriever for item in
                                                                  sublist]
                    batch_reader_token_text_training_retriever = [reader_token_text_training_retriever[
                                                                  (mmm * self.opt.retriever_n_context) + ppp:(
                                                                                                                         mmm * self.opt.retriever_n_context) + ppp + jumps]
                                                                  for mmm in range(len(passages))]
                    batch_reader_token_text_training_retriever = [item for sublist in
                                                                  batch_reader_token_text_training_retriever for item in
                                                                  sublist]
                    batch_passages = [item[ppp: ppp + jumps] for item in passages_train]
                    batch_label_training_retriever = None
                else:
                    # print(reader_token_text_training_retriever["input_ids"].shape, "reader_token_text_training_retriever")
                    batch_reader_token_text_training_retriever = {key: torch.concat([value[(mmm * self.opt.retriever_n_context) + ppp:(mmm * self.opt.retriever_n_context) + ppp+jumps] for mmm in range(len(passages))], dim=0)   for key, value in reader_token_text_training_retriever.items()}
                    # print(batch_reader_token_text_training_retriever["input_ids"].shape, "batch_reader_token_text_training_retriever")
                    batch_passages = [item[ppp: ppp + jumps] for item in passages_train]
                    reader_tokens_img_training_retriever[:][ppp: ppp + jumps]
                    batch_label_training_retriever = torch.concat([label_training_retriever[(mmm * self.opt.retriever_n_context) + ppp:(mmm * self.opt.retriever_n_context) + ppp+jumps] for mmm in range(len(passages))], dim=0)
                    batch_reader_tokens_img_training_retriever = torch.concat([reader_tokens_img_training_retriever[(mmm * self.opt.retriever_n_context) + ppp:(mmm * self.opt.retriever_n_context) + ppp+jumps] for mmm in range(len(passages))], dim=0)
                batch_cfg = {}
                batch_cfg["bsz"], batch_cfg["retriever_n_context"] = len(passages_train), jumps
                if self.opt.specific_tokens_for_retriever_loss:
                    list_labels_value = getattr(task, 'label', None)
                    if list_labels_value is None and not self.opt.use_targets and not self.opt.only_prob_retrieve_loss:
                        yes_token = self.reader_tokenizer.tokenizer(" Yes", add_special_tokens=False)["input_ids"][-1]
                        no_token = self.reader_tokenizer.tokenizer(" No", add_special_tokens=False)["input_ids"][-1]
                        specific_tokens = [no_token, yes_token]
                        list_labels_value = ["No", "Yes"]
                    elif self.opt.use_targets or self.opt.only_prob_retrieve_loss:
                        specific_tokens = []
                        list_labels_value = target.copy()
                        for label in target:

                            self.reader_tokenizer.tokenizer.padding_side = "right"
                            specific_tokens.append(
                                self.reader_tokenizer.tokenizer(label, add_special_tokens=False, max_length=5,
                                                                padding="max_length", truncation = True)["input_ids"])
                    else:
                        specific_tokens = []
                        for label in list_labels_value:
                            self.reader_tokenizer.tokenizer.padding_side = "right"
                            specific_tokens.append(self.reader_tokenizer.tokenizer(label, add_special_tokens=False, max_length=self.opt.max_new_tokens, padding="max_length", truncation = True)["input_ids"])

                    if self.opt.retriever_training_logits_from_generation:
                        labels_for_gen = []
                        for num in range(batch_cfg["bsz"] * batch_cfg["retriever_n_context"]):
                            index_query = int(np.floor((num) / batch_cfg["retriever_n_context"]))
                            iiii = list_labels_value.index(target[index_query])
                            labels_for_gen.append(specific_tokens[iiii])
                            if index_query is None:
                                raise Exception(f"{list_labels_value} Error {target[index_query]}")
                        batch_label_training_retriever = torch.tensor(labels_for_gen).cuda()

                else:
                    specific_tokens = None

                batch_gold_score, batch_retriever_score, batch_retriever_score_copied, query_emb, weights = self.cal_gold_score_for_ret_training(batch_cfg, query, query_enc, batch_passages, batch_reader_tokens_img_training_retriever, batch_reader_token_text_training_retriever, batch_label_training_retriever, iter_stats, specific_tokens, task, target, ppp)
                if ppp == 0:
                    final_weights = weights
                elif weights != None:
                    final_weights += weights
                if self.opt.train_retriever_text and self.opt.train_retriever_img:
                    if ppp < self.opt.retriever_n_context:
                        gold_score = torch.cat((gold_score, batch_gold_score), dim=-1)
                        retriever_score = torch.cat((retriever_score, batch_retriever_score), dim=-1)
                        retriever_score_copied = torch.cat((retriever_score_copied, batch_retriever_score_copied), dim=-1)
                    else:
                        gold_score_text = torch.cat((gold_score_text, batch_gold_score), dim=-1)
                        retriever_score_text = torch.cat((retriever_score_text, batch_retriever_score), dim=-1)
                        retriever_score_copied_text = torch.cat((retriever_score_copied_text, batch_retriever_score_copied), dim=-1)
                elif self.opt.train_retriever_img:
                    gold_score = torch.cat((gold_score, batch_gold_score), dim=-1)
                    retriever_score = torch.cat((retriever_score, batch_retriever_score), dim=-1)
                    retriever_score_copied = torch.cat((retriever_score_copied, batch_retriever_score_copied), dim=-1)
                elif self.opt.train_retriever_text:
                    gold_score_text = torch.cat((gold_score_text, batch_gold_score), dim=-1)
                    retriever_score_text = torch.cat((retriever_score_text, batch_retriever_score), dim=-1)
                    retriever_score_copied_text = torch.cat((retriever_score_copied_text, batch_retriever_score_copied),
                                                            dim=-1)


            if self.opt.weighting_kl:

                for index, query_iter in enumerate(query):
                    if query_iter not in self.model_query_answer_diversity.keys():
                        self.model_query_answer_diversity[query_iter] = []

                    self.model_query_answer_diversity[query_iter].append(final_weights[index])

                final_weights[final_weights > 0] = 1

            if "ppmean" in self.opt.gold_score_mode:

                if not self.opt.supervised_gold_score:
                    if not self.opt.add_prior_to_gold_score and not self.opt.add_prior_to_gold_score_with_ref:
                        gold_score_normalized = self.opt.normalize_factor_loss - self.batch_normalize_tensor(gold_score,
                                                                                                             new_max=self.opt.normalize_factor_loss)
                        gold_score = retriever_score_copied - gold_score_normalized
                    elif self.opt.add_prior_to_gold_score:
                        llm_gold_score = torch.nn.functional.log_softmax(gold_score, dim=-1)
                        retriever_score_gold_score = torch.nn.functional.log_softmax(retriever_score_copied, dim=-1)
                        gold_score = llm_gold_score + retriever_score_gold_score
                        if self.opt.train_retriever_text:
                            llm_gold_score_text = torch.nn.functional.log_softmax(gold_score_text, dim=-1)
                            retriever_score_gold_score_text = torch.nn.functional.log_softmax(retriever_score_copied_text, dim=-1)
                            gold_score_text = llm_gold_score_text + retriever_score_gold_score_text
                    else:
                        gold_score_copied = -gold_score.detach().clone()
                        retriever_score_copied_to_gold_score = (1 / retriever_score_copied) * gold_score_copied
                        retriever_loss = torch.mean(retriever_score * retriever_score_copied_to_gold_score)

                        # self.passage_encoder_for_train.eval()
                        # self.passage_encoder_for_train = self.passage_encoder_for_train.cuda()
                        # query_emb = self.partial_encoding(query_enc, self.retriever,
                        #                                   self.passage_encoder_for_train,
                        #                                   apply_dropout=False)
                        # gold_score_copied = gold_score.detach().clone()
                        # query_emb_copied = query_emb.detach().clone()
                        # gold_score_copied = -gold_score_copied.sum(dim=-1)
                        # y = torch.zeros_like(query_emb_copied).cuda()
                        # for i in range(gold_score_copied.shape[0]):
                        #     num_of_non_zero_vals = 0
                        #     for j in range(query_emb_copied.shape[1]):
                        #         if query_emb_copied[i][j] != 0:
                        #             y[i][j] = 1 / query_emb_copied[i][j]
                        #             num_of_non_zero_vals = num_of_non_zero_vals + 1
                        #         else:
                        #             y[i][j] = 0
                        #
                        #     y[i] = y[i] * (gold_score_copied[i] / num_of_non_zero_vals)
                        #
                        # retriever_loss = CustomMultiplicationLoss()(query_emb, y)




                        # reader_tokens_img_closed_book, reader_tokens_text_closed_book, labels_closed_book, _, _, _ = self.multi_modal_preprocess_input_to_reader(
                        #     query, query_text, [task.closed_book_passage[0], task.closed_book_passage[0]], target, task,
                        #     generation_mode=False, train_retriever=False, append_only_one_passage_to_query=False)
                        # gold_score_closed_book = self.perplexity_score_multi_modal(reader_tokens_img_closed_book,
                        #                                                            reader_tokens_text_closed_book,
                        #                                                            labels_closed_book,
                        #                                                            1, cfg["bsz"])
                        # gold_score_closed_book_expanded = gold_score_closed_book.expand(-1, cfg["retriever_n_context"])
                        # llm_gold_score = torch.stack((gold_score, gold_score_closed_book_expanded), dim=2)
                        # llm_gold_score = torch.softmax(llm_gold_score, dim=-1)
                        # llm_gold_score = llm_gold_score[:, :, 0]
                        # # llm_gold_score = torch.nn.functional.log_softmax(llm_gold_score, dim=-1)
                        # # retriever_score_gold_score = torch.nn.functional.log_softmax(retriever_score_copied, dim=-1)
                        # gold_score = llm_gold_score
                        # gold_score = llm_gold_score + retriever_score_gold_score


                        # gold_score = []
                        # for bbb in range(cfg["bsz"]):
                        #     batch_pass = []
                        #     for ccc in range(cfg["n_context"]):
                        #         if passages[bbb][ccc]['Pneumonia'] == 0 and passages[bbb][ccc]['Lung Opacity'] == 0 and target[bbb] == 'No':
                        #             batch_pass.append(1.0)
                        #         elif (passages[bbb][ccc]['Pneumonia'] == 1 or passages[bbb][ccc]['Lung Opacity'] == 1) and target[bbb] == 'Yes':
                        #             batch_pass.append(1.0)
                        #         else:
                        #             batch_pass.append(0.0)

                        #     gold_score.append(batch_pass)
                        #
                        # gold_score = torch.tensor(gold_score).cuda()

        if not self.opt.qwen_model and not self.opt.pixtral_model:
            cfg["bsz"] = reader_ids.size(0)
            cfg["n_context"] = n_context_training

            reader_ids_training = reader_ids[:, :n_context_training].contiguous()
            reader_mask_training = reader_mask[:, :n_context_training].contiguous()

            reader_ids_training = reader_ids_training.view(reader_ids.size(0), -1)
            reader_mask_training = reader_mask_training.view(reader_mask.size(0), -1)


            if self.opt.use_gradient_checkpoint_reader:
            # if False:
                if self.opt.reader_model_type == 'flamingo':
                    for name, param in self.reader.named_parameters():
                        if name in self.params_to_optimize_reader:
                            param.requires_grad = True
                else:
                    self.reader.gradient_checkpointing_enable()

            if self.opt.reader_model_type == 'flamingo' and self.opt.multi_modal:

                autocast = self.get_autocast(
                    self.opt.precision, cache_enabled=False)  # if fsdp, disable cache to save memory


                if self.opt.precision == 'fp16' or self.opt.precision == 'bf16':
                    # reader_tokens_img = reader_tokens_img.half()
                    # print(f"load {self.opt.precision} ")
                    cast_dtype = self.get_cast_dtype(self.opt.precision)
                    reader_tokens_img = reader_tokens_img.to('cuda', dtype=cast_dtype, non_blocking=True)

                with autocast():
                    reader_output = self.reader(
                    vision_x=reader_tokens_img,
                    lang_x=reader_ids.cuda(),
                    attention_mask=reader_mask.cuda().to(torch.float32),
                    labels=labels.cuda(),
            )
            else:
                reader_output = self.reader(
                    input_ids=reader_ids_training,
                    attention_mask=reader_mask_training,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    use_cache=False,
                )
            reader_loss = reader_output[0]

            if self.opt.use_gradient_checkpoint_reader:
            # if False:
                if self.opt.reader_model_type == 'flamingo':
                    for name, param in self.reader.named_parameters():
                        if name in self.params_to_optimize_reader:
                            # print("disable gradients")
                            param.requires_grad = False
                else:
                    self.reader.gradient_checkpointing_enable()
        else:
            reader_loss = torch.tensor([0])
        if train_retriever:

            if self.opt.multi_modal:

                # retriever_score = retriever_score / np.sqrt(query_emb.size(-1))
                if gold_score is not None:
                    gold_score = gold_score.float()
                    retriever_score = retriever_score.float()
                    if self.opt.gold_score_mode == "emdr":
                        retriever_loss = self.logprob(retriever_score, gold_score, labels)
                    else:
                        if not self.opt.add_prior_to_gold_score_with_ref:
                            if self.opt.weighting_kl:
                               if self.opt.symmetric_kl:
                                   retriever_loss = self.kldivloss_w(gold_score, retriever_score, final_weights.cuda())
                               else:
                                    retriever_loss = self.kldivloss_w(retriever_score, gold_score, final_weights.cuda())
                            else:
                                if self.opt.symmetric_kl:
                                    retriever_loss = self.kldivloss(gold_score, retriever_score)
                                else:
                                    if self.opt.train_retriever_img:
                                        retriever_loss = self.kldivloss(retriever_score, gold_score)
                                    if self.opt.train_retriever_text:
                                        retriever_loss_text = self.kldivloss(retriever_score_text, gold_score_text)



                        else:
                            retriever_loss = retriever_loss
                        # retriever_loss = self.kldivloss_adaptive(retriever_score, gold_score, target)


        # self.reader.reset_score_storage()
        iter_stats["loss/reader_loss"] = (reader_loss.item(), len(query))
        if retriever_loss is not None:
            iter_stats["loss/retriever_loss"] = (retriever_loss.item(), len(query))

        iter_stats["runtime/forward"] = (time.time() - forward_start, 1)

        return reader_loss, retriever_loss, retriever_loss_text

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)

    def kldivloss_w(self, score, gold_score, weights):
        gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        weights_expanded = torch.unsqueeze(weights, dim=-1).expand_as(score)
        kl_div_loss = torch.nn.KLDivLoss( reduction="none")(score, gold_score)

        # Apply the weights to the loss
        weighted_loss = kl_div_loss * weights_expanded

        # Sum or average the loss
        final_loss = weighted_loss.mean()
        return final_loss

    def kldivloss_adaptive(self, score, gold_score, targets):

        torch_target = []
        for ttt in targets:
            if "Yes" in ttt:
                torch_target.append(1)
            else:
                torch_target.append(0)

        targets = torch.tensor(torch_target)

        yes_scores = gold_score[targets == 1]
        no_scores = gold_score[targets == 0]

        yes_scores = torch.softmax(yes_scores / 0.1, dim=-1)
        no_scores = torch.softmax(no_scores / self.opt.temperature_gold, dim=-1)

        combined_softmax_gold_scores = torch.zeros_like(gold_score)
        combined_softmax_gold_scores[targets == 1] = yes_scores
        combined_softmax_gold_scores[targets == 0] = no_scores



        # gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, combined_softmax_gold_scores)

    def kldivloss_balanced(self, score, gold_score, targets):

        torch_target = []
        for ttt in targets:
            if "Yes" in ttt:
                torch_target.append(1)
            else:
                torch_target.append(0)

        targets = torch.tensor(torch_target)

        yes_scores = gold_score[targets == 1]
        no_scores = gold_score[targets == 0]

        yes_scores = torch.softmax(yes_scores / 0.1, dim=-1)
        no_scores = torch.softmax(no_scores / self.opt.temperature_gold, dim=-1)

        combined_softmax_gold_scores = torch.zeros_like(gold_score)
        combined_softmax_gold_scores[targets == 1] = yes_scores
        combined_softmax_gold_scores[targets == 0] = no_scores

        # gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, combined_softmax_gold_scores)



    def logprob(self, score, gold_score, labels):
        with torch.no_grad():
            repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
            repeated_labels[repeated_labels == IGNORE_INDEX] = 0

            mask_labels = labels >= 0

            gold_log_prob = torch.nn.functional.log_softmax(gold_score / self.opt.temperature_gold, dim=-1)
            gold_log_probs = torch.gather(gold_log_prob, dim=-1, index=repeated_labels[..., None]).view(
                gold_log_prob.size(0), -1
            )
            gold_log_probs = gold_log_probs.view(score.size(0), score.size(1), -1)

        log_score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        log_prob = gold_log_probs + log_score[..., None]
        logsumprobs = torch.logsumexp(log_prob, dim=1)
        loss = -1 * torch.sum(logsumprobs * mask_labels) / torch.sum(mask_labels)

        return loss

    @torch.no_grad()
    def compute_reader_loss_and_logits(self, tokens, decoder_input_ids, labels):
        cfg = self.reader.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        reader_loss = self.reader(
            input_ids=tokens["input_ids"].cuda().view(tokens["input_ids"].size(0), -1),
            attention_mask=tokens["attention_mask"].cuda().view(tokens["attention_mask"].size(0), -1),
            decoder_input_ids=decoder_input_ids.cuda(),
            labels=labels.cuda(),
            use_cache=False,
        )
        return reader_loss[0].cpu().item(), reader_loss[1]


    @torch.no_grad()
    def compute_multimodal_reader_loss_and_logits(self, tokens, decoder_input_ids, labels):
        cfg = self.reader.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        reader_loss = self.reader(
            input_ids=tokens["input_ids"].cuda().view(tokens["input_ids"].size(0), -1),
            attention_mask=tokens["attention_mask"].cuda().view(tokens["attention_mask"].size(0), -1),
            decoder_input_ids=decoder_input_ids.cuda(),
            labels=labels.cuda(),
            use_cache=False,
        )
        return reader_loss[0].cpu().item(), reader_loss[1]

    @torch.no_grad()
    def generate(self, tokens, query, choices=None):
        cfg = self.reader.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        tokens = {k: v.view(v.size(0), -1) for k, v in tokens.items()}

        bos_token_id = None

        prefix_allowed_tokens_fn = None
        if self.opt.decoder_prompt_format is not None:
            prefix_str = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
            prefix_allowed_tokens_fn = self.get_prefix_allowed_tokens_fn(prefix_str)

        outputs = self.reader.generate(
            input_ids=tokens["input_ids"].cuda(),
            attention_mask=tokens["attention_mask"].cuda(),
            num_return_sequences=1,
            max_length=self.opt.generation_max_length,
            min_length=self.opt.generation_min_length,
            num_beams=self.opt.generation_num_beams,
            length_penalty=self.opt.generation_length_penalty,
            forced_bos_token_id=bos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        return outputs

    def generate_flamingo(self, tokens_text, token_img, specific_tokens = None, labels= None,bsz = None, n_context = None,task = None, targets = None,  choices=None):
        weights = None
        self.reader = self.reader.eval()
        # print(token_img.shape, "token_img.shape")
        # print(tokens_text["input_ids"].shape, "tokens_text")
        with torch.no_grad():
            autocast = self.get_autocast(
                self.opt.precision, cache_enabled=False)

            if self.opt.precision == 'fp16' or self.opt.precision == 'bf16':
                # reader_tokens_img = reader_tokens_img.half()
                # print(f"load {self.opt.precision} ")
                cast_dtype = self.get_cast_dtype(self.opt.precision)
                token_img = token_img.to('cuda', dtype=cast_dtype, non_blocking=True)
            else:
                token_img = token_img.cuda()

            with autocast():
                output = self.reader.generate(
                    vision_x=token_img,
                    lang_x=tokens_text["input_ids"].cuda(),
                    attention_mask=tokens_text["attention_mask"].cuda(),
                    max_new_tokens=self.opt.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        if specific_tokens != None and labels != None and bsz != None and n_context != None:
            list_labels_value = getattr(task, 'label', None)
            pad_token = \
            self.reader_tokenizer.tokenizer(self.reader_tokenizer.tokenizer.pad_token, add_special_tokens=False)[
                "input_ids"][-1]
            if list_labels_value is not None:
                new_labels = torch.zeros((output['scores'][0].shape[0], 1))
                for index_output_to_label in range(output['scores'][0].shape[0]):
                    target_index = int(np.floor((index_output_to_label) / n_context))
                    new_labels[index_output_to_label, :] = list_labels_value.index(targets[target_index])

                logits_all = torch.zeros((output['scores'][0].shape[0], len(list_labels_value)))
                for num_label, label in enumerate(list_labels_value):
                    label_tokens = specific_tokens[num_label]
                    logits_label = None
                    counter_vals = 0
                    for num_token, token in enumerate(label_tokens):
                        if token == pad_token or num_token > (self.opt.max_new_tokens - 1):
                            continue
                        counter_vals = counter_vals + 1
                        logits_first_word = output['scores'][num_token]
                        value = logits_first_word[:, [token]]
                        if logits_label is None:
                            logits_label = value
                        else:
                            logits_label = logits_label + value
                    logits_label = logits_label / counter_vals
                    # print(logits_label.shape)
                    # print(logits_all.shape)
                    logits_all[:, num_label] = logits_label.squeeze()

                reader_logit = torch.softmax(logits_all, dim=-1)
                labels_spe = new_labels.clone().flatten().long()
            elif self.opt.use_targets:
                new_labels = torch.ones((output['scores'][0].shape[0], 1))
                labels_spe = new_labels.clone().flatten().long()
                logits_all = torch.zeros((output['scores'][0].shape[0], 2))
                for num_label in range(bsz * n_context):
                    index_token = int(np.floor((num_label) / n_context))
                    label_tokens = specific_tokens[index_token]
                    logits_label = None
                    logits_neg_label = None
                    counter_vals = 0
                    for num_token, token in enumerate(label_tokens):
                        if token == pad_token or num_token > (self.opt.max_new_tokens - 1):
                            continue
                        counter_vals = counter_vals + 1
                        logits_first_word = output['scores'][num_token]
                        value = logits_first_word[num_label, [token]].clone()
                        logits_first_word[num_label, [token]] = float('-inf')
                        # Calculate the max excluding the specified index
                        max_value_neg = torch.max(logits_first_word[num_label, :])
                        # Restore the original value to the tensor
                        logits_first_word[num_label, [token]] = value

                        if logits_label is None:
                            logits_label = value
                        else:
                            logits_label = logits_label + value

                        if logits_neg_label is None:
                            logits_neg_label = max_value_neg
                        else:
                            logits_neg_label = logits_label + max_value_neg

                    logits_label = logits_label / counter_vals
                    logits_neg_label = logits_neg_label / counter_vals
                    # print(logits_label.shape)
                    # print(logits_all.shape)
                    logits_all[num_label, 1] = logits_label.squeeze()
                    logits_all[num_label, 0] = logits_neg_label.squeeze()

                reader_logit = torch.softmax(logits_all, dim=-1)
            elif self.opt.only_prob_retrieve_loss:
                logits_all = torch.zeros((output['scores'][0].shape[0], 1))
                for num_label in range(bsz * n_context):
                    index_token = int(np.floor((num_label) / n_context))
                    label_tokens = specific_tokens[index_token]
                    counter_vals = 0
                    logits_label = None
                    for num_token, token in enumerate(label_tokens):
                        if token == pad_token or num_token > (self.opt.max_new_tokens - 1):
                            continue
                        counter_vals = counter_vals + 1
                        logits_first_word = output['scores'][num_token]
                        value = logits_first_word[num_label, [token]]
                        if logits_label is None:
                            logits_label = value
                        else:
                            logits_label = logits_label + value

                    logits_label = logits_label / counter_vals
                    logits_all[num_label, 0] = logits_label.squeeze()
            else:
                logits_first_word = output['scores'][0]
                logits_first_word = logits_first_word[:, specific_tokens]
                labels_spe = labels.clone()

                for i, token in enumerate(specific_tokens):
                    labels_spe[labels_spe == token] = i


                reader_logit = torch.softmax(logits_first_word, dim=-1)

            if self.opt.weighting_kl:
                _, predicted_classes = torch.max(reader_logit.view(bsz, n_context, -1), 2)
                uniformity_check = (predicted_classes == predicted_classes[:, 0:1]).all(dim=1)
                weights = torch.where(uniformity_check, torch.tensor(0.0), torch.tensor(1.0))

            if self.opt.only_prob_retrieve_loss:
                token_loss = logits_all.view(bsz, n_context, -1).cuda()
            else:
                token_loss = nn.functional.cross_entropy(
                    reader_logit.cuda(),
                    labels_spe.cuda(),
                    reduction="none",
                )
            gold_score = token_loss.view(bsz, n_context, -1)
            z = (labels.view(bsz, n_context, -1) > -1).sum(dim=-1)
            if not self.opt.only_prob_retrieve_loss:
                gold_score = -gold_score.sum(dim=-1)
            else:
                gold_score = gold_score.sum(dim=-1)
        else:
            gold_score = None

        return output['sequences'], output['scores'], gold_score, weights


    def get_prefix_allowed_tokens_fn(self, prefix_str: Optional[str] = None):
        if prefix_str:
            prefix_tokens_ids = self.reader_tokenizer.batch_encode_plus(prefix_str, add_special_tokens=False)[
                "input_ids"
            ]

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
                if input_ids.shape[-1] > len(prefix_tokens_ids[batch_id]):
                    return self.READER_ALL_TOKENS

                return prefix_tokens_ids[batch_id][input_ids.shape[-1] - 1]

        else:
            prefix_allowed_tokens_fn = None

        return prefix_allowed_tokens_fn

def select_crossattention_scores(scores, mode):
    if "eval" in mode:
        return scores[mode[len("eval") :]]
    elif "std" in mode:
        return scores[mode[len("std") :]]

def _to_cuda_float16(tok_dict):
    return {k: v.half().cuda() for k, v in tok_dict.items()}
def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}
