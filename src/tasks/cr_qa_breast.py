import random
from src.util import load_jsonl_file
from src.evaluation import exact_match_score, f1_score, normalize_answer
from src.options import Options
from collections import defaultdict
from src.tasks.base import BaseTask
import json
import numpy as np
import torch

from src.util import accuracy_score

class Task(BaseTask):
    metrics = ["exact_match", "f1", "eval_loss"]

    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()
        # passages_path = "/cs/labs/tomhope/nirm/atlas_data_dir/data/vqa_rad/dev._pass_pne_small.jsonl"
        # passages_path = "/cs/labs/tomhope/nirm/atlas_data_dir/data/vqa_rad/dev._pass.jsonl"
        # passages_path = "/cs/labs/tomhope/nirm/atlas_data_dir/data/vqa_rad/pass_overfit.jsonl"
        random.seed(opt.seed)
        self.question = '<image>'
        self.question_text = 'Does this breast ultrasound image show signs of cancer?'
        self.opt = opt
        passages_path = "/cs/labs/tomhope/nirm/atlas_data_dir/data/vqa_rad/dev_pass_pne_one_ex_pne.jsonl"
        self.targets = ["Yes", "No"]
        self.answers = ["Yes", "No"]
        self.add_retrieve_text = opt.add_retrieve_text_to_prompt
        self.qa_prompt_format_str = opt.qa_prompt_format
        self.closed_book_passage =[ [{'img_path': "", 'text': "", 'answers': ""}]]
        self.tokenizer = None
        self.instructions = ""
        if opt.use_file_passages and passages_path is not None:
            self.passages_to_use = load_jsonl_file(passages_path)
        else:
            self.passages_to_use = [{'img_path': "", 'text': "", 'answers': ""}]
        self.counter = 0

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


    def set_tokeinzer_create_labels(self, reader_tokenizer):
        self.tokenizer = reader_tokenizer
        # self.zero_unwanted_class, self.targets_labels = self.create_labels()

    def create_labels(self):
        labels = {}
        pad_token = self.tokenizer(self.tokenizer.pad_token, add_special_tokens=False)['input_ids'][0]
        tensor_zeros = torch.zeros((1, self.opt.max_new_tokens, self.tokenizer.vocab_size))
        for i in range(len(self.targets)):
            prompt_tokens = self.tokenizer(self.targets[i],
                                           max_length=self.opt.max_new_tokens,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt",
                                           add_special_tokens=False
                                           )["input_ids"][0]

            labels[self.targets[i]] = prompt_tokens
            labels[self.targets[i]][labels[self.targets[i]] == pad_token] = -100

            for j, token in enumerate(prompt_tokens):
                tensor_zeros[:, j, token] = 1

        return tensor_zeros, labels

    def cal_acc(self, pred_files, name_file = None):

        postive_class, negative_class  = 'Ye', "No"
        gt = []
        pred = []
        dict_query_id = {}
        for i, p_file in enumerate(pred_files):

            pred_model = p_file['generation']

            if negative_class in pred_model:
                pred.append(0)
                # pred.append(1)
            elif postive_class in pred_model:
                pred.append(1)
            else:
                pred.append(-1)

            gt_exp = p_file['answers']
            if negative_class in gt_exp:
                gt.append(0)
                # gt.append(1)
            elif postive_class in gt_exp:
                gt.append(1)
                # gt.append(0)
            else:
                gt.append(-1)

        return accuracy_score(np.array(gt), np.array(pred))

    def create_passage_qwen(self, passage, query):

        training_retriever_final_prompts = []
        final_prompts = []
        for batch_p_q in zip(passage, query):
            pti, q = batch_p_q[0], batch_p_q[1]
            qs_no_training_ret = self.instructions
            promp_closed_book = 'Does this breast ultrasound image show signs of cancer?'
            for i, ppp in enumerate(pti):
                # prompt_ret = f"You are a knowledgeable medical assistant. Based on the provided image and the most similar description from PubMed '{text}'. Please answer the question: Does this breast ultrasound image show signs of cancer?"
                prompt_ret = 'Does this breast ultrasound image show signs of cancer?'

                training_retriever_final_prompts.append(prompt_ret)

            final_prompts.append(promp_closed_book)

        return final_prompts, training_retriever_final_prompts

    def create_input_qwen_model(self, passage, query):

        final_prompts, training_retriever_final_prompt = self.create_passage_qwen(passage, query)

        return final_prompts, training_retriever_final_prompt


    def create_input_flamingo_model(self, passage, query, target):

        end_answer = "<|endofchunk|>"
        final_prompts, training_retriever_final_prompt = self.create_query_passages_flamingo(passage, query)
        if target != None:
            final_prompts = [final_prompts[i] + ' ' + ttt + end_answer for i, ttt in enumerate(target)]
            if len(training_retriever_final_prompt) > 0:
                training_retriever_final_prompt = [training_retriever_final_prompt[i] + ' ' + target[int(np.floor((i) / len(passage[0])))] + end_answer for i in range(len(target)*len(passage[0]))]


        return final_prompts, training_retriever_final_prompt



    def create_labels_to_flamingo(self, labels, tokenizer):
        answer_token_id = tokenizer("Answer:", add_special_tokens=False)["input_ids"][-1]
        end_token_id = tokenizer("</s>", add_special_tokens=False)["input_ids"][-1]

        l = torch.where(labels == answer_token_id)[1]
        if len(l) == 0:
            end_idx = torch.where(labels == end_token_id)[1][-1]
            start_answer = end_idx - 5
        else:
            answer_token_id = tokenizer("Answer:", add_special_tokens=False)["input_ids"][-1]
            last_answer_idx = torch.where(labels == answer_token_id)[1][-1]
            start_answer = last_answer_idx + 1

        # print(labels)
        # print(self.tokenizer.decode(labels[0]))
        end_idx = torch.where(labels == end_token_id)[1][-1]

        start_pad = end_idx - 1
        labels[0, :start_answer] = -100
        labels[0, start_pad:] = -100
        return labels

    def truncate_sentence(self, sentence, max_tokens = 50):
        """
        Truncate a sentence based on the maximum number of tokens.

        Args:
            sentence (str): The input sentence to process.
            max_tokens (int): The maximum number of tokens allowed.
            tokenizer_model (str): The tokenizer model to use (default is "bert-base-uncased").

        Returns:
            tuple: A tuple containing the number of tokens in the original sentence,
                   the truncated sentence, and the number of tokens in the truncated sentence.
        """
        # Load tokenizer
        # Tokenize the sentence
        tokens = self.tokenizer(sentence, add_special_tokens=False)['input_ids']
        num_tokens = len(tokens)

        # Truncate tokens if necessary
        truncated_tokens = tokens[:max_tokens]
        truncated_sentence = self.tokenizer.decode(truncated_tokens)

        return truncated_sentence
    def create_query_passages_flamingo(self, passage, query):

        end_answer = "<|endofchunk|>"
        final_prompts = []
        training_retriever_final_prompts = []
        for batch_p_q in zip(passage, query):
            pti, q = batch_p_q[0], batch_p_q[1]
            prompt = self.instructions
            for i, ppp in enumerate(pti):
                train_retriever_prompt = self.instructions
                if ppp["text"] != '' or ppp['img_path'] != '':
                    if self.add_retrieve_text and ppp["text"] != '':
                        # background_text = ''.join([char for char in ppp["text"] if not char.isupper()])
                        characters_to_remove = "_&!():"
                        background_text = ''.join([char for char in ppp["text"] if char not in characters_to_remove])
                        answer = "background:" + self.truncate_sentence(background_text.strip()) + '\n\n'
                    else:
                        answer = "background"


                    prompt += self.question + end_answer
                    train_retriever_prompt += self.question + answer + end_answer

                    training_retriever_final_prompts.append(train_retriever_prompt + q)

            final_prompts.append(prompt + q)

            # if self.counter == 0:
            #     print("final prompt: ", final_prompts[-1])
            # else:
            #     self.counter = self.counter + 1

        return final_prompts, training_retriever_final_prompts
        #

    @staticmethod
    def batch_iterator(data_iterator, batch_size, drop_last=False, shuffle=False):
        if shuffle:
            data_iterator = Task.shuffle_iterator(data_iterator)
        batch = defaultdict(lambda: [])
        batch["__size__"] = 0
        batch_counter = 0
        for example in data_iterator:
            for k, v in example.items():
                batch[k].append(v)
            batch["__size__"] += 1
            if batch["__size__"] == batch_size:
                batch_counter += 1
                yield batch
                batch = defaultdict(lambda: [])
                batch["__size__"] = 0
        if batch["__size__"] > 0 and not drop_last:
            yield batch

    @staticmethod
    def data_iterator(filenames, world_rank=-1, world_size=-1, repeat_if_less_than_world_size=False, *args, **kwargs):
        if isinstance(filenames, str):
            filenames = [filenames]

        def _iter():
            # iterate over files
            return (line for filename in filenames for line in open(filename, encoding="utf-8"))

        def _stop():
            # stop iterating over data when at least one example has been fed to each worker
            return (total_yielded >= world_size) if repeat_if_less_than_world_size else (total_yielded > 0)

        total_yielded = 0
        while not _stop():
            for line in _iter():
                total_yielded += 1
                if world_rank > -1 and total_yielded % world_size != world_rank:
                    continue
                example = json.loads(line)
                yield example

    @staticmethod
    def shuffle_iterator(dataset):
        ddd = list(dataset)

        yes_dicts = [d for d in ddd if "Yes" in d["target"]]
        no_dicts = [d for d in ddd if "No" in d["target"]]

        random.shuffle(yes_dicts)
        random.shuffle(no_dicts)

        min_len = min(len(yes_dicts), len(no_dicts))

        shuffle_dict = []
        for i in range(min_len):
            shuffle_dict.append(yes_dicts[i])
            shuffle_dict.append(no_dicts[i])

        for x in shuffle_dict:
            yield x
    def extract_model_answer_from_pred(self, pred):
        answer = pred[pred.rindex("Answer:") + 8:]
        print("answer:", answer)
        answer.replace("<|endofchunk|>", '')
        answer.replace("\n", '')
        answer.replace("10", '')
        answer.replace("1", '')
        answer.replace("<|endofchunk|>", '')
        print("answer after post processing:", answer)
        return answer


    def get_qa_prompt(self, question: str) -> str:
        preprocessed_question = f"<image>Question: {question.strip()} Answer:"
        return preprocessed_question

    def process(self, example, *args, **kwargs):
        instruct = self.instructions
        question_preprocessed = self.get_qa_prompt(self.question_text)
        if "target" in example:
            target = example["target"]
        elif "answers" in example:
            target = example["answers"]
        else:
            target = None

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = self.get_qa_prompt(example["image_path"])
        if target is not None:
            example["target"] = f"<extra_id_0> {target}"

        return {
            "query": example["image_path"],
            "instructions": instruct,
            "query_text": question_preprocessed,
            "target": target,
            "passages": self.passages_to_use,
            "answers": example["answers"],
            "metadata": example,
        }

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
