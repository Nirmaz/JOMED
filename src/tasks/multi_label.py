from src.util import load_jsonl_file
from src.evaluation import exact_match_score, f1_score, normalize_answer
from src.options import Options
from src.tasks.base import BaseTask
import numpy as np
import torch
import re

class Task(BaseTask):
    metrics = ["exact_match", "f1", "eval_loss"]


    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()

        self.question = '<image>'

        self.opt = opt
        passages_path = ""
        self.add_retrieve_text = opt.add_retrieve_text_to_prompt
        self.qa_prompt_format_str = opt.qa_prompt_format
        self.label = ['0', "1", '2', "3", '4', "5", '6', "7", '8', "9", '10', "11", '12', "13", "14"]
        self.targets = ['0', "1", '2', "3", '4', "5", '6', "7", '8', "9", '10', "11", '12', "13", "14"]
        self.answers = ['0', "1", '2', "3", '4', "5", '6', "7", '8', "9", '10', "11", '12', "13", "14"]
        self.closed_book_passage =[ [{'img_path': "", 'text': "", 'answers': ""}]]
        self.instructions = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. "
        self.instructions = ""
        # self.instructions = ""
        if opt.use_file_passages and passages_path is not None:
            self.passages_to_use = load_jsonl_file(passages_path)
        else:
            self.passages_to_use = [{'img_path': "", 'text': "", 'answers': ""}]
        self.counter = 0

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_tokeinzer_create_labels(self, reader_tokenizer):
        self.tokenizer = reader_tokenizer

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

        end_idx = torch.where(labels == end_token_id)[1][-1]

        start_pad = end_idx - 1
        labels[0, :start_answer] = -100
        labels[0, start_pad:] = -100
        return labels

    def calculate_f1(self, X, y):
        # Tokenize the strings
        pred_tokens = set(X.lower().split())
        true_tokens = set(y.lower().split())

        # Calculate the precision and recall
        common_tokens = pred_tokens & true_tokens
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        recall = len(common_tokens) / len(true_tokens) if true_tokens else 0

        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def extract_numbers(self,text):
        """
        Extract all numbers from a string and return them as a list of integers.

        Args:
            text (str): Input string containing letters and numbers

        Returns:
            list: List of integers found in the string
        """
        # Find all sequences of digits in the string
        numbers = re.findall(r'\d+', text)

        # Convert string numbers to integers
        return [int(num) for num in numbers]

    def accuracy_score2(self, y_true, y_pred, normalize=True, sample_weight=None):
        """
        Replicates sklearn's accuracy_score for multi-label classification.

        For multi-label case, sklearn uses EXACT MATCH (subset accuracy):
        - A sample is considered correct ONLY if ALL labels match exactly
        - This is the most strict interpretation of accuracy

        Args:
            y_true: True binary labels (n_samples, n_labels)
            y_pred: Predicted binary labels (n_samples, n_labels)
            normalize: If True, return fraction. If False, return count
            sample_weight: Sample weights (not commonly used)

        Returns:
            float or int: Accuracy score
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Check if all labels match for each sample (exact match)
        exact_matches = np.all(y_true == y_pred, axis=1)

        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            correct_samples = np.sum(exact_matches * sample_weight)
            total_weight = np.sum(sample_weight)
            if normalize:
                return correct_samples / total_weight
            else:
                return correct_samples
        else:
            correct_samples = np.sum(exact_matches)
            total_samples = len(y_true)
            if normalize:
                return correct_samples / total_samples
            else:
                return correct_samples

    def cal_acc_yes_no(self, pred_files, name_file=None):

        postive_class, negative_class = 'yes', "no"
        gt = []
        pred = []
        dict_query_id = {}
        try:
            for i, p_file in enumerate(pred_files):

                pred_model = p_file['generation']

                class_indices = []
                pred_answer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                gt_answer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                pred_model_eval = self.extract_numbers(pred_model)
                for x in pred_model_eval:
                    for ccc in range(len(pred_answer)):
                        if int(ccc) == x:
                            pred_answer[int(ccc)] = 1


                gt_exp = p_file['answers']

                for x in gt_exp.split(','):
                    x = x.strip()
                    if x:
                        try: # Skip empty strings
                            idx = int(x)
                        except:
                            continue
                        if 0 <= idx < len(gt_answer):
                            gt_answer[idx] = 1

                gt.append(gt_answer)
                pred.append(pred_answer)

            return self.accuracy_score2(np.array(gt), np.array(pred))

        except:
            return 0

    def cal_acc(self, pred_files, name_file = None):


        answers = []
        questions = []
        generations = []
        inferece_gpt = False
        type_set = pred_files[0]['metadata']['image_path'].split('/')[-2]
        # if (type_set == "images_test" and len(pred_files) < 500) and ("debug" in name_file) and ((self.opt.train_retriever_img and "img" in name_file) or (self.opt.train_retriever_text and "text" in name_file)):
        if (type_set == "images_test" and len(pred_files) < 500) and ("debug" in name_file) and ("fused" in name_file) and ("open" in name_file) and ("text" not in name_file) and ("img" not in name_file):
            return 0.0
            # for i, p_file in enumerate(pred_files):
            #     generations.append(p_file['generation'].replace(",", "").replace(".", ""))
            #     answers.append(p_file["answers"])
            #     questions.append(p_file["query"])
            #
            # total_count, count_Open_1, count_Close_1, count_Open_0, count_Close_0 = run_gpt_eval(questions,generations,answers, self.opt)
            # total_acc = (count_Open_1 + count_Close_1) / total_count
            # closed_acc = (count_Close_1) / (count_Close_0 + count_Close_1 + 0.003)
            # open_acc = (count_Open_1) / (count_Open_0 + count_Open_1 + 0.003)
            #
            # print("total acc: ", total_acc)
            # print("closed acc: ", closed_acc)
            # print("open acc: ", open_acc)
            # return total_acc
        else:
            return self.cal_acc_yes_no(pred_files)






    def create_passage_qwen(self, passage, query):

        training_retriever_final_prompts = []
        final_prompts = []
        for batch_p_q in zip(passage, query):
            pti, q = batch_p_q[0], batch_p_q[1]

            qs_no_training_ret = self.instructions
            promp_closed_book = q
            for i, ppp in enumerate(pti):
                # prompt_ret = f"You are a knowledgeable medical assistant. Based on the provided image and the most similar description from PubMed '{text}'. Please answer the question: Does this breast ultrasound image show signs of cancer?"
                prompt_ret = q

                training_retriever_final_prompts.append(prompt_ret)

            final_prompts.append(promp_closed_book)

        return final_prompts, training_retriever_final_prompts

    def create_input_qwen_model(self, passage, query):

        final_prompts, training_retriever_final_prompt = self.create_passage_qwen(passage, query)

        return final_prompts, training_retriever_final_prompt

    def truncate_sentence(self, sentence, max_tokens=50):
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
                    if self.add_retrieve_text:
                        # background_text = ''.join([char for char in ppp["text"] if not char.isupper()])
                        characters_to_remove = "_&!("
                        background_text = ''.join([char for char in ppp["text"] if char not in characters_to_remove])
                        answer = "background:" + self.truncate_sentence(background_text.strip()) + '\n\n'
                    else:
                        answer = "background"

                    prompt += self.question + "background" + end_answer
                    train_retriever_prompt += self.question + answer + end_answer
                    training_retriever_final_prompts.append(train_retriever_prompt + q)

            final_prompts.append(prompt + q)

            # if self.counter == 0:
            #     print("final prompt: ", final_prompts[-1])
            # else:
            #     self.counter = self.counter + 1

        return final_prompts, training_retriever_final_prompts

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
        if self.opt.qwen_model:
            return question
        preprocessed_question = f"<image>Question: {question.strip()} Answer:"
        return preprocessed_question

    def process(self, example, *args, **kwargs):
        instruct = self.instructions
        question_preprocessed = self.get_qa_prompt(example["question"])
        if "target" in example:
            target = example["target"]
        elif "answer" in example:
            target = example["answer"]
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
            "answers": example["answer"],
            "metadata": example,
        }

    def evaluation(self, prediction, ground_truths):
        # print(ground_truths, "ground truth")
        # print(prediction, "ground truth")
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
