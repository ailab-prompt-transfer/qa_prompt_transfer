import torch
import torch.nn as nn

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoConfig
from .modeling_t5 import T5ForConditionalGeneration

from transformers import T5Tokenizer

import collections

import re
import string


class PromptT5(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptT5, self).__init__()

        try:
            if config.get("model", "model_size") == "small":
                model = "t5-small"
                ckp = "T5SmallForMaskedLM"
                self.hidden_size = 512
            elif config.get("model", "model_size") == "large":
                model = "t5-large"
                ckp = "T5LargeForMaskedLM"
                self.hidden_size = 1024
            elif config.get("model", "model_size") == "b3":
                model = "t5-b3"
                ckp = "T5B3ForMaskedLM"
                self.hidden_size = 1024
            elif config.get("model", "model_size") == "small_v1_1":
                # https://huggingface.co/google/t5-v1_1-small
                model = "google/t5-v1_1-small"
                ckp = "T5Smallv1_1ForMaskedLM"
                self.hidden_size = 512
            else:
                model = "t5-base"
                ckp = "T5ForMaskedLM"
                self.hidden_size = 768
        except:
            model = "t5-base"
            ckp = "T5ForMaskedLM"
            self.hidden_size = 768

        self.model = model

        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.plmconfig = AutoConfig.from_pretrained(model)
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")

        # debug
        # print(f'model : {model}')
        # print(f'ckp: {ckp}')
        # print(f'hidden_size: {self.hidden_size}')
        # print(f'plmconfig: {self.plmconfig}')

        self.init_model_path = str(ckp) + "/" + "PromptT5_init_params"

        ##############
        ###Save a PLM + add prompt -->save --> load again
        # Build model and save it
        if os.path.exists(self.init_model_path + "/pytorch_model.bin"):
            print(f'Is exists? {os.path.exists(self.init_model_path+"/pytorch_model.bin")}')
            self.encoder = T5ForConditionalGeneration.from_pretrained(self.init_model_path, config=self.plmconfig)
            print(f"DONE !!!! ")
        else:
            print(f'Is exists? {os.path.exists(self.init_model_path+"/pytorch_model.bin")}')
            self.encoder = T5ForConditionalGeneration.from_pretrained(model, config=self.plmconfig)

            os.mkdir(self.init_model_path)
            torch.save(self.encoder.state_dict(), str(self.init_model_path) + "/pytorch_model.bin")
            print("Save Done")

            self.encoder = T5ForConditionalGeneration.from_pretrained(self.init_model_path, config=self.plmconfig)
        self.freeze_lm()

    def freeze_lm(self):
        for name, param in self.encoder.named_parameters():
            if not ("prompt_embedding" in name or "label_embedding" in name):
                param.requires_grad = False

    def init_prompt_emb(self, init_ids, **kwargs):
        self.encoder.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long).to(kwargs["gpu_list"][kwargs["local_rank"]]))

    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output=False, **kwargs):
        if mode == "train":
            if prompt_emb_output == True:
                # Wrong Code
                print("PromptT5.py line: 102 exit()")
                ####
            else:
                output = self.encoder(input_ids=data["inputx"], labels=data["label"])
                performance = kwargs["performance"]

                if int(kwargs["step"] % 1000) == 0:
                    # gen = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), no_repeat_ngram_size = config.getint("eval","no_repeat_ngram_size"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))
                    # https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
                    gen = self.encoder.generate(
                        input_ids=data["inputx"],
                        num_beams=config.getint("eval", "num_beams"),
                        output_scores=True,
                        return_dict_in_generate=True,
                        min_length=config.getint("eval", "min_length"),
                        max_length=config.getint("eval", "max_length"),
                        no_repeat_ngram_size=config.getint("eval", "no_repeat_ngram_size"),
                    )

                    # print('현재 prompt params')
                    # print(self.encoder.prompt_embeddings.weight.data)
                    performance = train_f1_em(gen["sequences"], data["text_label"], self.model)

                return {"loss": output["loss"], "performance": performance}
        # elif mode == 'valid':
        #     with torch.no_grad():
        #         output = self.encoder(input_ids=data['inputx'], labels=data['label'])
        #         return {'valid_loss': output['loss']}
        elif mode == "test" or mode == "valid":
            with torch.no_grad():
                # https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
                num_beams = config.getint("eval", "num_beams")
                output = self.encoder.generate(
                    input_ids=data["inputx"],
                    num_beams=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    min_length=config.getint("eval", "min_length"),
                    max_length=config.getint("eval", "max_length"),
                    no_repeat_ngram_size=config.getint("eval", "no_repeat_ngram_size"),
                )

                acc_result = calc_f1_em(output["sequences"], data["label"], acc_result, self.model)
            return {"acc_result": acc_result}


#########################################################################################################
#########################################################################################################
# reference : https://github.com/google-research/albert/blob/b772393d3dae115b493258ce8e37c17b2cc62100/squad_utils.py#L1129
# 동일한 코드른 다른 모델들도 사용하는 것 같았음


def metric_max_over_ground_truths(metric_fn, hyp, ref):
    """Computes the maximum of the metric over all ground truths."""
    if type(ref) == str:
        ref = [ref]
    return max(metric_fn(hyp, ground_truth) for ground_truth in ref)


def train_f1_em(summary, reference, model):
    tokenizer = T5Tokenizer.from_pretrained(model)
    em = 0
    cnt = 0
    f1 = 0

    for i in range(len(summary)):
        hyp = tokenizer.decode(summary[i], skip_special_tokens=True)
        ref = reference[i]

        # em 계산
        em += compute_em(hyp, ref)
        f1 += compute_f1(hyp, ref)
        cnt += 1

    result = f1 / cnt * 100  # performance : f1 score 
    return result


def calc_f1_em(summary, reference, acc_result, model):
    # print(summary)
    # print(reference)

    tokenizer = T5Tokenizer.from_pretrained(model)
    em = 0
    f1 = 0
    cnt = 0

    if acc_result is None:
        acc_result = {"em": 0, "f1": 0, "total_cnt": 0}

    for i in range(len(summary)):
        hyp = tokenizer.decode(summary[i], skip_special_tokens=True)
        ref = reference[i]

        # em 계산
        em += metric_max_over_ground_truths(compute_em, hyp, ref)  # compute_em(hyp, ref)
        f1 += metric_max_over_ground_truths(compute_f1, hyp, ref)  # compute_f1(hyp, ref)
        cnt += 1

        # # # ***
        # print(f"hyp: {hyp}")
        # print(f"ref: {ref}")
        # print(f"em: {(metric_max_over_ground_truths(compute_em, hyp, ref))}, f1: {(metric_max_over_ground_truths(compute_f1, hyp, ref))}")
        # print()


    acc_result["em"] += em
    acc_result["f1"] += f1
    acc_result["total_cnt"] += cnt

    return acc_result


def normalize_answer_v2(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer_v2(s).split()


def compute_f1(hyp, ref):
    hyp_tokens = get_tokens(hyp)
    ref_tokens = get_tokens(ref)
    common = collections.Counter(ref_tokens) & collections.Counter(hyp_tokens)
    num_same = sum(common.values())
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(ref_tokens == hyp_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(hyp_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_em(hyp, ref):
    return int(normalize_answer_v2(hyp) == normalize_answer_v2(ref))
