import json

import gin
import numpy as np
import torch
from jinja2 import Template
from promptsource.templates import DatasetTemplates
import copy
from src.data.dataset import Dataset

@gin.configurable
class GSM8KDataset(Dataset):
    def __init__(
        self,
        include_templates="original",
        ignore_templates=[],
        max_pretemplate_examples_per_dataset=None,
        round_robin_template=False,
        **kwargs,
    ):
        """
        include_templates: list, str
            list: list of the template names to use.
            str: "original" to use templates for original task, or "all" to use all templates.
            when using str, you can also specify ignore_templates.
        ignore_templates: list
            list of indices of the templates to ignore.
        max_pretemplate_examples_per_dataset: int
            Maximum number of examples to use from the dataset, before applying tempaltes. Useful for few-shot learning.
        """
        self.include_templates = include_templates
        self.ignore_templates = ignore_templates
        self.max_pretemplate_examples_per_dataset = max_pretemplate_examples_per_dataset
        self.round_robin_template = round_robin_template
        super().__init__(**kwargs)

    def load_data(self):
        super().load_data()
        # self._templates = self._get_templates(
        #     DatasetTemplates(*self.dataset_path[1:]),
        #     self.include_templates,
        #     self.ignore_templates,
        # )

    def process_data(self):
        self._examples = [example for example in self._examples]

    def truncate_dataset(self):
        if self.max_examples_per_dataset is not None:
            if len(self._examples) > self.max_examples_per_dataset:
                all_example_template_idx_tuples = list(
                    range(len(self._examples))
                )
                self._example_template_idx_tuples = self._rng.choice(
                    all_example_template_idx_tuples,
                    self.max_examples_per_dataset,
                    replace=False,
                ).tolist()
            else:
                self._example_template_idx_tuples = list(
                    range(len(self._examples))
                )

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        example_idx = idx
        example = self._examples[example_idx]
        input_str, target_str = example['question'], example['answer']
        input_str = FEW_SHOT + "Question: " + input_str + "\nAnswer: "
        gt_answer = extract_answer(target_str)
        input_ids = self.tokenize(input_str)
        target_ids = self.tokenize(target_str)
        
        tokenized_example = {
            "example_idx": example_idx,
            "input_str": input_str,
            "target_str": target_str,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "references": gt_answer,
        }
        tokenized_example = {
            k: v for k, v in tokenized_example.items() if v is not None
        }
        # add additional keys to tokenized_example
        tokenized_example.update(super().__getitem__(idx))
        tokenized_example.update({f"_{key}": value for key, value in example.items()})
        return tokenized_example
    
FEW_SHOT= "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72\n\nQuestion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10\n\nQuestion: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n#### 5\n\nQuestion: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nAnswer: Maila read 12 x 2 = 24 pages today.\nSo she was able to read a total of 12 + 24 = 36 pages since yesterday.\nThere are 120 - 36 = 84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.\n#### 42\n\nQuestion: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?\nAnswer: He writes each friend 3*2=6 pages a week\nSo he writes 6*2=12 pages every week\nThat means he writes 12*52=624 pages a year\n#### 624\n\nQuestion: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?\nAnswer: There are 80/100 * 10 = 8 more purple flowers than yellow flowers.\nSo in Mark's garden, there are 10 + 8 = 18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = 28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = 7 green flowers.\nSo in total Mark has 28 + 7 = 35 plants in his garden.\n#### 35\n\nQuestion: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?\nAnswer: He eats 32 from the largest pizzas because 2 x 16 = 32\nHe eats 16 from the small pizza because 2 x 8 = 16\nHe eats 48 pieces because 32 + 16 = 48\n#### 48\n\nQuestion: Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?\nAnswer: To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=6 pounds.\nNext, he added another 2 pounds of jelly beans, bringing the weight to 6+2=8 pounds.\nAnd finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=16 pounds.\n#### 16\n\n"

import json
import os
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)