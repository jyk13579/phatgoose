import json

import gin
import numpy as np
import torch
from jinja2 import Template
from promptsource.templates import DatasetTemplates
import copy
from src.data.dataset import Dataset

@gin.configurable
class LlemmaDataset(Dataset):
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
        # import pdb; pdb.set_trace()
        example_idx = idx
        example = self._examples[example_idx]
        input_str, target_str = example['text'], example['text']
        
        input_ids = self.tokenize(input_str)
        target_ids = self.tokenize(target_str)
        
        tokenized_example = {
            "example_idx": example_idx,
            "input_str": input_str,
            "target_str": target_str,
            "input_ids": input_ids,
            "target_ids": target_ids
        }
        tokenized_example = {
            k: v for k, v in tokenized_example.items() if v is not None
        }
        # add additional keys to tokenized_example
        tokenized_example.update(super().__getitem__(idx))
        tokenized_example.update({f"_{key}": value for key, value in example.items()})
        return tokenized_example




if __name__ == "__main__":
    gin_config = """
    D/P3SOCIALIQA/P3SocialIQADataset:
        dataset_batch_size = 8
        dataset_path = ["huggingface", "social_i_qa"]

    D/P3SOCIALIQA/TRAIN/build.cls = @P3SocialIQADataset
    D/P3SOCIALIQA/TRAIN/P3SocialIQADataset:
        dataset_split = "train"
        max_examples_per_dataset = 500_000
        max_seq_len = 512
        template_selection = "all"

    D/P3SOCIALIQA/EVAL/build.cls = @P3SocialIQADataset
    D/P3SOCIALIQA/EVAL/P3SocialIQADataset:
        dataset_split = "validation"
        max_examples_per_dataset = 10
        metrics = ["accuracy"]
        quick_evaluation = True
        template_names_to_ignore = ["Check if a random answer is valid or not"]
        template_selection = "original"
        info = {"inference_mode": "multiple_choice"}
    """
    gin.parse_config(gin_config)
    dataset_name = "D/P3SOCIALIQA/TRAIN"
    with gin.config_scope(dataset_name):
        dataset = build(scope_name=dataset_name)
    tokenizer = AutoTokenizer.from_pretrained("google/t5-large-lm-adapt")
    dataset.set_tokenizer(tokenizer)
    import ipdb

    ipdb.set_trace()
