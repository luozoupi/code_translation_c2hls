from __future__ import annotations

import torch

from scripts.train_small_lm_setup_router import (
    CompletionCollator,
    CompletionDataset,
)


class _Tokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del text, kwargs
        return {"input_ids": [3, 4, 5]}

    def encode(self, text, **kwargs):
        del kwargs
        return {"A": [6], "B": [7]}[text]


def test_completion_dataset_supervises_only_the_router_decision() -> None:
    dataset = CompletionDataset(
        [
            {
                "record_id": "example",
                "prompt": "choose",
                "completion": "B",
            }
        ],
        _Tokenizer(),
        max_length=16,
    )
    example = dataset[0]
    assert example["input_ids"] == [3, 4, 5, 7]
    assert example["labels"] == [-100, -100, -100, 7]


def test_completion_collator_masks_padding() -> None:
    collator = CompletionCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": [3, 6],
                "labels": [-100, 6],
            },
            {
                "input_ids": [3, 4, 7],
                "labels": [-100, -100, 7],
            },
        ]
    )
    assert torch.equal(
        batch["input_ids"],
        torch.tensor([[3, 6, 0], [3, 4, 7]]),
    )
    assert torch.equal(
        batch["attention_mask"],
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )
    assert batch["labels"][0, 2].item() == -100
