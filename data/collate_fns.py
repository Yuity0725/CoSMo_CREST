from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class PaddingCollateFunction(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch: List[tuple]):
        reference_images, target_images, modifiers, lengths, bert = zip(*batch)
        bert_feature = pad_sequence(bert, batch_first=True)
        reference_images = torch.stack(reference_images, dim=0)
        target_images = torch.stack(target_images, dim=0)
        seq_lengths = torch.tensor(lengths).long()
        modifiers = pad_sequence(modifiers, padding_value=self.padding_idx, batch_first=True)
        return reference_images, target_images, modifiers, seq_lengths, bert_feature


class PaddingCollateFunctionTest(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    @staticmethod
    def _collate_test_dataset(batch):
        reference_images, ids = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        return reference_images, ids

    def _collate_test_query_dataset(self, batch):
        reference_images, ref_attrs, modifiers, target_attrs, lengths, bert = zip(*batch)
        reference_images = torch.stack(reference_images, dim=0)
        bert_feature = pad_sequence(bert, batch_first=True)
        seq_lengths = torch.tensor(lengths).long()
        modifiers = pad_sequence(modifiers, padding_value=self.padding_idx, batch_first=True)
        return reference_images, ref_attrs, modifiers, target_attrs, seq_lengths, bert_feature

    def __call__(self, batch: List[tuple]):
        num_items = len(batch[0])
        if num_items > 2:
            return self._collate_test_query_dataset(batch)
        else:
            return self._collate_test_dataset(batch)