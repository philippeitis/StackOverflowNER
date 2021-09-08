# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

import logging
import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, DistributedSampler
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from create_freq_vector_info import read_file

logger = logging.getLogger(__name__)


class InputExample:
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, freq_ids, block_ids, label_ids, label_ids_ctc, label_ids_block):
        self.input_ids = input_ids
        self.input_freq_ids = freq_ids
        self.input_mask = input_mask
        self.block_ids = block_ids
        self.label_ids = label_ids
        self.label_ids_ctc = label_ids_ctc
        self.label_ids_block = label_ids_block


def read_examples_from_file(file_path, mode):
    guid_index = 1
    examples = []

    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.strip().split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[1:])
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_block_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_block_id=0,
        pad_token_label_id=-100,
        sequence_a_block_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_block_id` define the block id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    try:
        word_to_id = json.load(open("word_to_id.json", "r"))
    except FileNotFoundError:
        read_file(Path(__file__).parent / Path("Freq_Vector.txt"))
        word_to_id = json.load(open("word_to_id.json", "r"))
    word_id_pad = word_to_id["***PADDING***"]

    # print("*********************************")
    # print(label_map)
    # print("*********************************")

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        label_ids_ctc = []
        label_ids_block = []
        word_freq_ids = []
        for word, label in zip(example.words, example.labels):
            # print("*********************************")
            # print(word)
            # print(label)
            # print("*********************************")
            word_tokens = tokenizer.tokenize(word)

            if word not in word_to_id:
                word_id = word_to_id["UNK"]
            else:
                word_id = word_to_id[word]

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label[0]]] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_ids_ctc.extend([label_map[label[1]]] + [pad_token_label_id] * (len(word_tokens) - 1))
                label_ids_block.extend([label_map[label[2]]] + [pad_token_label_id] * (len(word_tokens) - 1))
                word_freq_ids.append(word_id)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()  # num_added_tokens()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            label_ids_ctc = label_ids_ctc[: (max_seq_length - special_tokens_count)]
            label_ids_block = label_ids_block[: (max_seq_length - special_tokens_count)]
            word_freq_ids = word_freq_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        word_freq_ids += [word_id_pad]
        label_ids += [pad_token_label_id]
        label_ids_ctc += [pad_token_label_id]
        label_ids_block += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            word_freq_ids += [word_id_pad]
            label_ids += [pad_token_label_id]
            label_ids_ctc += [pad_token_label_id]
            label_ids_block += [pad_token_label_id]
        block_ids = [sequence_a_block_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            word_freq_ids += [word_id_pad]
            label_ids += [pad_token_label_id]
            label_ids_ctc += [pad_token_label_id]
            label_ids_block += [pad_token_label_id]
            block_ids += [cls_token_block_id]
        else:
            tokens = [cls_token] + tokens
            word_freq_ids = [word_id_pad] + word_freq_ids
            label_ids = [pad_token_label_id] + label_ids
            label_ids_ctc = [pad_token_label_id] + label_ids_ctc
            label_ids_block = [pad_token_label_id] + label_ids_block
            block_ids = [cls_token_block_id] + block_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            word_freq_ids = ([word_id_pad] * padding_length) + word_freq_ids
            block_ids = ([pad_token_block_id] * padding_length) + block_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            label_ids_ctc = ([pad_token_label_id] * padding_length) + label_ids_ctc
            label_ids_block = ([pad_token_label_id] * padding_length) + label_ids_block
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            word_freq_ids += [word_id_pad] * padding_length
            block_ids += [pad_token_block_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            label_ids_ctc += [pad_token_label_id] * padding_length
            label_ids_block += [pad_token_label_id] * padding_length

        while len(word_freq_ids) != max_seq_length:
            word_freq_ids.append(word_id_pad)

        # print(len(word_freq_ids), len(input_ids))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(word_freq_ids) == max_seq_length
        assert len(block_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_ids_ctc) == max_seq_length
        assert len(label_ids_block) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("block_ids: %s", " ".join([str(x) for x in block_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        #     logger.info("label_ids_ctc: %s", " ".join([str(x) for x in label_ids_ctc]))
        #     logger.info("label_ids_block: %s", " ".join([str(x) for x in label_ids_block]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                freq_ids=word_freq_ids,
                block_ids=block_ids,
                label_ids=label_ids,
                label_ids_ctc=label_ids_ctc,
                label_ids_block=label_ids_block
            )
        )
    return features


def get_labels(path, *special_labels):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        labels.extend(special_labels)
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class FeatureConverter:
    FEATURE_ATTRS = tuple()

    def examples_to_dataset(self, examples):
        return self.features_to_dataset(self.examples_to_features(examples))

    def examples_to_features(self, examples):
        return convert_examples_to_features(
            examples,
            self.labels,
            self.args.max_seq_length,
            self.tokenizer,
            cls_token_at_end=self.args.model_type in ["xlnet"],
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_block_id=2 if self.args.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=self.args.model_type in ["roberta"],
            # roberta uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=self.args.model_type in ["xlnet"],
            # pad on the left for xlnet
            pad_token=self.tokenizer.pad_token_id,
            pad_token_block_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )

    @classmethod
    def features_to_dataset(cls, features):
        return TensorDataset(
            *[torch.tensor([getattr(f, attr) for f in features], dtype=torch.long) for attr in cls.FEATURE_ATTRS]
        )

    def load_and_cache_examples(self, path, mode=""):
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        name = Path(self.args.model_name_or_path).stem
        cache_file = Path(self.args.data_dir) / Path(f"cached_{mode}_{name}_{self.args.max_seq_length}")
        print(cache_file)
        if cache_file.exists() and not self.args.overwrite_cache:
            # logger.info("Loading features from cached file %s", cached_features_file)

            features = torch.load(cache_file)
        else:
            # logger.info("Creating features from dataset file at %s", args.data_dir)
            path = path or Path(self.args.data_dir) / Path(f"{mode}.txt")
            examples = read_examples_from_file(path, mode)
            features = self.examples_to_features(examples)
            # if args.local_rank in [-1, 0]:
            #     logger.info("Saving features into cached file %s", cached_features_file)
            #     torch.save(features, cached_features_file)

        if self.args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        return self.features_to_dataset(features)

    def batch_to_input(self, batch):
        raise NotImplemented()

    def predict(self, eval_dataset):
        # Note that DistributedSampler samples randomly
        if self.args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
        else:
            eval_sampler = DistributedSampler(eval_dataset)

        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        # logger.info("***** Running evaluation %s *****", prefix)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = self.batch_to_input(batch)
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(self.label_map[out_label_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])

        results = {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        return results, preds_list
