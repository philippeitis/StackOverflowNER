"""
Entry point for NER classification of text.
"""
import argparse
import io
import sys
from pathlib import Path
from typing import List, TextIO

from utils_seg import InputExample
from utils_ctc.prediction_ctc import prediction_on_token_input, CTCModel
from utils_ctc.config_ctc import parameters_ctc

from utils_preprocess.anntoconll import conll_from_sentences
from utils_preprocess.format_markdown import tokenize_and_annotate_post_body

from segmenter import Segmenter
from ner import NER

import click

TEMP_DIR = Path("temp_files")
SEGMENTER_INPUT_FILE = TEMP_DIR / Path("segmenter_ip.txt")
SEGMENTER_OUTPUT_FILE = TEMP_DIR / Path("segmenter_preds.txt")
NER_INPUT_FILE = TEMP_DIR / Path("ner_ip.txt")
NER_OUTPUT_FILE = Path("ner_preds.txt")


def conll_from_input(input_file, output_folder):
    post_id = 0

    for line in open(input_file):
        if not line.strip():
            continue

        padded = str(post_id).zfill(6)
        conll_file = output_folder / Path(f"{padded}_conll.txt")
        post_id += 1

        # if "--INLINE_CODE_BEGIN---" in line:
        # 	print(post_id)

        text, annotations = tokenize_and_annotate_post_body(line, padded)
        # To write the standoffs:
        #         text_file.write_text(text)
        #         ann_file.write_text("\n".join(str(annotation) for annotation in annotations))
        try:
            lines = conll_from_sentences([text], annotations)
        except Exception as e:
            print(e)
            continue

        conll_file.write_text(lines)


def segments_from_input(input_file, ctc_model):
    post_id = 0

    segments = SegmenterInput.default()
    for line in open(input_file):
        if not line.strip():
            continue

        post_id += 1

        # if "--INLINE_CODE_BEGIN---" in line:
        # 	print(post_id)

        segment = segment_from_sentence(line, str(post_id).zfill(6), ctc_model)
        if segment:
            segments += segment
    return segments


def segment_from_sentence(sentence, post_id, ctc_model):
    text, annotations = tokenize_and_annotate_post_body(sentence, str(post_id).zfill(6))
    try:
        lines = conll_from_sentences([text], annotations)
        return SegmenterInput.from_conll(lines, ctc_model)
    except Exception as e:
        print(e)


# def read_all_conll_files(conll_folder: Path, ctc_model):
#     def is_conll_file(path):
#         return path.is_file() and path.suffix == ".txt"
#
#     text_files = sorted([f for f in conll_folder.iterdir() if is_conll_file(f)])
#     return [SegmenterInput.from_path(path, ctc_model) for path in sorted(text_files)]


class SegmentItem:
    def __init__(self, word, md, ctc):
        self.md = md
        self.ctc = ctc
        self.word = word
        self.labels = ("O", f"CTC_PRED:{self.ctc}", f"md_label:{self.md}")

    def __str__(self):
        return f"{self.word}\tO\tCTC_PRED:{self.ctc}\tmd_label:{self.md}"


class SegmenterInput:
    def __init__(self, segments: List[List[SegmentItem]]):
        self.segments = segments

    @classmethod
    def default(cls):
        return cls([])

    @classmethod
    def from_path(cls, path: Path, ctc_model):
        segment_items = []
        for line in path.read_text().splitlines():
            segment_items.append(line.strip().split())
        return cls.from_conll(segment_items, ctc_model)

    @classmethod
    def from_conll(cls, lines: List, ctc_model):
        segment_items = [[]]
        for line in lines:
            if not line:
                segment_items.append([])
                continue

            word, md = line
            if not word or not md:
                print(line)
                continue

            ctc = prediction_on_token_input(word, ctc_model)

            if md != "O":
                md = "Name"
            segment_items[-1].append(SegmentItem(word, md, ctc))
        if not segment_items[-1]:
            segment_items.pop()
        return cls(segment_items)

    def to_examples(self, mode=""):
        guid_index = 1
        examples = []
        for segments in self.segments:
            words = []
            labels = []
            for segment in segments:
                if segment.word.startswith("-DOCSTART-"):
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    words.append(segment.word)
                    labels.append(segment.labels)
            if words:
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
        return examples

    def write_to_file(self, output: Path):
        output.write_text(
            "\n".join(
                "".join(f"{si}\n" for si in segment_items)
                for segment_items in self.segments
            )
        )

    def __add__(self, other):
        return SegmenterInput(self.segments + other.segments)

    def to_ner_input(self, segment_preds):
        items = []
        for segment_list, pred_list in zip(self.segments, segment_preds):
            ner_items = []
            for segment, pred in zip(segment_list, pred_list):
                if pred != "O":
                    pred = "Name"

                ner_items.append(NERItem(segment.word, pred, segment.ctc))
            items.append(ner_items)
        return NERInput(items)


class NERItem:
    def __init__(self, word, seg, ctc):
        self.seg = seg
        self.ctc = ctc
        self.word = word
        self.labels = ("O", f"CTC_PRED:{self.ctc}", f"pred_seg_label:{self.seg}")

    def __repr__(self):
        return f"NERItem({self.word} {self.seg} {self.ctc})"

    def __str__(self):
        return f"{self.word}\tO\tCTC_PRED:{self.ctc}\tpred_seg_label:{self.seg}"


class NERInput:
    def __init__(self, items: List[List[NERItem]]):
        self.items = items

    def write_to_file(self, output: Path):
        output.write_text(
            "\n".join(
                "".join(f"{si}\n" for si in segment_items)
                for segment_items in self.items
            )
        )

    def to_examples(self, mode=""):
        guid_index = 1
        examples = []
        for sentence in self.items:
            words = []
            labels = []
            for token in sentence:
                if token.word.startswith("-DOCSTART-"):
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    words.append(token.word)
                    labels.append(token.labels)
            if words:
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
        return examples

    def write_predictions(self, preds, writer: TextIO):
        is_first = True
        for sentence, pred_list in zip(self.items, preds):
            if not is_first:
                writer.write("\n")
            for token, pred in zip(sentence, pred_list):
                writer.write(f"{token.word} {pred}\n")
            is_first = False


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_file_with_so_body",
        default='xml_filted_body.txt',
        type=str,
    )

    return parser.parse_args()


@click.group()
def cli():
    pass


def ner_from_segments(segments, output):
    # conll_from_input(input_file, conll_folder)
    # si = sum(read_all_conll_files(conll_folder, ctc_model), SegmenterInput.default())
    segmenter = Segmenter()
    ner = NER()

    si = segments

    # Segmentation
    results, preds = segmenter.predict(
        segmenter.examples_to_dataset(si.to_examples())
    )

    ni = si.to_ner_input(preds)
    results, preds = ner.evaluate(
        ner.examples_to_dataset(ni.to_examples())
    )

    if output:
        with open(output, "w") as s:
            ni.write_predictions(preds, s)
        print("=" * 80)
        print(f"\tPredictions written to {output}")
        print("=" * 80)
    else:
        s = io.StringIO()
        ni.write_predictions(preds, s)
        print(s.getvalue())
        s.close()


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("path", type=click.Path(exists=True, dir_okay=False), default=Path("xml_filted_body.txt"))
@click.option("--output", "-o", "output", type=click.Path(), default=None)
@click.argument('base_args', nargs=-1, type=click.UNPROCESSED)
def ner_from_path(path: click.Path, output, base_args):
    sys.argv = [sys.argv[0], *base_args]

    train_file = parameters_ctc['train_file']
    test_file = parameters_ctc['test_file']

    ctc_model = CTCModel(train_file, test_file)
    # conll_from_input(input_file, conll_folder)
    # si = sum(read_all_conll_files(conll_folder, ctc_model), SegmenterInput.default())
    ner_from_segments(segments_from_input(path, ctc_model), output)


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument("sentence")
@click.option("--output", "-o", "output", type=click.Path(), default=None)
@click.argument('base_args', nargs=-1, type=click.UNPROCESSED)
def ner_from_str(sentence, output, base_args):
    sys.argv = [sys.argv[0], *base_args]

    train_file = parameters_ctc['train_file']
    test_file = parameters_ctc['test_file']

    ctc_model = CTCModel(train_file, test_file)

    # Segmentation
    ner_from_segments(segment_from_sentence(sentence, "000000", ctc_model), output)


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument('base_args', nargs=-1, type=click.UNPROCESSED)
def ner_repl(base_args):
    sys.argv = [sys.argv[0], *base_args]

    train_file = parameters_ctc['train_file']
    test_file = parameters_ctc['test_file']

    ctc_model = CTCModel(train_file, test_file)

    segmenter = Segmenter()
    ner = NER()

    while True:
        sentence = input("Sentence to annotate: ")

        # Segmentation
        si = segment_from_sentence(sentence, "000000", ctc_model)
        results, preds = segmenter.predict(
            segmenter.examples_to_dataset(si.to_examples())
        )

        ni = si.to_ner_input(preds)
        results, preds = ner.evaluate(
            ner.examples_to_dataset(ni.to_examples())
        )

        s = io.StringIO()
        ni.write_predictions(preds, s)
        print(s.getvalue())
        s.close()


if __name__ == '__main__':
    cli()
