#!/usr/bin/env python

# Convert text and standoff annotations into CoNLL format.

import re

import sys
from pathlib import Path

# assume script in brat tools/ directory, extend path to find sentencesplit.py
sys.path.append(str(Path(__file__).parent))
sys.path.append('.')

from sentencesplit import sentencebreaks_to_newlines

EMPTY_LINE_RE = re.compile(r'^\s*$')
CONLL_LINE_RE = re.compile(r'^\S+\t\d+\t\d+.')

import stokenizer  # JT: Dec 6
from map_text_to_char import map_text_to_char  # JT: Dec 6

NO_SPLIT = True
SINGLE_CLASS = None
ANN_SUFFIX = ".ann"
OUT_SUFFIX = "conll"
VERBOSE = False


def argparser():
    import argparse

    ap = argparse.ArgumentParser(
        description='Convert text and standoff annotations into CoNLL format.'
    )
    ap.add_argument('-a', '--annsuffix', default=ANN_SUFFIX,
                    help='Standoff annotation file suffix (default "ann")')
    ap.add_argument('-c', '--singleclass', default=SINGLE_CLASS,
                    help='Use given single class for annotations')
    ap.add_argument('-n', '--nosplit', default=NO_SPLIT, action='store_true',
                    help='No sentence splitting')
    ap.add_argument('-o', '--outsuffix', default=OUT_SUFFIX,
                    help='Suffix to add to output files (default "conll")')
    ap.add_argument('-v', '--verbose', default=VERBOSE, action='store_true',
                    help='Verbose output')
    # ap.add_argument('text', metavar='TEXT', nargs='+',
    #                 help='Text files ("-" for STDIN)')
    return ap


def init_globals():
    global NO_SPLIT, SINGLE_CLASS, ANN_SUFFIX, OUT_SUFFIX, VERBOSE

    ap = argparser()
    args = ap.parse_args(sys.argv[1:])
    NO_SPLIT = args.nosplit
    SINGLE_CLASS = args.singleclass
    ANN_SUFFIX = args.annsuffix
    OUT_SUFFIX = args.outsuffix
    VERBOSE = args.verbose


def read_sentence(f):
    """Return lines for one sentence from the CoNLL-formatted file.

    Sentences are delimited by empty lines.
    """

    lines = []
    for l in f:
        lines.append(l)
        if EMPTY_LINE_RE.match(l):
            break
        if not CONLL_LINE_RE.search(l):
            raise ValueError(
                'Line not in CoNLL format: "%s"' %
                l.rstrip('\n'))
    return lines


def strip_labels(lines):
    """Given CoNLL-format lines, strip the label (first TAB-separated field)
    from each non-empty line.

    Return list of labels and list of lines without labels. Returned
    list of labels contains None for each empty line in the input.
    """

    labels, stripped = [], []

    labels = []
    for l in lines:
        if EMPTY_LINE_RE.match(l):
            labels.append(None)
            stripped.append(l)
        else:
            fields = l.split('\t')
            labels.append(fields[0])
            stripped.append('\t'.join(fields[1:]))

    return labels, stripped


def attach_labels(labels, lines):
    """Given a list of labels and CoNLL-format lines, affix TAB-separated label
    to each non-empty line.

    Returns list of lines with attached labels.
    """

    assert len(labels) == len(
        lines), "Number of labels (%d) does not match number of lines (%d)" % (len(labels), len(lines))

    attached = []
    for label, line in zip(labels, lines):
        empty = EMPTY_LINE_RE.match(line)
        assert (label is None and empty) or (label is not None and not empty)

        if empty:
            attached.append(line)
        else:
            attached.append('%s\t%s' % (label, line))

    return attached


def conll_from_path(path):
    """Convert plain text into CoNLL format."""
    lines = path.read_text().splitlines()
    if NO_SPLIT:
        sentences = lines
    else:
        sentences = []
        for line in lines:
            line = sentencebreaks_to_newlines(line)
            sentences.extend([s for s in NEWLINE_TERM_REGEX.split(line) if s])

    if ANN_SUFFIX:
        annotations = get_annotations(path)
    else:
        annotations = None

    return conll_from_sentences(sentences, annotations)


def conll_from_sentences(sentences, annotations=None):
    """Convert plain text into CoNLL format."""
    lines = []

    offset = 0
    # print(sentences)
    # JT: Feb 19: added it for resolving char encoding issues
    fixed_sentences = []
    for s in sentences:
        # print(s)
        # fixed_s = ftfy.fix_text(s)
        # # print(fixed_s)
        # fixed_sentences.append(fixed_s)
        fixed_sentences.append(s)

    # for s in sentences:
    for s in fixed_sentences:
        tokens = stokenizer.tokenize(s)
        # Possibly apply timeout?
        # try:
        #     tokens = stokenizer.tokenize(s)
        # except stokenizer.TimedOutExc as e:
        #     try:
        #         print("***********using ark tokenizer")
        #         tokens = ark_twokenize.tokenizeRawTweetText(s)
        #     except Exception as e:
        #         print(e)
        token_w_pos = map_text_to_char(s, tokens, offset)

        for t, pos in token_w_pos:
            if not t.isspace():
                lines.append(('O', pos, pos + len(t), t))

        lines.append(tuple())

        offset += len(s)

    # add labels (other than 'O') from standoff annotation if specified
    if annotations:
        lines = relabel(lines, annotations)

    # lines = [[l[0], str(l[1]), str(l[2]), l[3]] if l else l for l in lines] #JT: Dec 6
    return [(line[3], line[0]) if line else line for line in lines]


def relabel(lines, annotations):
    # TODO: this could be done more neatly/efficiently
    offset_label = {}

    for tb in annotations:
        for i in range(tb.start, tb.end):
            if i in offset_label:
                print("Warning: overlapping annotations in ", file=sys.stderr)
            offset_label[i] = tb

    prev_label = None
    for i, l in enumerate(lines):
        if not l:
            prev_label = None
            continue
        tag, start, end, token = l

        # TODO: warn for multiple, detailed info for non-initial
        label = None
        for o in range(start, end):
            if o in offset_label:
                if o != start:
                    print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                        token, offset_label[o].text), file=sys.stderr)
                label = offset_label[o].type
                break

        if label is not None:
            if label == prev_label:
                tag = 'I-' + label
            else:
                tag = 'B-' + label
        prev_label = label

        lines[i] = [tag, start, end, token]

    # optional single-classing
    if SINGLE_CLASS:
        for l in lines:
            if l and l[0] != 'O':
                l[0] = l[0][:2] + SINGLE_CLASS

    return lines


def process_files(files, output_directory, phase_name=""):
    suffix = OUT_SUFFIX.replace(".", "") + "_" + phase_name.replace("/", "")

    for path in files:
        try:
            lines = '\n'.join(
                '\t'.join(line) for line in
                conll_from_path(path)
            )

        except Exception as e:
            print(e)
            continue

        # TODO: better error handling
        if lines is None:
            print(f"file at {path} could not be tokenized")
            continue
        file_name = output_directory / Path(f"{path.stem}_{suffix}.txt")
        file_name.write_text(lines)


TEXTBOUND_LINE_RE = re.compile(r'^T\d+\t')


def parse_textbounds(f):
    """Parse textbound annotations in input, returning a list of Textbound."""
    from .format_markdown import Annotation

    textbounds = []

    for line in f:
        line = line.rstrip('\n')
        if not TEXTBOUND_LINE_RE.search(line):
            continue

        id_, type_offsets, text = line.split('\t')
        type_, start, end = type_offsets.split()
        start, end = int(start), int(end)
        textbounds.append(Annotation(None, type_, start, end, text))
    return textbounds


def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check
    for t1 in textbounds:
        for t2 in textbounds:
            if t1 is t2:
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # eliminate shorter
            if t1.end - t1.start > t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[t2] = True
            else:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[t1] = True

    return [t for t in textbounds if t not in eliminate]


def get_annotations(path: Path):
    path = path.with_suffix(ANN_SUFFIX)
    textbounds = parse_textbounds(path.read_text().splitlines())
    return eliminate_overlaps(textbounds)


def convert_standoff_to_conll(source_directory_ann, output_directory_conll):
    init_globals()
    files = [f for f in source_directory_ann.iterdir() if f.suffix == ".txt" and f.is_file()]
    process_files(files, output_directory_conll)


if __name__ == '__main__':
    own_path = Path(__file__)
    source_directory_ann = own_path / Path("../temp_files/standoff_files/")
    output_directory_conll = own_path / Path("../temp_files/conll_files/")
    convert_standoff_to_conll(source_directory_ann, output_directory_conll)
