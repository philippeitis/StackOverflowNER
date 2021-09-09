import flask
from flask import request
from marshmallow import Schema, fields

from main import segment_from_sentence
from ner import NER
from segmenter import Segmenter
from utils_ctc.prediction_ctc import CTCModel
from utils_ctc.config_ctc import parameters_ctc

app = flask.Flask(__name__)
app.config["DEBUG"] = True

_SEGMENTER = None
_NER = None
_CTC_MODEL = None


class NERGet(Schema):
    text = fields.Str()


class BILOUNERChunk(Schema):
    text = fields.Str(required=True)
    type = fields.Str(required=True)


class NERChunk(Schema):
    text = fields.Str(required=True)
    type = fields.Str(required=True)
    start = fields.Integer(required=True)


class BILOUNERResult(Schema):
    entities = fields.List(
        fields.Nested(BILOUNERChunk()),
        required=True
    )


class NERResult(Schema):
    entities = fields.List(
        fields.Nested(NERChunk()),
        required=True
    )


@app.route('/', methods=['GET'])
def home():
    return ""


def get_ner():
    global _SEGMENTER, _NER, _CTC_MODEL
    train_file = parameters_ctc['train_file']
    test_file = parameters_ctc['test_file']

    _CTC_MODEL = _CTC_MODEL or CTCModel(train_file, test_file)
    _SEGMENTER = _SEGMENTER or Segmenter()
    _NER = _NER or NER()
    return _CTC_MODEL, _SEGMENTER, _NER


@app.route('/bilou_ner', methods=['GET'])
def bilou_ner_route():
    form_data = request.get_json()
    errors = NERGet().validate(form_data)
    if errors:
        return {"message": "Malformed input."}, 400

    ctc_model, segmenter, ner = get_ner()

    sentence = form_data["text"]
    si = segment_from_sentence(sentence, "000000", ctc_model)
    results, preds = segmenter.predict(
        segmenter.examples_to_dataset(si.to_examples())
    )

    ni = si.to_ner_input(preds)
    results, preds = ner.predict(
        ner.examples_to_dataset(ni.to_examples())
    )

    try:
        sentence, preds = next(zip(ni.items, preds))
    except StopIteration:
        return {"message": "Unknown failure"}, 500

    result_schema = BILOUNERResult()

    json_item = []
    for token, pred in zip(sentence, preds):
        json_item.append({"text": token.word, "type": pred})

    return result_schema.dumps(result_schema.load({"entities": json_item})), 200


@app.route('/ner', methods=['GET'])
def ner_route():
    form_data = request.get_json()
    errors = NERGet().validate(form_data)
    if errors:
        return {"message": "Malformed input."}, 400

    ctc_model, segmenter, ner = get_ner()

    sentence = form_data["text"]
    si = segment_from_sentence(sentence, "000000", ctc_model)
    results, preds = segmenter.predict(
        segmenter.examples_to_dataset(si.to_examples())
    )

    ni = si.to_ner_input(preds)
    results, preds = ner.predict(
        ner.examples_to_dataset(ni.to_examples())
    )

    try:
        spans = ni.bilou_to_spans(preds, [sentence])[0]
    except StopIteration:
        return {"message": "Unknown failure"}, 500

    result_schema = NERResult()

    json_item = []
    for start, text, tag in spans:
        json_item.append({"text": text, "type": tag, "start": start})

    return result_schema.dumps(result_schema.load({"entities": json_item})), 200


@app.route('/', methods=['GET'])
def main():
    return ""


if __name__ == '__main__':
    app.run()
