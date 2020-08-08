import flask
from flask import Flask, jsonify, request
import requests

import torch
import torch.nn

from tokenizers import Tokenizer
from nn.model import LSTMModel
from utils.data_utils import predict_sentence


app = Flask(__name__)
tokenizer = Tokenizer.from_file("app/materials/french_pretrained_tokenizer")
model = LSTMModel(tokenizer.get_vocab_size(), 300, 256)
model.load_state_dict(torch.load('app/materials/french_pretrained_model.pt'))


@app.route('/generate', methods = ['POST'])
def run():
    data = request.get_json(force=True)
    data = predict_sentence(model, tokenizer, data['text'], 30, 1, "cpu")
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)    