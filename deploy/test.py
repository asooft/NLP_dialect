from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from sklearn.base import TransformerMixin
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
from nltk.corpus import stopwords
import tashaphyne.normalize as normalize
import emoji
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


def replace_newlines(txt):
    return txt.replace("\n", " ")


def remove_tag(txt):
    return re.sub(r"@\w+\s*", "", txt)


def remove_links(txt):
    return re.sub(r"https?\S+\s*", "", txt)


def remove_english(txt):
    return re.sub(r"[a-zA-Z]+\s*", "", txt)


def remove_emoji(txt):
    return emoji.replace_emoji(txt, "")


def remove_punctuation(txt):
    return re.sub(r"[^\w\s]|[_]", "", txt)


def map_laughter(txt):
    return re.sub(r"(هه)ه+", "هه", txt)


def remove_repeated_letters(txt):
    return re.sub(r"(.)\1{2,}", r"\1", txt)


def remove_numbers(txt):
    return re.sub(r"\d+", "", txt)


def normalize_arabic(txt):
    return normalize.normalize_searchtext(txt)


def remove_stop_words(txt, stop_words):
    return " ".join([word for word in word_tokenize(txt) if word not in stop_words])


def remove_repeated_spaces(txt):
    return re.sub(r"\s{2,}", " ", txt).strip()


def preprocessing(txt):
    txt = replace_newlines(txt)
    txt = remove_tag(txt)
    txt = remove_links(txt)
    txt = remove_english(txt)
    txt = remove_emoji(txt)
    txt = remove_punctuation(txt)
    txt = map_laughter(txt)
    txt = normalize_arabic(txt)
    txt = remove_repeated_letters(txt)
    txt = remove_numbers(txt)
    ar_stop_words = set(stopwords.words("arabic"))
    ar_stop_words = [normalize_arabic(word) for word in ar_stop_words]
    txt = remove_stop_words(txt, stop_words=ar_stop_words)
    txt = remove_repeated_spaces(txt)

    return txt


class preprocessing_class(TransformerMixin):
    def transform(self, X, **transform_params):
        return [preprocessing(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


ml_model = joblib.load(r"D:\ITI\NLP\Deliverables\deploy\ml_model.joblib")

dl_path = r"C:\Users\dell\Downloads\trial_0"
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
model = BertForSequenceClassification.from_pretrained(dl_path)
dl_pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

dl_label_dict = {
    "LABEL_0": "EG",
    "LABEL_1": "LB",
    "LABEL_2": "LY",
    "LABEL_3": "MA",
    "LABEL_4": "SD",
}

ml_label_dict = {0: "EG", 1: "LY", 2: "LB", 3: "SD", 4: "MA"}


def dl_predict_dialect(text, pipe, label_dict):
    text = preprocessing(text)
    prediction = pipe(text)[0]
    output = label_dict[prediction["label"]]
    return output


app = Flask(__name__)


@app.route("/")
def upload_f():
    return render_template("upload.html")


import keras

# keras.preprocessing
def finds():

    txt = request.form.get("entered")
    switch_value = request.form.get("switch")
    print(switch_value)
    if switch_value == "on":
        pred = dl_predict_dialect(txt, dl_pipe, dl_label_dict)
    else:
        label = ml_model.predict(pd.DataFrame({"text": [txt]}))[0]
        pred = ml_label_dict[label]

    return pred


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        val = finds()
        return render_template("pred.html", ss=val)


if __name__ == "__main__":
    app.run()
