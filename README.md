# Digitizer

A simple convolutional neural net to recognize digits from the [Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits). Users draw digits on an interface and query the model to see what the neural net thinks the digit is.

## Interface example

![Digitizer interface](digitizer_example.png?raw=true = 200x)

## Usage

- `arch.py` computes the model and saves it to disk. The current model contains a Conv2D layer with 100 filters, a MaxPooling2D filter with size 2, followed by a flatten layer, and a Dense layer with 1000 units.  
- `a.py` uses the flask package to query the neural net based on user input. To start the server, run `export FLASK_APP=a.py`, followed by `flask run`. The server should start on `localhost:5000`.

## Packages used

keras, numpy, and flask.

