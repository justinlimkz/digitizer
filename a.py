from flask import Flask, flash, redirect
from flask import render_template
from flask import request
import numpy as np
import keras
import flask

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def hello(name=None):
	return render_template('index.html', name=name)

@app.route('/handle_data')
def handle_data():
	a = flask.request.args.get('pixelData').split(',')
	a = np.array(a).reshape(32, 32).astype(int)
	'''
	b = np.zeros((8, 8))
	for i in range(8):
		for j in range(8):
			sm = 0
			for ii in range(4):
				for jj in range(4):
					sm += a[4*i+ii, 4*j+jj]
			b[i][j] = sm
	'''
	model = keras.models.load_model("model")
	#b = b.reshape(1, 8, 8, 1)
	a = a.reshape(1, 32, 32, 1)
	classes = model.predict(a)
	result = "I think this digit is: " + str(np.argmax(classes))
	return flask.jsonify({"result":result})
