from flask import Flask, render_template, request
from model import load_model, predict_ensembles

LIST_MODEL = tuple()

app = Flask(__name__)

with app.app_context():
	if not LIST_MODEL:
		LIST_MODEL = load_model()
print("Load model successfully")


@app.route('/')
def main():
	data = {
		"comment": "",
		"label": ""
	}
	return render_template('main.html', data=data)


@app.route('/predict', methods=['POST'])
def predict():
	data = {
		"comment": "",
		"label": ""
	}
	
	results = {}
	
	comment = request.form.get('comment')
	print(comment)
	data['label'], results = predict_ensembles(comment, list_modes=LIST_MODEL)
	data['comment'] = comment
	return render_template('main.html', data=data, results=results)


if __name__ == '__main__':
	app.run()
