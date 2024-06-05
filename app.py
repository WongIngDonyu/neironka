from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static'

model = tf.keras.models.load_model('my_model.h5')
img_height = 512
img_width = 512


@app.route('/')
def home():
    return render_template('first.html')


@app.route('/error21')
def error21():
    return render_template('error21.html')


@app.route('/error53')
def error53():
    return render_template('error53.html')


@app.route('/error55')
def error55():
    return render_template('error55.html')


@app.route('/error56')
def error56():
    return render_template('error56.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = load_img(filepath, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    y_prob = model.predict(img_array)
    predicted_class = int(tf.argmax(y_prob, axis=-1))
    if predicted_class == 0:
        return redirect(url_for('error21'))
    elif predicted_class == 1:
        return redirect(url_for('error53'))
    elif predicted_class == 2:
        return redirect(url_for('error55'))
    elif predicted_class == 3:
        return redirect(url_for('error56'))
    else:
        return redirect(url_for('error'))


@app.route('/error')
def error():
    return "Error detected"


if __name__ == '__main__':
    app.run(debug=True)