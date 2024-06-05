import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
print(tf.__version__)
model = tf.keras.models.load_model('my_model.h5')
image_path = '231.png'
img_height = 256
img_width = 256
img = load_img(image_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
y_prob = model.predict(img_array)
y_classes = y_prob.argmax(axis=-1)
print(y_classes)

