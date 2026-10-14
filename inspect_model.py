import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
json_file = open(r'C:\Users\LCS\Desktop\landslides\t2\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(r'C:\Users\LCS\Desktop\landslides\t2\model.h5')
print("Model loaded successfully.")

def predict_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((50,50))
    img = np.array(img)
    im2arr = img.reshape(1,50,50,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    prediction = model.predict(test)
    y_pred1 = np.argmax(prediction, axis=1)[0]
    return y_pred1, prediction

# Let's inspect some images from images/0
dir0 = r"C:\Users\LCS\Desktop\landslides\t2\images\0"
print("\n--- Predictions on class 0 (landslide) ---")
if os.path.exists(dir0):
    files = [f for f in os.listdir(dir0) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
    for f in files:
        path = os.path.join(dir0, f)
        pred, raw = predict_img(path)
        print(f"{f}: pred={pred}, raw={raw}")

# Let's inspect some images from images/1
dir1 = r"C:\Users\LCS\Desktop\landslides\t2\images\1"
print("\n--- Predictions on class 1 (no landslide) ---")
if os.path.exists(dir1):
    files = [f for f in os.listdir(dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
    for f in files:
        path = os.path.join(dir1, f)
        pred, raw = predict_img(path)
        print(f"{f}: pred={pred}, raw={raw}")
