import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model




def predict(inputs, model_path='Animal_Classification.h5'):    
    try:
        model = load_model(model_path)
        pred = model.predict(inputs)
        return pred
    except Exception as e:
        print(f"An error occured: {e}")

labels = {0: "Cat",
          1: "Dog",
          2: "Wild"}


def processed_img(imagepath):
    img = load_img(imagepath, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, [0])
    pred = predict(img)
    y_class = pred.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()


def run():
    img_file = st.file_uploader('Choose an Image', type=['jpg', 'png'])
    if img_file is not None:
        img = Image.open(img_file).resize((250,250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, 'wb') as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            st.success(f'This is a {result}')
            
st.title('CATS AND DOGS CLASSIFICATION')


if __name__=='__main__':
    run()
    