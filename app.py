from flask import Flask, render_template, request, jsonify
#import opencv as cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import layers
from keras.preprocessing.image import load_img, img_to_array
from app import app
import os
import pyrebase
# firebase config
import random
import time
from collections import OrderedDict

uri = "mongodb+srv://notsokamka:mongodb@cluster.ojpzq3j.mongodb.net/?retryWrites=true&w=majority"
config = {  

  "apiKey": "AIzaSyD3tY1feZB3fQLwKVxsl9s-suDUHJ7WzVs",
  "authDomain": "rinnegan-435d6.firebaseapp.com",
  "databaseURL": "https://rinnegan-435d6-default-rtdb.firebaseio.com",
  "projectId": "rinnegan-435d6",
  "storageBucket": "rinnegan-435d6.appspot.com",
  "messagingSenderId": "480644787341",
  "appId": "1:480644787341:web:6e5fe779cf94506b7dc6c4",
  "measurementId": "G-G37V6ZD223"
}

# initializing app
firebase = pyrebase.initialize_app(config)
database = firebase.database()
model = load_model(r"C:\Users\deepv\OneDrive\Desktop\bio\App\model.h5")
app = Flask(__name__)

read=(database.child("Vitals").child("Patient1").get()).val()
v1, v2, v3 = read.values()


@app.route('/', methods=['GET'])
def hello_world():
        #values = read.values()
        #return render_template('index.html', v1=values[0], v2=values[1], v3=values[2])
        return render_template('index.html',v_1=v1,v_2=v2,v_3=v3)

@app.route('/', methods=['POST'])
def predictions():
    #print("deep1")
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    print(image_path)
    #print(v1, v2, v3)
    # Preprocess the uploaded image
    #preprocessed_image = preprocess_image(image_path)
    loaded_image = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(loaded_image)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions with the preprocessed image
    prediction = model.predict(img_array.numpy())
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary'] 
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction))
    
    #imagefile.save((remove_last_4_characters(image_path))+"_"+(str(predicted_class)+"_"+str(confidence)+".jpg"))
                   
    return jsonify({"class": predicted_class, "confidence": confidence})

def remove_last_4_characters(input_string):
    if len(input_string) >= 4:
        modified_string = input_string[:-4]
    else:
        modified_string = input_string
    return modified_string
 

if __name__ == '__main__':
    app.run(port=3000, debug=True)
