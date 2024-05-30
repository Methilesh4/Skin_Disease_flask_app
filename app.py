from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image



app = Flask(__name__)

disease_info = {
    
    "Bacal Cell Cercinoma": {
        "Symptoms": "Basal Cell Carcinoma typically appears as a shiny, translucent bump or a pearly nodule on the skin. It may bleed easily and form an open sore.",
        "Precautions": "To reduce the risk of Basal Cell Carcinoma, protect your skin from UV radiation, avoid tanning beds, and perform regular skin checks."
    },
    "Eczema": {
        "Symptoms": "Eczema looks different for everyone. And your flare-ups won’t always happen in the same area.No matter which part of your skin eczema affects, it's almost always itchy. The itching sometimes starts before the rash",
        "Precautions": "To prevent Actinic Keratosis, avoid excessive sun exposure, wear sunscreen, and protective clothing when outdoors."
    },
    "Melanoma Photos": {
        "Symptoms": "Melanoma, also known as eczema, may cause dry, itchy, and inflamed skin. It can result in red or brown patches, blisters, and scaling of the skin.",
        "Precautions": "Managing Melanoma involves keeping the skin moisturized, avoiding triggers like allergens and irritants, and using prescribed medications as directed."
    }
}

model1 = tf.keras.models.load_model('lmy_model.h5')
print(model1.count_params())
print(model1.summary())
encoder1 = joblib.load("encoder.pkl")
   

labels = ["Bacal Cell Cercinoma","Eczema",'Melanoma']
def preprocessing(imagep):
    img_rows = 224
    img_cols = 224

    
    img_path = imagep
    img = image.load_img(img_path, target_size=(img_rows, img_cols))
    print(img)
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    
    return x

@app.route('/')
def main():
    Sucess = False
    Unsucess = False

    return render_template("index.html" , submitSucess= Sucess, submitUnsucess= Unsucess)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    Sucess = False
    Unsucess = False

    img = request.files["skinImage"]
   
    if img:
        image_path = 'static/img/uploaded_image.png'  # Specify the path to save the uploaded image
        img.save(image_path)
    print(image_path,"jj")
    # img_array = np.frombuffer(img, np.uint8)
    # uploaded_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    prediction = model1.predict(preprocessing(image_path))
    print(prediction)
    print(max(prediction[0]),prediction[0].argmax(),"ppp")
    predicted_label = labels[prediction[0].argmax()]

    symptoms = ""
    precautions = ""
    
    
    
    
    if predicted_label == 'Bacal Cell Cercinoma':
        symptoms= "Basal Cell Carcinoma typically appears as a shiny, translucent bump or a pearly nodule on the skin. It may bleed easily and form an open sore.",
        precautions= "To reduce the risk of Basal Cell Carcinoma, protect your skin from UV radiation, avoid tanning beds, and perform regular skin checks."
    elif predicted_label == 'Eczema':
        print("e")
        symptoms = "Eczema looks different for everyone. And your flare-ups won’t always happen in the same area.No matter which part of your skin eczema affects, it's almost always itchy. The itching sometimes starts before the rash"
        precautions = "To prevent Actinic Keratosis, avoid excessive sun exposure, wear sunscreen, and protective clothing when outdoors."
    elif predicted_label == 'Melanoma':
        symptoms = "Melanoma, also known as eczema, may cause dry, itchy, and inflamed skin. It can result in red or brown patches, blisters, and scaling of the skin."
        precautions = "Managing Melanoma involves keeping the skin moisturized, avoiding triggers like allergens and irritants, and using prescribed medications as directed."

    
    # _, img_encoded = cv2.imencode(".jpg", uploaded_image)
    # img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return render_template("predicted.html", predicted=predicted_label, symptoms=symptoms, precautions=precautions, submitSucess= Sucess, submitUnsucess= Unsucess)



if __name__ == "__main__":
    app.run(port = 3000 ,debug =True)