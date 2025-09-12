from flask import Flask, request, jsonify, render_template, render_template_string
import tensorflow as tf
import numpy as np
import base64, re
import cv2
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# --- 1. Load your trained model ---

MODEL_PATH = "eye_model.h5"  # Change to your model path
model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def check_image(image_json):

    # Decode Base64 -> PIL -> NumPy
    image = Image.open(BytesIO(base64.b64decode(image_json))).convert("RGB")
    img_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    eyes = eye_cascade.detectMultiScale(gray, 1.3,5)

    if len(faces) > 0:
        return "Face"
    elif len(eyes) == 1:
        return "One Eye"
    elif len(eyes) > 1:
        return "Two Eyes"
    else:
        return None

def preprocess_eye(json_image, target_size=(128,128), eye='left'):
    """
    Takes a Base64 JSON image (whole face or both eyes) and extracts one eye.
    Args:
        json_image: base64 encoded image (from request.get_json()["image"])
        target_size: resize shape (H, W) for the model
        eye: "left" or "right"
    Returns:
        eye_img: preprocessed numpy array ready for prediction
    """
    pil_img = Image.open(BytesIO(base64.b64decode(json_image))).convert("RGB")
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # --- Detect face first (optional, helps narrow search) ---
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # take first detected face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_np[y:y+h, x:x+w]
    else:
        roi_gray = gray
        roi_color = img_np

    # --- Detect eyes inside ROI ---
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes) == 0:
        raise ValueError("No eyes detected in image")

    # Sort eyes left-to-right
    eyes = sorted(eyes, key=lambda e: e[0])

    if eye == "left":
        (ex, ey, ew, eh) = eyes[0]
    else:
        (ex, ey, ew, eh) = eyes[-1]  # rightmost eye

    # Extract and resize eye
    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
    eye_img = cv2.resize(eye_img, target_size)
    eye_img = eye_img.astype("float32") / 255.0  # normalize
    eye_img = np.expand_dims(eye_img, axis=0)    # (1, H, W, C)

    return eye_img

@app.route("/")
def index():
    return render_template("eye_detect.html")


@app.route("/predict", methods=["POST"])
# --- 2. Preprocessing function ---
def predict():
    data = request.get_json()
    
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data["image"].split(",")[1]  # remove 'data:image/png;base64,'
    check = check_image(image_data)
    if check == None:
        return jsonify({"error": "Image too complicated"})
    
    if check == "One Eye":
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.resize((128, 128))  # match your model's expected size
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
    else:
        image = preprocess_eye(image_data, target_size=(128,128), eye='left')



    
    # Predict
    prediction = model.predict(image)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    if predicted_class == 0:
        html_snippet = render_template_string("""
            <div class="prediction-block">
                  <h2> This is a non-blue eyed image </h2>
                <img src=""> 
                <a href="" type="button">Try again?</a>                           

            </div>

        """)
    else:
        html_snippet = render_template_string("""
            <div class="prediction-block">
                <h2> This is a blue eyed image </h2>
                <img src="/static/Photos/elegant_blueeye.jpg">
                <a href="" type="button">Try again?</a>
            </div>
        """)
    
    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence,
        'html_snippet': html_snippet
    })

if __name__ == "__main__":
    app.run(debug=True)
