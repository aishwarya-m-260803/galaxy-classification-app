from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json
import hashlib
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = "galaxy_secret_key"

# Load model
model = tf.keras.models.load_model("galaxy_resnet_model.h5")
classes = ["Elliptical", "Spiral", "Irregular"]

# Path to users database
USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/home-page")
def home_page():
    return render_template("home.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        # Validation
        if not username or not password:
            return render_template("register.html", error="Username and password are required")
        
        if len(username) < 3:
            return render_template("register.html", error="Username must be at least 3 characters")
        
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters")
        
        # Load existing users
        users = load_users()
        
        # Check if username already exists
        if username in users:
            return render_template("register.html", error="Username already exists. Choose a different one.")
        
        # Register new user
        users[username] = {
            "password": hash_password(password),
            "created": str(__import__('datetime').datetime.now())
        }
        save_users(users)
        
        # Redirect to login
        return redirect(url_for("login"))
    
    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        # Load users
        users = load_users()
        
        # Check credentials
        if username in users and users[username]["password"] == hash_password(password):
            session["user"] = username
            session["history"] = []
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


# Root route: show landing/home when not logged in, otherwise send to predict
@app.route("/")
def root():
    if "user" in session:
        return redirect(url_for("predict"))
    return render_template("home.html")

# ---------------- HOME ----------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("predict.html")

    # POST handler for image prediction
    try:
        if "image" not in request.files:
            return render_template("predict.html", error="No image file provided")
        
        file = request.files["image"]
        
        if file.filename == "":
            return render_template("predict.html", error="No image selected")
        
        # Validate file type
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            return render_template("predict.html", error="Invalid file type. Please upload JPG or PNG image")
        
        # Open and validate image
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess for model
        img = preprocess_image(image)
        
        # ============================================================
        # PREDICTION LOGIC SECTION - CONFIDENCE VALIDATION READY
        # ============================================================
        
        # Make prediction - returns probability distribution for all classes
        pred = model.predict(img, verbose=0)
        # pred shape: (1, 3) where 3 is the number of classes
        # pred contains: [probability_elliptical, probability_spiral, probability_irregular]
        
        # STEP 1: Obtain prediction probabilities
        # pred is a numpy array with raw confidence scores for each class
        
        # STEP 2: Calculate predicted class and confidence
        class_idx = np.argmax(pred)  # Get index of class with highest probability
        confidence = round(float(np.max(pred)) * 100, 2)  # Convert highest probability to percentage
        
        # ============================================================

        prediction = classes[class_idx]

        # Save to history
        if "history" not in session:
            session["history"] = []
        session["history"].append(prediction)
        session.modified = True

        # Encode image to base64 for display
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Galaxy information
        galaxy_info = {
            "Spiral": "Spiral galaxies have rotating arms and active star formation.",
            "Elliptical": "Elliptical galaxies are smooth and contain older stars.",
            "Irregular": "Irregular galaxies lack a defined structure."
        }

        return render_template(
            "result.html",
            prediction=prediction,
            info=galaxy_info[prediction],
            history=session["history"],
            image_base64=img_base64
        )
    
    except Exception as e:
        return render_template("predict.html", error=f"Error processing image: {str(e)}")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("root"))

if __name__ == "__main__":
    app.run(debug=True)
