#  Galaxy Morphology Classification System

A Flask-based deep learning web application that classifies galaxy images into Elliptical, Spiral, and Irregular types using a ResNet50 model.

---

##  Key Features

###  Deep Learning
- ResNet50-based trained model (.h5)
- 3-class classification: Elliptical, Spiral, Irregular
- Confidence scoring for predictions
- Image preprocessing (224×224, ImageNet normalization)

### Authentication
- User registration & login system
- SHA256 password hashing
- Session-based authentication
- JSON-based user storage

###  Image Processing
- Supports JPG, PNG, GIF, BMP
- Automatic RGB conversion & validation
- Real-time image prediction

###  User Functionality
- Prediction history tracking (per session)
- Galaxy information with results
- Interactive dashboard & prediction flow

---

##  Project Structure

```
galaxy_app/
├── app.py
├── model/                  # Place model here
├── training/train.ipynb    # Model training (Colab)
├── templates/
├── static/
├── requirements.txt
└── users.json
```

---

##  Model Setup

Download the trained model:

👉 https://drive.google.com/file/d/1OArnFOBxRlEvP1JELbvzOJNb9vhBz5G2/view?usp=sharing

Place it inside:
```
model/galaxy_resnet_model.h5
```

---

##  Run the Project

```
git clone https://github.com/aishwarya-m-260803/galaxy-classification-app.git
cd galaxy_app
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000/`

---

##  Training (Optional)

Training code is available in:

```
training/train.ipynb
```

- Run in Google Colab  
- Modify dataset / model  
- Export `.h5` for deployment  

---

##  Tech Stack

- Flask (Backend)
- TensorFlow / Keras
- OpenCV, Pillow
- HTML, CSS (Jinja2)

---

## Author

Aishwarya M
