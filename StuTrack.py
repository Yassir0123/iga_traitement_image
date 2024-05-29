#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Set paths
IGAREGISTER_PATH = r'C:\Users\SAMSUNG\igaregister'
EXPORT_PATH = r'C:\Users\SAMSUNG\Downloads'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained models
model_face_id = joblib.load("trained_model_face_id.pkl")
model_face_class = joblib.load("trained_model_face_class.pkl")
model_face_firstname = joblib.load("trained_model_face_firstname.pkl")
model_face_lastname = joblib.load("trained_model_face_lastname.pkl")
model_face_age = joblib.load("trained_model_face_age.pkl")
model_face_filiere = joblib.load("trained_model_face_filiere.pkl")

# Functions for face detection and attribute prediction
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return detected_faces, gray

def predict_attributes(detected_faces, gray):
    predictions = []
    for (x, y, w, h) in detected_faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        face_flat = face.flatten().reshape(1, -1)
        
        id_prediction = model_face_id.predict(face_flat)[0]
        class_prediction = model_face_class.predict(face_flat)[0]
        firstname_prediction = model_face_firstname.predict(face_flat)[0]
        lastname_prediction = model_face_lastname.predict(face_flat)[0]
        age_prediction = model_face_age.predict(face_flat)[0]
        filiere_prediction = model_face_filiere.predict(face_flat)[0]
        
        predictions.append({
            "ID": id_prediction,
            "Class": class_prediction,
            "Firstname": firstname_prediction,
            "Lastname": lastname_prediction,
            "Age": age_prediction,
            "Filiere": filiere_prediction
        })
    return predictions

# Functions for training models
def capture_faces_from_folders(main_folder):
    faces = []
    labels = []
    classes = []
    firstnames = []
    lastnames = []
    ages = []
    filieres = []
    student_id = 0

    for student_folder in os.listdir(main_folder):
        if not student_folder.startswith('.'):
            student_folder_path = os.path.join(main_folder, student_folder)
            if os.path.isdir(student_folder_path):
                firstname, lastname, age, filiere = student_folder.split('_')
                class_id = input(f"Enter the class for student in folder '{student_folder}': ")
                print(f"Processing folder: {student_folder}")
                print(f"Extracted Info - Firstname: {firstname}, Lastname: {lastname}, Age: {age}, Filiere: {filiere}, Class: {class_id}, ID: {student_id}")
                for file in os.listdir(student_folder_path):
                    if file.endswith((".jpg", ".png", ".jpeg")):
                        img_path = os.path.join(student_folder_path, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Failed to load image at {img_path}")
                            continue
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        gray = cv2.equalizeHist(gray)
                        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                        for (x, y, w, h) in detected_faces:
                            face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                            faces.append(face)
                            labels.append(student_id)
                            classes.append(class_id)
                            firstnames.append(firstname)
                            lastnames.append(lastname)
                            ages.append(age)
                            filieres.append(filiere)
                student_id += 1
    return faces, labels, classes, firstnames, lastnames, ages, filieres

def train_and_save_model(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Model Accuracy: {accuracy_score(y_test, y_pred)}")
    joblib.dump(model, os.path.join(EXPORT_PATH, f"trained_model_{model_name}.pkl"))

def train_models():
    faces, labels, classes, firstnames, lastnames, ages, filieres = capture_faces_from_folders(IGAREGISTER_PATH)
    X_face = np.array(faces)
    X_face_flat = X_face.reshape(len(X_face), -1)

    train_and_save_model(X_face_flat, labels, "face_id")
    train_and_save_model(X_face_flat, classes, "face_class")
    train_and_save_model(X_face_flat, firstnames, "face_firstname")
    train_and_save_model(X_face_flat, lastnames, "face_lastname")
    train_and_save_model(X_face_flat, ages, "face_age")
    train_and_save_model(X_face_flat, filieres, "face_filiere")

    st.success("All models trained and saved successfully!")

def login_page():
    st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)

    username = st.text_input("Username", key="login_username", help="Enter your username")
    password = st.text_input("Password", type="password", key="login_password", help="Enter your password")

    if st.button("LOGIN", key="login", help="Click to login", use_container_width=True):
        if username == 'admin' and password == 'admin123':
            st.success("Logged in as Admin")
            st.session_state.logged_in = True
            st.session_state.role = "admin"
            st.experimental_rerun()
        elif username == 'user' and password == 'user123':
            st.success("Logged in as User")
            st.session_state.logged_in = True
            st.session_state.role = "user"
            st.experimental_rerun()
        else:
            st.error("Invalid Username or Password")

def admin_page():
    st.subheader("Admin Page")
    st.markdown("<h2 class='title'>Face Recognition Training</h2>", unsafe_allow_html=True)
    
    # Form to create folders
    st.markdown("<div class='form-group'>", unsafe_allow_html=True)
    firstname = st.text_input("Enter Firstname:", key='firstname', help="Enter the student's firstname")
    lastname = st.text_input("Enter Lastname:", key='lastname', help="Enter the student's lastname")
    age = st.text_input("Enter Age:", key='age', help="Enter the student's age")
    filiere = st.text_input("Enter Filiere:", key='filiere', help="Enter the student's filiere")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Create Folder", key='create-folder', help="Create a new folder with the provided information"):
        if firstname and lastname and age and filiere:
            folder_name = f"{firstname}_{lastname}_{age}_{filiere}"
            folder_path = os.path.join(IGAREGISTER_PATH, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            st.success(f"Folder '{folder_name}' created successfully!")
        else:
            st.error("Please fill in all fields to create a folder.")
    
    existing_folders = [f for f in os.listdir(IGAREGISTER_PATH) if os.path.isdir(os.path.join(IGAREGISTER_PATH, f))]
    selected_folder = st.selectbox("Select Folder to Upload Images", existing_folders, help="Select a folder to upload images to")

    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key='upload-images', help="Upload images to the selected folder")
    if st.button("Upload Images", key='upload-button', help="Upload the selected images to the specified folder"):
        if selected_folder and uploaded_images:
            folder_path = os.path.join(IGAREGISTER_PATH, selected_folder)
            for uploaded_image in uploaded_images:
                img_array = np.frombuffer(uploaded_image.read(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_filename = os.path.join(folder_path, uploaded_image.name)
                cv2.imwrite(img_filename, img)
            st.success("Images uploaded successfully!")
        else:
            st.error("Please select a folder and upload images before clicking 'Upload Images'.")

    if st.button("Train Models", key='train-models', help="Train the models with the current data in the 'igaregister' folder"):
        train_models()
    
    if st.button("Go to Face Detection", key='go-to-detection'):
        st.session_state.page = "face_detection"
        st.experimental_rerun()

def face_detection_page():
    st.subheader("Face Detection")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Faces"):
            detected_faces, gray = detect_faces(image)
            predictions = predict_attributes(detected_faces, gray)

            for prediction in predictions:
                st.write(f"ID: {prediction['ID']}, Class: {prediction['Class']}, Firstname: {prediction['Firstname']}, Lastname: {prediction['Lastname']}, Age: {prediction['Age']}, Filiere: {prediction['Filiere']}")

def main():
    # Custom CSS styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            font-family: 'Arial', sans-serif;
            background-color: #f9f9f9;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            color: #333;
            margin-bottom: 1.5rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-control {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .btn-primary {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            text-align: center;
            display: inline-block;
            font-size: 1rem;
            margin: 0.5rem 0;
            cursor: pointer;
            border-radius: 4px;
            text-decoration: none;
        }
        .btn-primary:hover {
            background-color: #45a049;
        }
        .select-folder {
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Face Recognition Application</h1>", unsafe_allow_html=True)
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.role == "admin":
            if "page" not in st.session_state:
                st.session_state.page = "admin"

            if st.session_state.page == "admin":
                admin_page()
            elif st.session_state.page == "face_detection":
                face_detection_page()
        elif st.session_state.role == "user":
            face_detection_page()

if __name__ == "__main__":
    main()


# In[ ]:




