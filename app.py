import streamlit as st
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os
from google.cloud import storage
import numpy as np
from PIL import Image
import random
import cv2
from streamlit_option_menu import option_menu
import requests

#References
#Citations: Victoryhb (2022), Streamlit-option-menu, github. https://github.com/victoryhb/streamlit-option-menu

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Creating a horizontal navigation bar 
selected = option_menu(
    menu_title= None,
    options=["Home Page", "DTSC 691 Project", "Resume", "Other Projects"],
    orientation="horizontal",
)

# Display content based on the button clicked
if selected == "Home Page":
    st.subheader("About me")
    st.markdown("""Hello! My name is Ryuei Matsui, a machine learning enthusiast and aspiring data scientist with an interest in applying machine learning to real-world problems,
         particularly in the healthcare sector. I am particularly interested in the potential of AI to enhance medical diagnostics, making healthcare more accessible and effective.
         I attend Eastern University and currently attending Eastern University's data science program. In my free time, I enjoy hiking, snowboarding, and spending time with friends and family.""") 
    st.subheader("About the Project")
    st.markdown("""In this project, I have developed a deep learning model to detect Tuberculosis from X-ray scans using a convolutional neural network (CNN) model.
        This project aims to improve early detection and assist healthcare professionals in producing a model that will accurately diagnose Tuberculosis more efficiently.""") 
                
  # Footer or Contact Information (Optional)
    st.markdown("---")
    st.subheader("Contact Information")
    st.markdown("""
    - **Email**: rmatsui101@gmail.com  
    - **LinkedIn**: www.linkedin.com/in/rmatsui
    
    """) 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
if selected == "DTSC 691 Project":
    st.title('Machine Learning Project: Detecting Tuberculosis from X-ray Images')
    st.subheader("About the Model")
    st.write("""Using a Convolutional neural network (CNN) model, this model detects Tuberculosis from a random batch of 10 X-ray images collected from the 
             training data. With an accuracy of 99%, the model will predict the classification based on binary output of a scale from 0 to 1.
             A confidence score of over 0.5 will result in a postive case and a score below 0.5 will result in a normal case. """)
    # Load the pre-trained model
    # Use relative path for loading the model
    model_path = "tuberculosisclassification2.keras"
    if not os.path.exists(model_path):
        url = "https://github.com/rmatsui1/Tuberculosis_Detection_Model/raw/main/tuberculosisclassification2.keras"
        response = requests.get(url)
        with open(model_path, "wb") as file:
            file.write(response.content)

    model = tf.keras.models.load_model(model_path)

    # Importing training data for image selection
    training_images = "./trainingdataweb"
    image_files = [f for f in os.listdir(training_images) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Function to preprocess images for prediction
    def preprocess_image(img, target_size=(512, 512)):
        img = img.resize(target_size)
        img = np.array(img)
        if img.ndim == 2:  # Convert grayscale to RGB if necessary
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype('float32') / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    # Function to predict the image label (Normal or Positive)
    def predict_image(img):
        target_size = model.input_shape[1:3]  # Get model's expected input size
        preprocessed_img = preprocess_image(img, target_size=target_size)
        prediction = model.predict(preprocessed_img)
        if prediction[0][0] > 0.5:
            return 'Positive (Tuberculosis Detected)'
        else:
            return 'Normal (No Tuberculosis Detected)'

    # Streamlit Interface for Image Selection and Prediction
    st.subheader("Please Select an X-ray Scan Image and Click on the Make Prediction Button.")

    # Generate a random batch of 10 images if not already in session_state
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = random.sample(image_files, 10)

    # Display the selected images and allow users to select images for batch prediction
    num_columns = 5
    columns = st.columns(num_columns)

    selected_image_paths = []
    for idx, img_name in enumerate(st.session_state.selected_images):
        col = columns[idx % num_columns] 
        img_path = os.path.join(training_images, img_name)
        img = Image.open(img_path)

        # Display the image in the selected column with a checkbox for selection
        with col:
            if st.checkbox("Select", key=img_name):
                selected_image_paths.append(img_path)
            st.image(img, caption="", use_container_width=True)

    # Button to trigger batch prediction for selected images
    predict_button = st.button("Make Prediction")
    if predict_button and selected_image_paths:
        # Loop through the selected images and make predictions
        for img_path in selected_image_paths:
            img = Image.open(img_path)
            img_array = preprocess_image(img)

            # Make prediction
            prediction = model.predict(img_array)

            confidence = prediction[0][0]  # Assuming binary output (positive or negative)
            if confidence > 0.5:
                st.write(f"{os.path.basename(img_path)}: Positive for Tuberculosis (Confidence: {confidence * 100:.2f}%)")
            else:
                st.write(f"{os.path.basename(img_path)}: Negative for Tuberculosis (Confidence: {confidence * 100:.2f}%)")
   

    # --- File Uploader for Individual Images ---
    st.markdown("### Upload Your Own X-ray Image for Prediction")
    uploaded_files = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    # If files are uploaded, make predictions on them
    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_container_width=True)

            # Make prediction for the uploaded file
            prediction = predict_image(img)
            st.write(f"Prediction for {uploaded_file.name}: {prediction}")

# --- Resume Page ---
if selected == "Resume":
    st.title("Ryuei Matsui")

    # Education Section
    st.subheader("Education")
    st.markdown("""
    **Eastern University**, Private University of Pennsylvania | St. Davids, PA  
    **Master of Science (M.S)** in Data Science | *Expected Graduation: December 2025*  
    Cumulative GPA: 3.90  
    Relevant Coursework: Machine Learning, Statistical Modeling, Data Analytics in R, Business in Data

    **Binghamton University**, State University of New York | Binghamton, NY  
    **Bachelor of Science (B.S)** in Financial Economics, Harpur College of Arts and Sciences | *Jan 2019 - May 2022*
    """)

    # Skills Section
    st.subheader("Skills")
    st.markdown("""
    **Programming Languages**: Python | R | SQL | Oracle BI | Tableau | Excel | PowerPoint  
    **Data Science Technologies**: Statistics, Time Series, Cluster Analysis, Regressions (Multi, Logistic), Data mining, Forecasting, Visualizations, Decision Trees, Support Vector Models, Random Forest
    """)

    # Work Experience Section
    st.subheader("Work Experience")
    st.markdown("""
    **NYC Parks Capital Projects** | *May 2024 - Aug 2024*  
    **Integrated Data Solutions Team Member Intern**
    - Leveraged Oracle BI to create and integrate drill-down charts into an interactive dashboard for 600+ active NYC Park projects, including metrics on status, fiscal years, boroughs, and project details.
    - Utilized SQL Developer to query and clean data from the Parks database, handling datasets of over 500,000 rows of Parks project information.
    - Configured and set up the Unifier environment, gaining expertise in front-end navigation and backend administration to streamline project management processes.
    """)
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
if selected == "Other Projects":
    st.subheader("Other Projects")
    st.markdown("""**GitHub Repository**:(https://github.com/rmatsui1/PortfolioProjects/tree/main
    """)
    


