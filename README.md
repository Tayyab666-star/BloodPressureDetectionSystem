# BP Fuel AI - Blood Pressure Estimation App

This project demonstrates a web application built with Streamlit that estimates blood pressure. It combines traditional patient health information (collected via a questionnaire) with (simulated) computer vision features derived from a webcam or uploaded image/video. The application then provides personalized health tips based on the estimated blood pressure and questionnaire responses.

**Disclaimer:** This app is for educational and demonstration purposes only and does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.

## Features

* **Interactive Questionnaire:** Collects user's age, gender, diet, exercise habits, smoking/alcohol status, and previous health conditions.
* **(Simulated) Computer Vision Integration:** Allows users to capture an image via webcam or upload an image/video, from which hypothetical computer vision features (e.g., facial redness, estimated heart rate from video) are simulated and used in the prediction model.
* **Machine Learning Model:** Utilizes a Random Forest Regressor trained on a cardiovascular disease dataset to estimate systolic and diastolic blood pressure.
* **Personalized Health Tips:** Provides suggestions and warnings based on the estimated blood pressure category (Normal, Elevated, Hypertension Stage 1 & 2, Hypotension) and questionnaire responses.
* **User-Friendly Interface:** Built with Streamlit, featuring an intuitive UI with custom CSS.

## Project Structure
├── cardio_train.csv           # The dataset used for training (download separately)
├── cardio_train_model.py      # Python script to train ML models and save them
├── app.py # The main Streamlit web application
└── requirements.txt           # List of Python dependencies
└── README.md                  # This README file


## Setup and Running the Application

Follow these steps to set up and run the application locally:

### Prerequisites

Before you start, ensure you have Python (3.8+) installed.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Tayyab666-star/BloodPressureDetectionSystem.git)
    cd BloodPressureDetectionSystem.git
    ```
    (Replace `YourUsername/YourRepoName` with your actual GitHub username and repository name.)

2.  **Download the Dataset:**
    The `cardio_train.csv` dataset is not included directly in the repository due to size.
    * Download it from Kaggle: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
    * Make sure to unzip the downloaded file.
    * Place the `cardio_train.csv` file directly into your project's root directory (the same folder as `streamlit_bp_app_enhanced.py`).

3.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Install all required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Step-by-Step Execution

1.  **Train and Save the Machine Learning Models:**
    You need to run the training script *once* to generate the `.pkl` model files.
    ```bash
    python cardio_train_model.py
    ```
    This script will print messages about data loading, preprocessing, model training, evaluation, and finally, indicate that the `.pkl` files (`systolic_bp_model.pkl`, `diastolic_bp_model.pkl`, `data_preprocessor.pkl`) have been saved.

2.  **Run the Streamlit Application:**
    Once the `.pkl` files are generated, you can launch the Streamlit app:
    ```bash
    streamlit run streamlit_bp_app_enhanced.py
    ```
    This command will open the application in your default web browser (usually at `http://localhost:8501`).

## How to Use the App

1.  **Fill out the Questionnaire:** Complete the patient questionnaire in Step 1 and click "Submit Questionnaire & Continue."
2.  **Grant Camera Access:** If prompted by your browser, allow access to your webcam.
3.  **Capture or Upload Image/Video:** In Step 2, either take a picture using your webcam or upload a clear image/short video of a face.
4.  **View Results:** The app will process the input, estimate blood pressure, and display personalized health tips based on your questionnaire data and the estimated blood pressure.

## Future Enhancements (Ideas)

* **Real Computer Vision Integration:** Replace simulated CV features with actual facial analysis for more accurate blood pressure estimation (e.g., using PPG from video).
* **More Comprehensive Questionnaire:** Include fields for height, weight, cholesterol, glucose, and other vital signs directly.
* **Database Integration:** Store user data and predictions securely for historical tracking.
* **User Authentication:** Implement login/signup for personalized user experiences.
* **Deployment:** Deploy the Streamlit app to a cloud platform (e.g., Streamlit Community Cloud, Heroku, AWS).

---
