import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import joblib # To load our saved models and preprocessor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline # Make sure Pipeline is imported

st.set_page_config(page_title="BP Fuel AI - Blood Pressure Estimation", layout="centered")

# --- Custom CSS for enhanced UI ---
st.markdown("""
<style>
    .main {
        background-color: #f4f6fa; /* Light grey background */
        padding: 20px;
    }
    .stButton>button {
        background-color: #28a745; /* Green for primary action */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838; /* Darker green on hover */
    }
    .stForm {
        background-color: #ffffff; /* White form background */
        padding: 2em;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Soft shadow */
        margin-bottom: 2em;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stRadio>div>label {
        border-radius: 8px; /* Corrected typo from 8box to 8px */
        border: 1px solid #ced4da;
        padding: 0.5em;
    }
    .stAlert {
        border-radius: 8px;
        padding: 1em;
    }
    h1 {
        color: #0056b3; /* Dark blue for main title */
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        margin-bottom: 1em;
    }
    h2, h3, h4, h5, h6 {
        color: #007bff; /* Lighter blue for sections */
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #ced4da;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Centered title with an icon
st.markdown(
    "<h1 style='text-align: center;'>❤️‍🩹 BP Fuel AI</h1>",
    unsafe_allow_html=True
)

st.write("---") # Visual separator

st.info("👋 Welcome! This application helps estimate your blood pressure using your personal health data and (simulated) insights from your webcam/uploaded media. Please fill out the questionnaire and provide an image/video.")

# --- Load Models and Preprocessor (only once when the app starts) ---
@st.cache_resource # Use st.cache_resource for heavy objects like models
def load_ml_assets():
    try:
        systolic_model = joblib.load('systolic_bp_model.pkl')
        diastolic_model = joblib.load('diastolic_bp_model.pkl')
        data_preprocessor = joblib.load('data_preprocessor.pkl')
        
        # This list comes from the X.columns.tolist() used during training.
        # It's crucial for the `predict_blood_pressure_new_data` function.
        # The order matters! Ensure it matches the exact feature set and order from your training script.
        feature_columns_for_prediction = [
            'gender', 'height', 'weight', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'age_years', 'FacialRednessIndex', 'EyeAreaRatio', 'SkinToneVariability',
            'EstimatedHeartRate_CV', 'PPG_SignalNoiseRatio'
        ]
        return systolic_model, diastolic_model, data_preprocessor, feature_columns_for_prediction
    except FileNotFoundError:
        st.error("❌ Error: Trained models or preprocessor files not found. "
                 "Please ensure 'systolic_bp_model.pkl', 'diastolic_bp_model.pkl', "
                 "and 'data_preprocessor.pkl' are in the same directory as this app. "
                 "You need to run the model training script at least once to generate them.")
        st.stop() # Stop the app if models aren't found
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading ML assets: {e}")
        st.stop()

systolic_model, diastolic_model, data_preprocessor, ml_feature_columns = load_ml_assets()

# --- Helper function to prepare input data for the ML models ---
def prepare_features_for_prediction(user_data, cv_features):
    # Map Streamlit inputs to model's expected features
    # Fill in placeholder/default values for features not collected directly from UI
    data_for_prediction = {
        'age_years': user_data['age'],
        # Gender: Map 'Male': 1, 'Female': 2 based on the dataset's encoding (1 for male, 2 for female)
        'gender': 1 if user_data['gender'] == 'Male' else (2 if user_data['gender'] == 'Female' else 0), # 0 for 'Other' or unknown
        
        # Placeholder/Default values for features not asked in the questionnaire:
        'height': 170, # Default height if not collected
        'weight': 70,  # Default weight if not collected (adjust as needed)
        
        # Cholesterol and Gluc are typically 1=normal, 2=above normal, 3=well above normal
        # Since not in questionnaire, we'll assume normal (1)
        'cholesterol': 1,
        'gluc': 1, 
        
        'smoke': 1 if user_data['smoker'] == 'Yes' else 0,
        'alco': 1 if user_data['alcohol'] == 'Yes' else 0,
        # Simplified 'active' based on exercise frequency
        'active': 1 if user_data['exercise'] in ["Daily", "Few times a week"] else 0,
        
        'FacialRednessIndex': cv_features.get('FacialRednessIndex', 0.5),
        'EyeAreaRatio': cv_features.get('EyeAreaRatio', 0.03),
        'SkinToneVariability': cv_features.get('SkinToneVariability', 0.005),
        'EstimatedHeartRate_CV': cv_features.get('EstimatedHeartRate_CV', 75),
        'PPG_SignalNoiseRatio': cv_features.get('PPG_SignalNoiseRatio', 20.0)
    }

    # Create DataFrame in the exact order expected by the preprocessor
    input_df = pd.DataFrame([data_for_prediction], columns=ml_feature_columns)

    # Transform the input data using the loaded preprocessor
    processed_input = data_preprocessor.transform(input_df)
    return processed_input

# --- BP Estimation Function (using ML model) ---
# This function will now take the ML models and preprocessor
def estimate_bp_from_frame(frame, user_data):
    # --- IMPORTANT: SIMULATED CV FEATURE EXTRACTION ---
    # In a real application, you'd integrate actual computer vision
    # libraries (like OpenCV with advanced algorithms or deep learning models)
    # to derive these features from the 'frame'.
    # For this example, we continue to simulate them based on the frame dimensions
    # to make them non-static but still random for demonstration purposes.
    # The actual 'frame' (image data) is passed but not deeply analyzed here.
    if frame is None:
        st.warning("No valid image frame to process for CV features. Skipping BP estimation.")
        return None, None # Return None for BP if frame is invalid

    # Basic check for frame dimensions for pseudo-randomness
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        seed_val = frame.shape[0] * frame.shape[1]
    else:
        seed_val = 42 # Default seed if frame is invalid/empty

    np.random.seed(seed_val)
    simulated_cv_features = {
        'FacialRednessIndex': np.random.uniform(0.4, 0.7),
        'EyeAreaRatio': np.random.uniform(0.02, 0.04),
        'SkinToneVariability': np.random.uniform(0.003, 0.007),
        'EstimatedHeartRate_CV': np.random.randint(65, 95),
        'PPG_SignalNoiseRatio': np.random.uniform(15, 25.0)
    }

    # Prepare data for prediction using questionnaire and simulated CV features
    processed_input = prepare_features_for_prediction(user_data, simulated_cv_features)

    # Make predictions using the loaded ML models
    predicted_systolic = systolic_model.predict(processed_input)[0]
    predicted_diastolic = diastolic_model.predict(processed_input)[0]

    return predicted_systolic, predicted_diastolic

# --- Questionnaire Section ---
st.header("📝 Step 1: Patient Questionnaire")
st.markdown("Please provide accurate information for a better estimation.")

with st.form("questionnaire_form", clear_on_submit=False): # Keep form data after submit
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30, help="Your current age in years.")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Your biological gender.")
        diet = st.selectbox("How would you describe your diet?", ["Healthy", "Average", "Unhealthy"], help="Your typical eating habits.")
        salt_intake = st.selectbox("Salt intake?", ["Low", "Moderate", "High"], help="How much salt do you typically consume?")
    
    with col2:
        exercise = st.selectbox("How often do you exercise?", ["Daily", "Few times a week", "Rarely", "Never"], help="Your physical activity level.")
        smoker = st.radio("Do you smoke?", ["Yes", "No"], horizontal=True, help="Current smoking status.")
        alcohol = st.radio("Do you consume alcohol?", ["Yes", "No"], horizontal=True, help="Current alcohol consumption status.")
        # We need to map the prev_conditions to the dataset's 'Medical Condition' if it was a feature
        # For simplicity, here we map the relevant ones for tips, but it's not a model input directly from this
        prev_conditions = st.multiselect("Previous health conditions", ["Hypertension", "Diabetes", "Heart Disease", "None"], help="Select any pre-existing medical conditions.")
        
    st.markdown("---")
    submitted = st.form_submit_button("Submit Questionnaire & Continue")

if submitted:
    st.session_state['questionnaire'] = {
        "age": age,
        "gender": gender,
        "diet": diet,
        "salt_intake": salt_intake,
        "exercise": exercise,
        "smoker": smoker,
        "alcohol": alcohol,
        "prev_conditions": prev_conditions
    }
    st.success("✅ Questionnaire saved! Please proceed to the next step.")
    st.balloons() # Visual feedback

# Only show the next step if questionnaire is submitted
if 'questionnaire' in st.session_state:
    st.header("📸 Step 2: Capture or Upload Your Face")
    st.markdown("This step uses (simulated) computer vision to enhance the prediction. Provide a clear image of your face.")

    col_cam, col_upload = st.columns(2)

    with col_cam:
        st.markdown("#### Webcam Capture")
        camera_image = st.camera_input("Take a picture now")
        if camera_image: # Only process if image is taken
            with st.spinner("Analyzing image..."):
                file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Captured Image", use_container_width=True)
                
                systolic, diastolic = estimate_bp_from_frame(frame, st.session_state['questionnaire'])
                if systolic is not None and diastolic is not None:
                    st.session_state['bp_result'] = (systolic, diastolic)
                    st.success(f"**Estimated Blood Pressure:** {int(systolic)} / {int(diastolic)} mmHg")
                else:
                    st.error("Could not estimate BP from the captured image. Please try again.")

    with col_upload:
        st.markdown("#### File Upload")
        uploaded_file = st.file_uploader(
            "Upload an image or short video (4-5s)",
            type=["jpg", "jpeg", "png", "mp4", "avi"],
            key="main_file_uploader",
            help="Upload a clear image of your face or a short video for analysis."
        )
        if uploaded_file is not None:
            with st.spinner("Processing uploaded file..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = None
                if uploaded_file.type.startswith("image/"):
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                elif uploaded_file.type.startswith("video/"):
                    # Save to a temporary file for OpenCV to read
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(file_bytes)
                        temp_video_path = tfile.name
                    video = cv2.VideoCapture(temp_video_path)
                    ret, frame = video.read() # Read the first frame
                    video.release()
                    import os
                    os.unlink(temp_video_path) # Clean up temp file
                
                if frame is not None:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Processed Frame", use_container_width=True)
                    systolic, diastolic = estimate_bp_from_frame(frame, st.session_state['questionnaire'])
                    if systolic is not None and diastolic is not None:
                        st.session_state['bp_result'] = (systolic, diastolic)
                        st.success(f"**Estimated Blood Pressure:** {int(systolic)} / {int(diastolic)} mmHg")
                    else:
                        st.error("Could not estimate BP from the uploaded file. Please try another file.")
                else:
                    st.error("Could not process the uploaded file. Ensure it's a valid image or video.")

# --- Health Tips Section ---
def generate_tips(age, diet, salt_intake, exercise, smoker, alcohol, prev_conditions, systolic=None, diastolic=None):
    tips = []
    
    # Blood Pressure Category Based Suggestions
    if systolic is not None and diastolic is not None:
        if systolic < 90 or diastolic < 60:
            tips.append("### ⚠️ Low Blood Pressure (Hypotension) Suggestions:")
            tips.append("-   **Consult a Doctor:** If you frequently experience low BP symptoms (dizziness, fainting), see a healthcare professional for diagnosis and treatment.")
            tips.append("-   **Stay Hydrated:** Drink plenty of fluids throughout the day to prevent dehydration, which can lower blood pressure.")
            tips.append("-   **Increase Salt (with caution):** Discuss with your doctor if increasing sodium intake slightly is appropriate for you.")
            tips.append("-   **Small, Frequent Meals:** Eating smaller, low-carb meals can help prevent sudden drops in BP after eating.")
            tips.append("-   **Avoid Sudden Movements:** Rise slowly from sitting or lying positions to prevent orthostatic hypotension.")
            tips.append("---")
        elif (systolic >= 90 and systolic <= 120) and (diastolic >= 60 and diastolic <= 80):
            tips.append("### ✅ Normal Blood Pressure Suggestions:")
            tips.append("-   **Maintain Healthy Habits:** Keep up your balanced diet, regular exercise, and stress management practices.")
            tips.append("-   **Regular Check-ups:** Even with normal BP, routine medical check-ups are important for overall health monitoring.")
            tips.append("-   **Monitor Trends:** Be aware of any changes in your readings over time.")
            tips.append("---")
        elif (systolic > 120 and systolic <= 129) and (diastolic >= 60 and diastolic <= 80):
            tips.append("### 💛 Elevated Blood Pressure Suggestions (Pre-Hypertension):")
            tips.append("-   **Lifestyle Changes are Key:** This is a crucial stage to prevent hypertension. Focus on lifestyle modifications.")
            tips.append("-   **DASH Diet Emphasis:** Strictly follow the DASH diet principles.")
            tips.append("-   **Increase Physical Activity:** Aim for at least 150 minutes of moderate-intensity exercise per week.")
            tips.append("-   **Limit Sodium & Alcohol:** Reduce salt intake and moderate alcohol consumption.")
            tips.append("-   **Stress Management:** Implement stress-reduction techniques like meditation or yoga.")
            tips.append("---")
        elif (systolic >= 130 and systolic <= 139) or (diastolic >= 80 and diastolic <= 89):
            tips.append("### 🧡 High Blood Pressure (Hypertension Stage 1) Suggestions:")
            tips.append("-   **Consult a Doctor:** Your doctor may recommend lifestyle changes and possibly medication.")
            tips.append("-   **Consistent Monitoring:** Monitor your blood pressure at home regularly and keep a log for your doctor.")
            tips.append("-   **Dietary Adjustments:** Seriously commit to low-sodium, heart-healthy foods (DASH diet).")
            tips.append("-   **Regular Exercise:** Make physical activity a consistent part of your routine.")
            tips.append("---")
        elif systolic >= 140 or diastolic >= 90:
            tips.append("### ❤️‍🔥 High Blood Pressure (Hypertension Stage 2 or Hypertensive Crisis) Suggestions:")
            tips.append("-   **URGENT Medical Attention:** **Immediately consult a doctor.** This level of blood pressure often requires medication and significant lifestyle changes.")
            tips.append("-   **Do NOT Self-Treat:** Do not ignore these readings. Follow your doctor's instructions meticulously.")
            tips.append("-   **Emergency if Symptoms:** If these readings are accompanied by symptoms like chest pain, severe headache, shortness of breath, or numbness/weakness, **seek emergency medical care immediately.**")
            tips.append("---")

    # General Questionnaire-based tips (these remain valuable irrespective of BP category)
    tips.append("### General Health Tips based on Questionnaire:")

    if age >= 50:
        tips.append("-   **Age Factor:** Regular blood pressure monitoring becomes even more critical after age 50. Discuss screening frequency with your doctor.")
    if diet == "Unhealthy":
        tips.append("-   **Diet Improvement:** Adopting a healthier diet rich in fruits, vegetables, and lean proteins is vital. Limit processed foods, unhealthy fats, and added sugars.")
    elif diet == "Average":
        tips.append("-   **Diet Tweaks:** Small improvements like reducing sugary drinks and increasing fiber can significantly impact your health.")
    if salt_intake == "High":
        tips.append("-   **Sodium Reduction:** Be mindful of hidden salts in processed foods. Cooking at home allows for better control of sodium intake.")
    if exercise in ["Rarely", "Never"]:
        tips.append("-   **Increase Activity:** Start with short walks and gradually increase duration and intensity. Aim for at least 30 minutes of moderate activity most days.")
    elif exercise == "Few times a week":
        tips.append("-   **Boost Exercise:** Consider increasing the duration or intensity of your workouts, or try new activities to stay motivated.")
    if smoker == "Yes":
        tips.append("-   **Quit Smoking:** Smoking damages blood vessels and significantly increases heart disease risk. Quitting is the best gift you can give your health.")
    if alcohol == "Yes":
        tips.append("-   **Moderate Alcohol:** If you drink, do so in moderation (up to one drink per day for women, two for men). Excessive alcohol raises blood pressure.")
    
    if "Hypertension" in prev_conditions:
        tips.append("-   **Manage Existing Hypertension:** Continue to diligently follow your doctor's treatment plan for hypertension, including medication and lifestyle changes.")
    if "Diabetes" in prev_conditions:
        tips.append("-   **Diabetes Management:** Tightly control your blood sugar levels, as diabetes is a major risk factor for cardiovascular complications.")
    if "Heart Disease" in prev_conditions:
        tips.append("-   **Cardiologist Consults:** Maintain regular follow-ups with your cardiologist and adhere strictly to your prescribed medications and lifestyle regimen.")
    
    # Final general wellness tips
    tips.append("---")
    tips.append("### Overall Wellness Reminders:")
    tips.append("-   **Stress Management:** Practice relaxation techniques such as deep breathing, meditation, or yoga to help manage stress levels, which can impact BP.")
    tips.append("-   **Quality Sleep:** Aim for 7-9 hours of consistent, good-quality sleep each night. Poor sleep can contribute to high blood pressure.")
    tips.append("-   **Regular Check-ups:** Always prioritize regular visits to your healthcare provider for comprehensive health assessments and personalized advice.")
    tips.append("-   **Seek Professional Advice:** Remember, this app is for informational purposes. For any health concerns or before making changes to your treatment plan, consult a qualified medical professional.")

    return tips

if 'bp_result' in st.session_state and 'questionnaire' in st.session_state:
    st.markdown("---")
    st.subheader(":bulb: Personalized Health Tips and Recommendations")
    
    q_data = st.session_state['questionnaire']
    s_bp, d_bp = st.session_state['bp_result']
    
    # Only generate tips if BP results are valid (not None)
    if s_bp is not None and d_bp is not None:
        tips_list = generate_tips(
            q_data['age'],
            q_data['diet'],
            q_data['salt_intake'],
            q_data['exercise'],
            q_data['smoker'],
            q_data['alcohol'],
            q_data['prev_conditions'],
            s_bp,
            d_bp
        )
        
        for tip in tips_list:
            if "High Blood Pressure Alert!" in tip or "URGENT Medical Attention!" in tip:
                st.error(tip) # Use error styling for critical alerts
            elif "Low Blood Pressure Alert!" in tip or "Elevated Blood Pressure Suggestions" in tip or "High Blood Pressure (Hypertension Stage 1)" in tip:
                st.warning(tip) # Use warning for concerning but not emergency levels
            elif "Normal Blood Pressure Suggestions" in tip:
                st.success(tip) # Use success for normal range
            elif "---" in tip: # For separators in tips
                st.markdown(tip)
            else:
                st.info(tip) # Use info styling for general tips and other sections
    else:
        st.warning("BP estimation was not successful. Please ensure you've provided a clear image/video.")

st.markdown("---")
st.caption("Disclaimer: This app is for educational and demonstration purposes only and does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.")
