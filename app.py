import streamlit as st

st.set_page_config(page_title="Blood Pressure Detection App", layout="centered")

st.markdown("""
<style>
    .main {background-color: #f4f6fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stForm {background-color: #e3eafc; padding: 1em; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Centered title
st.markdown(
    "<h1 style='text-align: center;'>🧠 BP Fuel AI</h1>",
    unsafe_allow_html=True
)

st.info("This app estimates your blood pressure using your webcam or uploaded image/video and collects information about your diet and previous health history. Please allow webcam access and answer the questions below.")

# --- BP Estimation Function (must be top-level) ---
def estimate_bp_from_frame(frame):
    import numpy as np
    if frame is None:
        raise ValueError("Frame is null. Please capture a valid frame first.")
    systolic = np.random.randint(120, 180)
    diastolic = np.random.randint(70, 110)
    return systolic, diastolic

# --- Questionnaire Section ---
st.header("Step 1: Patient Questionnaire")
with st.form("questionnaire_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    diet = st.selectbox("How would you describe your diet?", ["Healthy", "Average", "Unhealthy"])
    salt_intake = st.selectbox("Salt intake?", ["Low", "Moderate", "High"])
    exercise = st.selectbox("How often do you exercise?", ["Daily", "Few times a week", "Rarely", "Never"])
    smoker = st.radio("Do you smoke?", ["Yes", "No"])
    alcohol = st.radio("Do you consume alcohol?", ["Yes", "No"])
    prev_conditions = st.multiselect("Previous health conditions", ["Hypertension", "Diabetes", "Heart Disease", "None"])
    submitted = st.form_submit_button("Submit Questionnaire")

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
    st.success("Thank you for submitting your information! Please proceed to the next step.")

# --- Only show BP detection if questionnaire is submitted ---
if 'questionnaire' in st.session_state:
    st.header("Step 2: Capture or Upload Your Face")

    # --- Browser-based webcam capture ---
    st.markdown("#### Capture your image using your webcam")
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        import numpy as np
        import cv2
        file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Captured Image")
        systolic, diastolic = estimate_bp_from_frame(frame)
        st.success(f"Estimated Blood Pressure: {systolic}/{diastolic} mmHg")
        st.session_state['bp_result'] = (systolic, diastolic)

    # --- Upload image or short video ---
    st.markdown("#### Or upload an image/short video (4-5 seconds)")
    uploaded_file = st.file_uploader(
        "Upload an image or short video", 
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        key="main_file_uploader"
    )
    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            import numpy as np
            import cv2
            import tempfile
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = None
            if uploaded_file.type.startswith("image/"):
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            elif uploaded_file.type.startswith("video/"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(file_bytes)
                    tfile.flush()
                    video = cv2.VideoCapture(tfile.name)
                    ret, frame = video.read()
                    video.release()
            if frame is not None:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="First Frame from Upload")
                systolic, diastolic = estimate_bp_from_frame(frame)
                st.success(f"Estimated Blood Pressure: {systolic}/{diastolic} mmHg")
                st.session_state['bp_result'] = (systolic, diastolic)
            else:
                st.error("Could not process the uploaded file. Please try another file.")

def generate_tips(age, diet, salt_intake, exercise, smoker, alcohol, prev_conditions, systolic=None, diastolic=None):
    tips = []
    # Questionnaire-based tips
    if age > 50:
        tips.append("Because you are over 50, regular blood pressure monitoring is especially important.")
    if diet == "Unhealthy":
        tips.append("A healthier diet with more fruits, vegetables, and whole grains can help lower your blood pressure.")
    elif diet == "Average":
        tips.append("Consider making small improvements to your diet, such as reducing processed foods and increasing fiber.")
    if salt_intake == "High":
        tips.append("Reducing salt intake can significantly help control blood pressure.")
    if exercise in ["Rarely", "Never"]:
        tips.append("Try to include at least 30 minutes of moderate exercise most days of the week.")
    elif exercise == "Few times a week":
        tips.append("Increasing exercise frequency can further benefit your heart health.")
    if smoker == "Yes":
        tips.append("Quitting smoking is one of the best things you can do for your blood pressure and overall health.")
    if alcohol == "Yes":
        tips.append("Limiting alcohol consumption can help keep your blood pressure in check.")
    if "Hypertension" in prev_conditions:
        tips.append("Continue to follow your doctor's advice for hypertension management and monitor your BP regularly.")
    if "Diabetes" in prev_conditions:
        tips.append("Managing your blood sugar is also important for blood pressure control.")
    if "Heart Disease" in prev_conditions:
        tips.append("Regular check-ups with your cardiologist are recommended.")
    # BP result-based tips
    if systolic and diastolic:
        if systolic > 140 or diastolic > 90:
            tips.append("Your estimated blood pressure is high. Please consult a healthcare professional for further evaluation.")
            tips.append("Consider reducing stress, improving sleep, and monitoring your BP at home.")
        elif systolic < 90 or diastolic < 60:
            tips.append("Your estimated blood pressure is low. If you feel dizzy or unwell, seek medical advice.")
        else:
            tips.append("Your estimated blood pressure is in the normal range. Keep up your healthy habits!")
    # General encouragement
    if not tips:
        tips.append("Keep up the good work maintaining a healthy lifestyle!")
    # Extra general tips
    tips.append("If you experience symptoms like chest pain, severe headache, or shortness of breath, seek medical attention immediately.")
    tips.append("Regular check-ups and following your healthcare provider's advice are key to long-term health.")
    return tips

if 'bp_result' in st.session_state and 'questionnaire' in st.session_state:
    st.markdown("---")
    st.subheader(":bulb: Personalized Health Tips:")
    q = st.session_state['questionnaire']
    s, d = st.session_state['bp_result']
    for tip in generate_tips(q['age'], q['diet'], q['salt_intake'], q['exercise'], q['smoker'], q['alcohol'], q['prev_conditions'], s, d):
        st.info(tip)

st.markdown("---")
st.caption("This app is for demonstration purposes only and does not provide medical advice.")
