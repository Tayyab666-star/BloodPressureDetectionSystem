import streamlit as st
import cv2
import numpy as np
import tempfile
import time

st.set_page_config(page_title="Blood Pressure Detection App", layout="centered")

# --- Improved UI Section ---
st.markdown("""
<style>
    .main {background-color: #f4f6fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stForm {background-color: #e3eafc; padding: 1em; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title(":heart: Blood Pressure Detection using Computer Vision and Questionnaire")

st.info("This app estimates your blood pressure using your webcam and collects information about your diet and previous health history. Please allow webcam access and answer the questions below.")

# --- Computer Vision Section ---
def get_webcam_frame():
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Let camera warm up
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

def estimate_bp_from_frame(frame):
    # Placeholder: In real scenario, use a trained model or algorithm
    # Here, we simulate BP estimation using random values
    np.random.seed(int(time.time()))
    systolic = np.random.randint(110, 140)
    diastolic = np.random.randint(70, 90)
    return systolic, diastolic

st.header("Step 1: Capture Your Face")
frame = None
if st.button("📸 Capture from Webcam"):
    frame = get_webcam_frame()
    if frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Captured Image")
        systolic, diastolic = estimate_bp_from_frame(frame)
        st.success(f"Estimated Blood Pressure: {systolic}/{diastolic} mmHg")
    else:
        st.error("Could not access webcam. Please try again.")
else:
    st.warning("Click the button to capture an image from your webcam.")

# --- Questionnaire Section ---
st.header("Step 2: Patient Questionnaire")
with st.form("questionnaire_form"):
    st.markdown(":pencil: **Please fill out the following details:**")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    diet = st.selectbox("How would you describe your diet?", ["Healthy", "Average", "Unhealthy"])
    salt_intake = st.selectbox("Salt intake?", ["Low", "Moderate", "High"])
    exercise = st.selectbox("How often do you exercise?", ["Daily", "Few times a week", "Rarely", "Never"])
    smoker = st.radio("Do you smoke?", ["Yes", "No"])
    alcohol = st.radio("Do you consume alcohol?", ["Yes", "No"])
    prev_conditions = st.multiselect("Previous health conditions", ["Hypertension", "Diabetes", "Heart Disease", "None"])
    submitted = st.form_submit_button("Submit Questionnaire")

# --- Few-Shot Prompting for Personalized Tips ---
def generate_tips(age, diet, salt_intake, exercise, smoker, alcohol, prev_conditions):
    tips = []
    # Few-shot prompting logic (simple rule-based for demo)
    if age > 50:
        tips.append("Consider regular blood pressure monitoring due to your age.")
    if diet == "Unhealthy":
        tips.append("Try to incorporate more fruits and vegetables into your diet.")
    if salt_intake == "High":
        tips.append("Reduce salt intake to help control blood pressure.")
    if exercise in ["Rarely", "Never"]:
        tips.append("Regular physical activity can help lower blood pressure.")
    if smoker == "Yes":
        tips.append("Quitting smoking greatly reduces cardiovascular risk.")
    if alcohol == "Yes":
        tips.append("Limit alcohol consumption for better blood pressure control.")
    if "Hypertension" in prev_conditions:
        tips.append("Continue to follow your doctor's advice for hypertension management.")
    if not tips:
        tips.append("Keep up the good work maintaining a healthy lifestyle!")
    return tips

if submitted:
    st.subheader(":clipboard: Summary of Your Answers:")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Diet:** {diet}")
    st.write(f"**Salt Intake:** {salt_intake}")
    st.write(f"**Exercise Frequency:** {exercise}")
    st.write(f"**Smoker:** {smoker}")
    st.write(f"**Alcohol Consumption:** {alcohol}")
    st.write(f"**Previous Conditions:** {', '.join(prev_conditions) if prev_conditions else 'None'}")
    st.success("Thank you for submitting your information!")
    # Show personalized tips
    st.markdown("---")
    st.subheader(":bulb: Personalized Health Tips:")
    for tip in generate_tips(age, diet, salt_intake, exercise, smoker, alcohol, prev_conditions):
        st.info(tip)

st.markdown("---")
st.caption("This app is for demonstration purposes only and does not provide medical advice.")
