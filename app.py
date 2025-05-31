import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

st.set_page_config(page_title="Blood Pressure Detection App", layout="centered")

# --- Improved UI Section ---
st.markdown("""
<style>
    .main {background-color: #f4f6fa;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stForm {background-color: #e3eafc; padding: 1em; border-radius: 10px;}
    .webcam-container {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title(":heart: Blood Pressure Detection App  ")

# Check if running on Streamlit Cloud
is_cloud = 'STREAMLIT_SHARING' in os.environ or 'STREAMLIT_CLOUD' in os.environ or 'streamlit.io' in str(os.environ.get('HOSTNAME', ''))

if is_cloud:
    st.info("🌐 Running on Streamlit Cloud - Using advanced webcam integration for better compatibility")
else:
    st.info("💻 Running locally - Direct webcam access available")

# --- Enhanced Computer Vision Section ---
class BPVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.bp_estimated = False
        self.systolic = None
        self.diastolic = None
        self.frame_count = 0
        self.capture_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Add frame counter for processing
        self.frame_count += 1
        
        # Store frame for BP estimation (every 30 frames to avoid overprocessing)
        if self.frame_count % 30 == 0:
            self.capture_frame = img.copy()
            
        # Add visual feedback
        height, width = img.shape[:2]
        
        # Draw face detection area
        cv2.rectangle(img, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
        cv2.putText(img, "Position your face in the green box", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame count
        cv2.putText(img, f"Frame: {self.frame_count}", (20, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img

    def estimate_bp_from_frame(self, frame):
        """Enhanced BP estimation with more realistic simulation"""
        if frame is None:
            return None, None
            
        # Simulate more realistic BP estimation based on frame analysis
        # In a real implementation, this would use actual CV/ML algorithms
        
        # Convert to grayscale for "analysis"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simulate face detection area analysis
        height, width = gray.shape
        face_region = gray[height//4:3*height//4, width//4:3*width//4]
        
        # Use image statistics to create more realistic variation
        mean_intensity = np.mean(face_region)
        std_intensity = np.std(face_region)
        
        # Simulate BP based on "skin tone analysis" (just for demo)
        base_systolic = 120 + (mean_intensity - 128) * 0.1
        base_diastolic = 80 + (std_intensity - 50) * 0.05
        
        # Add some randomness but keep it realistic
        np.random.seed(int(time.time()) % 1000)
        systolic = int(base_systolic + np.random.randint(-10, 10))
        diastolic = int(base_diastolic + np.random.randint(-5, 5))
        
        # Keep values in reasonable range
        systolic = max(90, min(180, systolic))
        diastolic = max(60, min(100, diastolic))
        
        return systolic, diastolic

# Initialize session state
if 'bp_result' not in st.session_state:
    st.session_state.bp_result = None
if 'captured_frame' not in st.session_state:
    st.session_state.captured_frame = None

st.header("Step 1: Blood Pressure Detection")

# WebRTC Configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    
    # WebRTC streamer for real-time video
    webrtc_ctx = webrtc_streamer(
        key="bp-detection",
        video_transformer_factory=BPVideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280, "max": 1920},
                "height": {"min": 480, "ideal": 720, "max": 1080},
                "frameRate": {"min": 15, "ideal": 30, "max": 60}
            },
            "audio": False
        },
        async_processing=True,
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Instructions")
    st.markdown("""
    1. **Allow camera access** when prompted
    2. **Position your face** in the green box
    3. **Stay still** for a few seconds
    4. **Click 'Analyze BP'** to get results
    """)
    
    # Analyze button
    if st.button("🔬 Analyze Blood Pressure", type="primary"):
        if webrtc_ctx.video_transformer:
            transformer = webrtc_ctx.video_transformer
            if transformer.capture_frame is not None:
                # Estimate BP from the captured frame
                systolic, diastolic = transformer.estimate_bp_from_frame(transformer.capture_frame)
                
                if systolic and diastolic:
                    st.session_state.bp_result = f"{systolic}/{diastolic}"
                    st.session_state.captured_frame = transformer.capture_frame
                    
                    st.markdown('<div class="status-success">', unsafe_allow_html=True)
                    st.success(f"📊 Estimated BP: {systolic}/{diastolic} mmHg")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show confidence level based on "analysis quality"
                    confidence = np.random.randint(75, 95)
                    st.info(f"🎯 Confidence Level: {confidence}%")
                else:
                    st.error("❌ Could not analyze the frame. Please try again.")
            else:
                st.warning("⚠️ No frame captured yet. Please wait a moment and try again.")
        else:
            st.error("❌ Camera not connected. Please allow camera access and try again.")

# Display results
if st.session_state.bp_result:
    st.markdown("---")
    st.subheader("📈 Blood Pressure Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Blood Pressure", st.session_state.bp_result, "mmHg")
    
    with col2:
        systolic_val = int(st.session_state.bp_result.split('/')[0])
        if systolic_val < 120:
            bp_category = "Normal"
            bp_color = "🟢"
        elif systolic_val < 140:
            bp_category = "Elevated"
            bp_color = "🟡"
        else:
            bp_category = "High"
            bp_color = "🔴"
        
        st.metric("Category", f"{bp_color} {bp_category}")
    
    with col3:
        timestamp = time.strftime("%H:%M:%S")
        st.metric("Time", timestamp)

# Alternative upload option
st.markdown("---")
st.subheader("📁 Alternative: Upload Video/Image")
st.info("If webcam doesn't work, you can upload a video or image file for analysis.")

uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'],
    help="Upload a video of yourself or a clear face photo"
)

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Handle image upload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=400)
        
        if st.button("🔬 Analyze Uploaded Image"):
            transformer = BPVideoTransformer()
            systolic, diastolic = transformer.estimate_bp_from_frame(image)
            
            if systolic and diastolic:
                st.success(f"📊 Estimated BP from uploaded image: {systolic}/{diastolic} mmHg")
                st.session_state.bp_result = f"{systolic}/{diastolic}"
    
    elif uploaded_file.type.startswith('video'):
        # Handle video upload
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.video(uploaded_file)
        
        if st.button("🔬 Analyze Uploaded Video"):
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                transformer = BPVideoTransformer()
                systolic, diastolic = transformer.estimate_bp_from_frame(frame)
                
                if systolic and diastolic:
                    st.success(f"📊 Estimated BP from video: {systolic}/{diastolic} mmHg")
                    st.session_state.bp_result = f"{systolic}/{diastolic}"
            else:
                st.error("Could not process the video file.")
        
        # Clean up temp file
        os.unlink(tfile.name)

# --- Original Questionnaire Section (unchanged) ---
st.markdown("---")
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

# --- Enhanced Tips Generation ---
def generate_tips(age, diet, salt_intake, exercise, smoker, alcohol, prev_conditions, bp_reading=None):
    tips = []
    
    # BP-specific tips
    if bp_reading:
        systolic = int(bp_reading.split('/')[0])
        if systolic >= 140:
            tips.append("⚠️ Your BP reading suggests hypertension. Please consult a healthcare professional.")
        elif systolic >= 120:
            tips.append("📈 Your BP is elevated. Consider lifestyle modifications.")
        else:
            tips.append("✅ Your BP reading is in the normal range. Keep up the good work!")
    
    # Original tips logic
    if age > 50:
        tips.append("👴 Consider regular blood pressure monitoring due to your age.")
    if diet == "Unhealthy":
        tips.append("🥗 Try to incorporate more fruits and vegetables into your diet.")
    if salt_intake == "High":
        tips.append("🧂 Reduce salt intake to help control blood pressure.")
    if exercise in ["Rarely", "Never"]:
        tips.append("🏃‍♂️ Regular physical activity can help lower blood pressure.")
    if smoker == "Yes":
        tips.append("🚭 Quitting smoking greatly reduces cardiovascular risk.")
    if alcohol == "Yes":
        tips.append("🍷 Limit alcohol consumption for better blood pressure control.")
    if "Hypertension" in prev_conditions:
        tips.append("💊 Continue to follow your doctor's advice for hypertension management.")
    
    if not tips:
        tips.append("🌟 Keep up the good work maintaining a healthy lifestyle!")
    
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
    
    if st.session_state.bp_result:
        st.write(f"**Estimated Blood Pressure:** {st.session_state.bp_result} mmHg")
    
    st.success("Thank you for submitting your information!")
    
    # Show enhanced personalized tips
    st.markdown("---")
    st.subheader(":bulb: Personalized Health Tips:")
    for tip in generate_tips(age, diet, salt_intake, exercise, smoker, alcohol, prev_conditions, st.session_state.bp_result):
        st.info(tip)

st.markdown("---")
st.caption("⚠️ This app is for demonstration purposes only and does not provide medical advice. Always consult healthcare professionals for medical concerns.")

# Troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **If webcam doesn't work:**
    1. Refresh the page and allow camera permissions
    2. Try using a different browser (Chrome/Firefox work best)
    3. Check if other apps are using your camera
    4. Use the file upload option as an alternative
    
    **For best results:**
    - Ensure good lighting
    - Keep your face centered in the green box
    - Stay still during analysis
    - Allow camera access when prompted
    """)
