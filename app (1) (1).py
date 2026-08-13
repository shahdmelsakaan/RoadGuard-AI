import streamlit as st
import tensorflow as tf
import cv2
import numpy as np


# =========================
# Page Settings
# =========================

st.set_page_config(
    page_title="Drive Safety",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Drive Safety")
st.subheader("Driver Drowsiness Detection System")


# =========================
# Load Models
# =========================

@st.cache_resource
def load_models():

    eye_model = tf.keras.models.load_model(
        "eye_model_deploy.keras",
        safe_mode=False
    )

    mouth_model = tf.keras.models.load_model(
        "mouth_model.keras",
        safe_mode=False
    )

    return eye_model, mouth_model


@st.cache_resource
def load_face_detector():

    detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    return detector


# =========================
# Load Everything
# =========================

try:

    eye_model, mouth_model = load_models()
    face_detector = load_face_detector()

except Exception as e:

    st.error("❌ Error loading the models.")
    st.code(str(e))
    st.stop()


# =========================
# Face Detection
# =========================

def detect_face(frame):

    height, width = frame.shape[:2]

    face_detector.setInputSize(
        (width, height)
    )

    _, faces = face_detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    # Get largest face
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    return face


# =========================
# Extract Eye and Mouth
# =========================

def extract_regions(frame, face):

    x, y, w, h = face[:4].astype(int)

    # YuNet landmarks
    right_eye = face[4:6]
    left_eye = face[6:8]

    right_mouth = face[10:12]
    left_mouth = face[12:14]


    # -------------------------
    # Eye region
    # -------------------------

    eye_x1 = int(
        min(right_eye[0], left_eye[0])
        - 0.35 * w
    )

    eye_x2 = int(
        max(right_eye[0], left_eye[0])
        + 0.35 * w
    )

    eye_y1 = int(
        min(right_eye[1], left_eye[1])
        - 0.15 * h
    )

    eye_y2 = int(
        max(right_eye[1], left_eye[1])
        + 0.25 * h
    )


    # -------------------------
    # Mouth region
    # -------------------------

    mouth_x1 = int(
        min(right_mouth[0], left_mouth[0])
        - 0.20 * w
    )

    mouth_x2 = int(
        max(right_mouth[0], left_mouth[0])
        + 0.20 * w
    )

    mouth_y1 = int(
        min(right_mouth[1], left_mouth[1])
        - 0.20 * h
    )

    mouth_y2 = int(
        max(right_mouth[1], left_mouth[1])
        + 0.35 * h
    )


    # Keep inside image
    eye_x1 = max(0, eye_x1)
    eye_y1 = max(0, eye_y1)

    eye_x2 = min(frame.shape[1], eye_x2)
    eye_y2 = min(frame.shape[0], eye_y2)

    mouth_x1 = max(0, mouth_x1)
    mouth_y1 = max(0, mouth_y1)

    mouth_x2 = min(frame.shape[1], mouth_x2)
    mouth_y2 = min(frame.shape[0], mouth_y2)


    # Crop
    eye_crop = frame[
        eye_y1:eye_y2,
        eye_x1:eye_x2
    ]

    mouth_crop = frame[
        mouth_y1:mouth_y2,
        mouth_x1:mouth_x2
    ]


    return (
        eye_crop,
        mouth_crop,
        (eye_x1, eye_y1, eye_x2, eye_y2),
        (mouth_x1, mouth_y1, mouth_x2, mouth_y2)
    )


# =========================
# Eye Prediction
# =========================

def predict_eye(eye_crop):

    eye_crop = cv2.resize(
        eye_crop,
        (64, 64)
    )

    # Deployment model expects 3 channels
    if len(eye_crop.shape) == 2:

        eye_crop = cv2.cvtColor(
            eye_crop,
            cv2.COLOR_GRAY2RGB
        )

    else:

        eye_crop = cv2.cvtColor(
            eye_crop,
            cv2.COLOR_BGR2RGB
        )


    eye_crop = eye_crop.astype(
        np.float32
    )

    eye_crop = eye_crop / 255.0

    eye_crop = np.expand_dims(
        eye_crop,
        axis=0
    )


    prediction = eye_model.predict(
        eye_crop,
        verbose=0
    )[0][0]


    if prediction >= 0.5:

        state = "Open"

    else:

        state = "Closed"


    return state, float(prediction)


# =========================
# Mouth Prediction
# =========================

def predict_mouth(mouth_crop):

    mouth_crop = cv2.resize(
        mouth_crop,
        (64, 64)
    )


    if len(mouth_crop.shape) == 2:

        mouth_crop = cv2.cvtColor(
            mouth_crop,
            cv2.COLOR_GRAY2RGB
        )

    else:

        mouth_crop = cv2.cvtColor(
            mouth_crop,
            cv2.COLOR_BGR2RGB
        )


    mouth_crop = mouth_crop.astype(
        np.float32
    )

    mouth_crop = mouth_crop / 255.0

    mouth_crop = np.expand_dims(
        mouth_crop,
        axis=0
    )


    prediction = mouth_model.predict(
        mouth_crop,
        verbose=0
    )[0][0]


    if prediction >= 0.5:

        state = "Yawn"

    else:

        state = "No Yawn"


    return state, float(prediction)


# =========================
# Final Decision
# =========================

def get_final_result(
    eye_state,
    mouth_state
):

    if (
        eye_state == "Closed"
        and mouth_state == "Yawn"
    ):

        return "DROWSY"


    elif eye_state == "Closed":

        return "POSSIBLE DROWSINESS"


    elif mouth_state == "Yawn":

        return "POSSIBLE DROWSINESS"


    else:

        return "ALERT"


# =========================
# Input
# =========================

st.divider()

st.write("### Choose Input")

input_method = st.radio(
    "Input method:",
    [
        "📷 Camera",
        "📁 Upload Image"
    ]
)


image_file = None


if input_method == "📷 Camera":

    image_file = st.camera_input(
        "Take a picture"
    )

else:

    image_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# =========================
# Process Image
# =========================

if image_file is not None:

    file_bytes = np.asarray(
        bytearray(image_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "❌ Could not read the image."
        )

        st.stop()


    # Detect face
    face = detect_face(image)


    if face is None:

        st.warning(
            "⚠️ No face detected. "
            "Please make sure your face is clearly visible."
        )

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Input Image"
        )

        st.stop()


    # Extract regions
    (
        eye_crop,
        mouth_crop,
        eye_box,
        mouth_box
    ) = extract_regions(
        image,
        face
    )


    # Predictions
    eye_state, eye_score = predict_eye(
        eye_crop
    )

    mouth_state, mouth_score = predict_mouth(
        mouth_crop
    )


    # Final result
    final_result = get_final_result(
        eye_state,
        mouth_state
    )


    # =========================
    # Draw Results
    # =========================

    output_image = image.copy()


    # Face box
    x, y, w, h = face[:4].astype(int)

    cv2.rectangle(
        output_image,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        2
    )


    # Eye box
    ex1, ey1, ex2, ey2 = eye_box

    cv2.rectangle(
        output_image,
        (ex1, ey1),
        (ex2, ey2),
        (0, 255, 0),
        2
    )


    # Mouth box
    mx1, my1, mx2, my2 = mouth_box

    cv2.rectangle(
        output_image,
        (mx1, my1),
        (mx2, my2),
        (0, 0, 255),
        2
    )


    # =========================
    # Results
    # =========================

    st.divider()

    st.subheader("🔍 Detection Result")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Eye Status",
            eye_state
        )


    with col2:

        st.metric(
            "Mouth Status",
            mouth_state
        )


    with col3:

        st.metric(
            "Final Result",
            final_result
        )


    # Warning
    if final_result == "DROWSY":

        st.error(
            "🚨 DROWSY — Driver may be drowsy!"
        )

    elif final_result == "POSSIBLE DROWSINESS":

        st.warning(
            "⚠️ POSSIBLE DROWSINESS"
        )

    else:

        st.success(
            "✅ ALERT — Driver appears alert."
        )


    # =========================
    # Display Images
    # =========================

    st.divider()

    col1, col2, col3 = st.columns(3)


    with col1:

        st.image(
            cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Detection",
            use_container_width=True
        )


    with col2:

        st.image(
            cv2.cvtColor(
                eye_crop,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Eye: {eye_state}",
            use_container_width=True
        )


    with col3:

        st.image(
            cv2.cvtColor(
                mouth_crop,
                cv2.COLOR_BGR2RGB
            ),
            caption=f"Mouth: {mouth_state}",
            use_container_width=True
        )


    # =========================
    # Scores
    # =========================

    st.divider()

    st.subheader("📊 Prediction Scores")


    score_col1, score_col2 = st.columns(2)


    with score_col1:

        st.write(
            f"Eye Score: `{eye_score:.4f}`"
        )


    with score_col2:

        st.write(
            f"Mouth Score: `{mouth_score:.4f}`"
        )


# =========================
# Footer
# =========================

st.divider()

st.caption(
    "Drive Safety Project — Driver Drowsiness Detection"
)