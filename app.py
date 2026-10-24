 
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Smart Quality Inspection AI",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Smart Quality Inspection AI")
st.subheader("DRL Mini Project | Intelligent Factory Inspection")
st.write(
    "Inspect products using computer vision, compare them with a "
    "known-good reference, and highlight suspicious visual differences."
)

# ---------------- SIDEBAR SETTINGS ----------------

with st.sidebar:
    st.header("Inspection Settings")

    product_name = st.text_input(
        "Product Name",
        "PCB Product"
    )

    sensitivity = st.slider(
        "Difference Sensitivity",
        min_value=5,
        max_value=100,
        value=25
    )

    min_area = st.slider(
        "Minimum Suspicious Area",
        min_value=20,
        max_value=2000,
        value=120
    )

    st.info(
        "Use images of the same product with similar lighting "
        "and camera position."
    )


# ---------------- IMAGE LOADING ----------------

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


# ---------------- IMAGE ALIGNMENT ----------------

def align_images(reference, test):
    reference_gray = cv2.cvtColor(
        reference, cv2.COLOR_RGB2GRAY
    )

    test_gray = cv2.cvtColor(
        test, cv2.COLOR_RGB2GRAY
    )

    orb = cv2.ORB_create(nfeatures=3000)

    kp1, des1 = orb.detectAndCompute(reference_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    if des1 is None or des2 is None:
        return test, False

    if len(kp1) < 8 or len(kp2) < 8:
        return test, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(des2, des1, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    if len(good_matches) < 8:
        return test, False

    source_points = np.float32([
        kp2[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    reference_points = np.float32([
        kp1[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        5.0
    )

    if matrix is None:
        return test, False

    height, width = reference.shape[:2]

    aligned = cv2.warpPerspective(
        test,
        matrix,
        (width, height)
    )

    return aligned, True


# ---------------- DEFECT DIFFERENCE ANALYSIS ----------------

def inspect_product(reference, test, sensitivity, min_area):
    height, width = reference.shape[:2]

    test = cv2.resize(test, (width, height))

    aligned, alignment_success = align_images(
        reference,
        test
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_RGB2GRAY
    )

    test_gray = cv2.cvtColor(
        aligned,
        cv2.COLOR_RGB2GRAY
    )

    reference_gray = cv2.GaussianBlur(
        reference_gray, (5, 5), 0
    )

    test_gray = cv2.GaussianBlur(
        test_gray, (5, 5), 0
    )

    difference = cv2.absdiff(
        reference_gray,
        test_gray
    )

    _, mask = cv2.threshold(
        difference,
        sensitivity,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8)
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = aligned.copy()
    suspicious_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            x, y, box_width, box_height = (
                cv2.boundingRect(contour)
            )

            suspicious_regions.append({
                "X position": x,
                "Y position": y,
                "Width": box_width,
                "Height": box_height,
                "Area": int(area)
            })

            # Draw red box around the visual difference
            cv2.rectangle(
                annotated,
                (x, y),
                (x + box_width, y + box_height),
                (255, 0, 0),
                3
            )

            cv2.putText(
                annotated,
                "CHECK",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    return (
        aligned,
        difference,
        mask,
        annotated,
        suspicious_regions,
        alignment_success
    )


# ---------------- DASHBOARD ----------------

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("---")
st.header("Product Image Inspection")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Good Reference Product")

    reference_file = st.file_uploader(
        "Upload a known-good product image",
        type=["jpg", "jpeg", "png"],
        key="reference"
    )

with col2:
    st.subheader("2. Product to Inspect")

    test_file = st.file_uploader(
        "Upload the product image to inspect",
        type=["jpg", "jpeg", "png"],
        key="test"
    )

if reference_file:
    st.image(
        load_image(reference_file),
        caption="Good reference image",
        use_container_width=True
    )

if test_file:
    st.image(
        load_image(test_file),
        caption="Product to inspect",
        use_container_width=True
    )


# ---------------- RUN INSPECTION ----------------

if st.button(
    "🔍 Run Quality Inspection",
    type="primary",
    use_container_width=True
):

    if reference_file is None or test_file is None:
        st.warning(
            "Please upload BOTH images before starting inspection."
        )

    else:
        try:
            reference = load_image(reference_file)
            test = load_image(test_file)

            with st.spinner("Inspecting product images..."):

                (
                    aligned,
                    difference,
                    mask,
                    annotated,
                    regions,
                    alignment_success
                ) = inspect_product(
                    reference,
                    test,
                    sensitivity,
                    min_area
                )

            st.markdown("---")
            st.header("Inspection Result")

            if regions:
                st.error(
                    f"⚠️ {len(regions)} suspicious visual "
                    "region(s) detected."
                )

            else:
                st.success(
                    "No large visual differences detected "
                    "at the selected sensitivity."
                )

            if not alignment_success:
                st.warning(
                    "Automatic image alignment was unsuccessful. "
                    "The images were compared at the same size. "
                    "Position differences may affect the result."
                )

            st.info(
                "Red boxes indicate visual differences, not "
                "confirmed manufacturing defects. Shadows, lighting, "
                "and product movement can also trigger alerts."
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.subheader("Suspicious Areas")

                st.image(
                    annotated,
                    caption="Red boxes show areas to review",
                    use_container_width=True
                )

            with result_col2:
                st.subheader("Difference Mask")

                st.image(
                    mask,
                    caption="White areas indicate image differences",
                    use_container_width=True
                )

            if regions:
                st.subheader("Detected Region Details")

                st.dataframe(
                    pd.DataFrame(regions),
                    use_container_width=True,
                    hide_index=True
                )

            with st.expander("View pixel difference image"):
                st.image(
                    difference,
                    caption="Grayscale difference map",
                    use_container_width=True
                )

            if regions:
                status = "Review required"
            else:
                status = "No large visual differences"

            st.session_state.history.append({
                "Product": product_name,
                "Inspection Result": status,
                "Suspicious Regions": len(regions)
            })

        except Exception as error:
            st.error(f"Inspection error: {error}")


# ---------------- INSPECTION STATISTICS ----------------

st.markdown("---")
st.header("Inspection Dashboard")

history = st.session_state.history

total = len(history)

flagged = sum(
    item["Inspection Result"] == "Review required"
    for item in history
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Products Inspected", total)
metric2.metric("Products Flagged", flagged)
metric3.metric("Confirmed Defects", "Not verified")

if history:
    st.subheader("Product Inspection Table")

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Inspection Report",
        history_df.to_csv(index=False).encode("utf-8"),
        file_name="inspection_report.csv",
        mime="text/csv"
    )


# ---------------- DRL PROJECT EXPLANATION ----------------

with st.expander("How this fits into the DRL project"):

    st.write(
        "The computer-vision stage compares the product with a "
        "known-good reference and identifies suspicious areas."
    )

    st.write(
        "A future Deep Reinforcement Learning agent can use "
        "inspection outcomes as observations and learn actions "
        "such as inspect again, send for human review, or accept "
        "a product, using suitable rewards and penalties."
    )

    st.warning(
        "The current code performs computer-vision comparison. "
        "It does not train a DRL agent or classify every defect type."
    )
