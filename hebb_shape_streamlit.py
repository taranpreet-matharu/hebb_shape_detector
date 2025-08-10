import streamlit as st
import cv2 as cv
import numpy as np
import os
import pandas as pd
import common
from PIL import Image
from io import BytesIO
import base64
import textwrap

st.header("Shape Detector - Hebb")

INPUT_SHAPE = (50, 50)
RECTANGLE = "Rectangle"
TRIANGLE = "Triangle"

# -----------------------
# Session state defaults
# -----------------------
if "training_results" not in st.session_state:
    st.session_state.training_results = {
        "weights": np.full(INPUT_SHAPE, 0),
        "biasWeight": 0,
    }


# -----------------------
# Utility functions
# -----------------------
def pass_from_net(matrix, output, prev_weights, prev_bias_weight):
    delta = matrix * output
    weights = prev_weights + delta
    bias_weight = prev_bias_weight + (1 * output)
    return {"weights": weights, "biasWeight": bias_weight}


def train_net(dir_path, output):
    for file_name in os.listdir(dir_path):
        image = cv.imread(os.path.join(dir_path, file_name), cv.IMREAD_GRAYSCALE)
        matrix = np.where(np.asarray(image, dtype="int64") == 255, -1, 1)
        st.session_state.training_results = pass_from_net(
            matrix,
            output,
            st.session_state.training_results["weights"],
            st.session_state.training_results["biasWeight"],
        )


def get_shape(value):
    return RECTANGLE if value == 1 else TRIANGLE


def image_to_html(img_path, width=50):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_b64}" width="{width}"/>'


def test_net(dir_path, actual_shape):
    results = []
    for file_name in os.listdir(dir_path):
        image = cv.imread(os.path.join(dir_path, file_name), cv.IMREAD_GRAYSCALE)
        matrix = np.where(np.asarray(image, dtype="int64") == 255, -1, 1)
        weights = st.session_state.training_results["weights"]
        bias_weight = st.session_state.training_results["biasWeight"]
        output_from_net = common.calculate_output(matrix, weights, bias_weight)
        result = common.calc_bipolar_activation(output_from_net)
        result_shape = get_shape(result)

        if actual_shape is None:
            file_name_without_ext = file_name.rsplit(".", 1)[0]
            actual_shape_calc = (
                RECTANGLE
                if int(file_name_without_ext.rsplit("_", 1)[1]) == 1
                else TRIANGLE
            )
        else:
            actual_shape_calc = actual_shape

        results.append(
            {
                "image": image,
                "imageHTML": image_to_html(os.path.join(dir_path, file_name)),
                "output": output_from_net,
                "result": result,
                "shape": result_shape,
                "actual_shape": actual_shape_calc,
                "correct": result_shape == actual_shape_calc,
            }
        )
    return results


def render_table(df, title):
    table_html = textwrap.dedent(
        """
        <table style="width:100%; border-collapse: collapse;" border="1">
            <tr>
                <th style="padding: 8px;">Image</th>
                <th style="padding: 8px;">Output from Net</th>
                <th style="padding: 8px;">Result</th>
                <th style="padding: 8px;">Shape</th>
            </tr>
    """
    )
    for _, row in df.iterrows():
        table_html += textwrap.dedent(
            f"""
            <tr>
                <td style="padding: 8px;">{row['Image']}</td>
                <td style="padding: 8px;">{row['Output']}</td>
                <td style="padding: 8px;">{row['Result']}</td>
                <td style="padding: 8px;">{row['Shape']}</td>
            </tr>
        """
        )
    table_html += "</table>"
    st.subheader(title)
    st.markdown(table_html, unsafe_allow_html=True)


def display_results(results, total_r=None, total_t=None):
    # split correct / incorrect
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    c_df = pd.DataFrame(
        {
            "Image": [x["imageHTML"] for x in correct],
            "Output": [x["output"] for x in correct],
            "Result": [x["result"] for x in correct],
            "Shape": [x["shape"] for x in correct],
        }
    )

    i_df = pd.DataFrame(
        {
            "Image": [x["imageHTML"] for x in incorrect],
            "Output": [x["output"] for x in incorrect],
            "Result": [x["result"] for x in incorrect],
            "Shape": [x["shape"] for x in incorrect],
        }
    )

    total = len(results)
    if total_r is None:
        total_r = len([x for x in results if x["actual_shape"] == RECTANGLE])
    if total_t is None:
        total_t = len([x for x in results if x["actual_shape"] == TRIANGLE])

    r_c = len([x for x in correct if x["actual_shape"] == RECTANGLE])
    r_i = len([x for x in incorrect if x["actual_shape"] == RECTANGLE])

    overall_stats = pd.DataFrame(
        {
            "Shape": ["Overall", RECTANGLE, TRIANGLE],
            "Total Count": [total, total_r, total_t],
            "❌ Count": [len(incorrect), r_i, len(incorrect) - r_i],
            "✅ Count": [len(correct), r_c, len(correct) - r_c],
            "❌ %": [
                f"{(len(incorrect) / total) * 100:.2f}%",
                f"{(r_i / total_r) * 100:.2f}%" if total_r else "N/A",
                (
                    f"{((len(incorrect) - r_i) / total_t) * 100:.2f}%"
                    if total_t
                    else "N/A"
                ),
            ],
            "✅ %": [
                f"{(len(correct) / total) * 100:.2f}%",
                f"{(r_c / total_r) * 100:.2f}%" if total_r else "N/A",
                f"{((len(correct) - r_c) / total_t) * 100:.2f}%" if total_t else "N/A",
            ],
        }
    )

    st.table(overall_stats)
    render_table(i_df, "Incorrectly classified images")
    render_table(c_df, "Correctly classified images")


# -----------------------
# UI - Controls (container 1)
# -----------------------
controls_container = st.container()
with controls_container:
    col1, col2 = st.columns(2)

    with col1:
        train_clicked = st.button("Train Neural Net", key="btn_train")
        test_clicked = st.button("Test Neural Net with Trained Data", key="btn_test")

    with col2:
        predict_clicked = st.button("Predict Shapes", key="btn_predict")
        clear_clicked = st.button("Clear Data", key="btn_clear", type="primary")


# -----------------------
# Results container (container 2)
# -----------------------
results_container = st.container()
results_slot = results_container.empty()  # dedicate a slot below the controls

# Clear data if requested
if clear_clicked:
    results_slot.empty()

# Train
if train_clicked:
    results_slot.empty()
    with results_slot.container():
        with st.spinner("Training neural network..."):
            train_net(os.path.join("shapes", "triangles"), -1)
            st.write("Network trained for triangles")
            train_net(os.path.join("shapes", "rectangles"), 1)
            st.write("Network trained for rectangles")
        st.success("Training complete.")

# Test with trained data
if test_clicked:
    results_slot.empty()
    rect_results = test_net(os.path.join("shapes", "rectangles"), RECTANGLE)
    tri_results = test_net(os.path.join("shapes", "triangles"), TRIANGLE)
    with results_slot.container():
        display_results(rect_results + tri_results)

# Predict (mixed)
if predict_clicked:
    results_slot.empty()
    results = test_net(os.path.join("shapes", "mixed"), None)
    with results_slot.container():
        # pass totals if you want specific totals (optional)
        display_results(results, total_r=91, total_t=109)
