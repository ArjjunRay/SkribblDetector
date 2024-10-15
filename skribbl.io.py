import streamlit as st
st.write("Hello World")

from streamlit_drawable_canvas import st_canvas

# Specify canvas parameters in application

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

stroke_color = st.sidebar.color_picker("Stroke color hex: ")

bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component

canvas_result = st_canvas(

    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_color=bg_color,

    update_streamlit=realtime_update,

    height=500,

    drawing_mode="freedraw",

    point_display_radius=0,

    key="canvas",

)
import numpy as np
from week2_code import Neuron # Adjust the import based on your file structure

def test_neuron_forward():
    # Test case 1: Known weights and bias
    num_inputs = 3
    neuron = Neuron(num_inputs)
    neuron.weights = np.array([0.5, -0.2, 0.1])
    neuron.bias = 0.4
    inputs = np.array([1.0, 2.0, 3.0])
    expected_output = np.dot(neuron.weights, inputs) + neuron.bias
    output = neuron.forward(inputs)
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test case 2: Zero weights and bias
    neuron.weights = np.zeros(num_inputs)
    neuron.bias = 0.0
    inputs = np.array([1.0, 2.0, 3.0])
    expected_output = 0.0
    output = neuron.forward(inputs)
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    # Test case 3: Random weights and bias
    neuron.weights = np.random.randn(num_inputs)
    neuron.bias = np.random.randn()
    inputs = np.random.randn(num_inputs)
    expected_output = np.dot(neuron.weights, inputs) + neuron.bias
    output = neuron.forward(inputs)
    assert np.isclose(output, expected_output), f"Expected {expected_output}, but got {output}"

if __name__ == "__main__":
    test_neuron_forward()
    print("All tests passed!")