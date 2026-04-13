import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(page_title="ML Visualizer", layout="wide")

st.title("🧠 ML Foundations Visualizer")

# Sidebar Navigation
module = st.sidebar.selectbox("Choose Module", [
    "Vector Similarity",
    "Function & Derivative",
    "Gradient Descent"
])

# ==============================
# MODULE 1: VECTOR SIMILARITY
# ==============================
if module == "Vector Similarity":
    st.header("📐 Vector Similarity")

    v1 = np.array([
        st.number_input("v1 x", value=1),
        st.number_input("v1 y", value=2)
    ])

    v2 = np.array([
        st.number_input("v2 x", value=2),
        st.number_input("v2 y", value=1)
    ])

    dot = np.dot(v1, v2)
    cos_sim = dot / (np.linalg.norm(v1) * np.linalg.norm(v2))
    dist = np.linalg.norm(v1 - v2)

    st.write(f"Dot Product: {dot}")
    st.write(f"Cosine Similarity: {cos_sim:.3f}")
    st.write(f"Euclidean Distance: {dist:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, v1[0]], y=[0, v1[1]],
                             mode='lines+markers', name='v1'))
    fig.add_trace(go.Scatter(x=[0, v2[0]], y=[0, v2[1]],
                             mode='lines+markers', name='v2'))

    st.plotly_chart(fig)

# ==============================
# MODULE 2: FUNCTION VISUALIZER
# ==============================
elif module == "Function & Derivative":
    st.header("📊 Function & Derivative")

    x = sp.symbols('x')
    expr = st.text_input("Enter function (e.g. x**2 + 3*x)", "x**2")

    try:
        f = sp.sympify(expr)
        derivative = sp.diff(f, x)

        st.write(f"Derivative: {derivative}")

        f_lambd = sp.lambdify(x, f)
        d_lambd = sp.lambdify(x, derivative)

        x_vals = np.linspace(-10, 10, 100)
        y_vals = f_lambd(x_vals)
        dy_vals = d_lambd(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="f(x)")
        ax.plot(x_vals, dy_vals, label="f'(x)")
        ax.legend()

        st.pyplot(fig)

    except:
        st.error("Invalid function")

# ==============================
# MODULE 3: GRADIENT DESCENT
# ==============================
elif module == "Gradient Descent":
    st.header("📉 Gradient Descent")

    lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    iterations = st.slider("Iterations", 10, 200, 50)
    x_val = st.number_input("Starting Point", value=5.0)

    def f(x):
        return x**2

    def grad(x):
        return 2*x

    history = []

    for i in range(iterations):
        x_val = x_val - lr * grad(x_val)
        history.append(x_val)

    st.write(f"Final Value: {x_val}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history, mode='lines', name="x updates"))
    st.plotly_chart(fig)