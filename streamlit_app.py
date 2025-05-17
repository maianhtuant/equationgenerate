import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Polynomial Curve Fitting Web App")
st.write("Input your x and y values and choose the degree of the polynomial to fit a curve.")

# Input fields
x_input = st.text_input("Enter x values (comma-separated):", "1, 4, 8, 12")
y_input = st.text_input("Enter y values (comma-separated):", "3000, 4500, 500, 5000")
degree = st.number_input("Degree of polynomial:", min_value=1, max_value=10, value=3)

if st.button("Fit Curve"):
    try:
        x = np.array([float(i.strip()) for i in x_input.split(",")])
        y = np.array([float(i.strip()) for i in y_input.split(",")])

        if len(x) != len(y):
            st.error("x and y must have the same number of values.")
        else:
            coeffs = np.polyfit(x, y, deg=degree)
            poly_func = np.poly1d(coeffs)

            # Display polynomial equation
            equation = "y = " + " + ".join([f"{c:.4f}x^{i}" if i > 0 else f"{c:.4f}"
                                             for i, c in enumerate(coeffs[::-1])][::-1])
            st.markdown(f"**Fitted Equation:**  {equation}")

            # Plotting
            x_plot = np.linspace(min(x), max(x), 500)
            y_plot = poly_func(x_plot)
            fig, ax = plt.subplots()
            ax.plot(x, y, 'ro', label='Input Points')
            ax.plot(x_plot, y_plot, 'b-', label='Polynomial Fit')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Polynomial Curve Fit')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
