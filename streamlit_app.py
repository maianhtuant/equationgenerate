import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.title("Polynomial Curve Visualization Web App")

# Option selector
option = st.radio("Choose an option:", ["User-Defined Extrema Points","Custom Polynomial Fit"])

if option == "Custom Polynomial Fit":
    st.write("Input your x and y values and choose the degree of the polynomial to fit a curve.")

    x_input = st.text_input("Enter x values (comma-separated):", "0, 4, 8, 12")
    y_input = st.text_input("Enter y values (comma-separated):", "3000, 4500, 500, 6000")
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

                equation = "y = " + " + ".join(
                    [f"{c:.2f}x^{i}" if i > 0 else f"{c:.2f}" for i, c in enumerate(coeffs[::-1])][::-1]
                )
                st.markdown(f"**Fitted Equation:**  {equation}")

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

elif option == "User-Defined Extrema Points":
    st.write("Define extrema points and the desired polynomial degree.")
    degree = st.number_input("Degree of polynomial (min 2):", min_value=2, max_value=10, value=2)

    min_input = st.text_input("Enter minimum point(s) (e.g. (x1,y1), (x2,y2)):", "")
    max_input = st.text_input("Enter maximum point(s) (e.g. (x1,y1), (x2,y2)):", "")

    extrema_points = []
    try:
        if min_input:
            for pair in min_input.split(")"):
                if pair.strip():
                    x, y = map(float, pair.strip("(), ").split(","))
                    extrema_points.append((x, y))
        if max_input:
            for pair in max_input.split(")"):
                if pair.strip():
                    x, y = map(float, pair.strip("(), ").split(","))
                    extrema_points.append((x, y))
    except Exception:
        st.error("Invalid format. Use (x, y)")

    if st.button("Generate Polynomial Curve"):
        try:
            if degree == 2 and len(extrema_points) == 1:
                h, k = extrema_points[0]
                is_min = "Minimum" if min_input else "Maximum"
                a = 1.0 if is_min == "Minimum" else -1.0
                b = -2 * a * h
                c = k - a * h**2 - b * h

                def poly_func(x):
                    return a * x**2 + b * x + c

                x_vals = np.linspace(h - 5, h + 5, 500)
                y_vals = poly_func(x_vals)

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals, label='Parabola (Degree 2)', color='blue')
                ax.plot([h], [k], 'ro', label='Extremum Point')
                ax.set_title('Parabola with Vertex at (x, y)')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                st.markdown(f"**Parabola Equation:**  y = {a:.3f}x² + {b:.3f}x + {c:.3f}")

            elif degree == 3 and len(extrema_points) == 2:
                (x1, y1), (x2, y2) = extrema_points

                A = np.array([
                    [x1**3, x1**2, x1, 1],
                    [x2**3, x2**2, x2, 1],
                    [3*x1**2, 2*x1, 1, 0],
                    [3*x2**2, 2*x2, 1, 0],
                ])
                B = np.array([y1, y2, 0, 0])

                coeffs = np.linalg.solve(A, B)

                def poly_func(x):
                    return coeffs[0]*x**3 + coeffs[1]*x**2 + coeffs[2]*x + coeffs[3]

                x_vals = np.linspace(min(x1, x2) - 2, max(x1, x2) + 2, 500)
                y_vals = poly_func(x_vals)

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals, label='Cubic Curve', color='blue')
                ax.plot([x1, x2], [y1, y2], 'ro', label='Extrema Points')
                ax.set_title('Cubic Polynomial with Given Min and Max')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                a, b, c, d = coeffs
                st.markdown(f"**Cubic Equation:**  y = {a:.3f}x³ + {b:.3f}x² + {c:.3f}x + {d:.3f}")

            else:
                X = []
                Y = []

                if degree == 4 and len(extrema_points) == 3:
                    X = []
                    Y = []
                    for x, y in extrema_points:
                        X.append([x ** 4, x ** 3, x ** 2, x, 1])  # f(x) = y
                        Y.append(y)
                        X.append([4 * x ** 3, 3 * x ** 2, 2 * x, 1, 0])  # f'(x) = 0
                        Y.append(0)
                else:
                    X = []
                    Y = []
                    for x, y in extrema_points:
                        X.append([x ** i for i in range(degree, -1, -1)])
                        Y.append(y)
                        row_deriv = [degree * i * x ** (degree - i - 1) if i < degree else 0 for i in range(degree + 1)]
                        X.append(row_deriv)
                        Y.append(0)

                X = np.array(X)
                Y = np.array(Y)

                # Always allow least squares fitting regardless of constraints
                coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]

                def poly_func(x):
                    return sum(c * x**(degree - i) for i, c in enumerate(coeffs))

                all_x = [pt[0] for pt in extrema_points]
                x_vals = np.linspace(min(all_x) - 2, max(all_x) + 2, 500)
                y_vals = poly_func(x_vals)

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals, label=f'Degree-{degree} Polynomial', color='blue')
                ax.plot(all_x, [pt[1] for pt in extrema_points], 'ro', label='Extrema Points')
                ax.set_title(f'Degree-{degree} Polynomial with Custom Extrema')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                terms = [f"{c:.3f}x^{degree - i}" if degree - i > 0 else f"{c:.3f}" for i, c in enumerate(coeffs)]
                equation = "y = " + " + ".join(terms)
                st.markdown(f"**Polynomial Equation:**  {equation}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
