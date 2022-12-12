import streamlit as st
import numpy as np
import matplotlib.pylab as plt


st.title("Simulation[tm]")
st.write("Here is our super important simulation")


x = st.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
y = st.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

st.write(f"x={x} y={y}")

#builds line chart using streamlit built-in function
values = np.cumprod(1 + np.random.normal(x, y, (100,10)), axis=0)
st.line_chart(values)

#builds line chart using matplotlib
fig, ax = plt.subplots()

for i in range(values.shape[1]):
    ax.plot(values[:, i])

plt.title(f"x={x} y={y}")
st.pyplot(fig)