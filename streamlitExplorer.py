import streamlit as st
import numpy as np
import matplotlib.pylab as plt


st.title("Simulation[tm]")
st.write("Here is our super important simulation")

#Adds sidebar header and explainer
st.sidebar.markdown("## Controls")
st.sidebar.markdown("Change **parameters** bellow to refresh both plots")

#Adds slider on sidebar
x = st.sidebar.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
y = st.sidebar.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

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