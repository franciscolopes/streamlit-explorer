import streamlit as st
import numpy as np
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


st.title("Simulation[tm]")
st.write("Here is our super important simulation")

#Adds sidebar header and explainer
st.sidebar.markdown("## Controls")
st.sidebar.markdown("Change **parameters** bellow to refresh both plots")

#Adds slider on sidebar
xAxis = st.sidebar.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
yAxis = st.sidebar.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)
n_est = st.sidebar.slider("n_est", min_value=1, max_value=5_000, step=1)


st.write(f"x={xAxis} y={yAxis}")

#simulated data
values = np.cumprod(1 + np.random.normal(xAxis, yAxis, (100,10)), axis=0)
n = 1000
np.random.seed(42)
x = np.linspace(0, 6, n)
X = np.linspace(0, 6, n)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.random(n) * 0.3

#builds line chart using streamlit built-in function
st.line_chart(values)

#builds line chart using matplotlib
fig, ax = plt.subplots()
for i in range(values.shape[1]):
    ax.plot(values[:, i])

plt.title(f"x={xAxis} y={yAxis}")
st.pyplot(fig)

#Shows code snippet to end user
with st.echo():
    mod1 = DecisionTreeRegressor(max_depth=4)
    y1 = mod1.fit(X,y).predict(X)
    y2 = AdaBoostRegressor(mod1, n_estimators=n_est).fit(X, y).predict(X)

#Add model predictions plot 
fig2, ax2 = plt.subplots()
ax2.scatter(x, y, alpha=0.1)
ax2.plot(x, y1, label="just a tree")
ax2.plot(x, y2, label=f"adaboost-{n_est}")
ax2.legend()
st.pyplot(fig2)


