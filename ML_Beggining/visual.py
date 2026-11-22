
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "CGPA": [6, 7, 8, 9, 10, 7, 8, 9, 6.5, 8.5],
    "IQ": [100, 110, 120, 130, 140, 115, 125, 135, 105, 128],
    "Score": [55, 65, 75, 85, 95, 68, 78, 88, 60, 82]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[["CGPA", "IQ"]]
y = df["Score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R2 Score:", model.score(X, y))

# Create mesh grid for regression plane
cgpa_range = np.linspace(df.CGPA.min(), df.CGPA.max(), 30)
iq_range = np.linspace(df.IQ.min(), df.IQ.max(), 30)

CGPA, IQ = np.meshgrid(cgpa_range, iq_range)

grid = pd.DataFrame({
    "CGPA": CGPA.ravel(),
    "IQ": IQ.ravel()
})

Z = model.predict(grid).reshape(CGPA.shape)

# 3D Visualization
fig = go.Figure()

# Actual data points
fig.add_trace(go.Scatter3d(
    x=df["CGPA"],
    y=df["IQ"],
    z=df["Score"],
    mode="markers",
    marker=dict(size=6, color="red"),
    name="Actual Data"
))

# Regression plane
fig.add_trace(go.Surface(
    x=CGPA,
    y=IQ,
    z=Z,
    colorscale="Viridis",
    opacity=0.6,
    name="Regression Plane"
))

fig.update_layout(
    title="3D Multiple Linear Regression",
    scene=dict(
        xaxis_title="CGPA",
        yaxis_title="IQ",
        zaxis_title="Placement Score"
    ),
    width=1000,
    height=700
)

fig.show()
