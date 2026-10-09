import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved model at startup
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("project.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features submitted from the HTML form
        # Make sure the input names match your HTML form inputs
        feature_values = [float(x) for x in request.form.values()]

        # Reshape data for a single sample prediction: shape (1, n_features)
        final_input = [np.array(feature_values)]

        # Generate prediction
        prediction = model.predict(final_input)[0]

        return render_template(
            "project.html", prediction_text=f"Predicted Result: {prediction}"
        )

    except Exception as e:
        return render_template("project.html", prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)