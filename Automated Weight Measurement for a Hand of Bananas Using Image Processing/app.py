from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from predict import predict_weight  # Your custom prediction function

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Home page route (default)
@app.route("/")
def home():
    return render_template("home.html")

# ✅ Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    predicted_weight = None
    error_message = None
    image_name = None

    if request.method == "POST":
        if "file" not in request.files:
            error_message = "No file uploaded."
        else:
            file = request.files["file"]
            if file.filename == "":
                error_message = "No selected file."
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                predicted_weight = predict_weight(file_path)
                image_name = filename

                if predicted_weight is None:
                    error_message = "Invalid image. Please upload a valid hand of banana image."

    return render_template("index.html",
                           predicted_weight=predicted_weight,
                           error_message=error_message,
                           image_name=image_name)

# ✅ About page
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
