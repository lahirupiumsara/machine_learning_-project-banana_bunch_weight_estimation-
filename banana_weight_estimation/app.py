from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from predict import predict_weight

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_weight = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Predict weight
        predicted_weight = predict_weight(file_path)

    return render_template("index.html", predicted_weight=predicted_weight)

if __name__ == "__main__":
    app.run(debug=True)
