from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import numpy as np
import os
import requests
from PIL import Image, ImageDraw, ImageFont
import json
import time
import csv
import uuid

# Model and file configuration
MODEL_NAME = "my_model" 
TF_SERVING_URL = os.environ.get(
    "TF_SERVING_URL",
    f"http://tf-serving:8501/v1/models/{MODEL_NAME}:predict"  # Default REST endpoint
)

# Metrics and feedback data
DATA_FOLDER = "data"  
METRICS_FILE = os.path.join(DATA_FOLDER, "metrics.json")  # JSON file path for monitoring metrics
FEEDBACK_FILE = os.path.join(DATA_FOLDER, "feedback.csv")  # CSV file path for saved user feedback

# Class names
CLASS_NAMES = {
    1: "OryxGazella",          
    2: "StruthioCamelus",      
    3: "PhacochoerusAfricanus" 
}

# Flask app 
app = Flask(__name__)  # Create the Flask application instance
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")  # Needed for flash messages

# Application folders and allowed extensions
app.config["RESULTS_FOLDER"] = "results"  # output images
app.config["ORIGINALS_FOLDER"] = "originals"  # original images
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}  # supported image formats

# Ensure required folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs(app.config["ORIGINALS_FOLDER"], exist_ok=True)

# File validation function: check whether the uploaded file has an allowed image extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# TensorFlow Serving helper functions
def send_to_tf_serving(image_np):
    """
    Sends an image to TensorFlow Serving.
    """
    input_data = np.expand_dims(image_np, axis=0).astype(np.uint8) # batch dimension

    payload = {
        "instances": input_data.tolist() # converts NumPy to Python list
    }

    try:
        response = requests.post(TF_SERVING_URL, json=payload, timeout=60) # send image to the REST endpoint
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"TensorFlow Serving error: {str(e)}")
    
    return response.json()


def parse_detections(result, score_threshold=0.30):
    """
    Parses object detection output returned by TensorFlow Serving.
    Expected keys: detection_boxes, detection_scores, detection_classes, num_detections
    Returns a filtered list of detections above the given threshold.
    """
    # Extract predictions from JSON
    predictions = result.get("predictions", [])
    if not predictions:
        return [] 

    pred = predictions[0]
    boxes = pred.get("detection_boxes", [])
    scores = pred.get("detection_scores", [])
    classes = pred.get("detection_classes", [])
    num_detections = int(pred.get("num_detections", len(scores)))

    detections = []

    for i in range(num_detections):  # Loop through each detected object
        if i >= len(scores) or i >= len(classes) or i >= len(boxes):
            break

        score = float(scores[i])

        if score < score_threshold:
            continue

        class_id = int(classes[i])
        label = CLASS_NAMES.get(class_id, f"class_{class_id}")

        detections.append({
            "box": boxes[i],   # [ymin, xmin, ymax, xmax]
            "score": score,
            "class_id": class_id,
            "label": label
        })

    return detections

def draw_boxes(image_np, detections):
    """
    Draws bounding boxes and class labels on the image.
    """
    # Convert NumPy image into a PIL image for annotation
    image = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    width, height = image.size

    # Draw a bounding box and label 
    for det in detections:
        ymin, xmin, ymax, xmax = det["box"]

        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)

        label_text = f"{det['label']}: {int(det['score'] * 100)}%"

        draw.rectangle([left, top, right, bottom], outline="cyan", width=3) # draw detection box

        try:
            bbox = draw.textbbox((left, top), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = 100, 20

        label_top = max(0, top - text_h - 4)

        draw.rectangle(
            [left, label_top, left + text_w + 6, label_top + text_h + 4],
            fill="cyan"
        )
      
        draw.text((left + 3, label_top + 2), label_text, fill="black", font=font)

    return np.array(image) # return the annotated image as NumPy array

# Metrics helper functions

def load_metrics():
    """
    Loads metrics from the metrics JSON file.
    """
    if not os.path.exists(METRICS_FILE):
        metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0,
            "latest_latency": 0,
            "success_rate": 0,
            "failure_rate": 0,
            "model_version": "1",
            "tf_serving_status": "Unknown",
            "gpu_enabled": "Configured via Docker",
            "last_request_time": "N/A",
            "app_version": "1.0"
        }

        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        return metrics

    with open(METRICS_FILE, "r", encoding="utf-8") as f: # Load the existing metrics
        metrics = json.load(f)

    # Current counters
    total = metrics.get("total_requests", 0)
    success = metrics.get("successful_requests", 0)
    failed = metrics.get("failed_requests", 0)

    # Percentage rates
    if total > 0:
        metrics["success_rate"] = round((success / total) * 100, 2)
        metrics["failure_rate"] = round((failed / total) * 100, 2)
    else:
        metrics["success_rate"] = 0
        metrics["failure_rate"] = 0

    return metrics


def save_metrics(metrics):
    """
    Saves the metrics to the metrics JSON file.
    """
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def update_metrics(success, latency_ms=None):
    """
    Updates metrics after an inference.
    """
    metrics = load_metrics() # load current metrics
    metrics["total_requests"] += 1  # increment the count

    if success:
        metrics["successful_requests"] += 1  # successful inference
        metrics["tf_serving_status"] = "Running"
    else:
        metrics["failed_requests"] += 1  # failed inference
        metrics["tf_serving_status"] = "Error"

    if latency_ms is not None:
        metrics["latest_latency"] = round(latency_ms, 2)

        # Average latency
        completed = metrics["successful_requests"]
        if completed == 1:
            metrics["average_latency"] = round(latency_ms, 2)
        elif completed > 1:
            old_avg = metrics["average_latency"]
            new_avg = ((old_avg * (completed - 1)) + latency_ms) / completed
            metrics["average_latency"] = round(new_avg, 2)

    total = metrics["total_requests"]
    success_count = metrics["successful_requests"]
    failed_count = metrics["failed_requests"]

    # Update rates
    if total > 0:
        metrics["success_rate"] = round((success_count / total) * 100, 2)
        metrics["failure_rate"] = round((failed_count / total) * 100, 2)
    else:
        metrics["success_rate"] = 0
        metrics["failure_rate"] = 0

    metrics["last_request_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_metrics(metrics)

# Feedback helper functions

def save_feedback(image_name, predicted_label, feedback_type, error_type, correct_label, missed_label, feedback_note):
    """
    Appends feedback to the feedback CSV file.
    """
    file_exists = os.path.exists(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "image_name",
                "predicted_label",
                "feedback_type",
                "error_type",
                "correct_label",
                "missed_label",
                "feedback_note",
                "timestamp"
            ])

        # Append a new feedback row
        writer.writerow([
            image_name,
            predicted_label,
            feedback_type,
            error_type,
            correct_label,
            missed_label,
            feedback_note,
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])


def load_feedback_stats():
    """
    Computes feedback statistics from the feedback CSV file.
    """
    # Default values
    stats = {
        "total_feedback": 0,
        "positive_feedback": 0,
        "negative_feedback": 0,
        "positive_feedback_rate": 0,
        "negative_feedback_rate": 0,
        "unique_images_with_feedback": 0,
        "latest_feedback_time": "N/A",
        "wrong_label_reports": 0,
        "missed_label_reports": 0
    }

    if not os.path.exists(FEEDBACK_FILE):
        return stats

    with open(FEEDBACK_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row]

    if not rows:
        return stats 

    # Only keep rows with valid feedback values
    valid_rows = [
        row for row in rows
        if row.get("feedback_type", "").strip().lower() in ["positive", "negative"]
    ]

    if not valid_rows:
        return stats

    stats["total_feedback"] = len(valid_rows) # total number of feedback entries

    stats["unique_images_with_feedback"] = len(
        set(row.get("image_name", "").strip() for row in valid_rows if row.get("image_name", "").strip()) # unique images with feedback
    )

    # Timestamps
    timestamps = [
        row.get("timestamp", "").strip()
        for row in valid_rows
        if row.get("timestamp", "").strip()
    ]
    if timestamps:
        stats["latest_feedback_time"] = timestamps[-1]

    for row in valid_rows:
        feedback_type = row.get("feedback_type", "").strip().lower()
        error_type = row.get("error_type", "").strip().lower()

        if feedback_type == "positive":
            stats["positive_feedback"] += 1
        elif feedback_type == "negative":
            stats["negative_feedback"] += 1

            # Type of negative feedback
            if error_type == "wrong_label":
                stats["wrong_label_reports"] += 1
            elif error_type == "missed_label":
                stats["missed_label_reports"] += 1

    total = stats["total_feedback"]
    if total > 0:
        stats["positive_feedback_rate"] = round((stats["positive_feedback"] / total) * 100, 2) # percentages
        stats["negative_feedback_rate"] = round((stats["negative_feedback"] / total) * 100, 2)

    return stats

# App Routes

@app.route("/")
def index():
    """
    Renders the home page.
    """
    # Load the upload form page
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    """
    Renders the metrics dashboard page.
    """
    metrics_data = load_metrics() # load inference metrics
    feedback_stats = load_feedback_stats() # load feedback statistics
    metrics_data.update(feedback_stats)
    return render_template("metrics.html", metrics=metrics_data)


@app.route("/originals/<filename>")
def serve_original(filename):
    """
    Serves the originally uploaded image from the originals folder.
    """
    return send_from_directory(app.config["ORIGINALS_FOLDER"], filename) # original image

@app.route("/results/<filename>")
def serve_result(filename):
    """
    Serves the annotated image from the results folder.
    """
    return send_from_directory(app.config["RESULTS_FOLDER"], filename) # annotated image

@app.route("/view_result")
def view_result():
    """
    Displays the latest inference result stored in session.
    """
    result_data = session.get("last_result")

    if not result_data:
        flash("No result is available to display.", "error")
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        original_image=result_data["original_image"],
        annotated_image=result_data["annotated_image"],
        detections=result_data["detections"],
        inference_time_s=result_data["inference_time_s"],
        object_count=result_data["object_count"]
    )

@app.route("/upload", methods=["POST"])
def upload():
    """
    Uploads image, validates the file and saves the original image.
    """
    if "file" not in request.files:
        flash("No file part in request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename): # validate extensions
        flash("Invalid file type. Please upload JPG, JPEG, or PNG only.", "error")
        return redirect(url_for("index"))

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}" # each file gets a unique name

    original_path = os.path.join(app.config["ORIGINALS_FOLDER"], unique_name)

    file.save(original_path) # save the file to originals

    return redirect(url_for("uploaded_file", filename=unique_name))

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """
    Runs inference, saves the annotated result, updates metrics,
    stores the result in session, and generates the results page.
    """
    original_path = os.path.join(app.config["ORIGINALS_FOLDER"], filename)

    if not os.path.exists(original_path):
        flash("Uploaded image could not be found.", "error")
        return redirect(url_for("index"))

    try:
        # Load the saved image and convert it to an RGB NumPy array
        image_np = np.array(Image.open(original_path).convert("RGB"))

        start_time = time.time()
        result = send_to_tf_serving(image_np)
        inference_time_s = time.time() - start_time

        # Count returned detections
        detections = parse_detections(result)
        object_count = len(detections)

        # Draw boxes and labels on a copy of the original image
        image_np_inferenced = draw_boxes(image_np.copy(), detections)

        output_filename = f"result_{filename}"
        output_path = os.path.join(app.config["RESULTS_FOLDER"], output_filename)

        # Save the annotated image
        annotated_image = Image.fromarray(image_np_inferenced)
        annotated_image.save(output_path)

        # Update metrics
        update_metrics(success=True, latency_ms=inference_time_s * 1000)

        result_data = {
            "original_image": filename,
            "annotated_image": output_filename,
            "detections": detections,
            "inference_time_s": round(inference_time_s, 2),
            "object_count": object_count
        }

        session["last_result"] = result_data

        # Generate the results page
        return render_template(
            "results.html",
            original_image=result_data["original_image"],
            annotated_image=result_data["annotated_image"],
            detections=result_data["detections"],
            inference_time_s=result_data["inference_time_s"],
            object_count=result_data["object_count"]
        )

    except Exception as e:
        # Record the failure 
        update_metrics(success=False)
        flash(f"Error during inference: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    """
    Validates feedback, stores it in the CSV file, and redirects back to the existing results page.
    """
    image_name = request.form.get("image_name", "").strip()
    feedback_type = request.form.get("feedback_type", "").strip().lower()
    error_type = request.form.get("error_type", "").strip().lower()

    predicted_label = request.form.get("predicted_label", "").strip()
    correct_label = request.form.get("correct_label", "").strip()
    missed_label = request.form.get("missed_label", "").strip()

    if not image_name:
        flash("Image name is missing from the feedback form.", "error")
        return redirect(url_for("index"))

    if feedback_type not in ["positive", "negative"]:
        flash("Please select valid feedback.", "error")
        return redirect(url_for("view_result"))

    if feedback_type == "positive": # positice feedback
        feedback_note = "Positive feedback recorded."
        error_type = ""
        predicted_label = ""
        correct_label = ""
        missed_label = ""

    else:
        # Negative feedback type selection
        if error_type not in ["wrong_label", "missed_label"]:
            flash("Please select a valid negative feedback category.", "error")
            return redirect(url_for("view_result"))

        if error_type == "wrong_label":
            # Wrong label
            if not predicted_label or not correct_label:
                flash("Please provide both the incorrect label and the correct label.", "error")
                return redirect(url_for("view_result"))
            missed_label = ""
            feedback_note = "Negative feedback recorded: wrong label."

        elif error_type == "missed_label":
            # Missing label
            if not missed_label:
                flash("Please enter the missed label.", "error")
                return redirect(url_for("view_result"))
            predicted_label = ""
            correct_label = ""
            feedback_note = "Negative feedback recorded: missed label."

    # Save feedback into the CSV file
    save_feedback(
        image_name=image_name,
        predicted_label=predicted_label,
        feedback_type=feedback_type,
        error_type=error_type,
        correct_label=correct_label,
        missed_label=missed_label,
        feedback_note=feedback_note
    )

    flash("Thank you for your feedback!", "success")
    return redirect(url_for("view_result"))


# App entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)