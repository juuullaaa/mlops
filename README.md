# Wildlife Object Detection System using Flask and TensorFlow Serving

## 1. Overview

This project presents an end-to-end machine learning system for wildlife object detection, designed following MLOps principles. The system integrates a Flask-based web application with TensorFlow Serving to enable scalable, real-time inference, alongside monitoring and collecting user feedback. The application is containerised using Docker, with the Flask interface and TensorFlow Serving deployed as separate services to ensure modularity and ease of deployment. 
Users can upload images through a web interface and receive annotated outputs containing detected objects, bounding boxes, and associated confidence scores. In addition, the system captures operational metrics and structured user feedback, supporting continuous evaluation and model improvement.

## 2. System Architecture

The system is composed of two primary components:

### Flask Application

The Flask application handles user interaction and coordinates the inference workflow. It accepts and validates image uploads and forwards pre-processed data to the inference service via a REST API. After receiving predictions, the application processes the outputs and generates bounding boxes, class labels, and confidence scores, displaying the annotated results to the user interface. Additionally, the Flask layer captures user feedback and logs systemmetrics, including the number of requests, latency, prediction outcomes, and system status, to support monitoring and continuous evaluation.

### TensorFlow Serving (Inference Layer)

TensorFlow Serving hosts the trained object detection model and performs inference using GPU acceleration enabled via Docker, ensuring low-latency predictions. The service exposes a REST API endpoint, enabling the integration with the Flask application and supporting scalable, low-latency predictions.

### Communication

Communication between the Flask application and TensorFlow Serving is implemented using a REST API over HTTP. The Flask application sends HTTP POST requests to the TensorFlow Serving endpoint, with image data encoded as JSON. The inference service processes the request and returns predictions in a structured JSON format containing bounding boxes, class labels, and confidence scores. REST was selected due to its simplicity, compatibility, and ease of integration with web-based systems. This solution allows for independent scaling and deployment of the Flask application and inference layer. Docker Compose is used to orchestrate the Flask application and TensorFlow Serving containers, enabling communication between services and ensuring reproducibility and scalability.

## 3. Key Features
The system performs real-time object detection inference using TensorFlow Serving and automatically generates annotated images containing bounding boxes and class labels for detected objects, together with confidence scores for each prediction. The application also stores the original uploaded images and the annotated outputs for comparison. In addition, it provides a metrics dashboard for monitoring request volume, success and failure rates, inference latency, and service status. A structured feedback mechanism is included so that users can report positive results, incorrect labels, or missed detections, supporting continuous evaluation and future model improvement. The following components support these capabilities:

### Metrics and monitoring dashboard:
- Request volume tracking
- Success and failure rate monitoring
- Inference latency measurement
- Service status monitoring
### User feedback system:
- Collection of positive and negative feedback
- Reporting of incorrect labels and missed detections
- Aggregation of feedback for evaluation purposes
### Data Storage:
- Storage of uploaded images
- Storage of inference results
- Logging of system metrics
- Storage of user feedback

## 4. Directory Structure
```
file1/
│── app.py
│── Dockerfile
│── compose.yaml
│── requirements.txt
│
│── templates/
│ ├── index.html
│ ├── results.html
│ └── metrics.html
│
│── static/
│   └── css/
│       └── style.css
│
│── originals/
│── results/
│
│── models/
│   └── my_model/
│       └── 1/
│           ├── variables/
│           └── saved_model.pb
│
│── data/
│   ├── feedback.csv
│   └── metrics.json
│
│── screenshots/
```
## 5. Requirements

### System Requirements

- Docker Desktop with Linux containers enabled
- Docker Compose

### Hardware Requirements

- Multi-core CPU
- At least 8 GB RAM
- NVIDIA GPU with CUDA support for GPU-based TensorFlow Serving inference

### Python Dependencies

The Flask application uses the following Python libraries:

- Flask
- requests
- numpy
- Pillow
- Werkzeug

## 6. Running the Application

### Step 1: Build and Start Containers
From the project root directory, run:

```bash
docker compose up --build
```
### Step 2: Access the Web Interface
Once the containers are running, open the application in a browser at:
```
http://localhost:5000
```
