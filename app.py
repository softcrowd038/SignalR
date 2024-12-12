import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PIXELS_PER_CM = 2.54 * 300

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PIXELS_PER_CM = 2.54 * 300

def identify_shapes_and_calculate_area(contour):
    """Identify shape and calculate its area using the relevant formula."""
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        shape = "Triangle"
        area = cv2.contourArea(contour)
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        
        aspect_ratio = float(w) / h
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        area = w * h
    elif num_vertices > 4:
        shape = "Circle"
        radius = cv2.minEnclosingCircle(contour)[1]
        print(radius)
        area = np.pi * (radius ** 2)
    else:
        shape = "Unknown"
        area = cv2.contourArea(contour)

    return shape, area

def calculate_total_objects_area(image_path):
    """Calculate the area of the largest object in the image and identify its shape."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No shapes detected in the image.")

    
    largest_contour = max(contours, key=cv2.contourArea)

    
    shape, area = identify_shapes_and_calculate_area(largest_contour)

    
    contour_image = image.copy()
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    contour_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"largest_contour_{timestamp}.png")
    cv2.imwrite(contour_image_path, contour_image)

    return area, contour_image_path, shape

def analyze_colors(image_path):
    """Analyze dominant colors in the image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshaped_image = image.reshape((-1, 3))

    k = 4
    reshaped_image = np.float32(reshaped_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels_count = np.bincount(labels.flatten())
    total_pixels = len(labels)
    proportions = labels_count / total_pixels * 100
    colors = centers.astype(int)
    colors_hex = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in colors]

    color_proportions = [
        {"color": color_hex, "proportion": round(proportion, 2)}
        for color_hex, proportion in zip(colors_hex, proportions)
    ]
    return color_proportions

def create_color_bar(colors_proportions, width=500, height=100):
    """Generate a horizontal bar showing dominant colors."""
    bar = np.zeros((height, width, 3), dtype="uint8")
    start_x = 0

    for color_info in colors_proportions:
        proportion = color_info['proportion']
        color_hex = color_info['color']
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))

        end_x = start_x + int(proportion / 100 * width)
        cv2.rectangle(bar, (start_x, 0), (end_x, height), color_rgb[::-1], -1)
        start_x = end_x

    return bar

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and perform analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        total_area, contour_image_path, shape_info = calculate_total_objects_area(file_path)
        colors = analyze_colors(file_path)

        bar_image = create_color_bar(colors)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        bar_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"color_bar_{timestamp}.png")
        cv2.imwrite(bar_image_path, bar_image)

        response = {
            "total_area_cm2": total_area / PIXELS_PER_CM,
            "contour_image_url": f"/uploads/{os.path.basename(contour_image_path)}",
            "shape": shape_info,
            "colors": colors,
            "color_bar_url": f"/uploads/{os.path.basename(bar_image_path)}"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded or generated files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

