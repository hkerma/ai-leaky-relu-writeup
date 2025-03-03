from flask import Flask, request, render_template, jsonify
import torch
import os
from werkzeug.utils import secure_filename
import numpy as np
from model import NeuralNetwork

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pt", "pth"}

INPUT_LENGTH = 64



def hex_norm_to_char(norm_value):
    """Convert normalized value back to a character."""
    ascii_val = int(((norm_value + 127) / 254) * 255)
    return chr(ascii_val) if 0 <= ascii_val < 256 else "?"  # Handle invalid values safely

def reconstruct_sentences(processed_data):
    """Convert processed numerical representation back to text."""
    reconstructed_sentences = []

    for sentence_array in processed_data:
        # Remove padding (stop at first zero)
        unpadded_values = [val for val in sentence_array if val != 0]

        # Convert back to characters
        sentence = "".join(hex_norm_to_char(val) for val in unpadded_values)
        reconstructed_sentences.append(sentence)

    return reconstructed_sentences


def char_to_hex_norm(char):
    """Convert character to hex and normalize between -127 and 127."""
    hex_val = ord(char)  # Get ASCII/Unicode value
    return ((hex_val / 255) * 254) - 127  # Normalize to [-127, 127]

def process_sentences(filename):
    """Read file and process sentences."""
    processed_sentences = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            # Convert characters to normalized hex values
            hex_norm_values = [char_to_hex_norm(c) for c in line]
            
            # Pad with zeros to length 64
            if len(hex_norm_values) < 64:
                hex_norm_values += [0] * (64 - len(hex_norm_values))
            else:
                hex_norm_values = hex_norm_values[:64]  # Trim if too long
            
            processed_sentences.append(np.array(hex_norm_values, dtype=np.float32))

    return processed_sentences

# Function to check valid file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def model_inference(filepath):
# Load model and handle errors
    model = NeuralNetwork()
    result = {"success": None, "error": None}
    model_output = []

    try:
        state_dict = torch.load(filepath, map_location=torch.device("cpu"))

        # Check if the state_dict keys match the model
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())

        if model_keys != state_dict_keys:
            return jsonify({
                "error": "Model architecture mismatch",
                "expected_keys": list(model_keys),
                "provided_keys": list(state_dict_keys)
            }), 400

        model.load_state_dict(state_dict)
        model.eval()
        
        model_input = process_sentences("flag.txt")

        # Dummy input for inference
        for line in model_input:
            model_output.append(model(torch.from_numpy(line)).tolist())

        output = reconstruct_sentences(model_output)

        result["success"] = output
    
    except Exception as e:
        result["error"] = f"Error: {str(e)}"

    return result

@app.route("/", methods=["GET", "POST"])
def upload_file():
    messages = {"success": None, "error": None}

    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result = model_inference(filepath)
            messages["success"] = result["success"]
            messages["error"] = result["error"]  
    
    return render_template("index.html", messages=messages)

if __name__ == "__main__":
    app.run(debug=True)
