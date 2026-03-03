"""
Flask server for the DistilledPipeline.
"""

import logging
import os
import subprocess
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/tmp/ltx-2-19b-distilled-fp8.safetensors")
GEMMA_ROOT = os.environ.get("GEMMA_ROOT", "/tmp/gemma")
SPATIAL_UPSAMPLER_PATH = os.environ.get("SPATIAL_UPSAMPLER_PATH", "/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
QUANTIZATION = os.environ.get("QUANTIZATION", "fp8-cast")

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1536
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 24.0
DEFAULT_SEED = 10

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

temp_dir = Path("/tmp/ltx-server-uploads")
temp_dir.mkdir(parents=True, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/generatevideo", methods=["POST"])
def generatevideo():
    prompt = request.form.get("prompt", "A beautiful sunset over the ocean")
    seed = int(request.form.get("seed", DEFAULT_SEED))
    height = int(request.form.get("height", DEFAULT_HEIGHT))
    width = int(request.form.get("width", DEFAULT_WIDTH))
    num_frames = int(request.form.get("num_frames", DEFAULT_NUM_FRAMES))
    frame_rate = float(request.form.get("frame_rate", DEFAULT_FRAME_RATE))
    enhance_prompt = request.form.get("enhance_prompt", "false").lower() == "true"

    output_filename = request.form.get("output_filename", f"{uuid.uuid4()}.mp4")
    output_path = output_dir / output_filename

    uploaded_files = request.files.getlist("images")
    
    image_args = []
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file and uploaded_file.filename:
            ext = Path(uploaded_file.filename).suffix
            temp_path = temp_dir / f"{uuid.uuid4()}{ext}"
            uploaded_file.save(temp_path)
            
            frame_idx = request.form.get(f"images[{i}][frame_idx]", 0)
            strength = request.form.get(f"images[{i}][strength]", 0.8)
            
            image_args.extend(["--image", str(temp_path), str(frame_idx), str(strength)])

    cmd = [
        "python", "-m", "ltx_pipelines.distilled",
        "--checkpoint-path", CHECKPOINT_PATH,
        "--gemma-root", GEMMA_ROOT,
        "--spatial-upsampler-path", SPATIAL_UPSAMPLER_PATH,
        "--prompt", prompt,
        "--seed", str(seed),
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--frame-rate", str(frame_rate),
        "--output-path", str(output_path),
    ]

    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])

    if enhance_prompt:
        cmd.append("--enhance-prompt")

    cmd.extend(image_args)

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        for arg in image_args:
            if arg.startswith(str(temp_dir)):
                Path(arg).unlink(missing_ok=True)

        if not output_path.exists():
            return jsonify({"error": "Output file not created", "stderr": result.stderr}), 500

        return send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=output_filename,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e.stderr}")
        return jsonify({"error": e.stderr}), 500
    except Exception as e:
        logger.exception("Error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
