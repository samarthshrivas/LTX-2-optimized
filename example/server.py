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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/generatevideo", methods=["POST"])
def generatevideo():
    data = request.get_json() or {}

    prompt = data.get("prompt", "A beautiful sunset over the ocean")
    seed = data.get("seed", DEFAULT_SEED)
    height = data.get("height", DEFAULT_HEIGHT)
    width = data.get("width", DEFAULT_WIDTH)
    num_frames = data.get("num_frames", DEFAULT_NUM_FRAMES)
    frame_rate = data.get("frame_rate", DEFAULT_FRAME_RATE)
    enhance_prompt = data.get("enhance_prompt", False)

    images = data.get("images", [])

    output_filename = data.get("output_filename", f"{uuid.uuid4()}.mp4")
    output_path = output_dir / output_filename

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

    for img in images:
        img_path = img.get("path")
        frame_idx = img.get("frame_idx", 0)
        strength = img.get("strength", 0.8)
        if img_path:
            cmd.extend(["--image", img_path, str(frame_idx), str(strength)])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

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
