"""
Flask server for the DistilledPipeline with model hot-loading.
Supports both Text-to-Video (T2V) and Text+Image-to-Video (TI2V).
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, jsonify, request

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@dataclass
class ServerConfig:
    checkpoint_path: str = os.environ.get("CHECKPOINT_PATH", "/tmp/ltx-2-19b-distilled-fp8.safetensors")
    gemma_root: str = os.environ.get("GEMMA_ROOT", "/tmp/gemma")
    spatial_upsampler_path: str = os.environ.get("SPATIAL_UPSAMPLER_PATH", "/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors")
    output_dir: str = os.environ.get("OUTPUT_DIR", "./outputs")
    quantization: str = os.environ.get("QUANTIZATION", "fp8-cast")
    default_height: int = 1024
    default_width: int = 1536
    default_num_frames: int = 121
    default_frame_rate: float = 24.0
    default_seed: int = 10
    auto_load: bool = True


config = ServerConfig()
pipeline: DistilledPipeline | None = None
pipeline_lock = threading.Lock()


def get_quantization_policy():
    if config.quantization == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    elif config.quantization == "fp8-scaled-mm":
        return QuantizationPolicy.fp8_scaled_mm()
    return None


def initialize_pipeline():
    global pipeline
    with pipeline_lock:
        if pipeline is None:
            logger.info("Initializing DistilledPipeline...")
            pipeline = DistilledPipeline(
                checkpoint_path=config.checkpoint_path,
                spatial_upsampler_path=config.spatial_upsampler_path,
                gemma_root=config.gemma_root,
                loras=[],
                quantization=get_quantization_policy(),
            )
            logger.info("DistilledPipeline initialized successfully")
    return pipeline


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
    })


@app.route("/warm", methods=["POST"])
def warm():
    """Keep the model hot by preloading it."""
    try:
        initialize_pipeline()
        return jsonify({
            "status": "warmed",
            "message": "Pipeline is ready",
        })
    except Exception as e:
        logger.exception("Failed to warm up pipeline")
        return jsonify({"error": str(e)}), 500


@app.route("/generatevideo", methods=["POST"])
def generate():
    """
    Generate a video from a prompt.
    
    Supports both T2V (text-only) and TI2V (text + image conditioning).
    
    Request body (JSON):
    {
        "prompt": "A beautiful sunset over the ocean",  # required
        "seed": 10,                                     # optional, default: 10
        "height": 1024,                                 # optional, default: 1024
        "width": 1536,                                  # optional, default: 1536
        "num_frames": 121,                              # optional, default: 121
        "frame_rate": 24.0,                             # optional, default: 24.0
        "enhance_prompt": false,                        # optional, default: false
        "output_filename": "output.mp4",                # optional, default: output_{seed}.mp4
        
        // Image conditioning (TI2V) - leave empty for T2V
        "images": [
            {"path": "/path/to/image.jpg", "frame_idx": 0, "strength": 0.8}
        ]
    }
    """
    if pipeline is None:
        initialize_pipeline()

    data = request.get_json() or {}
    prompt = data.get("prompt", "A beautiful sunset over the ocean")
    seed = data.get("seed", config.default_seed)
    height = data.get("height", config.default_height)
    width = data.get("width", config.default_width)
    num_frames = data.get("num_frames", config.default_num_frames)
    frame_rate = data.get("frame_rate", config.default_frame_rate)
    enhance_prompt = data.get("enhance_prompt", False)

    images_raw = data.get("images", [])
    images: list[tuple[str, int, float]] = []
    mode = "t2v"
    
    if images_raw:
        mode = "ti2v"
        for img in images_raw:
            path = img.get("path")
            frame_idx = img.get("frame_idx", 0)
            strength = img.get("strength", 0.8)
            if path:
                images.append((path, frame_idx, strength))

    output_filename = data.get("output_filename", f"output_{seed}.mp4")
    output_path = Path(config.output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )

        return jsonify({
            "status": "success",
            "output_path": str(output_path),
            "prompt": prompt,
            "seed": seed,
            "mode": mode,
            "images": images if mode == "ti2v" else [],
        })
    except Exception as e:
        logger.exception("Failed to generate video")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if config.auto_load:
        logger.info("Auto-loading pipeline on startup...")
        initialize_pipeline()
    
    app.run(host="0.0.0.0", port=5000, debug=False)
