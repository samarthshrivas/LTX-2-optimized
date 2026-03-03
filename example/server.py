"""
Flask server for the DistilledPipeline with model hot-loading.
Supports both Text-to-Video (T2V) and Text+Image-to-Video (TI2V).
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from flask import Flask, jsonify, request

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

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


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate a video from a prompt.
    
    Supports both T2V (text-only) and TI2V (text + image conditioning).
    
    Request body (JSON):
    {