"""
Example script demonstrating how to run the DistilledPipeline programmatically.
"""

import logging

import torch
from tqdm import tqdm

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE


def run_with_args(args):
    """Run pipeline with parsed CLI arguments."""
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        quantization=args.quantization,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    with tqdm(total=args.num_frames, desc="Generating video frames", unit="frame") as pbar:
        video, audio = pipeline(
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=args.images,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
        )
        pbar.update(args.num_frames)

    from ltx_pipelines.utils.media_io import encode_video
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


def run_programmatically():
    """Run pipeline programmatically without CLI arguments."""
    checkpoint_path = "/tmp/ltx-2-19b-distilled-fp8.safetensors"
    gemma_root = "/tmp/gemma"
    spatial_upsampler_path = "/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    output_path = "output.mp4"
    prompt = "A beautiful sunset over the ocean"

    pipeline = DistilledPipeline(
        checkpoint_path=checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        quantization=QuantizationPolicy.fp8_cast(),
    )

    tiling_config = TilingConfig.default()
    num_frames = 121
    frame_rate = 24.0
    height = 1024
    width = 1536
    seed = 10
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    with tqdm(total=num_frames, desc="Generating video frames", unit="frame") as pbar:
        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            tiling_config=tiling_config,
            enhance_prompt=False,
        )
        pbar.update(num_frames)

    from ltx_pipelines.utils.media_io import encode_video
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    run_with_args(args)


if __name__ == "__main__":
    main()