python inference.py \
    --ckpt_path ./ckpts  \
    --input_dir /dir/to/input/videos \
    --output_dir /dir/to/output/videos \
    --enable_text_encoder   \
    --enable_captioner  \
    --enable_spatial_tiling \
    --enable_temporal_tiling    \
    --upscale=0 \
    --save_images  \
