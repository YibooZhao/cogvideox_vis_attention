python cogvideo_t2v.py \
    --model_path "THUDM/CogVideoX-5b" \
    --device "cuda:1" \
    --height 480 \
    --width 720 \
    --num_frames 20 \
    --num_inference_steps 50 \
    --guidance_scale 6 \
    --seed 42 \
    --prompt "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. " \
    --output_path "test.mp4" \
    --attn_map_output_dir "vis" \
    --fps 5 \
    #--save_tensor_dir "pt" \
