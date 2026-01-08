# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="./pretrained_models/OmniGen2"
python inference_chat.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 4.0 \
--instruction "A photo of a bench." \
--output_image_path outputs/output_und2.png \
--num_images_per_prompt 1