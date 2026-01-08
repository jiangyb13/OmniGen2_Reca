# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="/mnt/nfs/data/pretrained_models/OmniGen2"
python inference_reca.py \
--model_path $model_path \
--instruction "Please describe this image briefly." \
--input_image_path example_images/02.jpg \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 4.0 \
--output_image_path outputs/output_t2i.png \
--num_images_per_prompt 1