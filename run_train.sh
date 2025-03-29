export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROL_MODEL_DIR="lllyasviel/sd-controlnet-openpose"
export OUTPUT_DIR="custom_training"

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROL_MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="surreal" \
 --image_column="target" \
 --conditioning_image_column="pose" \
 --resolution=512 \
 --seed=42 \
 --learning_rate=5e-5 \
 --lr_scheduler cosine \
 --validation_image "./sample_inputs/teaser_pose.png" \
 --validation_prompt "a man standing in a kitchen with a large counter" \
 --checkpointing_steps 10000 \
 --checkpoints_total_limit 5 \
 --num_train_epochs 3 \
 --dataloader_num_workers=4 \
 --train_batch_size 4 \
 --validation_steps 2000 \
 --resume_from_checkpoint=latest \

