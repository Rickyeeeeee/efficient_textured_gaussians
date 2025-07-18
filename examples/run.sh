CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc \
    --data_dir /workspace/data/Datasets/dtu/DTU/scan24 \
    --result_dir results/dtu/2dgs/scan24 \
    --dataset colmap \
    --init_extent 1 \
    --init_type "sfm" \
    --model_type=2dgs \
    --init_num_pts 10000 \
    --strategy.cap-max 10000 \
    --texture_resolution 50 \
    --port 6070 --eval_steps -1 --disable_viewer --data_factor 2

# CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc \
#     --data_dir /workspace/data/Datasets/dtu/DTU/scan24 \
#     --pretrained_path PRETRAINED_PATH \
#     --result_dir results/dtu/textured_gaussians/scan24 \
#     --dataset colmap \
#     --init_type "pretrained" \
#     --model_type=textured_gaussians \
#     --init_num_pts 10000 \
#     --strategy.cap-max 10000 \
#     --strategy.refine-start-iter=1000000000000 \
#     --textured_rgb \
#     --textured_alpha \
#     --texture_resolution 50 \
#     --port 6070