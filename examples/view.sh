# scene="24"
# python textured_gaussians_viewer.py \
#     --dataset colmap \
#     --data_dir /workspace/data/Datasets/dtu/DTU/scan${scene} \
#     --data_factor 2 \
#     --ckpt /workspace/work/Outputs/dtu/ntex_rgba_pc10k_t1_t4/scan${scene}/ckpts/ckpt_29999.pt 

scene="room"
python textured_gaussians_viewer.py \
    --dataset colmap \
    --data_dir /workspace/data/Datasets/MipNerf360/${scene} \
    --data_factor 4 \
    --ckpt /workspace/work/Experiments/MipNerf360/ntex_anisotropic_rgba/pc10000/${scene}/ckpts/ckpt_29999.pt
    # /workspace/work/Experiments/MipNerf360/ntex_rgba/pc100000/${scene}/ckpts/ckpt_29999.pt 