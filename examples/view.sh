scene="24"
python textured_gaussians_viewer.py \
    --dataset colmap \
    --data_dir /workspace/data/Datasets/dtu/DTU/scan${scene} \
    --data_factor 2 \
    --ckpt /workspace/work/Outputs/dtu/ntex_rgba_pc100k_t1_t4/scan${scene}/ckpts/ckpt_29999.pt \
    /workspace/work/Outputs/dtu/ntex_rgba_pc100k_t4/scan${scene}/ckpts/ckpt_29999.pt 
    # /workspace/work/Outputs/dtu/textured_gaussians_rgb_pc10k_t16/scan${scene}/ckpts/ckpt_29999.pt \