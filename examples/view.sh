# scene="24"
# python textured_gaussians_viewer.py \
#     --dataset colmap \
#     --data_dir /workspace/data/Datasets/dtu/DTU/scan${scene} \
#     --data_factor 2 \
#     --ckpt /workspace/work/Outputs/dtu/ntex_rgba_pc10k_t1_t4/scan${scene}/ckpts/ckpt_29999.pt 

scene="garden"
python textured_gaussians_viewer.py \
    --dataset colmap \
    --data_dir /workspace/data/Datasets/MipNerf360/${scene} \
    --data_factor 4 \
    --ckpt /workspace/work/A2TG_Test/FixedPC_mcmc/MipNerf360/a2tg/pc50000/${scene}/ckpts/ckpt_29999.pt
    # --ckpt /workspace/work/A2TG_Correct/MipNerf360/ntex_full/pc10000/${scene}/ckpts/ckpt_2999.pt
    # --ckpt /workspace/work/FixedPC_mcmc_tex_abla/MipNerf360/ntex_full_tex4/pc50000/${scene}/ckpts/ckpt_29999.pt
    # --ckpt /workspace/work/FixedPC_mcmc_correct/MipNerf360/ntex_full/pc50000/${scene}/ckpts/ckpt_29999.pt
    # --ckpt /workspace/work/FixedMemory_mcmc_correct/db/ntex_full/pc200000/${scene}/ckpts/ckpt_29999.pt
    # /workspace/work/Experiments/MipNerf360/ntex_rgba/pc100000/${scene}/ckpts/ckpt_29999.pt 

# scene="playroom"
# python textured_gaussians_viewer.py \
#     --dataset colmap \
#     --data_dir /workspace/data/Datasets/tandt_db/db/${scene} \
#     --data_factor 4 \
#     --ckpt /workspace/work/A2TG_Correct/FixedPC_mcmc/db/ntex_full/pc100000/${scene}/ckpts/ckpt_29999.pt
    # --ckpt /workspace/work/A2TG_Correct/MipNerf360/ntex_full/pc10000/${scene}/ckpts/ckpt_2999.pt