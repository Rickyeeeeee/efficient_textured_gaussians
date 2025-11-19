ncu \
--set detailed \
--target-processes all \
--kernel-name "rasterize_to_pixels_fwd_packed_textured_gaussians_kernel" \
--import-source yes \
--source-folders $(pwd) \
--resolve-source-file rasterize_to_pixels_packed_textured_gaussians_2dgs_fwd.cu \
-f -o a2tg-no-double \
python textured_gaussians_viewer.py \
--dataset colmap \
--data_dir /workspace/data/Datasets/MipNerf360/room \
--data_factor 4 \
--ckpt /workspace/work/A2TG_Correct/FixedPC_mcmc/MipNerf360/ntex_full/pc100000/room/ckpts/ckpt_29999.pt