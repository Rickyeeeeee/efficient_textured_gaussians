ncu \
--set detailed \
--target-processes all \
--kernel-name "rasterize_to_pixels_fwd_2dgs_kernel" \
--import-source yes \
--source-folders $(pwd) \
--resolve-source-file rasterize_to_pixels_2dgs_fwd.cu \
-f -o 2dgs-lineinfo-source \
python 2dgs_viewer.py \
--ckpt /workspace/work/A2TG_Correct/FixedPC_mcmc/MipNerf360/2dgs_mcmc/pc100000/room/ckpts/ckpt_29999.pt