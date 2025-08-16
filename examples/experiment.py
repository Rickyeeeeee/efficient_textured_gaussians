import os

dataset_dir = "/workspace/data/Datasets/MipNerf360"
output_dir = "/workspace/work/Experiments/MipNerf360"
point_count = 20000
texture_resolution = 1
data_factor = 4
# scenes = [
#     'bicycle',
#     'bonsai',
#     'counter',
#     'garden',
#     'kitchen',
#     'room',
#     'stump'
# ]
scene = 'room'
has_rgb = True
has_alpha = True
render = False

method_name = "2dgs_mcmc"
cmd_2dgs = (
    f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
    f"{f"--ckpt {output_dir}/{method_name}/pc{point_count}/{scene}/ckpts/ckpt_29999.pt " if render else ""}"
    f"--max_steps 30000 "
    f"--eval_steps 7000 30000 "
    f"--save_steps 7000 30000 "
    f"--data_dir {dataset_dir}/{scene} "
    f"--result_dir {output_dir}/2dgs_mcmc/pc{point_count}/{scene} "
    f"--dataset colmap "
    f"--init_extent 1 "
    f"--init_type sfm "
    f"--model_type=2dgs "
    f"--init_num_pts {point_count} "
    f"--strategy.cap-max {point_count} "
    f"--port 6070 "
    f"--disable_viewer "
    f"--data_factor {data_factor} "
    f"--upscale_start_iter 100000000"
)
if os.path.isdir(f"{output_dir}/2dgs_mcmc/pc{point_count}/{scene}/videos") and not render:
    print(f"[INFO] Skipping scene {scene} for 2DGS as videos directory already exists.")
else:
    print(f"[INFO] Running command for scene {scene}: {cmd_2dgs}")
    os.system(cmd_2dgs)

method_name = f'ntex_{'rgb' if has_rgb else ''}{'a' if has_alpha else ''}'
method_name += '_enlarge'
cmd_ntex = (
    f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
    f"{f"--ckpt {output_dir}/{method_name}/pc{point_count}/{scene}/ckpts/ckpt_29999.pt " if render else ""}"
    f"--data_dir {dataset_dir}/{scene} "
    f"--pretrained_path {output_dir}/2dgs_mcmc/pc{point_count}/{scene}/ckpts/ckpt_29999.pt "
    f"--result_dir {output_dir}/{method_name}/pc{point_count}/{scene} "
    f"--dataset colmap "
    f"--init_type pretrained "
    f"--model_type=textured_gaussians "
    f"--init_num_pts {point_count} "
    f"--strategy.cap-max {point_count} "
    f"--strategy.refine-start-iter=1000000000000 "
    f"{'--textured_rgb ' if has_rgb else ''}"
    f"{'--textured_alpha ' if has_alpha else ''}"
    f"--texture_resolution {texture_resolution} "
    f"--port 6070 "
    f"--disable_viewer "
    f"--data_factor {data_factor} "
    f"--upscale_grad2d=0.00001 "
    f"--upscale_start_iter=0 "
    f"--upscale_stop_iter=1002 "
    f"--upscale_every=500 "
)
print(f"[INFO] Running command for scene {scene}: {cmd_ntex}")
os.system(cmd_ntex)

