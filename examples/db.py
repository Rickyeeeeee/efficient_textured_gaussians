import os
import itertools
import math

# Settings
dataset_dir = "/workspace/data/tandt_db/db"
output_dir = "/workspace/work/V100_Evaluations/db"
point_counts = [20000, 100000, 200000, 1000000]
texture_resolution = 4
tex_res_start = 1
tex_res_end = 4
data_factor = 1
scenes = [
    'drjohnson',
    'playroom'
]

# Hyperparameters
min_aspect_ratio = 4.0
max_scale_for_thin = 0.01
upscale_grad2d = 0.00002

has_rgb = True
has_alpha = True
render = False
is_eval = False

CUDA_DEVICE_ID=0

for scene, point_count in itertools.product(scenes, point_counts):

    method_name = "2dgs_mcmc"
    cmd_2dgs = (
        f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python simple_trainer_textured_gaussians.py mcmc "
        f"{f"--ckpt {output_dir}/{method_name}/pc{point_count}/{scene}/ckpts/ckpt_29999.pt " if render else ""}"
        f"--max_steps 30000 "
        f"--eval_steps 30000 "
        f"--save_steps 30000 "
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

    method_name = f'ntex_full'
    cmd_ntex = (
        f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python simple_trainer_textured_gaussians.py mcmc "
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
        f"--texture_resolution {tex_res_start} "
        f"--port 6070 "
        f"--disable_viewer "
        f"--data_factor {data_factor} "
        f"--min_aspect_ratio={min_aspect_ratio} "
        f"--max_scale_for_thin={max_scale_for_thin} "
        f"--upscale_grad2d={upscale_grad2d} "
        f"--upscale_start_iter=0 "
        f"--upscale_stop_iter={500*int(math.log2(tex_res_end))+2} "
        f"--upscale_every=500 "
    )
    print(f"[INFO] Running command for scene {scene}: {cmd_ntex}")
    os.system(cmd_ntex)

    method_name = f'textured_gaussians_{'rgb' if has_rgb else ''}{'a' if has_alpha else ''}'
    cmd_textured_gaussians = (
        f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python simple_trainer_textured_gaussians.py mcmc "
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
        f"--texture_resolution {tex_res_end} "
        f"--port 6070 "
        f"--disable_viewer "
        f"--data_factor {data_factor} "
        f"--upscale_grad2d=0.00002 "
        f"--upscale_start_iter=1000000 "
        f"--upscale_stop_iter=1002 "
        f"--upscale_every=500 "
    )
    if not os.path.isdir(f"{output_dir}/{method_name}/pc{point_count}/{scene}/videos") or render:
        print(f"[INFO] Running command for scene {scene}: {cmd_textured_gaussians}")
        os.system(cmd_textured_gaussians)
