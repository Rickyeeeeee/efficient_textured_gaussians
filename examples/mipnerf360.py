import os
import itertools

dataset_dir = "/workspace/data/Datasets/MipNerf360"
output_dir = "/workspace/work/Outputs/MipNerf360"
point_counts = [10000, 50000, 100000]
texture_resolution = 4
data_factor = 4
scenes = [
    'bicycle',
    'bonsai',
    'counter',
    'garden',
    'kitchen',
    'room',
    'stump'
]
has_rgb = True
has_alpha = True

is_eval = False

for scene, point_count in itertools.product(scenes, point_counts):
    cmd_2dgs = (
        f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
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
    if os.path.isdir(f"{output_dir}/2dgs_mcmc/pc{point_count}/{scene}/video"):
        print(f"[INFO] Running command for scene {scene}: {cmd_2dgs}")
        os.system(cmd_2dgs)

    cmd_textured_gaussians = (
        f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
        f"--data_dir {dataset_dir}/{scene} "
        f"--pretrained_path {output_dir}/2dgs_mcmc/pc{point_count}/{scene}/ckpts/ckpt_29999.pt "
        f"--result_dir {output_dir}/textured_gaussians_{'rgb' if has_rgb else ''}{'a' if has_alpha else ''}/pc{point_count}/{scene} "
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
        f"--data_factor {data_factor}"
    )
    if os.path.isdir(f"{output_dir}/textured_gaussians/pc{point_count}/{scene}/video"):
        print(f"[INFO] Running command for scene {scene}: {cmd_textured_gaussians}")
        os.system(cmd_textured_gaussians)
    
