import os
import itertools
import math

dataset_dir = "/workspace/data/Datasets/MipNerf360"
output_dir = "/workspace/work/Tuning/MipNerf360"
output_dir_for_2dgs = "/workspace/work/Experiments/MipNerf360"
point_counts = [10000, 1000000]
tex_res_start = 1
tex_res_end = 4
data_factor = 4
scenes = [
    # 'room',
    # 'bicycle',
    # 'bonsai',
    # 'counter',
    # 'garden',
    # 'kitchen',
    'stump'
]
has_rgb = True
has_alpha = True
render = False
is_eval = False

min_aspect_ratios = [4.0, 6.0, 10.0]
max_scale_for_thins = [0.01, 0.05, 0.1]

for scene, point_count,min_aspect_ratio, max_scale_for_thin in itertools.product(scenes, point_counts, min_aspect_ratios, max_scale_for_thins):

    method_name = f'ntex_{'rgb' if has_rgb else ''}{'a' if has_alpha else ''}_{min_aspect_ratio}_{max_scale_for_thin}'
    cmd_ntex = (
        f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
        f"{f"--ckpt {output_dir}/{method_name}/pc{point_count}/{scene}/ckpts/ckpt_29999.pt " if render else ""}"
        f"--data_dir {dataset_dir}/{scene} "
        f"--pretrained_path {output_dir_for_2dgs}/2dgs_mcmc/pc{point_count}/{scene}/ckpts/ckpt_29999.pt "
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
        f"--upscale_grad2d=0.00002 "
        f"--upscale_start_iter=0 "
        f"--upscale_stop_iter={500*int(math.log2(tex_res_end))+2} "
        f"--upscale_every=500 "
    )
    if not os.path.isdir(f"{output_dir}/{method_name}/pc{point_count}/{scene}/videos") or render:
        print(f"[INFO] Running command for scene {scene}: {cmd_ntex}")
        os.system(cmd_ntex)

