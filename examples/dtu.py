import os

dataset_dir = "/workspace/data/Datasets/dtu/DTU"
output_dir = "/workspace/work/Outputs/dtu"
point_count = 100000
texture_resolution = 4
scenes = ['scan24','scan37','scan40','scan55','scan63','scan65','scan69','scan83','scan97','scan105','scan106','scan110','scan114','scan118','scan122']

is_eval = False

for scene in scenes:
    cmd_2dgs = (
        f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
        # f"--ckpt {output_dir}/2dgs_mcmc_pc{point_count//1000}k_t{texture_resolution}/{scene}/ckpts/ckpt_29999.pt "
        f"--max_steps 30000 "
        f"--eval_steps 30000 "
        f"--save_steps 30000 "
        f"--data_dir {dataset_dir}/{scene} "
        f"--result_dir {output_dir}/2dgs_mcmc_pc{point_count//1000}k_t{texture_resolution}/{scene} "
        f"--dataset colmap "
        f"--init_extent 1 "
        f"--init_type sfm "
        f"--model_type=2dgs "
        f"--init_num_pts {point_count} "
        f"--strategy.cap-max {point_count} "
        f"--texture_resolution {texture_resolution} "
        f"--port 6070 "
        f"--eval_steps -1 "
        f"--disable_viewer "
        f"--data_factor 2"
    )
    # print(f"[INFO] Running command for scene {scene}: {cmd_2dgs}")
    # os.system(cmd_2dgs)

    cmd_textured_gaussians = (
        f"CUDA_VISIBLE_DEVICES=0 python simple_trainer_textured_gaussians.py mcmc "
        f"--data_dir {dataset_dir}/{scene} "
        f"--pretrained_path {output_dir}/2dgs_mcmc_pc{point_count//1000}k_t{texture_resolution}/{scene}/ckpts/ckpt_29999.pt "
        f"--result_dir {output_dir}/ntex_rgb_pc{point_count//1000}k_t{texture_resolution}/{scene} "
        f"--dataset colmap "
        f"--init_type pretrained "
        f"--model_type=textured_gaussians "
        f"--init_num_pts {point_count} "
        f"--strategy.cap-max {point_count} "
        f"--strategy.refine-start-iter=1000000000000 "
        f"--textured_rgb "
        f"--textured_alpha "
        f"--texture_resolution {texture_resolution} "
        f"--port 6070 "
        f"--eval_steps -1 "
        f"--disable_viewer "
        f"--data_factor 2"
    )
    print(f"[INFO] Running command for scene {scene}: {cmd_textured_gaussians}")
    os.system(cmd_textured_gaussians)
    break
