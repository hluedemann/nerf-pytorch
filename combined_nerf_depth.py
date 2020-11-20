import argparse
import os
import time

import imageio
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm

from nerf import (
    CfgNode,
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
)


def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def run_nerf(cfg, checkpoint, configargs):

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            render_trans=cfg.rendering.translation
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    checkpoint = torch.load(checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )
    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    render_poses = render_poses.float().to(device)

    # Evaluation loop
    times_per_image = []
    rgb_images = []
    depth_images = []
    disp_images = []
    for i, pose in enumerate(tqdm(render_poses)):
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            pose = pose[:3, :4]
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            rgb_coarse, disp_coarse, acc_coarse, depth_coarse, rgb_fine, disp_fine, acc_fine, depth_fine = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
            disp = disp_fine if disp_fine is not None else disp_coarse
            depth = depth_fine if depth_fine is not None else depth_coarse

        times_per_image.append(time.time() - start)

        rgb_images.append(rgb)
        depth_images.append(depth)
        disp_images.append(disp)
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")

    return rgb_images, depth_images, disp_images


def combine_images(rgb1, depth1, disp1, rgb2, depth2, disp2, configargs):

    os.makedirs(configargs.savedir, exist_ok=True)

    for i in range(len(rgb1)):

        disp1[i][torch.isnan(disp1[i])] = 0.0
        disp2[i][torch.isnan(disp2[i])] = 0.0
        idxs = disp1[i] >= disp2[i]

        rgb = torch.where(idxs[..., None].expand(rgb1[i].shape), rgb1[i], rgb2[i])
        if configargs.savedir:
            savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3], "blender")
            )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config1", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--config2", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint1",
        type=str,
        required=True,
        help="Checkpoint / first pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        required=True,
        help="Checkpoint / second pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--savedir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg1, cfg2 = None, None
    with open(configargs.config1, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg1 = CfgNode(cfg_dict)

    with open(configargs.config2, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg2 = CfgNode(cfg_dict)

    rgb1, depth1, disp1 = run_nerf(cfg1, configargs.checkpoint1, configargs)
    rgb2, depth2, disp2 = run_nerf(cfg2, configargs.checkpoint2, configargs)

    combine_images(rgb1, depth1, disp1, rgb2, depth2, disp2, configargs)


if __name__ == "__main__":
    main()
