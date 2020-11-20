import argparse
import os
import time

import imageio
import torch
import yaml
from tqdm import tqdm
from nerf.nerf_helpers import get_minibatches, ndc_rays
from nerf.nerf_helpers import sample_pdf_2 as sample_pdf
from nerf.volume_rendering_utils import volume_render_radiance_field
from nerf.train_utils import run_network
from eval_nerf import cast_to_image, cast_to_disparity_image

from nerf import (
    CfgNode,
    get_ray_bundle,
    load_blender_data,
    models,
    get_embedding_function,
)


def run_nerf(cfgs, checkpoint_dirs, save_dir):

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    render_poses = []

    for cfg in cfgs:
        # Load blender dataset
        _, _, render_p, hwf, _ = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            render_trans=cfg.rendering.translation
        )
        render_poses.append(render_p.float().to(device))

    print("Start rendering scenes ...\n")
    for p in tqdm(range(len(render_poses[0]))):

        radiance_fields_coarse = []
        z_values_coarse = []

        radiance_fields_fine = []
        z_values_fine = []
        ray_directions_fine = []

        for i, cfg in enumerate(cfgs):

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

            checkpoint = torch.load(checkpoint_dirs[i])
            model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])

            if "height" in checkpoint.keys():
                hwf[0] = checkpoint["height"]
            if "width" in checkpoint.keys():
                hwf[1] = checkpoint["width"]
            if "focal_length" in checkpoint.keys():
                hwf[2] = checkpoint["focal_length"]

            model_coarse.eval()

            with torch.no_grad():
                pose = render_poses[i][p]
                pose = pose[:3, :4]
                ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
                radiance_field_batched_coarse, z_batched_coarse, rd_batched_coarse = get_coarse_radiance_field(
                                                 hwf[0],
                                                 hwf[1],
                                                 hwf[2],
                                                 model_coarse,
                                                 ray_origins,
                                                 ray_directions,
                                                 cfg,
                                                 mode="validation",
                                                 encode_position_fn=encode_position_fn,
                                                 encode_direction_fn=encode_direction_fn
                                                 )

            radiance_fields_coarse.append(radiance_field_batched_coarse)
            z_values_coarse.append(z_batched_coarse)

        combined_rfs_coarse = get_combined_radiance_fields(radiance_fields_coarse)

        # Start rendering the finale image

        for i, cfg in enumerate(cfgs):

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

            checkpoint = torch.load(checkpoint_dirs[i])
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

            if model_fine:
                model_fine.eval()

            # Cache shapes now, for later restoration.
            restore_shapes = None

            radiance_fields = []
            z_values = []
            rds = []
            with torch.no_grad():
                pose = render_poses[i][p]
                pose = pose[:3, :4]
                ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
                radiance_field_batched, z_batched, rd_batched = get_fine_radiance_field(
                    hwf[0],
                    hwf[1],
                    hwf[2],
                    combined_rfs_coarse,
                    z_values_coarse[0],
                    model_fine,
                    ray_origins,
                    ray_directions,
                    cfg,
                    mode="validation",
                    encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn
                )
                # Cache shapes now, for later restoration.
            restore_shapes = [
                ray_directions.shape,
                ray_directions.shape[:-1],
                ray_directions.shape[:-1],
                ray_directions.shape[:-1]
            ]

            radiance_fields_fine.append(radiance_field_batched)
            z_values_fine.append(z_batched)
            ray_directions_fine.append(rd_batched)

        combined_rfs = get_combined_radiance_fields(radiance_fields_fine)

        pred = []
        for b in range(len(combined_rfs)):
            rgb, disp, acc, _, depth = volume_render_radiance_field(
                combined_rfs[b],
                z_values_fine[0][b],
                ray_directions_fine[0][b],
                radiance_field_noise_std=getattr(
                    cfg.nerf, "validation").radiance_field_noise_std,
                white_background=getattr(cfg.nerf, "validation").white_background,
            )
            pred.append((rgb, disp, acc, depth))

        synthesized_images = list(zip(*pred))
        synthesized_images = [
            torch.cat(image, dim=0) if image[0] is not None else (None)
            for image in synthesized_images
        ]

        if "validation" == "validation":

            synthesized_images = [
                image.view(shape) if image is not None else None
                for (image, shape) in zip(synthesized_images, restore_shapes)
            ]

            # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
            # (assuming both the coarse and fine networks are used).
            if model_fine:
                synthesized_images = tuple(synthesized_images)
            else:
                # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
                # set to None.
                synthesized_images = tuple(synthesized_images + [None, None, None])
        else:
            synthesized_images = tuple(synthesized_images)

        save_images(synthesized_images, save_dir, title=f"{p:04d}.png", disparity=True)


def save_images(synthesized_images, save_dir, title, disparity=False):

    os.makedirs(save_dir, exist_ok=True)
    if disparity:
        os.makedirs(os.path.join(save_dir, "disparity"), exist_ok=True)

    rgb_map, disp_map, acc_map, depth_map = synthesized_images
    savefile = os.path.join(save_dir, title)
    imageio.imwrite(
        savefile, cast_to_image(rgb_map[..., :3], "blender")
    )

    if disparity:
        savefile = os.path.join(save_dir, "disparity", title)
        imageio.imwrite(savefile, cast_to_disparity_image(disp_map))


def get_combined_radiance_fields(radiance_fields):

    rf_batched = []
    for b in range(len(radiance_fields[0])):
        idx = radiance_fields[0][b][..., -1] >= radiance_fields[1][b][..., -1]

        result = torch.where(idx[..., None].expand(radiance_fields[0][b].shape),
                             radiance_fields[0][b],
                             radiance_fields[1][b])
        for j in range(1, len(radiance_fields)-1):
            idx = result[..., -1] >= radiance_fields[j+1][b][..., -1]
            result = torch.where(idx[..., None].expand(result.shape),
                                 result,
                                 radiance_fields[j + 1][b])

        rf_batched.append(result)

    return rf_batched


def get_coarse_radiance_field(
        height,
        width,
        focal_length,
        model_coarse,
        ray_origins,
        ray_directions,
        options,
        mode="validation",
        encode_position_fn=None,
        encode_direction_fn=None
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)

    radiance_field_batched = []
    z_values_batched = []
    ray_directions_batched = []

    for i, ray_batch in enumerate(batches):
        # TESTED
        num_rays = ray_batch.shape[0]
        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        bounds = ray_batch[..., 6:8].view((-1, 1, 2))
        near, far = bounds[..., 0], bounds[..., 1]

        # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
        # when not enabling "ndc".
        t_vals = torch.linspace(
            0.0,
            1.0,
            getattr(options.nerf, mode).num_coarse,
            dtype=ro.dtype,
            device=ro.device,
        )
        if not getattr(options.nerf, mode).lindisp:
            z_vals = near * (1.0 - t_vals) + far * t_vals
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
        z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

        # pts -> (num_rays, N_samples, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        radiance_field = run_network(
            model_coarse,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )
        radiance_field_batched.append(radiance_field.cpu())
        z_values_batched.append(z_vals.cpu())
        ray_directions_batched.append(rd.cpu())

    return radiance_field_batched, z_values_batched, ray_directions_batched


def get_fine_radiance_field(
        height,
        width,
        focal_length,
        coarse_rf,
        coarse_z_vals,
        model_fine,
        ray_origins,
        ray_directions,
        options,
        mode="train",
        encode_position_fn=None,
        encode_direction_fn=None,
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)

    radiance_field_batched = []
    z_values_batched = []
    rd_batched = []

    for i, ray_batch in enumerate(batches):
        # TESTED
        num_rays = ray_batch.shape[0]
        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        bounds = ray_batch[..., 6:8].view((-1, 1, 2))
        near, far = bounds[..., 0], bounds[..., 1]

        z_vals = coarse_z_vals[i]

        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights_coarse,
            depth_coarse,
        ) = volume_render_radiance_field(
            coarse_rf[i],
            z_vals,
            rd.cpu(),
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
        )

        if getattr(options.nerf, mode).num_fine > 0:

            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                weights_coarse[..., 1:-1],
                getattr(options.nerf, mode).num_fine,
                det=(getattr(options.nerf, mode).perturb == 0.0),
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
            # pts -> (N_rays, N_samples + N_importance, 3)#

            pts = ro[..., None, :] + rd[..., None, :].cuda() * z_vals[..., :, None].cuda()

            radiance_field_fine = run_network(
                model_fine,
                pts,
                ray_batch,
                getattr(options.nerf, mode).chunksize,
                encode_position_fn,
                encode_direction_fn,
            )

            radiance_field_batched.append(radiance_field_fine.cpu())
            z_values_batched.append(z_vals.cpu())
            rd_batched.append(rd.cpu())

    return radiance_field_batched, z_values_batched, rd_batched


def main():

    # TODO: Include argument parser for options like disparity image

    config_dirs = [
        "pretrained/lego-lowres/config.yml",
        "pretrained/chair-lowres/config.yml",
        "pretrained/drums-lowres/config.yml"
    ]
    checkpoint_dirs = [
        "pretrained/lego-lowres/checkpoint199999.ckpt",
        "pretrained/chair-lowres/checkpoint155000.ckpt",
        "pretrained/drums-lowres/checkpoint199999.ckpt"

    ]
    save_dir = "cache/rendered/combined-radiance-lowres"

    # Read config file.
    cfgs = []
    for cfg_dir in config_dirs:

        with open(cfg_dir, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfgs.append(CfgNode(cfg_dict))

    run_nerf(cfgs, checkpoint_dirs, save_dir)

    print("Done!")


if __name__ == "__main__":
    main()

