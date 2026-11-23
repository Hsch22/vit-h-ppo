"""Run the released offline ViT-H-to-G1 model on one video."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video to G1 motion with the released offline model."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Offline model checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/preprocess"),
        help="Directory for reusable video preprocessing caches.",
    )
    parser.add_argument("-s", "--static-cam", action="store_true", help="Skip visual odometry.")
    parser.add_argument("--use-dpvo", action="store_true", help="Use DPVO instead of SimpleVO.")
    parser.add_argument("--f-mm", type=int, default=None, help="Full-frame focal length in mm.")
    parser.add_argument("--verbose", action="store_true", help="Save preprocessing overlays.")
    parser.add_argument(
        "--imgfeat-encoder",
        choices=("hmr2", "sam3d"),
        default=None,
        help=(
            "Image feature encoder. Omit this for a known release checkpoint; "
            "its recorded encoder is selected automatically. Custom checkpoints "
            "without encoder metadata must set this explicitly."
        ),
    )
    parser.add_argument(
        "--allow-encoder-mismatch",
        action="store_true",
        help="Allow an explicit experimental encoder/checkpoint mismatch.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute every managed preprocessing artifact; preserves camera_K.pt.",
    )
    parser.add_argument(
        "--force-recompute-imgfeat",
        action="store_true",
        help="Recompute only image features and their metadata.",
    )
    parser.add_argument("--sam3d-ckpt-path", type=str, default=None)
    parser.add_argument("--sam3d-mhr-path", type=str, default=None)
    parser.add_argument("--sam3d-root", type=str, default=None)
    parser.add_argument("--sam3d-img-ds", type=float, default=1.0)
    parser.add_argument("--sam3d-batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Overlap frames between adjacent windows (1=legacy hard stitch, 16=recommended cosine/SLERP blend).",
    )
    parser.add_argument(
        "--align-mode",
        choices=("yaw", "so3"),
        default="yaw",
        help="Inter-window rigid alignment mode ('yaw' preserves gravity-view vertical axis; 'so3' matches legacy 3D align).",
    )
    parser.add_argument(
        "--blend-mode",
        choices=("cosine", "linear"),
        default="cosine",
        help="Overlap blending schedule when --overlap > 1.",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("multidiffusion", "none", "inpainting"),
        default="multidiffusion",
        help="Overlap denoising synchronization mode ('multidiffusion'=recommended bidirectional joint denoising, 'none'=independent windows, 'inpainting'=causal prefix lock).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--singular-rewrite",
        choices=("none", "blender"),
        default="none",
        help="Optional visualization-only joint-angle rewrite.",
    )
    parser.add_argument(
        "--exp",
        default="offline/g1_training_set_exp1",
    )
    parser.add_argument("--model-latent-dim", type=int, default=1024)
    parser.add_argument("--model-num-layers", type=int, default=16)
    parser.add_argument("--model-num-heads", type=int, default=16)
    return parser.parse_args()


def build_preprocess_cfg(args):
    """Build config under a held cache lock; prefer preprocessing_session()."""
    from tools.video_preprocess import build_preprocess_cfg as _build

    return _build(args)


def _to_numpy_f32(x):
    import torch

    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _identity_T_w2c(length):
    T = np.repeat(np.eye(4, dtype=np.float32)[None], int(length), axis=0)
    return T


def add_video_camera_data(data, cfg):
    """Store full video-camera matrices for Blender render, without changing model inputs."""
    import torch
    from motion_lib.motion_utils.rotation_conversions import quaternion_to_matrix
    from tools.preprocess_cache import require_cache_lock
    from tools.video_preprocess import get_video_lwh

    require_cache_lock(cfg.output_dir)
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        T_w2c = _identity_T_w2c(length)
    else:
        traj = torch.load(cfg.paths.slam, map_location="cpu")
        if cfg.use_dpvo:
            traj = np.asarray(traj)
            if traj.shape != (length, 7) or not np.isfinite(traj).all():
                raise ValueError(
                    f"Expected DPVO camera trajectory ({length},7), got "
                    f"{traj.shape}: {cfg.paths.slam}"
                )
            traj_quat = torch.as_tensor(traj[:, [6, 3, 4, 5]], dtype=torch.float32)
            if (torch.linalg.vector_norm(traj_quat, dim=-1) < 1e-8).any():
                raise ValueError(f"Zero-length DPVO quaternion: {cfg.paths.slam}")
            R_w2c = quaternion_to_matrix(traj_quat).mT.detach().cpu().numpy().astype(np.float32)
            T_w2c = _identity_T_w2c(length)
            T_w2c[:, :3, :3] = R_w2c
            # DPVO returns T_c2w with the camera center C in world coordinates.
            # Its rigid inverse is [R_c2w.T, -R_c2w.T @ C].
            centers_world = np.asarray(traj[:, :3], dtype=np.float32)
            T_w2c[:, :3, 3] = -np.einsum("tij,tj->ti", R_w2c, centers_world)
        else:
            traj_np = np.asarray(traj, dtype=np.float32)
            if traj_np.shape != (length, 4, 4) or not np.isfinite(traj_np).all():
                raise ValueError(
                    f"Expected SimpleVO camera trajectory ({length},4,4), got "
                    f"{traj_np.shape}: {cfg.paths.slam}"
                )
            T_w2c = traj_np.copy()

    data["T_w2c"] = T_w2c[:length].astype(np.float32)
    data["image_wh"] = np.asarray([width, height], dtype=np.int32)


def make_prep_from_video_data(data, window, device, overlap=1):
    """Convert original VITH demo data dict into G1 pipeline inference tensors."""
    import torch

    from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from post_training.stitching import chunk_windows

    length = int(data["length"].item() if torch.is_tensor(data["length"]) else data["length"])

    obs = select_coco17_no_nose_ears(normalize_kp2d(data["kp2d"], data["bbx_xys"]))
    prep = {
        # Keep full-video tensors on CPU. Long videos can have large 1024-D ViT
        # features, so each sliding window is moved to GPU only when needed.
        "obs_full": obs.to(dtype=torch.float32).cpu(),
        "bbx_xys_full": data["bbx_xys"].to(dtype=torch.float32).cpu(),
        "K_fullimg_full": data["K_fullimg"].to(dtype=torch.float32).cpu(),
        "cam_angvel_full": data["cam_angvel"].to(dtype=torch.float32).cpu(),
        "f_imgseq_full": data["f_imgseq"].to(dtype=torch.float32).cpu(),
        "window": int(window),
        "overlap": int(overlap),
        "total_len": length,
    }
    chunks_meta = chunk_windows(length, window=int(window), overlap=int(overlap))
    prep["chunks_meta"] = chunks_meta
    print(f"[Info] Video length={length}, {len(chunks_meta)} chunks (window={window}, overlap={overlap})")
    return prep, length


def align_window_to_previous(
    pred_root_pos,
    pred_root_quat,
    pred_body_pos,
    previous_root_pos,
    previous_root_quat,
):
    """Rigidly align a later window, including its world-space body points."""
    from motion_lib.motion_utils.rotation_conversions import (
        matrix_to_quaternion,
        quaternion_to_matrix,
    )

    current_origin = pred_root_pos[0].clone()
    previous_rotation = quaternion_to_matrix(previous_root_quat)
    current_rotation = quaternion_to_matrix(pred_root_quat[0])
    delta_rotation = previous_rotation @ current_rotation.transpose(-1, -2)

    aligned_root_pos = (
        (pred_root_pos - current_origin) @ delta_rotation.transpose(-1, -2)
        + previous_root_pos
    )
    aligned_body_pos = pred_body_pos
    if pred_body_pos is not None:
        aligned_body_pos = (
            (pred_body_pos - current_origin) @ delta_rotation.transpose(-1, -2)
            + previous_root_pos
        )
    aligned_root_quat = matrix_to_quaternion(
        delta_rotation.unsqueeze(0) @ quaternion_to_matrix(pred_root_quat)
    )
    return aligned_root_pos, aligned_root_quat, aligned_body_pos


def infer_video(
    pipeline,
    prep,
    total_len,
    device="cuda",
    singular_rewrite="none",
    align_mode="so3",
    blend_mode="cosine",
    sync_mode="none",
):
    import torch
    import torch.nn.functional as F
    from motion_lib.motion_utils.rotation_conversions import (
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
    from post_training.stitching import sample_ddim_synchronized, stitch_sequence_yup
    from tools.inference_utils import reduce_singular_chains

    if sync_mode not in ("none", "multidiffusion", "inpainting"):
        raise ValueError(f"Unsupported sync_mode {sync_mode!r}")

    obs_full = prep["obs_full"]
    bbx_xys_full = prep["bbx_xys_full"]
    K_fullimg_full = prep["K_fullimg_full"]
    cam_angvel_full = prep["cam_angvel_full"]
    f_imgseq_full = prep["f_imgseq_full"]
    chunks_meta = prep["chunks_meta"]
    window = prep["window"]
    if prep.get("full_sequence", False):
        chunks_meta = [(0, total_len)]
        window = total_len
        print(
            f"[Info] full-sequence mode: one forward, length={total_len}; "
            f"network attention max_len remains {prep.get('trained_max_len', 120)}"
        )

    def pad_window(x, pad_len):
        if pad_len <= 0:
            return x
        return torch.cat([x, x[-1:].expand(pad_len, *x.shape[1:])], dim=0)

    n_chunks = len(chunks_meta)
    use_sync = sync_mode != "none" and n_chunks > 1

    if use_sync:
        if not (hasattr(pipeline, "denoiser3d") and hasattr(pipeline, "train_diffusion")):
            raise ValueError(f"sync_mode={sync_mode!r} requires a diffusion pipeline with denoiser3d and train_diffusion")
        from post_training.eval_compare import build_condition
        from post_training.eval_long_horizon import _decode_chunks_yup
        from post_training.policy import G1_PRED_X_DIM

        obs_b, bbx_b, K_b, ang_b, feat_b, len_b = [], [], [], [], [], []
        for start, actual_len in chunks_meta:
            end = start + actual_len
            pad_len = window - actual_len
            obs_b.append(pad_window(obs_full[start:end], pad_len))
            bbx_b.append(pad_window(bbx_xys_full[start:end], pad_len))
            K_b.append(pad_window(K_fullimg_full[start:end], pad_len))
            ang_b.append(pad_window(cam_angvel_full[start:end], pad_len))
            feat_b.append(pad_window(f_imgseq_full[start:end], pad_len))
            len_b.append(actual_len)

        dev = torch.device(device)
        use_bf16 = dev.type == "cuda"
        batch_gpu = {
            "obs": torch.stack(obs_b, dim=0).to(dev),
            "bbx_xys": torch.stack(bbx_b, dim=0).to(dev),
            "K_fullimg": torch.stack(K_b, dim=0).to(dev),
            "cam_angvel": torch.stack(ang_b, dim=0).to(dev),
            "f_imgseq": torch.stack(feat_b, dim=0).to(dev),
            "f_imgseq_mask": torch.ones((n_chunks,), dtype=torch.bool, device=dev),
            "length": torch.tensor(len_b, dtype=torch.long, device=dev),
        }
        f_cond, length_t, all_valid = build_condition(pipeline, batch_gpu, use_bf16=use_bf16)
        x_inits = torch.randn((n_chunks, window, G1_PRED_X_DIM), device=dev, dtype=torch.float32)
        ddim_steps = int(getattr(pipeline, "test_diffusion_steps", 50))
        ddim_eta = float(getattr(pipeline, "ddim_eta", 0.0))
        sync_w_mode = "cosine" if blend_mode == "cosine" else "uniform"
        x0_chunks = sample_ddim_synchronized(
            pipeline=pipeline,
            denoiser=pipeline.denoiser3d,
            f_cond=f_cond,
            length=length_t,
            x_init=x_inits,
            chunks_meta=chunks_meta,
            total_len=total_len,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            all_valid_length=all_valid,
            use_bf16=use_bf16,
            sync_mode=sync_mode,
            sync_weight_mode=sync_w_mode,
        )
        pelvis_h = float(getattr(getattr(getattr(pipeline, "args", None), "robot", None), "default_pelvis_height", 0.793))
        root0_chunks = torch.tensor([[[0.0, pelvis_h, 0.0]]], device=dev, dtype=torch.float32).expand(n_chunks, 1, 3)
        jp_chunks, rp_chunks, rq_chunks, bp_chunks = _decode_chunks_yup(
            pipeline, x0_chunks, batch_gpu["cam_angvel"], length_t, root0_chunks
        )
    else:
        jp_chunks = torch.zeros(n_chunks, window, 29, device=device)
        rp_chunks = torch.zeros(n_chunks, window, 3, device=device)
        rq_chunks = torch.zeros(n_chunks, window, 4, device=device)
        bp_chunks = None

        for chunk_idx, (start, actual_len) in enumerate(chunks_meta):
            end = start + actual_len
            pad_len = window - actual_len
            obs = pad_window(obs_full[start:end], pad_len).to(device)
            bbx_xys = pad_window(bbx_xys_full[start:end], pad_len).to(device)
            K_fullimg = pad_window(K_fullimg_full[start:end], pad_len).to(device)
            cam_angvel = pad_window(cam_angvel_full[start:end], pad_len).to(device)
            f_imgseq = pad_window(f_imgseq_full[start:end], pad_len).to(device)

            batch = {
                "length": torch.tensor([actual_len], device=device, dtype=torch.long),
                "obs": obs.unsqueeze(0),
                "bbx_xys": bbx_xys.unsqueeze(0),
                "K_fullimg": K_fullimg.unsqueeze(0),
                "cam_angvel": cam_angvel.unsqueeze(0),
                "f_imgseq": f_imgseq.unsqueeze(0),
            }

            outputs = pipeline.forward(batch, train=False)
            jp_chunks[chunk_idx, :actual_len] = outputs["pred_g1_joint_pos"][0, :actual_len]
            rp_chunks[chunk_idx, :actual_len] = outputs["pred_root_pos"][0, :actual_len]
            rq_chunks[chunk_idx, :actual_len] = outputs["pred_root_quat_w"][0, :actual_len]
            pred_body = outputs.get("pred_g1_body_pos_w")
            if pred_body is not None:
                pred_body = pred_body[0, :actual_len]
                if bp_chunks is None:
                    bp_chunks = torch.zeros(n_chunks, window, pred_body.shape[1], 3, device=device)
                bp_chunks[chunk_idx, :actual_len] = pred_body

    has_body = bp_chunks is not None
    bp_in = bp_chunks if has_body else torch.zeros(n_chunks, window, 1, 3, device=device)
    fk_fn = None
    if has_body and hasattr(pipeline, "endecoder") and hasattr(pipeline.endecoder, "fk_g1"):
        def _fk_g1_single(jp, rp, rq):
            bp, _ = pipeline.endecoder.fk_g1(jp.unsqueeze(0), rp.unsqueeze(0), rq.unsqueeze(0), world_up="y")
            return bp[0]
        fk_fn = _fk_g1_single

    stitched = stitch_sequence_yup(
        jp_c=jp_chunks,
        rp_c=rp_chunks,
        rq_c=rq_chunks,
        bp_c=bp_in,
        chunks_meta=chunks_meta,
        total_len=total_len,
        align_mode=align_mode,
        blend_mode=blend_mode,
        fk_fn=fk_fn,
    )
    joint_pos_out = stitched["joint_pos"]
    root_pos_out = stitched["root_pos_w"]
    root_quat_out = stitched["root_quat_w"]
    body_pos_out = stitched["body_pos_w"] if has_body else None

    # Match infer_amass2g1.py output convention: convert internal y-up frame to G1 z-up URDF world.
    M_pos = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        device=device,
        dtype=root_pos_out.dtype,
    )
    root_pos_out = root_pos_out @ M_pos
    body_pos_z = body_pos_out @ M_pos if body_pos_out is not None else None
    R_yup = quaternion_to_matrix(root_quat_out)
    root_quat_out = matrix_to_quaternion(M_pos.transpose(-1, -2) @ R_yup)
    root_quat_out = F.normalize(root_quat_out, dim=-1)

    assert singular_rewrite in ("none", "blender"), singular_rewrite
    joint_pos_np = joint_pos_out.cpu().numpy()
    if singular_rewrite == "blender":
        joint_pos_np = reduce_singular_chains(joint_pos_np)

    results = {
        "joint_pos": joint_pos_np,
        "root_pos_w": root_pos_out.cpu().numpy(),
        "root_quat_w": root_quat_out.cpu().numpy(),
    }
    if body_pos_z is not None:
        results["body_pos_w"] = body_pos_z.cpu().numpy()
    return results


def main():
    args = parse_args()

    import torch

    from tools.video_preprocess import load_data_dict, run_preprocess, preprocessing_session
    from tools.inference_utils import (
        inspect_checkpoint_imgfeat_encoder,
        load_pipeline,
        resolve_imgfeat_encoder,
    )
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if args.window != 120:
        raise ValueError("This release was trained with a 120-frame window; use --window 120.")
    if args.overlap < 1 or args.overlap >= args.window:
        raise ValueError(f"--overlap must satisfy 1 <= overlap < window ({args.window}), got {args.overlap}.")

    required_encoder, encoder_provenance = inspect_checkpoint_imgfeat_encoder(args.ckpt)
    args.imgfeat_encoder = resolve_imgfeat_encoder(
        args.imgfeat_encoder,
        required_encoder,
        encoder_provenance,
        allow_mismatch=args.allow_encoder_mismatch,
    )
    args.model_imgfeat_encoder = required_encoder

    _set_seed(args.seed)
    args.video = str(args.video)
    args.ckpt = str(args.ckpt)
    args.output_root = str(args.output_root)
    if args.output is None:
        output = Path("outputs/inference") / f"{Path(args.video).stem}_g1.npz"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with preprocessing_session(args) as cfg:
        run_preprocess(cfg)
        data = load_data_dict(cfg)
        add_video_camera_data(data, cfg)
    data["source_name"] = Path(args.video).stem
    data["source_video"] = args.video
    data["preprocess_dir"] = str(cfg.preprocess_dir)
    data["imgfeat_encoder"] = args.imgfeat_encoder
    data["fps"] = float(cfg.target_fps)

    prep, total_len = make_prep_from_video_data(
        data,
        window=args.window,
        device=args.device,
        overlap=args.overlap,
    )
    pipeline = load_pipeline(
        args.ckpt,
        exp=args.exp,
        device=args.device,
        overrides=[
            f"network.latent_dim={args.model_latent_dim}",
            f"network.num_layers={args.model_num_layers}",
            f"network.num_heads={args.model_num_heads}",
        ],
    )

    with torch.no_grad():
        results = infer_video(
            pipeline,
            prep,
            total_len,
            device=args.device,
            singular_rewrite=args.singular_rewrite,
            align_mode=args.align_mode,
            blend_mode=args.blend_mode,
            sync_mode=args.sync_mode,
        )

    required_lengths = {
        "joint_pos": len(results["joint_pos"]),
        "root_pos_w": len(results["root_pos_w"]),
        "root_quat_w": len(results["root_quat_w"]),
        "K_fullimg": len(data["K_fullimg"]),
        "T_w2c": len(data["T_w2c"]),
    }
    if "body_pos_w" in results:
        required_lengths["body_pos_w"] = len(results["body_pos_w"])
    mismatched = {name: value for name, value in required_lengths.items() if value != total_len}
    if mismatched:
        raise RuntimeError(
            f"Inference output length mismatch for {total_len} staged frames: {mismatched}"
        )

    np.savez(
        output,
        joint_pos=results["joint_pos"],
        root_pos_w=results["root_pos_w"],
        root_quat_w=results["root_quat_w"],
        **({"body_pos_w": results["body_pos_w"]} if "body_pos_w" in results else {}),
        K_fullimg=_to_numpy_f32(data.get("K_fullimg")),
        T_w2c=_to_numpy_f32(data.get("T_w2c")),
        image_wh=np.asarray(data["image_wh"], dtype=np.int32),
        source_video=args.video,
        preprocess_dir=str(data["preprocess_dir"]),
        ckpt=args.ckpt,
        exp=args.exp,
        imgfeat_encoder=args.imgfeat_encoder,
        model_latent_dim=np.array(args.model_latent_dim, dtype=np.int32),
        model_num_layers=np.array(args.model_num_layers, dtype=np.int32),
        model_num_heads=np.array(args.model_num_heads, dtype=np.int32),
        fps=np.array(float(cfg.target_fps), dtype=np.float32),
        source_name=data["source_name"],
        stitch_overlap=np.array(int(args.overlap), dtype=np.int32),
        stitch_align_mode=args.align_mode,
        stitch_blend_mode=args.blend_mode,
        stitch_sync_mode=args.sync_mode,
    )
    print(f"Saved: {output}")
    print(f"joint_pos: {results['joint_pos'].shape}")
    print(f"root_pos_w: {results['root_pos_w'].shape}")
    print(f"root_quat_w: {results['root_quat_w'].shape}")


if __name__ == "__main__":
    main()
