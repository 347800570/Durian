import os
import re
import numpy as np
import astra
import imageio.v2 as imageio
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SART Reconstruction for CT data")

    parser.add_argument(
        '--input_dir',
        required=True,
        type=str,
        help="Directory containing the input .npy projection files"
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help="Directory to save the output slices"
    )

    parser.add_argument(
        '--start_slice',
        type=int,
        default=0,
        help="Start slice index (default: 0)"
    )
    parser.add_argument(
        '--end_slice',
        type=int,
        default=768,
        help="End slice index (default: 768)"
    )
    parser.add_argument(
        '--step_slice',
        type=int,
        default=1,
        help="Slice step size (default: 1)"
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help="Number of full sweeps for SART iterations (default: 3)"
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=768,
        help="Size of the output reconstruction slices (default: 768)"
    )

    parser.add_argument(
        '--cor_offset',
        type=float,
        default=0.0,
        help="Single rotation center offset in detector pixels (default: 0.0)"
    )
    parser.add_argument(
        '--cor_scan_start',
        type=float,
        default=None,
        help="Start value for cor_offset scan"
    )
    parser.add_argument(
        '--cor_scan_end',
        type=float,
        default=None,
        help="End value for cor_offset scan"
    )
    parser.add_argument(
        '--cor_scan_step',
        type=float,
        default=None,
        help="Step value for cor_offset scan"
    )

    return parser.parse_args()


def extract_angle_index(filename):
    stem = os.path.splitext(filename)[0]
    match = re.search(r'(\d+)$', stem)
    if match is None:
        raise ValueError(f"Cannot parse angle index from filename: {filename}")
    return int(match.group(1))


def load_projection_stack_with_angles(input_dir):
    proj_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    if len(proj_files) == 0:
        raise ValueError(f"No .npy files found in directory: {input_dir}")

    proj_files = sorted(proj_files, key=extract_angle_index)
    angle_indices = [extract_angle_index(f) for f in proj_files]

    proj_list = []
    for file in proj_files:
        proj_data = np.load(os.path.join(input_dir, file)).astype(np.float32)
        proj_list.append(proj_data)

    proj = np.stack(proj_list, axis=0)

    print(f"Loaded projection stack with shape: {proj.shape}")
    print(f"Detected filename angle indices: {angle_indices[0]} ~ {angle_indices[-1]}")

    return proj, np.array(angle_indices, dtype=np.int32)


def prepare_parallel_unique_180(proj, angle_indices):
    n_theta = proj.shape[0]

    if n_theta == 361 and angle_indices[0] == 1 and angle_indices[-1] == 361:
        print("Detected 360-degree input from filenames 0001~0361.")
        print("Dropping the last endpoint projection (360° duplicate of 0°).")
        proj = proj[:-1]
        angle_indices = angle_indices[:-1]
        n_theta = proj.shape[0]

    if n_theta % 2 != 0:
        raise ValueError(
            "After endpoint handling, number of projections must be even for "
            f"360° -> 180° cropping, got {n_theta}."
        )

    half_n = n_theta // 2

    print(f"Cropping 360° data to unique 180° set: keep first {half_n} projections.")

    proj = proj[:half_n]
    angle_indices = angle_indices[:half_n]

    angles = np.linspace(0, np.pi, half_n, endpoint=False).astype(np.float32)

    print(f"Reconstruction will use {half_n} projections over [0, π).")
    return proj, angle_indices, angles


def build_cor_offsets(args):
    use_scan = (
        args.cor_scan_start is not None or
        args.cor_scan_end is not None or
        args.cor_scan_step is not None
    )

    if not use_scan:
        return [float(args.cor_offset)], False

    if args.cor_scan_start is None or args.cor_scan_end is None or args.cor_scan_step is None:
        raise ValueError(
            "When using COR scan mode, you must provide "
            "--cor_scan_start, --cor_scan_end, and --cor_scan_step together."
        )

    start = float(args.cor_scan_start)
    end = float(args.cor_scan_end)
    step = float(args.cor_scan_step)

    if step == 0:
        raise ValueError("--cor_scan_step must not be 0.")

    if end > start and step < 0:
        raise ValueError("For increasing scan range, --cor_scan_step must be positive.")
    if end < start and step > 0:
        raise ValueError("For decreasing scan range, --cor_scan_step must be negative.")

    offsets = []
    eps = abs(step) * 1e-8 + 1e-12
    current = start

    if step > 0:
        while current <= end + eps:
            offsets.append(float(current))
            current += step
    else:
        while current >= end - eps:
            offsets.append(float(current))
            current += step

    offsets = [0.0 if abs(v) < 1e-12 else float(v) for v in offsets]

    print(f"COR offsets to reconstruct: {offsets}")
    return offsets, True


def infer_decimal_places(values):
    max_decimals = 0
    for v in values:
        s = f"{v:.10f}".rstrip('0').rstrip('.')
        if '.' in s:
            decimals = len(s.split('.')[-1])
            max_decimals = max(max_decimals, decimals)
    return max_decimals


def format_cor_value(value, decimals):
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def shift_sinogram_horizontal(sino_slice, cor_offset):
    if abs(cor_offset) < 1e-12:
        return sino_slice.astype(np.float32, copy=False)

    n_angles, detector_cols = sino_slice.shape
    x = np.arange(detector_cols, dtype=np.float32)
    shifted = np.empty_like(sino_slice, dtype=np.float32)

    for i in range(n_angles):
        src_x = x - cor_offset
        shifted[i] = np.interp(src_x, x, sino_slice[i], left=0.0, right=0.0)

    return shifted.astype(np.float32)


def main():
    args = parse_args()

    proj, angle_indices = load_projection_stack_with_angles(args.input_dir)
    proj, used_angle_indices, angles = prepare_parallel_unique_180(proj, angle_indices)

    n_theta, H, W = proj.shape
    print(f"Projection stack used for reconstruction: {proj.shape}")
    print(f"Used filename angle indices: {used_angle_indices[0]} ~ {used_angle_indices[-1]}")

    if args.step_slice <= 0:
        raise ValueError(f"Invalid step_slice: {args.step_slice}. Must be > 0.")

    if args.start_slice < 0 or args.end_slice > H or args.start_slice >= args.end_slice:
        raise ValueError(
            f"Invalid slice range: start={args.start_slice}, end={args.end_slice}. "
            f"Must be within [0, {H}) and start < end."
        )

    cor_offsets, is_scan_mode = build_cor_offsets(args)
    cor_decimals = infer_decimal_places(cor_offsets)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(
        f"Reconstructing slices from {args.start_slice} to {args.end_slice - 1} "
        f"with step {args.step_slice}..."
    )

    for cor_offset in cor_offsets:
        cor_label = format_cor_value(cor_offset, cor_decimals)
        print(f"Running reconstruction with cor_offset = {cor_label}")

        for z_idx in range(args.start_slice, args.end_slice, args.step_slice):
            print(f"  Processing slice {z_idx} at cor_offset={cor_label}...")

            sino_slice = proj[:, z_idx, :]
            sino_slice_shifted = shift_sinogram_horizontal(sino_slice, cor_offset)

            det_spacing = 1.0
            proj_geom = astra.create_proj_geom('parallel', det_spacing, W, angles)
            vol_geom = astra.create_vol_geom(args.output_size, args.output_size)

            projector_id = astra.create_projector('linear', proj_geom, vol_geom)
            sinogram_id = astra.data2d.create('-sino', proj_geom, sino_slice_shifted)
            recon_id = astra.data2d.create('-vol', vol_geom, 0)

            cfg = astra.astra_dict('SART')
            cfg['ProjectorId'] = projector_id
            cfg['ProjectionDataId'] = sinogram_id
            cfg['ReconstructionDataId'] = recon_id
            alg_id = astra.algorithm.create(cfg)

            astra.algorithm.run(alg_id, args.iterations * n_theta)

            recon_slice = astra.data2d.get(recon_id)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sinogram_id)
            astra.data2d.delete(recon_id)
            astra.projector.delete(projector_id)

            recon_normalized = (
                (recon_slice - recon_slice.min()) /
                (recon_slice.max() - recon_slice.min() + 1e-8) * 255
            ).astype(np.uint8)

            if is_scan_mode:
                slice_dir = os.path.join(args.output_dir, f"slice_{z_idx:04d}")
                os.makedirs(slice_dir, exist_ok=True)
                output_path = os.path.join(
                    slice_dir,
                    f"slice_{z_idx:04d}_cor_{cor_label}.png"
                )
            else:
                output_path = os.path.join(
                    args.output_dir,
                    f"slice_{z_idx:04d}_cor_{cor_label}.png"
                )

            imageio.imwrite(output_path, recon_normalized)
            print(f"    Saved to: {output_path}")

    print("All specified slices have been reconstructed and saved.")


if __name__ == "__main__":
    main()
