import os
import re
import numpy as np
import astra
import imageio.v2 as imageio
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SART Reconstruction for CT data")

    parser.add_argument('--input_dir', required=True, type=str,
                        help="Directory containing the input .npy projection files")
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Directory to save the output slices")

    parser.add_argument('--start_slice', type=int, default=0,
                        help="Start slice index (default: 0)")
    parser.add_argument('--end_slice', type=int, default=768,
                        help="End slice index (default: 768)")
    parser.add_argument('--step_slice', type=int, default=1,
                        help="Slice step size (default: 1)")

    parser.add_argument('--iterations', type=int, default=1,
                        help="Number of full sweeps for SART iterations (default: 1)")
    parser.add_argument('--output_size', type=int, default=768,
                        help="Size of the output reconstruction slices (default: 768)")

    parser.add_argument('--relaxation', type=float, default=0.3,
                        help="Relaxation parameter for SART (default: 0.3)")
    parser.add_argument('--nonnegative_mode', type=str, default='per_iteration',
                        choices=['none', 'final', 'per_iteration'],
                        help="Nonnegative constraint mode (default: per_iteration)")

    parser.add_argument('--geom_type', type=str, default='parallel',
                        choices=['parallel', 'fanflat'],
                        help="Geometry type for single-run mode (default: parallel)")

    parser.add_argument('--parallel_projector_type', type=str, default='strip',
                        choices=['line', 'strip', 'linear'],
                        help="Projector type for parallel geometry (default: strip)")

    parser.add_argument('--fanflat_projector_type', type=str, default='line_fanflat',
                        choices=['line_fanflat', 'strip_fanflat'],
                        help="Projector type for fanflat geometry (default: line_fanflat)")

    parser.add_argument('--det_spacing', type=float, default=1.0,
                        help="Detector spacing (default: 1.0)")

    parser.add_argument('--cor_offset', type=float, default=0.0,
                        help="Rotation center offset in detector pixels, applied via geom_postalignment")

    parser.add_argument('--source_origin', type=float, default=500.0,
                        help="Source-to-origin distance for fanflat single-run mode")
    parser.add_argument('--origin_det', type=float, default=500.0,
                        help="Origin-to-detector distance for fanflat single-run mode")

    parser.add_argument('--scan_fanflat', action='store_true',
                        help="Enable fanflat geometry scan over source_origin and origin_det")

    parser.add_argument('--source_origin_scan_start', type=float, default=None,
                        help="Start value for source_origin scan")
    parser.add_argument('--source_origin_scan_end', type=float, default=None,
                        help="End value for source_origin scan")
    parser.add_argument('--source_origin_scan_step', type=float, default=None,
                        help="Step value for source_origin scan")

    parser.add_argument('--origin_det_scan_start', type=float, default=None,
                        help="Start value for origin_det scan")
    parser.add_argument('--origin_det_scan_end', type=float, default=None,
                        help="End value for origin_det scan")
    parser.add_argument('--origin_det_scan_step', type=float, default=None,
                        help="Step value for origin_det scan")

    parser.add_argument('--save_parallel_baseline', action='store_true',
                        help="Also save one parallel reconstruction as baseline during fanflat scan")

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
            f"After endpoint handling, number of projections must be even for 360° -> 180° cropping, got {n_theta}."
        )

    half_n = n_theta // 2
    print(f"Cropping 360° data to unique 180° set: keep first {half_n} projections.")

    proj = proj[:half_n]
    angle_indices = angle_indices[:half_n]
    angles = np.linspace(0, np.pi, half_n, endpoint=False).astype(np.float32)

    print(f"Reconstruction will use {half_n} projections over [0, π).")
    return proj, angle_indices, angles


def infer_decimal_places(values):
    max_decimals = 0
    for v in values:
        s = f"{v:.10f}".rstrip('0').rstrip('.')
        if '.' in s:
            decimals = len(s.split('.')[-1])
            max_decimals = max(max_decimals, decimals)
    return max_decimals


def format_value(value, decimals):
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def build_scan_list(start, end, step, name):
    if start is None or end is None or step is None:
        raise ValueError(f"When scanning {name}, start/end/step must all be provided.")

    start = float(start)
    end = float(end)
    step = float(step)

    if step == 0:
        raise ValueError(f"{name} scan step must not be 0.")
    if end > start and step < 0:
        raise ValueError(f"For increasing {name} scan, step must be positive.")
    if end < start and step > 0:
        raise ValueError(f"For decreasing {name} scan, step must be negative.")

    values = []
    eps = abs(step) * 1e-8 + 1e-12
    current = start

    if step > 0:
        while current <= end + eps:
            values.append(float(current))
            current += step
    else:
        while current >= end - eps:
            values.append(float(current))
            current += step

    values = [0.0 if abs(v) < 1e-12 else float(v) for v in values]
    return values


def build_fanflat_scan_pairs(args):
    if not args.scan_fanflat:
        return None

    so_values = build_scan_list(
        args.source_origin_scan_start,
        args.source_origin_scan_end,
        args.source_origin_scan_step,
        "source_origin"
    )
    od_values = build_scan_list(
        args.origin_det_scan_start,
        args.origin_det_scan_end,
        args.origin_det_scan_step,
        "origin_det"
    )

    pairs = []
    for so in so_values:
        for od in od_values:
            pairs.append((so, od))

    print(f"source_origin scan values: {so_values}")
    print(f"origin_det scan values: {od_values}")
    print(f"Total fanflat combinations: {len(pairs)}")

    return pairs, so_values, od_values


def apply_nonnegative_constraint(recon_slice):
    return np.maximum(recon_slice, 0.0)


def run_sart_with_constraint(alg_id, recon_id, total_steps, nonnegative_mode):
    if nonnegative_mode == 'per_iteration':
        for _ in range(total_steps):
            astra.algorithm.run(alg_id, 1)
            recon_slice = astra.data2d.get(recon_id)
            recon_slice = apply_nonnegative_constraint(recon_slice)
            astra.data2d.store(recon_id, recon_slice)
    else:
        astra.algorithm.run(alg_id, total_steps)


def normalize_to_uint8(recon_slice):
    return (
        (recon_slice - recon_slice.min()) /
        (recon_slice.max() - recon_slice.min() + 1e-8) * 255
    ).astype(np.uint8)


def apply_cor_to_proj_geom(proj_geom, cor_offset):
    if abs(cor_offset) < 1e-12:
        return proj_geom
    return astra.functions.geom_postalignment(proj_geom, cor_offset)


def resolve_projector_type(args, geom_type):
    if geom_type == 'parallel':
        return args.parallel_projector_type
    if geom_type == 'fanflat':
        return args.fanflat_projector_type
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def reconstruct_one_slice(sino_slice, angles, W, args, geom_type, source_origin=None, origin_det=None):
    det_spacing = args.det_spacing

    if geom_type == 'parallel':
        proj_geom = astra.create_proj_geom('parallel', det_spacing, W, angles)
    elif geom_type == 'fanflat':
        if source_origin is None or origin_det is None:
            raise ValueError("fanflat geometry requires source_origin and origin_det.")
        proj_geom = astra.create_proj_geom(
            'fanflat',
            det_spacing,
            W,
            angles,
            float(source_origin),
            float(origin_det)
        )
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    proj_geom = apply_cor_to_proj_geom(proj_geom, args.cor_offset)

    vol_geom = astra.create_vol_geom(args.output_size, args.output_size)
    projector_type = resolve_projector_type(args, geom_type)

    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sino_slice)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)

    cfg = astra.astra_dict('SART')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['option'] = cfg.get('option', {})
    cfg['option']['Relaxation'] = float(args.relaxation)

    alg_id = astra.algorithm.create(cfg)

    total_steps = args.iterations * len(angles)
    run_sart_with_constraint(
        alg_id=alg_id,
        recon_id=recon_id,
        total_steps=total_steps,
        nonnegative_mode=args.nonnegative_mode
    )

    recon_slice = astra.data2d.get(recon_id)

    if args.nonnegative_mode == 'final':
        recon_slice = apply_nonnegative_constraint(recon_slice)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(recon_id)
    astra.projector.delete(projector_id)

    return recon_slice


def save_image(output_path, recon_slice):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.imwrite(output_path, normalize_to_uint8(recon_slice))
    print(f"    Saved to: {output_path}")


def main():
    args = parse_args()

    if args.relaxation <= 0:
        raise ValueError(f"Invalid relaxation: {args.relaxation}. Must be > 0.")
    if args.det_spacing <= 0:
        raise ValueError(f"Invalid det_spacing: {args.det_spacing}. Must be > 0.")
    if args.step_slice <= 0:
        raise ValueError(f"Invalid step_slice: {args.step_slice}. Must be > 0.")

    proj, angle_indices = load_projection_stack_with_angles(args.input_dir)
    proj, used_angle_indices, angles = prepare_parallel_unique_180(proj, angle_indices)

    n_theta, H, W = proj.shape
    print(f"Projection stack used for reconstruction: {proj.shape}")
    print(f"Used filename angle indices: {used_angle_indices[0]} ~ {used_angle_indices[-1]}")

    if args.start_slice < 0 or args.end_slice > H or args.start_slice >= args.end_slice:
        raise ValueError(
            f"Invalid slice range: start={args.start_slice}, end={args.end_slice}. "
            f"Must be within [0, {H}) and start < end."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"Reconstructing slices from {args.start_slice} to {args.end_slice - 1} "
        f"with step {args.step_slice}..."
    )
    print(f"Parallel projector: {args.parallel_projector_type}")
    print(f"Fanflat projector: {args.fanflat_projector_type}")
    print(f"Relaxation: {args.relaxation}")
    print(f"Nonnegative mode: {args.nonnegative_mode}")
    print(f"Detector spacing: {args.det_spacing}")
    print(f"COR offset (geom_postalignment): {args.cor_offset}")

    fanflat_scan_info = build_fanflat_scan_pairs(args)

    if fanflat_scan_info is None:
        print(f"Single-run geometry mode: {args.geom_type}")
        if args.geom_type == 'fanflat':
            print(f"source_origin={args.source_origin}, origin_det={args.origin_det}")

        for z_idx in range(args.start_slice, args.end_slice, args.step_slice):
            print(f"  Processing slice {z_idx}...")
            sino_slice = proj[:, z_idx, :]

            if args.geom_type == 'parallel':
                recon_slice = reconstruct_one_slice(
                    sino_slice=sino_slice,
                    angles=angles,
                    W=W,
                    args=args,
                    geom_type='parallel'
                )
                output_path = os.path.join(
                    args.output_dir,
                    f"slice_{z_idx:04d}_parallel.png"
                )
            else:
                recon_slice = reconstruct_one_slice(
                    sino_slice=sino_slice,
                    angles=angles,
                    W=W,
                    args=args,
                    geom_type='fanflat',
                    source_origin=args.source_origin,
                    origin_det=args.origin_det
                )

                so_decimals = infer_decimal_places([args.source_origin])
                od_decimals = infer_decimal_places([args.origin_det])
                so_label = format_value(args.source_origin, so_decimals)
                od_label = format_value(args.origin_det, od_decimals)

                output_path = os.path.join(
                    args.output_dir,
                    f"slice_{z_idx:04d}_so_{so_label}_od_{od_label}.png"
                )

            save_image(output_path, recon_slice)

    else:
        pairs, so_values, od_values = fanflat_scan_info
        so_decimals = infer_decimal_places(so_values)
        od_decimals = infer_decimal_places(od_values)

        for z_idx in range(args.start_slice, args.end_slice, args.step_slice):
            print(f"  Processing slice {z_idx} for fanflat scan...")
            sino_slice = proj[:, z_idx, :]

            slice_dir = os.path.join(args.output_dir, f"slice_{z_idx:04d}")
            os.makedirs(slice_dir, exist_ok=True)

            if args.save_parallel_baseline:
                print(f"    Saving parallel baseline for slice {z_idx}...")
                recon_parallel = reconstruct_one_slice(
                    sino_slice=sino_slice,
                    angles=angles,
                    W=W,
                    args=args,
                    geom_type='parallel'
                )
                parallel_path = os.path.join(
                    slice_dir,
                    f"slice_{z_idx:04d}_parallel.png"
                )
                save_image(parallel_path, recon_parallel)

            for so, od in pairs:
                so_label = format_value(so, so_decimals)
                od_label = format_value(od, od_decimals)

                print(f"    fanflat so={so_label}, od={od_label}")

                recon_slice = reconstruct_one_slice(
                    sino_slice=sino_slice,
                    angles=angles,
                    W=W,
                    args=args,
                    geom_type='fanflat',
                    source_origin=so,
                    origin_det=od
                )

                output_path = os.path.join(
                    slice_dir,
                    f"slice_{z_idx:04d}_so_{so_label}_od_{od_label}.png"
                )
                save_image(output_path, recon_slice)

    print("All specified slices have been reconstructed and saved.")


if __name__ == "__main__":
    main()