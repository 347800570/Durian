import os
import datetime
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid, PYDICOM_IMPLEMENTATION_UID
# =========================
# 作用：这个脚本用于将npy文件转换为DICOM文件。
# 使用：配置输入路径和输出路径即可。
# =========================

def to_zyx(volume: np.ndarray, input_order: str) -> np.ndarray:
    """
    Convert volume to (Z, Y, X) for writing as DICOM slices.
    input_order:
      - "xyz": volume[x, y, z]
      - "zyx": volume[z, y, x]
    """
    o = input_order.lower()
    if o == "zyx":
        return volume
    if o == "xyz":
        # (X, Y, Z) -> (Z, Y, X)
        return np.transpose(volume, (2, 1, 0))
    raise ValueError(f"Unsupported input_order: {input_order}")


def robust_float_to_uint8(
    vol_f: np.ndarray,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
):
    """
    Robustly map float volume -> uint8 using percentile clipping:
      clip to [lo, hi], then map to [0,255].
    Returns: (vol_u8, lo, hi)
    """
    lo = float(np.percentile(vol_f, p_lo))
    hi = float(np.percentile(vol_f, p_hi))

    if hi <= lo:
        lo = float(vol_f.min())
        hi = float(vol_f.max())
        if hi <= lo:
            # Degenerate case: constant image
            return np.zeros_like(vol_f, dtype=np.uint8), lo, hi

    vol_clip = np.clip(vol_f, lo, hi)
    vol_norm = (vol_clip - lo) / (hi - lo)  # [0,1]
    vol_u8 = (vol_norm * 255.0 + 0.5).astype(np.uint8)
    return vol_u8, lo, hi


def npy_xyz_to_dicom_u8_series(
    npy_path: str,
    out_dir: str,
    spacing_mm: float = 0.14,  # 0.14 mm
    origin_xyz_mm=(0.0, 0.0, 0.0),
    patient_name="fdk_recon",
    patient_id="12345",
    study_description="NPY Reconstruction",
    series_description="NPY->DICOM uint8",
    series_number=1,
    input_order="xyz",
    # uint8 映射参数
    p_lo: float = 0.5,
    p_hi: float = 99.5,
):
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"NPY not found: {npy_path}")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Output dir not found (must exist): {out_dir}")

    vol = np.load(npy_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")

    # Convert to (Z, Y, X)
    vol_zyx = to_zyx(vol, input_order=input_order)
    z, y, x = vol_zyx.shape
    vol_f = vol_zyx.astype(np.float32, copy=False)

    print("Volume stats (float input):")
    print("  shape(Z,Y,X):", vol_f.shape)
    print("  min/max:", float(vol_f.min()), float(vol_f.max()))
    print("  mean/std:", float(vol_f.mean()), float(vol_f.std()))
    print(f"  p{p_lo}/p{p_hi}:", float(np.percentile(vol_f, p_lo)), float(np.percentile(vol_f, p_hi)))

    # Robust mapping float -> uint8
    vol_u8, lo, hi = robust_float_to_uint8(vol_f, p_lo=p_lo, p_hi=p_hi)

    print("Mapping to uint8 using:")
    print(f"  lo={lo}, hi={hi}  -> [0..255]")

    # Shared UIDs (IMPORTANT for stacking into one volume)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    # Date/time
    now = datetime.datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")

    # SOP class: keep CTImageStorage so Slicer treats it as CT scalar volume
    sop_class_uid = CTImageStorage

    sx = sy = sz = float(spacing_mm)
    ox, oy, oz = [float(v) for v in origin_xyz_mm]

    # Orientation (simple axial)
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Window for uint8: you can start with full range
    window_center = 127.5
    window_width = 255.0

    for k in range(z):
        filename = os.path.join(out_dir, f"IM-{k+1:04d}.dcm")

        # File meta
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationVersion = b"\x00\x01"
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False  # Explicit VR

        # Patient / Study / Series
        ds.PatientName = patient_name
        ds.PatientID = patient_id

        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid

        ds.StudyDate = study_date
        ds.StudyTime = study_time
        ds.StudyDescription = study_description

        ds.SeriesNumber = int(series_number)
        ds.SeriesDescription = series_description
        ds.Modality = "CT"

        # SOP / Instance
        ds.SOPClassUID = sop_class_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = k + 1
        ds.ImageType = r"DERIVED\PRIMARY\AXIAL"

        # Pixel matrix
        ds.Rows = y
        ds.Columns = x
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # uint8 pixel format
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0  # 0 = unsigned

        # IMPORTANT: do NOT write RescaleSlope/RescaleIntercept
        # so that Slicer shows raw uint8 values.

        # Spatial (CRITICAL for stacking)
        ds.PixelSpacing = [sy, sx]  # [row_spacing, col_spacing]
        ds.SliceThickness = sz
        ds.SpacingBetweenSlices = sz
        ds.ImageOrientationPatient = iop
        ds.ImagePositionPatient = [ox, oy, oz + k * sz]

        # Window for display
        ds.WindowCenter = float(window_center)
        ds.WindowWidth = float(window_width)

        # Pixel data
        ds.PixelData = vol_u8[k].tobytes()

        ds.save_as(filename, write_like_original=False)

    print(f"\nDone. Wrote {z} uint8 DICOM slices to: {out_dir}")
    print(f"StudyUID={study_uid}")
    print(f"SeriesUID={series_uid}")


if __name__ == "__main__":
    # 你可以改成 input()，也可以保留固定路径
    npy_path = r"D:\Experiment\Durian\project_CTrec\data\Durian_002_20260316\3_rec\FDK\shift_left_px=-55\recon_512.npy"
    out_dir = r"D:\Experiment\Durian\project_CTrec\data\Durian_002_20260316\3_rec\FDK\shift_left_px=-55\dicom"

    npy_xyz_to_dicom_u8_series(
        npy_path=npy_path,
        out_dir=out_dir,
        spacing_mm=0.14,
        input_order="xyz",
        series_description="XYZ(512^3) 0.14mm vox -> DICOM uint8",
        p_lo=0.5,
        p_hi=99.5,
    )
