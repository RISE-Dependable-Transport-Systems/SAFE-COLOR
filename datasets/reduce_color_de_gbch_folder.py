import argparse
import os
import glob
import numpy as np
from PIL import Image as PILImage
from skimage import io, color, img_as_ubyte
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging
from scipy.optimize import minimize_scalar

###############################################################################
# 1. Macbeth ColorChecker LAB (D65)
###############################################################################
macbeth_lab = np.array([
    [37.99,  13.56,  14.06],   # Dark Skin
    [65.71,  18.13,  17.81],   # Light Skin
    [49.93,  -4.88, -21.93],   # Blue Sky
    [43.14, -13.10,  21.91],   # Foliage
    [55.11,   8.84, -25.40],   # Blue Flower
    [70.72, -33.40,  -0.20],   # Bluish Green
    [62.66,  36.07,  57.10],   # Orange
    [40.02,  10.41, -45.96],   # Purplish Blue
    [51.12,  48.24,  16.25],   # Moderate Red
    [30.33,  22.98, -21.59],   # Purple
    [72.53, -23.71,  57.26],   # Yellow Green
    [71.94,  19.36,  67.85],   # Orange Yellow
    [28.78,  14.18, -50.30],   # Blue
    [55.26, -38.34,  31.37],   # Green
    [42.10,  53.38,  28.19],   # Red
    [81.73,   4.04,  79.82],   # Yellow
    [51.94,  49.99, -14.57],   # Magenta
    [49.04, -28.63, -28.64],   # Cyan
    [96.54,  -0.48,   1.23],   # White (N9.5)
    [81.26,  -0.53,   0.03],   # Neutral 8 (N8)
    [66.77,  -0.73,  -0.52],   # Neutral 6.5 (N6.5)
    [50.87,  -0.13,  -0.27],   # Neutral 5 (N5)
    [35.66,  -0.46,  -0.48],   # Neutral 3.5 (N3.5)
    [20.46,  -0.08,  -0.31]    # Black (N2)
])

###############################################################################
# 2. ΔE2000 (vectorized)
###############################################################################
def delta_e_cie2000_vectorized(Lab1, Lab2):
    """
    Computes ΔE2000 for matching shapes (..., 3).
    """
    L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]

    L_avg = 0.5 * (L1 + L2)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = 0.5 * (C1 + C2)

    G = 0.5 * (1 - np.sqrt((C_avg**7) / (C_avg**7 + 25**7)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    C_avg_p = 0.5 * (C1p + C2p)

    h1p = np.degrees(np.arctan2(b1, a1p))
    h1p = np.where(h1p < 0, h1p + 360, h1p)
    h2p = np.degrees(np.arctan2(b2, a2p))
    h2p = np.where(h2p < 0, h2p + 360, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    H_avg_p = np.where(np.abs(h1p - h2p) > 180,
                       (h1p + h2p + 360) / 2,
                       (h1p + h2p) / 2)

    T = (1
         - 0.17 * np.cos(np.radians(H_avg_p - 30))
         + 0.24 * np.cos(np.radians(2 * H_avg_p))
         + 0.32 * np.cos(np.radians(3 * H_avg_p + 6))
         - 0.20 * np.cos(np.radians(4 * H_avg_p - 63)))

    Sl = 1 + ((0.015 * (L_avg - 50)**2)
              / np.sqrt(20 + (L_avg - 50)**2))
    Sc = 1 + 0.045 * C_avg_p
    Sh = 1 + 0.015 * C_avg_p * T

    delta_theta = 30 * np.exp(-((H_avg_p - 275) / 25)**2)
    Rc = 2 * np.sqrt((C_avg_p**7) / (C_avg_p**7 + 25**7))
    Rt = -Rc * np.sin(2 * np.radians(delta_theta))

    dE = np.sqrt(
        (dLp / Sl)**2
        + (dCp / Sc)**2
        + (dHp / Sh)**2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return dE

###############################################################################
# 3. Converting Macbeth from LAB to sRGB so we can apply transformations in sRGB
###############################################################################
def lab_to_srgb_batch(lab_patches):
    """
    lab_patches shape: (N,3)
    Return sRGB in shape (N,3), float [0,1].
    """
    # Expand dims to treat it like an image
    lab_image = lab_patches.reshape((1, -1, 3))
    srgb_image = color.lab2rgb(lab_image)
    srgb_patches = srgb_image.reshape((-1, 3))
    return srgb_patches

###############################################################################
# 4. The Single-Parameter Transforms
###############################################################################
def linearize_srgb(srgb):
    """
    Convert sRGB to linear RGB.
    sRGB input should be in [0,1].
    """
    threshold = 0.04045
    linear = np.where(srgb <= threshold, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear

def delinearize_srgb(linear_rgb):
    """
    Convert linear RGB to sRGB.
    linear_rgb input should be in [0,1].
    """
    threshold = 0.0031308
    srgb = np.where(linear_rgb <= threshold, linear_rgb * 12.92, 1.055 * (linear_rgb ** (1/2.4)) - 0.055)
    return srgb

def apply_gamma_srgb(srgb, param):
    """
    Apply gamma correction to sRGB image with proper linearization and delinearization.
    param: Gamma value.
    """
    # Linearize sRGB
    linear_rgb = linearize_srgb(np.clip(srgb, 0, 1))
    # Apply gamma
    gamma_corrected = linear_rgb ** param
    # Delinearize back to sRGB
    srgb_corrected = delinearize_srgb(np.clip(gamma_corrected, 0, 1))
    return np.clip(srgb_corrected, 0, 1)

def apply_brightness_srgb(srgb, param):
    """
    Apply brightness adjustment to sRGB image.
    param in [-0.5 ... 0.5], shift all channels.
    """
    return np.clip(srgb + param, 0, 1)

def apply_contrast_srgb(srgb, param):
    """
    Apply contrast adjustment to sRGB image.
    param >1 increases contrast, param <1 decreases contrast.
    Pivot around 0.5.
    """
    return np.clip(0.5 + param * (srgb - 0.5), 0, 1)

def apply_hue_srgb(srgb, param):
    """
    Apply hue rotation to sRGB image.
    param in degrees: [-180 ... 180]
    """
    hsv = color.rgb2hsv(np.clip(srgb, 0, 1))
    hue_shift = param / 360.0
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
    return color.hsv2rgb(hsv)

def apply_mean_pull_srgb(srgb_image, alpha):
    """
    Pull all colors in an sRGB image toward the image's mean color in LAB space
    by a factor alpha in [0,1].
    alpha=0   => all pixels become the mean color
    alpha=1   => no change
    """
    # Convert to LAB
    lab_image = color.rgb2lab(np.clip(srgb_image, 0, 1))

    # Compute the image's mean LAB
    mean_lab = np.mean(lab_image.reshape(-1, 3), axis=0)
    # # Compute mean of Macbeth patches
    # mean_lab = np.mean(macbeth_lab, axis=0)

    # Pull pixels toward mean
    lab_out = mean_lab + alpha * (lab_image - mean_lab)

    # Convert back to sRGB
    srgb_out = color.lab2rgb(lab_out)
    return np.clip(srgb_out, 0, 1)

def apply_saturation_srgb(srgb, param):
    """
    Apply a saturation factor to an sRGB image.
    param=1 => no change, param>1 => more saturated, param<1 => less saturated, 0 => grayscale
    """
    hsv = color.rgb2hsv(np.clip(srgb, 0, 1))
    hsv[..., 1] = np.clip(hsv[..., 1] * param, 0, 1)
    return color.hsv2rgb(hsv)

###############################################################################
# 5. Single function to measure mean ΔE on Macbeth for a given transform
###############################################################################
def measure_delta_e_for_transform(macbeth_srgb_ref, macbeth_lab_ref,
                                  param, transform):
    """
    Apply a transform to Macbeth sRGB patches and compute the mean ΔE2000.
    """
    if transform == "no_transform":
        srgb_distorted = macbeth_srgb_ref  # No change
    elif transform == "gamma":
        srgb_distorted = apply_gamma_srgb(macbeth_srgb_ref, param)
    elif transform == "brightness":
        srgb_distorted = apply_brightness_srgb(macbeth_srgb_ref, param)
    elif transform == "contrast":
        srgb_distorted = apply_contrast_srgb(macbeth_srgb_ref, param)
    elif transform == "hue":
        srgb_distorted = apply_hue_srgb(macbeth_srgb_ref, param)
    elif transform == "pull_mean":
        srgb_distorted = apply_mean_pull_srgb(macbeth_srgb_ref, param)
    elif transform == "saturation":
        srgb_distorted = apply_saturation_srgb(macbeth_srgb_ref, param)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    # Convert distorted sRGB to LAB
    lab_distorted = color.rgb2lab(srgb_distorted.reshape(1, -1, 3)).reshape(-1, 3)

    # Measure ΔE2000 vs reference LAB
    dE = delta_e_cie2000_vectorized(lab_distorted, macbeth_lab_ref)

    return np.mean(dE)

###############################################################################
# 6. Parameter Search Methods
###############################################################################
def find_param_for_delta_e_optimization(macbeth_srgb, macbeth_lab,
                                        transform: str,
                                        target_de: float,
                                        param_bounds: tuple,
                                        tolerance: float = 0.1):
    """
    Use scipy.optimize.minimize_scalar to find the parameter that achieves target ΔE.
    """
    if transform == "no_transform":
        best_param = 0.0
        best_de = measure_delta_e_for_transform(macbeth_srgb, macbeth_lab, best_param, transform)
        return best_param, best_de

    def objective(param):
        current_de = measure_delta_e_for_transform(macbeth_srgb, macbeth_lab, param, transform)
        return (current_de - target_de) ** 2  # Minimize squared difference

    result = minimize_scalar(
        objective,
        bounds=param_bounds,
        method='bounded',
        options={'xatol': tolerance}
    )

    best_param = result.x
    best_de = measure_delta_e_for_transform(macbeth_srgb, macbeth_lab, best_param, transform)

    return best_param, best_de

###############################################################################
# 7. Apply transform to a real image with the found param
###############################################################################
def apply_transform_to_image(img_srgb, param, transform):
    """
    Apply the specified transform to the input sRGB image.
    """
    if transform == "no_transform":
        return img_srgb  # No change
    elif transform == "gamma":
        return apply_gamma_srgb(img_srgb, param)
    elif transform == "brightness":
        return apply_brightness_srgb(img_srgb, param)
    elif transform == "contrast":
        return apply_contrast_srgb(img_srgb, param)
    elif transform == "hue":
        return apply_hue_srgb(img_srgb, param)
    elif transform == "pull_mean":
        return apply_mean_pull_srgb(img_srgb, param)
    elif transform == "saturation":
        return apply_saturation_srgb(img_srgb, param)
    else:
        raise ValueError(f"Unknown transform: {transform}")

###############################################################################
# 8. Image Processing Function (For Parallelization)
###############################################################################
def process_single_image(img_path, out_folder, transform, param):
    """
    Process a single image: apply the transform and save the result
    with the same JPEG quantization (if it's a JPEG).
    """
    try:
        # 1) Check if the file is JPG/JPEG
        ext = os.path.splitext(img_path)[1].lower()
        qtables = None  # Will store original quantization if present

        if ext in ('.jpg', '.jpeg'):
            # Open with Pillow to get quantization tables
            with PILImage.open(img_path) as pil_img:
                # Not all JPEGs have quantization tables accessible,
                # so we check if this attribute exists and is non-empty
                if hasattr(pil_img, 'quantization') and pil_img.quantization:
                    qtables = pil_img.quantization

        # 2) Read the image data for processing with scikit-image
        #    (We do this second read so we can keep the workflow consistent.)
        img_srgb_in = io.imread(img_path)

        # Convert to float [0,1] if needed
        if img_srgb_in.dtype.kind == 'u':
            img_srgb_in = img_srgb_in.astype(np.float32) / 255.0
        else:
            img_srgb_in = img_srgb_in.astype(np.float32)

        # If grayscale or RGBA, adapt
        if len(img_srgb_in.shape) == 2:
            # Grayscale => replicate channels
            img_srgb_in = np.stack([img_srgb_in]*3, axis=-1)
        elif img_srgb_in.shape[-1] == 4:
            # RGBA => drop alpha
            img_srgb_in = img_srgb_in[..., :3]

        # 3) Apply the chosen transform
        img_srgb_out = apply_transform_to_image(img_srgb_in, param, transform)
        # Convert back to 8-bit
        img_srgb_out_8u = img_as_ubyte(np.clip(img_srgb_out, 0, 1))

        # 4) Save with quantization tables if it was JPEG
        filename = os.path.basename(img_path)
        out_path = os.path.join(out_folder, filename)

        if qtables is not None:
            # We have original JPEG quantization data
            pil_out = PILImage.fromarray(img_srgb_out_8u, mode='RGB')
            # Use the exact tables and preserve subsampling
            pil_out.save(out_path, qtables=qtables)
        else:
            # If not JPEG, or missing quantization data, fall back to scikit-image
            io.imsave(out_path, img_srgb_out_8u, check_contrast=False)

        return f"Saved: {out_path}"
    except Exception as e:
        return f"Failed to process {img_path}: {e}"


###############################################################################
# 9. Main script
###############################################################################
def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Apply a color transform (gamma, brightness, contrast, hue, pull_mean, saturation, or no_transform) so that Macbeth chart hits a target ΔE, then process images in a folder."
    )
    parser.add_argument("input_folder", help="Path to folder with images to process.")
    parser.add_argument("--transform", choices=["gamma","brightness","contrast","hue","pull_mean","saturation","no_transform"],
        default="gamma", help="Type of color transform to use.")
    parser.add_argument("-d", "--deltaE", type=float, default=5.0,
                        help="Target mean ΔE on Macbeth (ignored if no_transform).")
    parser.add_argument("-t", "--tolerance", type=float, default=0.001,
                        help="Convergence tolerance on ΔE (ignored if no_transform).")
    # Ranges for each transform
    parser.add_argument("--gamma_range", type=float, nargs=2, default=[0.1, 10.0],
                        help="Min/Max range for gamma.")
    parser.add_argument("--brightness_range", type=float, nargs=2, default=[-1.0, 1.0],
                        help="Min/Max range for brightness shift.")
    parser.add_argument("--contrast_range", type=float, nargs=2, default=[0.0, 30.0],
                        help="Min/Max range for contrast factor.")
    parser.add_argument("--hue_range", type=float, nargs=2, default=[-180,180],
                        help="Min/Max range for hue rotation (degrees).")
    parser.add_argument("--pull_mean_range", type=float, nargs=2, default=[0.0, 1.0],
                        help="Param range (alpha) for pull_mean transform.")
    parser.add_argument("--saturation_range", type=float, nargs=2, default=[0.0, 2.0],
                        help="Min/Max range for saturation factor.")
    parser.add_argument("-p", "--processes", type=int, default=os.cpu_count(),
                        help="Number of parallel processes to use (default: number of CPU cores).")
    args = parser.parse_args()

    input_folder = os.path.normpath(args.input_folder)
    if not os.path.exists(input_folder):
        logging.error(f"Input folder '{input_folder}' does not exist.")
        return

    # 1) Convert Macbeth from LAB to sRGB
    macbeth_srgb = lab_to_srgb_batch(macbeth_lab)

    # 2) Determine param range (except for no_transform, which doesn't use param)
    if args.transform == "no_transform":
        # We can use dummy bounds for no_transform
        param_min, param_max = (0.0, 0.0)
    elif args.transform == "gamma":
        param_min, param_max = args.gamma_range
    elif args.transform == "brightness":
        param_min, param_max = args.brightness_range
    elif args.transform == "contrast":
        param_min, param_max = args.contrast_range
    elif args.transform == "hue":
        param_min, param_max = args.hue_range
    elif args.transform == "pull_mean":
        param_min, param_max = args.pull_mean_range
    elif args.transform == "saturation":
        param_min, param_max = args.saturation_range
    else:
        raise ValueError("Unknown transform")

    best_param, final_de = find_param_for_delta_e_optimization(
        macbeth_srgb, macbeth_lab,
        transform=args.transform,
        target_de=args.deltaE,
        param_bounds=(param_min, param_max),
        tolerance=args.tolerance
    )

    # 3) Verify final ΔE (for no_transform, this is just the original chart's ΔE vs reference)
    final_de = measure_delta_e_for_transform(macbeth_srgb, macbeth_lab,
                                             best_param, args.transform)
    logging.info(f"Transform: {args.transform}")
    logging.info(f"Found param = {best_param:.4f}")
    logging.info(f"Achieved mean ΔE ~ {final_de:.4f}")

    # 4) Create output folder
    transform_name = args.transform
    if transform_name == "gamma":
        transform_name = "gamma-inc" if best_param >= 1.0 else "gamma-dec"
    elif transform_name == "brightness":
        transform_name = "brightness-inc" if best_param >= 0.0 else "brightness-dec"
    elif transform_name == "contrast":
        transform_name = "contrast-inc" if best_param >= 1.0 else "contrast-dec"
    elif transform_name == "saturation":
        transform_name = "saturation-inc" if best_param >= 1.0 else "saturation-dec"
    elif transform_name == "no_transform":
        transform_name = "no_transform"

    target_de_str = f"{args.deltaE:.1f}".replace(".", "_")
    # input_basename = os.path.basename(os.path.abspath(input_folder.rstrip(os.sep)))
    # out_folder = os.path.join(
    #     os.path.dirname(os.path.abspath(input_folder)),
    #     f"{transform_name}_{param_str}_{input_basename}"
    # )
    out_folder = f"{transform_name}_deltaE{target_de_str}_{input_folder}"
    os.makedirs(out_folder, exist_ok=True)

    # 5) Gather all image paths
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    if not image_paths:
        logging.warning("No images found in the input folder!")
        return

    logging.info(f"Processing {len(image_paths)} images with {args.processes} parallel processes...")

    # 6) Prepare partial function with fixed parameters
    process_func = partial(
        process_single_image, 
        out_folder=out_folder, 
        transform=args.transform, 
        param=best_param
    )

    # 7) Use ProcessPoolExecutor to process images in parallel
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        future_to_img = {executor.submit(process_func, img_path): img_path for img_path in image_paths}
        for future in as_completed(future_to_img):
            result = future.result()
            if result.startswith("Saved:"):
                pass #logging.info(result)
            else:
                logging.error(result)

    logging.info("All done.")

if __name__ == "__main__":
    main()
