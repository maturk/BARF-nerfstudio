import numpy as np
import os
import sys
import time
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import ipdb
import types
import termcolor
import socket
import contextlib
from easydict import EasyDict as edict
from PIL import Image
import cv2

import os
import json
import imageio
from pathlib import Path
import numpy as np
import shutil
from scipy.spatial.transform import Rotation as R

import concurrent.futures
import matplotlib.pyplot as plt
import colour


from colour_checker_detection import colour_checkers_coordinates_segmentation, detect_colour_checkers_segmentation
from colour_checker_detection.detection.segmentation import adjust_image


def generate_density_shards(opt):
    """Generates N shards of images based on density.

    Args:
          waypoints (np.ndarray): Nx7 array of waypoints

    Returns: shard_id2image_filenames (dict): dictionary mapping shard id to list of waypoint ids
    """
    n_shards = opt["n_shards"]
    print(f"Beginning density sharding with n_shards={n_shards}")
    images_dir = Path(opt["base_dir"]) / "data" / opt["scene"] / f'{opt["resolution"]}x'
    image_filenames = {idx: val for idx, val in enumerate(sorted(os.listdir(images_dir)))}

    shard_id2image_filenames = {}

    for idx, image_path in image_filenames.items():
        shard_id = idx % n_shards
        if shard_id in shard_id2image_filenames:
            shard_id2image_filenames[shard_id].append(image_path)
        else:
            shard_id2image_filenames[shard_id] = [image_path]
    print(f"Finished density sharding with {len(shard_id2image_filenames)} shards.")
    return shard_id2image_filenames


def generate_radial_shards(opt, waypoints):
    """Generate shards of images based on radial angle.

    Args:
            waypoints (np.ndarray): Nx7 array of waypoints
            n_shards (int): number of shards to generate
    Returns: shard_id2image_filenames (dict): dictionary mapping shard id to list of waypoints
    """
    n_shards = opt["n_shards"]
    print(f"Beginning radial sharding with n_shards={n_shards}")

    images_dir = Path(opt["base_dir"]) / "data" / opt["scene"] / f'{opt["resolution"]}x'
    image_filenames = {idx: val for idx, val in enumerate(sorted(images_dir.iterdir()))}

    raw_shard_id2image_filenames = {}
    shard_id2image_filenames = {}
    for waypoint_id in range(waypoints.shape[0]):
        C2W = pose2transform(*waypoints[waypoint_id])
        rotation = C2W[:3, :3]
        direction_vector = np.matmul(rotation, -np.array([0, 0, 1]))
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        shard_id = int((np.rad2deg(angle) + 180) // (360 / n_shards))

        if shard_id in raw_shard_id2image_filenames:
            raw_shard_id2image_filenames[shard_id].append(image_filenames[waypoint_id])
        else:
            raw_shard_id2image_filenames[shard_id] = [image_filenames[waypoint_id]]

    min_shard_id = min(raw_shard_id2image_filenames.keys())
    for raw_shard_id, v in raw_shard_id2image_filenames.items():
        shard_id = raw_shard_id - min_shard_id
        shard_id2image_filenames[shard_id] = v

    print(f"Finished radial sharding with {len(shard_id2image_filenames)} shards.")
    return shard_id2image_filenames


def correct_robot_center_of_attention(waypoints):
    rotated_waypoints = waypoints.copy()
    for i in range(len(waypoints)):
        C2W = pose2transform(*waypoints[i])
        theta_1 = np.deg2rad(225)
        C2W = np.dot(rot_X(theta_1), C2W)
        rotated_waypoints[i, :] = np.array(transform2pose(C2W))
    return rotated_waypoints


def init_shard_dir(opt, shard_dirname, image_filenames):
    shard_dir = Path(opt["base_dir"]) / "exp" / (opt["scene"] + f"_{opt['resolution']}x") / shard_dirname
    src_images_dir = Path(opt["base_dir"]) / "data" / opt["scene"] / f'{opt["resolution"]}x'
    shard_images_dir = shard_dir / "images"

    os.makedirs(shard_images_dir, exist_ok=True)

    for image_path in image_filenames:
        src = os.path.join(src_images_dir, image_path)
        dst = os.path.join(shard_images_dir, image_path)
        shutil.copyfile(src, dst)
    return shard_dir


def load_waypoints(opt):
    waypoints_fp = Path(opt["base_dir"]) / "data" / opt["scene"] / "waypoints.npy"
    with open(waypoints_fp, "r") as fin:
        waypoints = np.loadtxt(fin)
    n_waypoints = waypoints.shape[0]
    print(f"Number of waypoints: {n_waypoints}")
    return waypoints


def init_colmap_shard(opt, shard_dir, shard_id, waypoints, shard_id2image_filenames):
    models_dir = shard_dir / "sensor" / "sparse/0"
    images_dir = Path(opt["base_dir"]) / "data" / opt["scene"] / f'{opt["resolution"]}x'

    os.makedirs(models_dir, exist_ok=True)

    # Create empty points3D.txt file
    open(models_dir / "points3D.txt", "w")
    # Write intrinsics into cameras.txt
    shutil.copyfile(opt["cameras_txt_filepath"], models_dir / "cameras.txt")

    image_filenames = shard_id2image_filenames[shard_id]
    n_waypoints = len(image_filenames)
    if n_waypoints < 5:
        print(f" Skipping Shard {shard_id} because it only has {n_waypoints} waypoints")
        return
    print(f"Shard {shard_id} has {n_waypoints} waypoints")

    waypoints_idxs = [int(os.path.splitext(image_filename)[0]) for image_filename in image_filenames]
    curr_shard_waypoints = {idx: waypoints[idx] for idx in waypoints_idxs}
    translate, scale = get_tf_cams(curr_shard_waypoints, target_radius=opt["target_radius"])

    with open(models_dir / "images.txt", "w") as fout:
        fout.write("# Image list with two lines of data per image:\n")
        fout.write("# Image list with two lines of data per image:\n")
        fout.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fout.write(f"# Number of images: {n_waypoints}, mean observations per image: X\n")

        for i, img_name in enumerate(sorted(os.listdir(images_dir))):
            idx = int(os.path.splitext(img_name)[0])
            print(idx, img_name)
            if idx not in curr_shard_waypoints:
                continue
            C2W = pose2transform(*waypoints[idx])
            W2C = transform_pose(C2W, translate, scale)
            theta_2 = np.deg2rad(-180)
            W2C = np.dot(rot_Z(theta_2), W2C)
            assert np.isclose(np.linalg.det(W2C[:3, :3]), 1.0)
            x, y, z, qx, qy, qz, qw = transform2pose(W2C)
            fout.write(f"{i+1} {qw} {qx} {qy} {qz} {x} {y} {z} 1 {img_name}\n \n")


def generate_shards(opt):
    """
    Generate shards of waypoints.
    Sharding techniques:
        radial: Generate shards of waypoints based on radial angle. Must have waypoints a priori.
        density: Generates N shards of waypoints based on density.
    """
    if opt["sharding"] == "radial":
        waypoints = load_waypoints(opt)
        corrected_waypoints = correct_robot_center_of_attention(waypoints)
        shard_id2image_filenames = generate_radial_shards(opt, waypoints)
        n_shards = len(shard_id2image_filenames)
        for shard_id in shard_id2image_filenames.keys():
            shard_dir = init_shard_dir(opt, f"shard{shard_id+1}_of_{n_shards}", shard_id2image_filenames[shard_id])
            if opt["run_sensor_colmap"]:
                init_colmap_shard(opt, shard_dir, shard_id, corrected_waypoints, shard_id2image_filenames)

    elif opt["sharding"] == "density":
        if opt["run_sensor_colmap"]:
            waypoints = load_waypoints(opt)
            corrected_waypoints = correct_robot_center_of_attention(waypoints)
        shard_id2image_filenames = generate_density_shards(opt)
        n_shards = len(shard_id2image_filenames)

        for shard_id in shard_id2image_filenames.keys():
            shard_dir = init_shard_dir(opt, f"shard{shard_id+1}_of_{n_shards}", shard_id2image_filenames[shard_id])
            if opt["run_sensor_colmap"]:
                init_colmap_shard(opt, shard_dir, shard_id, corrected_waypoints, shard_id2image_filenames)

    else:
        print(f"Unrecognized Sharding Method: {opt['sharding']}")


def rename(opt):
    raw_images_dir = Path(opt["base_dir"]) / "data" / (opt["scene"]) / "raw_images"
    images_dir = Path(opt["base_dir"]) / "data" / (opt["scene"]) / "images"
    os.makedirs(images_dir, exist_ok=True)

    for idx, filename in enumerate(sorted(os.listdir(raw_images_dir))):
        #         print(images_dir/ f"{idx+start_idx:04}.jpg")
        shutil.copy(raw_images_dir / filename, images_dir / f"{idx:04}.jpg")


def downsample(opt):
    base_dir = Path(opt["base_dir"])
    data_dir = base_dir / "data" / opt["scene"]
    images_path = data_dir / "images"
    dst_dir = data_dir / f'{opt["resolution"]}x'
    os.makedirs(dst_dir, exist_ok=True)
    if opt["resolution"] == 1:
        print("For resolution 1, no downsampling is needed.")
        for image_path in sorted(Path(images_path).glob("*.jpg")):
            dst = str(dst_dir / f"{image_path.stem}.png")
            print(f"JPG to PNG: {dst}")
            image = cv2.imread(str(image_path))
            cv2.imwrite(dst, image)
    else:
        for image_path in sorted(Path(images_path).glob("*.jpg")):
            image = imageio.imread(image_path)
            dst = str(dst_dir / f"{image_path.stem}.png")
            save_image(dst, image_to_uint8(downsample_image(image, opt["resolution"])))


def blurriness_filter(opt):
    blur_filter_perc = opt["blur_filter_perc"]
    if blur_filter_perc == 0:
        print("No images filtered because blur_filter_perc is 0.")
        return

    base_dir = Path(opt["base_dir"])
    data_dir = base_dir / "data" / (opt["scene"])

    #     image_filenames = sorted((data_dir / "images").iterdir())

    images_paths = {idx: val for idx, val in enumerate(sorted((data_dir / "images").iterdir()))}
    #     print(list(images_paths.values()))

    print("Loading images.")

    def load_image(path):
        with path.open("rb") as f:
            return imageio.imread(f)

    images = list(map(load_image, list(images_paths.values())))

    print("Computing blur scores.")

    def variance_of_laplacian(image: np.ndarray) -> np.ndarray:
        """Compute the variance of the Laplacian which measure the focus."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    blur_scores = np.array([variance_of_laplacian(im) for im in images])
    blur_thres = np.percentile(blur_scores, blur_filter_perc)
    blur_filter_inds = np.where(blur_scores >= blur_thres)[0]
    blur_filter_scores = [blur_scores[i] for i in blur_filter_inds]

    blur_filter_inds = blur_filter_inds[np.argsort(blur_filter_scores)]
    blur_filter_scores = np.sort(blur_filter_scores)
    blur_filter_paths = [images_paths[i] for i in blur_filter_inds]

    print(f"Filtering {len(blur_filter_paths)} IDs: {[path.name for path in blur_filter_paths]}")
    num_filtered = len(blur_filter_paths)
    for blur_filter_path in blur_filter_paths:
        os.remove(blur_filter_path)
    print(f"Filtered {num_filtered} images")

    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.title(f"Least blurry among filtered: {images_paths[blur_filter_inds[-1]].name}")
    plt.imshow(images[blur_filter_inds[-1]])
    plt.subplot(122)
    plt.title(f"Most blurry among filtered: {images_paths[blur_filter_inds[0]].name}")
    plt.imshow(images[blur_filter_inds[0]])


def color_correct(opt):
    calibration_img_path = Path(opt["base_dir"]) / "data" / opt["scene"] / opt["calibration_relative_path"]
    # calibration_img_path = Path(opt["base_dir"])/"exp"/(
    #     opt["scene"]+f"_{opt['resolution']}x") / opt["calibration_relative_path"]
    calibration_image = colour.cctf_decoding(colour.io.read_image(calibration_img_path))

    colour.plotting.plot_image(colour.cctf_encoding(calibration_image))
    seg_results = detect_colour_checkers_segmentation(calibration_image, additional_data=True)
    assert len(seg_results) == 1
    colour_checker_swatches_data = seg_results[0]

    swatch_colours, colour_checker_image, swatch_masks = colour_checker_swatches_data.values

    # Using the additional data to plot the colour checker and masks.
    masks_i = np.zeros(colour_checker_image.shape)
    for i, mask in enumerate(swatch_masks):
        masks_i[mask[0] : mask[1], mask[2] : mask[3], ...] = 1
    colour.plotting.plot_image(colour.cctf_encoding(np.clip(colour_checker_image + masks_i * 0.25, 0, 1)))

    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]

    #     # NOTE: The reference swatches values as produced by the "colour.XYZ_to_RGB"
    #     # definition are linear by default.
    #     # See https://github.com/colour-science/colour-checker-detection/discussions/59
    #     # for more information.

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        REFERENCE_COLOUR_CHECKER.illuminant,
        D65,
        colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
    )

    swatches_xyY = colour.XYZ_to_xyY(
        colour.RGB_to_XYZ(swatch_colours, D65, D65, colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ)
    )

    colour_checker = colour.characterisation.ColourChecker(
        os.path.basename(calibration_img_path), dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_xyY)), D65
    )
    # print(colour_checker)

    colour.plotting.plot_multi_colour_checkers([REFERENCE_COLOUR_CHECKER, colour_checker])

    swatches_f = colour.colour_correction(swatch_colours, swatch_colours, REFERENCE_SWATCHES)
    swatches_f_xyY = colour.XYZ_to_xyY(
        colour.RGB_to_XYZ(swatches_f, D65, D65, colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ)
    )
    colour_checker = colour.characterisation.ColourChecker(
        "{0} - CC".format(os.path.basename(calibration_img_path)),
        dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_f_xyY)),
        D65,
    )

    colour.plotting.plot_multi_colour_checkers([REFERENCE_COLOUR_CHECKER, colour_checker])

    colour.plotting.plot_image(
        colour.cctf_encoding(colour.colour_correction(calibration_image, swatch_colours, REFERENCE_SWATCHES))
    )

    # For each image in the preprocessed dataset, color correct the image using the swatches from the calibration image
    image_dir = Path(opt["base_dir"]) / "data" / opt["scene"] / "images"
    cc_image_dir = Path(opt["base_dir"]) / "data" / opt["scene"] / "cc_images"
    os.makedirs(cc_image_dir, exist_ok=True)
    image_filenames = sorted([f for f in image_dir.iterdir() if f.is_file() and f.suffix in [".jpg", ".png"]])
    for image_filename in image_filenames:
        image = colour.cctf_decoding(colour.io.read_image(image_filename))
        corrected_image = colour.colour_correction(image, swatch_colours, REFERENCE_SWATCHES)
        dst_fp = str(cc_image_dir / image_filename.name)
        print(dst_fp)
        colour.io.write_image(corrected_image, dst_fp)

    colour_checkers, clusters, swatches, segmented_image = colour_checkers_coordinates_segmentation(
        calibration_image, additional_data=True
    ).values

    image_a = adjust_image(calibration_image, SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC["working_width"])

    colour.plotting.plot_image(
        colour.cctf_encoding(segmented_image), text_kwargs={"text": "Segmented Image", "color": "black"}
    )

    cv2.drawContours(image_a, swatches, -1, (1, 0, 1), 3)
    cv2.drawContours(image_a, clusters, -1, (0, 1, 1), 3)

    colour.plotting.plot_image(
        colour.cctf_encoding(image_a), text_kwargs={"text": "Swatches & Clusters", "color": "white"}
    )


def invert_quaternion(q):
    norm = q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2
    return [q[0] / norm, -q[1] / norm, -q[2] / norm, -q[3] / norm]


def filename2id(filename):
    return int(os.path.splitext(filename)[0][17:])


def get_tf_cams(waypoints, target_radius=1.0):
    cam_centers = [waypoints[i][:3] for i in range(len(waypoints))]

    def get_center_and_diag(cam_centers):
        cam_centers = np.vstack(cam_centers)
        center = np.mean(cam_centers, axis=0, keepdims=True)
        dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def transform_pose(C2W, translate, scale):
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.linalg.inv(C2W)


def pose2transform(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = np.array([x, y, z])
    return T


def transform2pose(T):
    x, y, z = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    qx, qy, qz, qw = r.as_quat()
    return x, y, z, qx, qy, qz, qw


def save_image(path, image: np.ndarray) -> None:
    if isinstance(path, str):
        path = Path(path)
    print(f"Saving {path}")
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("wb") as f:
        image = Image.fromarray(np.asarray(image))
        image.save(f, format=path.suffix.lstrip("."))


def random_subset_split(list, train_ratio, val_ratio, test_ratio):
    # Randomly split a list into train, val, and test subsets.
    # The ratios should sum to 1.
    assert train_ratio + val_ratio + test_ratio == 1
    np.random.shuffle(list)
    train_len = int(len(list) * train_ratio)
    val_len = int(len(list) * val_ratio)
    train = list[:train_len]
    val = list[train_len : train_len + val_len]
    test = list[train_len + val_len :]
    return train, val, test


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to a uint8 array."""
    if image.dtype == np.uint8:
        return image
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(f"Input image should be a floating type but is of type {image.dtype!r}")
    return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
    """Trim the image if not divisible by the divisor."""
    height, width = image.shape[:2]
    if height % divisor == 0 and width % divisor == 0:
        return image

    new_height = height - height % divisor
    new_width = width - width % divisor

    return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
    """Downsamples the image by an integer factor to prevent artifacts."""
    if scale == 1:
        return image

    height, width = image.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f"Image shape ({height},{width}) must be divisible by the" f" scale ({scale}).")
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
    return resized


def rot_X(theta):
    R = np.zeros((4, 4))
    R[0, 0] = 1
    R[1, 1] = np.cos(theta)
    R[1, 2] = -np.sin(theta)
    R[2, 1] = np.sin(theta)
    R[2, 2] = np.cos(theta)
    R[3, 3] = 1
    return R


def rot_Y(theta):
    R = np.zeros((4, 4))
    R[0, 0] = np.cos(theta)
    R[0, 2] = np.sin(theta)
    R[2, 0] = -np.sin(theta)
    R[2, 2] = np.cos(theta)
    R[3, 3] = 1
    return R


def rot_Z(theta):
    R = np.zeros((4, 4))
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    R[2, 2] = 1
    R[3, 3] = 1
    return R


# convert to colored strings
def red(message, **kwargs):
    return termcolor.colored(str(message), color="red", attrs=[k for k, v in kwargs.items() if v is True])


def green(message, **kwargs):
    return termcolor.colored(str(message), color="green", attrs=[k for k, v in kwargs.items() if v is True])


def blue(message, **kwargs):
    return termcolor.colored(str(message), color="blue", attrs=[k for k, v in kwargs.items() if v is True])


def cyan(message, **kwargs):
    return termcolor.colored(str(message), color="cyan", attrs=[k for k, v in kwargs.items() if v is True])


def yellow(message, **kwargs):
    return termcolor.colored(str(message), color="yellow", attrs=[k for k, v in kwargs.items() if v is True])


def magenta(message, **kwargs):
    return termcolor.colored(str(message), color="magenta", attrs=[k for k, v in kwargs.items() if v is True])


def grey(message, **kwargs):
    return termcolor.colored(str(message), color="grey", attrs=[k for k, v in kwargs.items() if v is True])


def get_time(sec):
    d = int(sec // (24 * 60 * 60))
    h = int(sec // (60 * 60) % 24)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    return d, h, m, s


def add_datetime(func):
    def wrapper(*args, **kwargs):
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(grey("[{}] ".format(datetime_str), bold=True), end="")
        return func(*args, **kwargs)

    return wrapper


def add_functionname(func):
    def wrapper(*args, **kwargs):
        print(grey("[{}] ".format(func.__name__), bold=True))
        return func(*args, **kwargs)

    return wrapper


def pre_post_actions(pre=None, post=None):
    def func_decorator(func):
        def wrapper(*args, **kwargs):
            if pre:
                pre()
            retval = func(*args, **kwargs)
            if post:
                post()
            return retval

        return wrapper

    return func_decorator


debug = ipdb.set_trace


class Log:
    def __init__(self):
        pass

    def process(self, pid):
        print(grey("Process ID: {}".format(pid), bold=True))

    def title(self, message):
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        print(magenta(message, bold=True))

    def options(self, opt, level=0):
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   " * level + cyan("* ") + green(key) + ":")
                self.options(value, level + 1)
            else:
                print("   " * level + cyan("* ") + green(key) + ":", yellow(value))

    def loss_train(self, opt, ep, lr, loss, timer):
        if not opt.max_epoch:
            return
        message = grey("[train] ", bold=True)
        message += "epoch {}/{}".format(cyan(ep, bold=True), opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr), bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss), bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)), bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)

    def loss_val(self, opt, loss):
        message = grey("[val] ", bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss), bold=True))
        print(message)


log = Log()


def update_timer(opt, timer, ep, it_per_ep):
    if not opt.max_epoch:
        return
    momentum = 0.99
    timer.elapsed = time.time() - timer.start
    timer.it = timer.it_end - timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean * momentum + timer.it * (1 - momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean * it_per_ep * (opt.max_epoch - ep)


# move tensors to device in-place


def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X


def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D


def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}


def restore_checkpoint(opt, model, load_name=None, resume=False):
    # resume can be True/False or epoch numbers
    assert (load_name is None) == (resume is not False)
    if resume:
        load_name = (
            "{0}/model.ckpt".format(opt.output_path)
            if resume is True
            else "{0}/model/{1}.ckpt".format(opt.output_path, resume)
        )
    checkpoint = torch.load(load_name, map_location=opt.device)
    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"], name)
        if child_state_dict:
            print("restoring {}...".format(name))
            child.load_state_dict(child_state_dict)
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print("restoring {}...".format(key))
            getattr(model, key).load_state_dict(checkpoint[key])
    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        if resume is not True:
            assert resume == (ep or it)
        print("resuming from epoch {0} (iteration {1})".format(ep, it))
    else:
        ep, it = None, None
    return ep, it


def save_checkpoint(opt, model, ep, it, latest=False, children=None):
    os.makedirs("{0}/model".format(opt.output_path), exist_ok=True)
    if children is not None:
        graph_state_dict = {k: v for k, v in model.graph.state_dict().items() if k.startswith(children)}
    else:
        graph_state_dict = model.graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            checkpoint.update({key: getattr(model, key).state_dict()})
    torch.save(checkpoint, "{0}/model.ckpt".format(opt.output_path))
    if not latest:
        shutil.copy(
            "{0}/model.ckpt".format(opt.output_path), "{0}/model/{1}.ckpt".format(opt.output_path, ep or it)
        )  # if ep is None, track it instead


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))


@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout:
            old_stdout, sys.stdout = sys.stdout, devnull
        if stderr:
            old_stderr, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            if stdout:
                sys.stdout = old_stdout
            if stderr:
                sys.stderr = old_stderr


def colorcode_to_number(code):
    ords = [ord(c) for c in code[1:]]
    ords = [n - 48 if n < 58 else n - 87 for n in ords]
    rgb = (ords[0] * 16 + ords[1], ords[2] * 16 + ords[3], ords[4] * 16 + ords[5])
    return rgb
