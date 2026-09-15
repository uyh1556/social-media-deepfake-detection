import io
import random
from pathlib import Path

import timm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as functional


STANDARD_NAME = "standard_center_crop_299"
LETTERBOX_NAME = "full_frame_letterbox_299"
FACE_ROI_NAME = "face_roi_mtcnn_1p5_299"
CONTROLLED_REENCODE_NAME = "canonical256_jpegq95_letterbox299"
MIXED_JPEG_REENCODE_NAME = "canonical256_jpegmixed_letterbox299"
LETTERBOX_FILL_RGB = (128, 128, 128)
FACE_ROI_FILL_RGB = LETTERBOX_FILL_RGB
FACE_ROI_REQUIRED_COLUMNS = {
    "roi_status",
    "roi_center_x_normalized",
    "roi_center_y_normalized",
    "roi_side_width_fraction",
    "roi_side_height_fraction",
}


class Letterbox:
    def __init__(
        self,
        size=299,
        interpolation=InterpolationMode.BICUBIC,
        fill=LETTERBOX_FILL_RGB,
    ):
        self.size = int(size)
        self.interpolation = interpolation
        self.fill = tuple(fill)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image, got {type(image).__name__}")
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image size: {image.size}")
        scale = min(self.size / width, self.size / height)
        resized_width = max(1, min(self.size, round(width * scale)))
        resized_height = max(1, min(self.size, round(height * scale)))
        resized = functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=self.interpolation,
            antialias=True,
        )
        horizontal = self.size - resized_width
        vertical = self.size - resized_height
        left = horizontal // 2
        top = vertical // 2
        return functional.pad(
            resized,
            [left, top, horizontal - left, vertical - top],
            fill=self.fill,
            padding_mode="constant",
        )

    def __repr__(self):
        return (
            f"Letterbox(size={self.size}, interpolation="
            f"{self.interpolation.value}, fill={self.fill})"
        )


class JpegRoundTrip:
    """Apply one deterministic in-memory JPEG encode/decode round trip."""

    def __init__(
        self,
        quality=95,
        subsampling=2,
        optimize=False,
        progressive=False,
    ):
        quality = int(quality)
        subsampling = int(subsampling)
        if not 1 <= quality <= 100:
            raise ValueError(f"JPEG quality must be 1-100, got {quality}")
        if subsampling not in {0, 1, 2}:
            raise ValueError(
                f"JPEG subsampling must be 0, 1, or 2, got {subsampling}"
            )
        self.quality = quality
        self.subsampling = subsampling
        self.optimize = bool(optimize)
        self.progressive = bool(progressive)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image, got {type(image).__name__}")
        buffer = io.BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=self.quality,
            subsampling=self.subsampling,
            optimize=self.optimize,
            progressive=self.progressive,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    def __repr__(self):
        return (
            f"JpegRoundTrip(quality={self.quality}, "
            f"subsampling={self.subsampling}, optimize={self.optimize}, "
            f"progressive={self.progressive})"
        )


class RandomJpegRoundTrip:
    """Apply one JPEG round trip using a uniformly sampled quality."""

    def __init__(
        self,
        qualities,
        subsampling=2,
        optimize=False,
        progressive=False,
    ):
        self.qualities = tuple(int(value) for value in qualities)
        if not self.qualities:
            raise ValueError("At least one JPEG quality is required")
        if len(set(self.qualities)) != len(self.qualities):
            raise ValueError("JPEG quality choices must be unique")
        if any(not 1 <= value <= 100 for value in self.qualities):
            raise ValueError(
                f"JPEG qualities must be 1-100, got {self.qualities}"
            )
        self.subsampling = int(subsampling)
        if self.subsampling not in {0, 1, 2}:
            raise ValueError(
                "JPEG subsampling must be 0, 1, or 2, got "
                f"{self.subsampling}"
            )
        self.optimize = bool(optimize)
        self.progressive = bool(progressive)

    def __call__(self, image):
        quality = random.choice(self.qualities)
        return JpegRoundTrip(
            quality=quality,
            subsampling=self.subsampling,
            optimize=self.optimize,
            progressive=self.progressive,
        )(image)

    def __repr__(self):
        return (
            f"RandomJpegRoundTrip(qualities={self.qualities}, "
            f"subsampling={self.subsampling}, optimize={self.optimize}, "
            f"progressive={self.progressive}, sampling=uniform)"
        )


def interpolation_mode(value):
    modes = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
        "lanczos": InterpolationMode.LANCZOS,
    }
    if value not in modes:
        raise ValueError(f"Unsupported interpolation: {value}")
    return modes[value]


def letterbox_transforms(data_config):
    channels, height, width = tuple(data_config["input_size"])
    if channels != 3 or height != width:
        raise ValueError(
            f"Letterbox requires square RGB input, got {data_config['input_size']}"
        )
    letterbox = Letterbox(
        size=height,
        interpolation=interpolation_mode(data_config["interpolation"]),
    )
    evaluation = transforms.Compose(
        [
            letterbox,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config["mean"], std=data_config["std"]
            ),
        ]
    )
    training = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), evaluation]
    )
    return training, evaluation


def canonical_reencode_transform(
    data_config,
    *,
    canonical_size=256,
    jpeg_quality=95,
    jpeg_subsampling=2,
    jpeg_optimize=False,
    jpeg_progressive=False,
):
    """Build the deterministic canonical-size/JPEG/model-input transform."""
    channels, height, width = tuple(data_config["input_size"])
    if channels != 3 or height != width:
        raise ValueError(
            "Controlled re-encoding requires square RGB model input, got "
            f"{data_config['input_size']}"
        )
    canonical_size = int(canonical_size)
    if canonical_size < 1:
        raise ValueError("canonical_size must be positive")
    steps = [
        Letterbox(
            size=canonical_size,
            interpolation=interpolation_mode(data_config["interpolation"]),
        )
    ]
    if jpeg_quality is not None:
        steps.append(
            JpegRoundTrip(
                quality=jpeg_quality,
                subsampling=jpeg_subsampling,
                optimize=jpeg_optimize,
                progressive=jpeg_progressive,
            )
        )
    steps.extend(
        [
            Letterbox(
                size=height,
                interpolation=interpolation_mode(
                    data_config["interpolation"]
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config["mean"], std=data_config["std"]
            ),
        ]
    )
    return transforms.Compose(steps)


def canonical_reencode_transforms(
    data_config,
    *,
    canonical_size=256,
    jpeg_quality=95,
    jpeg_subsampling=2,
    jpeg_optimize=False,
    jpeg_progressive=False,
):
    evaluation = canonical_reencode_transform(
        data_config,
        canonical_size=canonical_size,
        jpeg_quality=jpeg_quality,
        jpeg_subsampling=jpeg_subsampling,
        jpeg_optimize=jpeg_optimize,
        jpeg_progressive=jpeg_progressive,
    )
    training = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), evaluation]
    )
    return training, evaluation


def canonical_mixed_reencode_transforms(
    data_config,
    *,
    canonical_size=256,
    train_jpeg_qualities=(75, 80, 85, 90, 95),
    validation_jpeg_quality=95,
    jpeg_subsampling=2,
    jpeg_optimize=False,
    jpeg_progressive=False,
):
    """Build mixed-quality training and fixed-quality validation transforms."""
    channels, height, width = tuple(data_config["input_size"])
    if channels != 3 or height != width:
        raise ValueError(
            "Mixed JPEG re-encoding requires square RGB model input, got "
            f"{data_config['input_size']}"
        )
    canonical_size = int(canonical_size)
    if canonical_size < 1:
        raise ValueError("canonical_size must be positive")
    interpolation = interpolation_mode(data_config["interpolation"])
    training = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            Letterbox(size=canonical_size, interpolation=interpolation),
            RandomJpegRoundTrip(
                qualities=train_jpeg_qualities,
                subsampling=jpeg_subsampling,
                optimize=jpeg_optimize,
                progressive=jpeg_progressive,
            ),
            Letterbox(size=height, interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config["mean"], std=data_config["std"]
            ),
        ]
    )
    evaluation = canonical_reencode_transform(
        data_config,
        canonical_size=canonical_size,
        jpeg_quality=validation_jpeg_quality,
        jpeg_subsampling=jpeg_subsampling,
        jpeg_optimize=jpeg_optimize,
        jpeg_progressive=jpeg_progressive,
    )
    return training, evaluation


def validate_face_roi_columns(frame):
    missing = FACE_ROI_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Face-ROI manifest missing columns: {sorted(missing)}"
        )
    unexpected = set(frame["roi_status"].dropna().unique()) - {
        "detected",
        "fallback_letterbox",
    }
    if unexpected:
        raise ValueError(f"Unexpected Face-ROI statuses: {sorted(unexpected)}")


def crop_face_roi(image, row, fallback_size=299):
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image).__name__}")
    status = row["roi_status"]
    if status == "fallback_letterbox":
        return Letterbox(size=fallback_size, fill=FACE_ROI_FILL_RGB)(image)
    if status != "detected":
        raise ValueError(f"Unsupported Face-ROI status: {status!r}")

    width, height = image.size
    center_x = float(row["roi_center_x_normalized"]) * width
    center_y = float(row["roi_center_y_normalized"]) * height
    side_from_width = float(row["roi_side_width_fraction"]) * width
    side_from_height = float(row["roi_side_height_fraction"]) * height
    side = max(1, int(round((side_from_width + side_from_height) / 2.0)))
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = left + side
    bottom = top + side

    canvas = Image.new("RGB", (side, side), FACE_ROI_FILL_RGB)
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        raise ValueError("Face-ROI lies completely outside the image.")
    visible = image.crop(
        (source_left, source_top, source_right, source_bottom)
    )
    canvas.paste(visible, (source_left - left, source_top - top))
    return canvas


def face_roi_transforms(data_config):
    channels, height, width = tuple(data_config["input_size"])
    if channels != 3 or height != width:
        raise ValueError(
            f"Face-ROI requires square RGB input, got {data_config['input_size']}"
        )
    resize = transforms.Resize(
        [height, width],
        interpolation=interpolation_mode(data_config["interpolation"]),
        antialias=True,
    )
    evaluation = transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config["mean"], std=data_config["std"]
            ),
        ]
    )
    training = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), evaluation]
    )
    return training, evaluation


def evaluation_transform_from_checkpoint(checkpoint):
    config = checkpoint["config"]
    name = config.get("preprocessing_name", STANDARD_NAME)
    data_config = config["data_config"]
    if name == LETTERBOX_NAME:
        _, evaluation = letterbox_transforms(data_config)
        return evaluation
    if name == FACE_ROI_NAME:
        _, evaluation = face_roi_transforms(data_config)
        return evaluation
    if name == CONTROLLED_REENCODE_NAME:
        preprocessing = config.get("preprocessing", {})
        _, evaluation = canonical_reencode_transforms(
            data_config,
            canonical_size=preprocessing["canonical_size"],
            jpeg_quality=preprocessing["jpeg_quality"],
            jpeg_subsampling=preprocessing["jpeg_subsampling"],
            jpeg_optimize=preprocessing["jpeg_optimize"],
            jpeg_progressive=preprocessing["jpeg_progressive"],
        )
        return evaluation
    if name == MIXED_JPEG_REENCODE_NAME:
        preprocessing = config.get("preprocessing", {})
        return canonical_reencode_transform(
            data_config,
            canonical_size=preprocessing["canonical_size"],
            jpeg_quality=preprocessing["validation_jpeg_quality"],
            jpeg_subsampling=preprocessing["jpeg_subsampling"],
            jpeg_optimize=preprocessing["jpeg_optimize"],
            jpeg_progressive=preprocessing["jpeg_progressive"],
        )
    if name == STANDARD_NAME or "preprocessing_name" not in config:
        return timm.data.create_transform(
            **data_config, is_training=False
        )
    raise ValueError(f"Unsupported checkpoint preprocessing: {name}")


def strip_leading_component(path, component):
    relative = Path(path)
    if component is None:
        return relative.as_posix()
    if not relative.parts or relative.parts[0] != component:
        raise ValueError(f"Path does not start with {component!r}: {path}")
    return Path(*relative.parts[1:]).as_posix()
