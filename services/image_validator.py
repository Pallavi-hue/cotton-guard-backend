from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def validate_image(image_bytes: bytes) -> dict:
    """
    Validate image bytes without saving an uploaded file to disk.

    Returns a structured result. This validates image integrity,
    not whether the image contains a cotton leaf.
    Normalizes EXIF orientation so phone captures are upright.
    """

    if not image_bytes:
        return {
            "valid": False,
            "reason": "EMPTY_IMAGE",
            "message": "No image data was provided.",
        }

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return {
            "valid": False,
            "reason": "IMAGE_TOO_LARGE",
            "message": "The image exceeds the 10 MB limit.",
        }

    try:
        # First pass: inspect the file structure.
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size

            if image_format not in ALLOWED_FORMATS:
                return {
                    "valid": False,
                    "reason": "UNSUPPORTED_FORMAT",
                    "message": "Use a JPEG, PNG, or WebP image.",
                }

            if width <= 0 or height <= 0:
                return {
                    "valid": False,
                    "reason": "INVALID_DIMENSIONS",
                    "message": "The image dimensions are invalid.",
                }

            if width * height > MAX_IMAGE_PIXELS:
                return {
                    "valid": False,
                    "reason": "IMAGE_TOO_MANY_PIXELS",
                    "message": "The image dimensions are too large.",
                }

            image.verify()

        # Second pass: fully decode the image and normalize EXIF orientation.
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            transposed = ImageOps.exif_transpose(image)
            image_rgb = (transposed if transposed is not None else image).convert("RGB")
            t_width, t_height = image_rgb.size

            return {
                "valid": True,
                "reason": None,
                "message": "Image validated successfully.",
                "format": image_format,
                "width": t_width,
                "height": t_height,
                "image_rgb": image_rgb,
            }

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return {
            "valid": False,
            "reason": "INVALID_IMAGE",
            "message": "The file is not a valid, readable image.",
        }
