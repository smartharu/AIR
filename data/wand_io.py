from wand.image import Image, IMAGE_TYPES
import wand
import torch

GRAYSCALE_TYPES = {
    "grayscale",
    "grayscalematte",
    "grayscalealpha",
}
GRAYSCALE_ALPHA_TYPE = "grayscalealpha" if "grayscalealpha" in IMAGE_TYPES else "grayscalematte"
GRAYSCALE_TYPE = "grayscale"
RGBA_TYPE = "truecoloralpha" if "truecoloralpha" in IMAGE_TYPES else "truecolormatte"
RGB_TYPE = "truecolor"
GAMMA_LCD = 45454


def predict_storage(dtype: torch.dtype, int_type: str = "short") -> str:
    if dtype in {torch.float, torch.float32, torch.float16}:
        storage = "float"
    elif dtype in {torch.double, torch.float64}:
        storage = "double"
    elif dtype == torch.uint8:
        storage = "char"
    else:
        storage = int_type
    return storage


def to_wand_image(img: torch.Tensor) -> wand.image:
    ch, h, w = img.shape
    assert (ch in {1, 3})
    if ch == 1:
        channel_map = "I"
    else:
        channel_map = "RGB"

    storage = predict_storage(img.dtype, int_type="long")

    arr = img.permute(1, 2, 0).detach().cpu().numpy()
    ret = Image.from_array(arr, channel_map=channel_map, storage=storage)
    if channel_map == "I":
        ret.type = "grayscale"
    return ret


def to_tensor(img: wand.image, dtype=torch.float32) -> torch.Tensor:
    if img.type in {RGB_TYPE, RGBA_TYPE}:
        channel_map = "RGB"
    elif img.type in {GRAYSCALE_TYPE, GRAYSCALE_ALPHA_TYPE}:
        channel_map = "R"
    else:
        assert (img.type in {RGB_TYPE, RGBA_TYPE, GRAYSCALE_TYPE, GRAYSCALE_ALPHA_TYPE})

    storage = predict_storage(dtype)
    w, h = img.size
    ch = len(channel_map)
    data = img.export_pixels(0, 0, w, h, channel_map=channel_map, storage=storage)
    x = torch.tensor(data, dtype=dtype).view(h, w, ch).permute(2, 0, 1).contiguous()
    del data
    return x
