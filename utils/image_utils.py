from PIL import Image
import numpy as np

def apply_mask(image: Image.Image, mask: np.ndarray):
    image = image.resize((mask.shape[1], mask.shape[0]))
    rgba = np.dstack((np.array(image), mask))
    return Image.fromarray(rgba)

def overlay_person(bg: Image.Image, person: Image.Image, x_offset: int, y_offset: int = 0):
    bg = bg.convert("RGBA")
    person = person.convert("RGBA")
    bg.paste(person, (x_offset, y_offset), mask=person)
    return bg

def split_stereo_image(image: Image.Image):
    w, h = image.size
    left = image.crop((0, 0, w // 2, h))
    right = image.crop((w // 2, 0, w, h))
    return left, right

def scale_image(image: Image.Image, scale: float):
    w, h = image.size
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)

def create_anaglyph(left: Image.Image, right: Image.Image):
    left = left.convert("RGB")
    right = right.convert("RGB")
    r, _, _ = left.split()
    _, g, b = right.split()
    anaglyph = Image.merge("RGB", (r, g, b))
    return anaglyph