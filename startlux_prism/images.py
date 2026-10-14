"""Images in a request's evidence, as StartLux-Decision v1.1 reads them (startlux_decide/images.py and the backends'
use of the model's processor, Apache-2.0): decoding, the processor's resize, each image's place in the prompt, its
tokens and the decoder's three-axis positions.

* Request: a top-level "images" list of base64 strings or data URIs (at most 8 by default, `--max-images`).  A string
  state (or the text of a JSON state) with exactly n "<image>" marks gets an image at each mark, in order; without marks
  the images come first (one image alone, several as "Image 1: ...", "Image 2: ..." lines), then a blank line and the
  state (place_images).  An empty list is no images.
* Pixels: each image is decoded with Pillow, converted to RGB, and resized with Pillow's bicubic filter so that both
  sides are multiples of 32 and it has 65,536 to 1,048,576 pixels at about its aspect ratio (smart_resize) - what
  transformers' image processor (its Pillow backend) and the package's llama.cpp engine feed the vision tower.  The
  tower (vision.py) rescales, normalizes and cuts it into 16-pixel patches.
* Tokens: in the prompt an image is <|vision_start|><|image_pad|><|vision_end|>, and its <|image_pad|> becomes one token
  per 32 x 32 pixels (rows x columns of the merged grid): a 1024 x 1024 image is 1,024 tokens.  The decoder reads the
  tower's features in place of those tokens' embeddings.
* Positions (Qwen3.5's M-RoPE, transformers' get_rope_index): every text token takes the next position on all three
  axes (time, height, width); an image's tokens all sit at the image's start position s on the time axis, at
  s + row and s + column on the others, and the text after the image continues at s + max(rows, columns).  So with
  images a token's position is no longer its index: Row.pos holds them, Row.last_pos the last prompt token's (a text
  token, the same on all three axes; a token generated after the prompt takes last_pos + 1 on every axis).

The image's identity (Image.digest: its bytes as sent and its size after the resize) goes into the prefix cache's key
and into the planner's sharing of evidence between requests, since image tokens are the same ids for every image.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import io
import math
import re
import threading

import numpy as np

from . import jevfmt as J
from . import prompt as P

IMAGE_MARK = "<image>"                      # in a string state: where the next image goes
SLOT = "\x00IMAGE\x00"                      # an image's place in a rendered question, before the token placeholder
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = "<|image_pad|>"                 # the token that becomes the image's tokens
FACTOR = 32                                 # pixels per image token side: 16-pixel patches, merged 2 x 2
MIN_PIXELS = 65536
MAX_PIXELS = 1 << 20
MAX_IMAGES = 8                              # images per request (startlux-decide serve's default, --max-images)
_DATA_URI = re.compile(r"^data:(image/[\w.+-]+)?(;[\w=.+-]+)*;base64,", re.I)


class ImageError(ValueError):
    """A request whose images cannot be read (HTTP 422), with the location of the offending field."""

    def __init__(self, message, loc=("body", "images")):
        super().__init__(message)
        self.loc = list(loc)


@dataclasses.dataclass(frozen=True)
class Settings:
    """How a server reads images.  refused: why it reads none ("" when it reads them)."""
    max_images: int = MAX_IMAGES
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS
    factor: int = FACTOR                    # pixels per image token side (patch size x spatial merge)
    pad_id: int = 248056                    # <|image_pad|> (config.json image_token_id)
    refused: str = ""


@dataclasses.dataclass(eq=False)
class Image:
    """One image of a request, as the vision tower reads it."""
    rgb: np.ndarray             # [height, width, 3] uint8 after the processor's resize
    digest: bytes               # identity: hash of the bytes as sent, and the size
    factor: int = FACTOR

    @property
    def rows(self):
        return self.rgb.shape[0] // self.factor

    @property
    def cols(self):
        return self.rgb.shape[1] // self.factor

    @property
    def tokens(self):
        return self.rows * self.cols


def place_images(text, n):
    """The state text with the place (SLOT) of n images (module docstring); raises ImageError for marks that do not
    match the images."""
    marks = text.count(IMAGE_MARK)
    if marks == n:
        return text.replace(IMAGE_MARK, SLOT)
    if marks:
        raise ImageError(f"the state has {marks} {IMAGE_MARK} marks for {n} images", ("body", "state"))
    head = SLOT if n == 1 else "\n".join(f"Image {k + 1}: {SLOT}" for k in range(n))
    return head + ("\n\n" + text if text.strip() else "")


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """The size the image processor resizes an image to (transformers' Qwen2-VL smart_resize): both sides multiples of
    `factor`, between min_pixels and max_pixels pixels, close to the original aspect ratio."""
    if max(height, width) / min(height, width) > 200:
        raise ImageError(f"an image's sides may differ by a factor of 200 at most, not {height} x {width}")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def decode(value):
    """A base64 string or a data URI -> the encoded image bytes.  Strict base64 first, in C; whatever that refuses gets
    v1.1's rule (whitespace removed, characters validated), so the strings accepted and their bytes are v1.1's
    (tests/test_vision.py).  v1.1's rule alone runs regular expressions over the whole string: 50 ms for 3 MB."""
    if not isinstance(value, str):
        raise ImageError(f"an image must be a base64 string or a data URI, not {type(value).__name__}")
    m = _DATA_URI.match(value)
    payload = value[m.end():] if m else value
    try:
        return binascii.a2b_base64(payload, strict_mode=True)
    except (binascii.Error, ValueError):
        pass
    try:
        return base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("an image must be a base64 string or a data URI") from None


_POOL, _POOL_LOCK = None, threading.Lock()  # decoding threads (Pillow releases the GIL while it decodes and resizes)


def _load(value, k, settings):
    from PIL import Image as PILImage, UnidentifiedImageError
    raw = decode(value)
    try:
        img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as e:
        raise ImageError(f"image {k + 1} is not a readable image ({e})") from None
    h, w = smart_resize(img.height, img.width, settings.factor, settings.min_pixels, settings.max_pixels)
    rgb = np.array(img.resize((w, h), resample=PILImage.Resampling.BICUBIC), dtype=np.uint8)
    digest = hashlib.blake2b(raw, digest_size=16).digest() + np.asarray([h, w], np.int32).tobytes()
    return Image(rgb, digest, settings.factor)


def load_images(values, settings=Settings()):
    """A request's "images" -> [Image]: decoded, converted to RGB and resized as the processor does (Pillow, bicubic),
    several at once in threads; raises ImageError for anything unreadable (the first such image)."""
    global _POOL
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        raise ImageError("this server cannot read images: install Pillow") from None
    if not isinstance(values, (list, tuple)):
        raise ImageError("images must be a list of base64 strings or data URIs")
    if len(values) > settings.max_images:
        raise ImageError(f"at most {settings.max_images} images per request, not {len(values)}")
    if len(values) < 2:
        return [_load(v, k, settings) for k, v in enumerate(values)]
    with _POOL_LOCK:
        if _POOL is None:
            from concurrent.futures import ThreadPoolExecutor
            _POOL = ThreadPoolExecutor(MAX_IMAGES, thread_name_prefix="startlux-prism-images")
    futures = [_POOL.submit(_load, v, k, settings) for k, v in enumerate(values)]
    return [f.result() for f in futures]


def prepare(state, values, settings):
    """A request's state and images -> (the state as text with the images' places, [Image]); raises ImageError (a
    server that reads no images refuses them with settings.refused)."""
    if settings.refused:
        raise ImageError(settings.refused)
    images = load_images(values, settings)
    return place_images(J.state_text(state), len(images)), images


# ---------------------------------------------------------------- tokens and positions

def positions(ids, offsets, images, start=0):
    """M-RoPE positions [3, len(ids)] int32 of a token sequence whose images' tokens start at `offsets`, and the next
    position after it (the get_rope_index rule, module docstring)."""
    n = len(ids)
    pos = np.empty((3, n), np.int32)
    cur, i = start, 0
    for off, im in zip(offsets, images):
        pos[:, i:off] = cur + np.arange(off - i, dtype=np.int32)
        cur += off - i
        r, c = im.rows, im.cols
        j = off + r * c
        pos[0, off:j] = cur
        pos[1, off:j] = cur + np.repeat(np.arange(r, dtype=np.int32), c)
        pos[2, off:j] = cur + np.tile(np.arange(c, dtype=np.int32), r)
        cur += max(r, c)
        i = j
    pos[:, i:] = cur + np.arange(n - i, dtype=np.int32)
    return pos, cur + n - i


def expand(ids, images, pad_id):
    """Prompt tokens with one <|image_pad|> per image -> (the tokens with each image's tokens in its place, the offset
    of each image's first token)."""
    ids = np.asarray(ids, np.int32)
    at = np.flatnonzero(ids == pad_id)
    if len(at) != len(images):
        raise ImageError(f"the prompt has {len(at)} {IMAGE_PAD} tokens for {len(images)} images: the evidence must not "
                         f"hold {IMAGE_PAD} as text")
    counts = np.ones(len(ids), np.int64)
    counts[at] = [im.tokens for im in images]
    offsets = at + np.concatenate([[0], np.cumsum(counts[at] - 1)[:-1]]).astype(np.int64) if len(at) else at
    return np.repeat(ids, counts), [int(o) for o in offsets]


def rows_for(tokenizer, records, keys, images, settings=Settings()):
    """Rendered records whose state holds the images' places (SLOT) -> prompt.Rows with the image tokens, positions
    and images (prompt.rows_for for prompts with images).  The images are in the evidence, so in every row's head;
    the question tails are text, positioned after the head."""
    heads, tails, orders = [], [], []
    for r in records:
        t, o = P.render_text(r)
        h, tl = P.split_prompt(t.replace(SLOT, IMAGE_PLACEHOLDER))
        heads.append(h)
        tails.append(tl)
        orders.append(o)
    head = {}
    for h in set(heads):
        ids, offsets = expand(tokenizer.encode_head(h), images, settings.pad_id)
        pos, nxt = positions(ids, offsets, images)
        head[h] = (ids, pos, nxt, tuple(zip(offsets, images)))
    out = []
    for k, r, h, t, o in zip(keys, records, heads, tokenizer.encode_tails(tails), orders):
        ids, pos, nxt, placed = head[h]
        if settings.pad_id in t:
            raise ImageError(f"a question holds {IMAGE_PAD} as text", ("body", "questions", k))
        tail = np.asarray(t, np.int32)
        tail_pos = np.broadcast_to(nxt + np.arange(len(t), dtype=np.int32), (3, len(t)))
        out.append(P.Row(k, r["type"], np.concatenate([ids, tail]).tolist(), len(o), o, r, len(ids),
                         pos=np.concatenate([pos, tail_pos], 1), images=placed))
    return out
