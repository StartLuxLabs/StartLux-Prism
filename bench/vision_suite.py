"""Requests with images, synthesized with Pillow (deterministic): shapes and colours, rendered text (an invoice, signs,
a shipping label), counting, several images per request (1 to 8, "<image>" marks and none), sizes from 32 x 32 to
2,400 x 1,600 pixels and aspect ratios up to 20:1.  Every question has a right answer (`answer`: the option id, or
"true" / "false" for yes/no, or the level index for scores), so a run can also be scored.

    requests() -> [(name, state, questions, images (data URIs), answers {question key: right answer})]
"""
import base64
import io
import random

from PIL import Image, ImageDraw, ImageFont

COLORS = {"red": (215, 35, 35), "green": (35, 165, 60), "blue": (35, 70, 215), "yellow": (240, 210, 30),
          "black": (20, 20, 20), "orange": (240, 130, 20)}


def _uri(im, kind="PNG"):
    b = io.BytesIO()
    im.save(b, kind, **({"quality": 90} if kind == "JPEG" else {}))
    return f"data:image/{kind.lower()};base64," + base64.b64encode(b.getvalue()).decode()


def _font(size):
    return ImageFont.load_default(size=size)


def shapes(big, small, w, h, bg=(245, 245, 240), kind="PNG"):
    """A large circle of colour `big` and a small square of colour `small`."""
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    r = int(min(w, h) * 0.38)
    cx, cy = int(w * 0.35), h // 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=COLORS[big])
    s = max(4, int(min(w, h) * 0.12))
    x0, y0 = int(w * 0.78), int(h * 0.2)
    d.rectangle((x0, y0, x0 + s, y0 + s), fill=COLORS[small])
    return _uri(im, kind)


def invoice(number, total, due, w=900, h=640):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), f"INVOICE #{number}", fill="black", font=_font(44))
    d.text((40, 110), "Acme Supplies Ltd. - Business plan, annual renewal", fill=(60, 60, 60), font=_font(24))
    d.line((40, 170, w - 40, 170), fill=(0, 0, 0), width=2)
    d.text((40, 200), "Seats: 11", fill="black", font=_font(28))
    d.text((40, 260), f"Payment due: {due}", fill="black", font=_font(28))
    d.text((40, h - 140), f"TOTAL: {total}", fill="black", font=_font(48))
    return _uri(im)


def sign(kind, w=500, h=500):
    im = Image.new("RGB", (w, h), (170, 200, 230))
    d = ImageDraw.Draw(im)
    c = (w // 2, h // 2)
    if kind == "STOP":
        d.regular_polygon((c, int(w * 0.42)), 8, rotation=22.5, fill=COLORS["red"])
        d.text((c[0] - 105, c[1] - 45), "STOP", fill="white", font=_font(90))
    elif kind == "YIELD":
        d.regular_polygon((c, int(w * 0.42)), 3, rotation=180, fill="white", outline=COLORS["red"], width=25)
        d.text((c[0] - 70, c[1] - 60), "YIELD", fill="black", font=_font(50))
    else:
        d.rectangle((40, h // 3, w - 40, 2 * h // 3), fill=COLORS["green"])
        d.text((c[0] - 80, c[1] - 40), "EXIT", fill="white", font=_font(80))
    return _uri(im)


def dots(n, w=600, h=400, seed=0):
    rng = random.Random(seed)
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    for i in range(n):
        x = 60 + (w - 120) * (i + 0.5) / n + rng.uniform(-10, 10)
        y = h / 2 + rng.uniform(-80, 80)
        d.ellipse((x - 30, y - 30, x + 30, y + 30), fill="black")
    return _uri(im)


def tile(color, w=96, h=96):
    return _uri(Image.new("RGB", (w, h), COLORS[color]))


def label(city, weight, w=1200, h=300):
    """A wide shipping label (4:1)."""
    im = Image.new("RGB", (w, h), (250, 245, 225))
    d = ImageDraw.Draw(im)
    d.rectangle((10, 10, w - 10, h - 10), outline="black", width=4)
    d.text((40, 40), f"SHIP TO: {city.upper()}", fill="black", font=_font(64))
    d.text((40, 170), f"WEIGHT: {weight} KG   FRAGILE", fill="black", font=_font(48))
    return _uri(im)


def strip(color, w=2000, h=100):
    """A 20:1 strip with a coloured block at its right end."""
    im = Image.new("RGB", (w, h), (230, 230, 230))
    ImageDraw.Draw(im).rectangle((w - 300, 10, w - 20, h - 10), fill=COLORS[color])
    return _uri(im)


YES_NO = lambda q: {"type": "noul", "instructions": q}                       # noqa: E731


def choice(q, options):
    return {"type": "choice", "instructions": q, "criteria": {o: None for o in options}}


def requests():
    out = []
    four = ["red", "green", "blue", "yellow"]
    sizes = [(640, 480), (320, 240), (1600, 1200), (256, 256), (2000, 500), (120, 900), (32, 32), (2400, 1600)]
    pairs = [("red", "blue"), ("green", "yellow"), ("blue", "red"), ("yellow", "green"), ("red", "green"),
             ("blue", "yellow"), ("green", "red"), ("yellow", "blue")]
    for k, ((w, h), (big, small)) in enumerate(zip(sizes, pairs)):
        qs = {"red": YES_NO("Is the large shape red?"),
              "small": choice("What colour is the small square?", four),
              "big": choice("What colour is the large circle?", four)}
        state = "Photo taken at delivery." if k % 2 else "Photo taken at delivery: <image>"
        out.append((f"shapes-{w}x{h}", state, qs, [shapes(big, small, w, h, kind="JPEG" if k % 3 == 2 else "PNG")],
                    {"red": "true" if big == "red" else "false", "small": small, "big": big}))
    for k, (number, total, due) in enumerate((("2291", "$1,140.00", "2026-11-03"), ("4410", "$86.50", "2026-12-15"),
                                               ("7302", "$12,900.00", "2027-01-31"))):
        totals = ["$1,140.00", "$86.50", "$12,900.00", "$114.00"]
        qs = {"total": {"type": "choice", "instructions": "What is the invoice total?",
                        "criteria": {f"t{i}": t for i, t in enumerate(totals)}},
              "number": YES_NO("Is the invoice number 2291?"),
              "due": choice("When is the payment due?", ["2026-11-03", "2026-12-15", "2027-01-31"])}
        state = {"ticket": "Customer disputes the amount on the attached invoice.", "attachment": "<image>"}
        out.append((f"invoice-{number}", state, qs, [invoice(number, total, due)],
                    {"total": f"t{totals.index(total)}", "number": "true" if number == "2291" else "false",
                     "due": due}))
    for kind in ("STOP", "YIELD", "EXIT"):
        out.append((f"sign-{kind}", "A road sign photographed by the driver.", {
            "says": choice("What does the sign say?", ["STOP", "YIELD", "EXIT", "GO"]),
            "red": YES_NO("Is the sign mostly red?")}, [sign(kind)],
                    {"says": kind, "red": "true" if kind == "STOP" else "false"}))
    for n in (1, 3, 5):
        out.append((f"dots-{n}", "", {"count": {"type": "score", "instructions": "How many black dots are there?",
                                                "criteria": ["one", "two", "three", "four", "five"]}},
                    [dots(n, seed=n)], {"count": n - 1}))
    out.append(("label-wide", "Shipping label scanned at the depot: <image>", {
        "city": choice("Where is the parcel going?", ["Berlin", "Paris", "Madrid", "Rome"]),
        "fragile": YES_NO("Is the parcel marked fragile?")}, [label("Paris", 12)],
                {"city": "Paris", "fragile": "true"}))
    out.append(("strip-20x1", "A status bar: <image>", {
        "colour": choice("What colour is the block at the end of the bar?", four)}, [strip("green")],
                {"colour": "green"}))
    # several images
    out.append(("two-images", "Two photos of the same delivery.", {
        "which": choice("Which image shows a green circle?", ["image 1", "image 2"]),
        "same": YES_NO("Do both images show a circle of the same colour?")},
                [shapes("red", "blue", 640, 480), shapes("green", "blue", 480, 640)], {"which": "image 2",
                                                                                         "same": "false"}))
    out.append(("marks-3", "Before: <image>\nAfter repair: <image>\nInvoice: <image>", {
        "total": {"type": "choice", "instructions": "What is the invoice total?",
                  "criteria": {"a": "$114.00", "b": "$1,140.00", "c": "$86.50"}},
        "after": choice("What colour is the large circle after the repair?", four)},
                [shapes("red", "red", 400, 300), shapes("blue", "red", 400, 300),
                 invoice("2291", "$1,140.00", "2026-11-03")], {"total": "b", "after": "blue"}))
    out.append(("four-signs", "Signs along the route, in order.", {
        "stop": choice("Which image shows a STOP sign?", ["image 1", "image 2", "image 3", "image 4"])},
                [sign("EXIT"), sign("YIELD"), sign("STOP"), sign("EXIT")], {"stop": "image 3"}))
    colors8 = ["red", "blue", "yellow", "black", "green", "orange", "blue", "red"]
    out.append(("eight-tiles", "Eight colour swatches.", {
        "green": choice("Which image is green?", [f"image {i + 1}" for i in range(8)]),
        "black": choice("Which image is black?", [f"image {i + 1}" for i in range(8)])},
                [tile(c) for c in colors8], {"green": "image 5", "black": "image 4"}))
    out.append(("eight-large", "Eight photos: " + " ".join("<image>" for _ in range(8)), {
        "red": YES_NO("Is the large circle in the first photo red?")},
                [shapes(b, s, 1024, 1024) for b, s in pairs], {"red": "true"}))
    return out
