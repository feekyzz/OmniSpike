"""Generate clean high-resolution redraws of the red-boxed infographic assets."""

from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parents[1] / "assets" / "redbox_hd_assets"
OUT.mkdir(parents=True, exist_ok=True)
RED = (215, 45, 55, 255)
BLUE = (38, 139, 214, 255)
PURPLE = (126, 73, 184, 255)
GREEN = (30, 155, 82, 255)
INK = (25, 39, 58, 255)


def canvas(size, transparent=False):
    return Image.new("RGBA", size, (255, 255, 255, 0 if transparent else 255))


def save_signal(name, values, color=RED, size=(1600, 700), width=11):
    im = canvas(size)
    d = ImageDraw.Draw(im)
    pad = 70
    mid = size[1] / 2
    amp = size[1] * 0.34
    pts = []
    for i, value in enumerate(values):
        x = pad + i * (size[0] - 2 * pad) / max(1, len(values) - 1)
        pts.append((x, mid - value * amp))
    d.line(pts, fill=color, width=width, joint="curve")
    im.save(OUT / name)


def signal_assets():
    rng = random.Random(20260826)
    n = 420
    noise = [0.16 * math.sin(i * 0.24) + rng.uniform(-0.58, 0.58) for i in range(n)]
    save_signal("01_noise_interference.png", noise)

    attenuation = [0.9 * math.exp(-i / 115) * math.sin(i * 0.17) for i in range(n)]
    save_signal("02_amplitude_attenuation.png", attenuation)

    loss = [0.60 * math.sin(i * 0.17) if not 165 <= i <= 265 else None for i in range(n)]
    im = canvas((1600, 700))
    d = ImageDraw.Draw(im)
    segments, current = [], []
    for i, value in enumerate(loss):
        if value is None:
            if current:
                segments.append(current); current = []
            continue
        current.append((70 + i * 1460 / (n - 1), 350 - value * 238))
    if current:
        segments.append(current)
    for segment in segments:
        d.line(segment, fill=RED, width=11, joint="curve")
    im.save(OUT / "03_signal_loss.png")

    drift = []
    for i in range(n):
        t = i / (n - 1)
        drift.append(-0.55 + 1.05 * t + (0.08 + 0.20 * t) * math.sin(i * 0.14) + rng.uniform(-0.025, 0.025))
    save_signal("04_drift_distortion.png", drift)

    colors = [BLUE, RED, BLUE, PURPLE]
    names = ["05_speed_waveform.png", "06_torque_waveform.png", "07_temperature_waveform.png", "08_load_waveform.png"]
    for j, (name, color) in enumerate(zip(names, colors)):
        vals = [0.52 * math.sin(i * (0.13 + j * 0.018) + j * 0.7) + 0.10 * math.sin(i * 0.037) for i in range(260)]
        save_signal(name, vals, color=color, size=(1200, 420), width=10)


def network_asset():
    rng = random.Random(73)
    size = (1600, 900)
    im = canvas(size, transparent=True)
    d = ImageDraw.Draw(im)
    nodes = []
    for _ in range(72):
        angle = rng.random() * math.tau
        radius = math.sqrt(rng.random())
        nodes.append((800 + math.cos(angle) * radius * 610, 430 + math.sin(angle) * radius * 300))
    for i, a in enumerate(nodes):
        nearest = sorted(range(len(nodes)), key=lambda k: (nodes[k][0]-a[0])**2 + (nodes[k][1]-a[1])**2)[1:5]
        for k in nearest:
            if k > i:
                color = rng.choice([(35, 85, 120, 150), (190, 62, 54, 160), (45, 135, 93, 150), (90, 90, 90, 125)])
                d.line((a, nodes[k]), fill=color, width=4)
    for x, y in nodes:
        r = rng.choice((7, 8, 10))
        d.ellipse((x-r, y-r, x+r, y+r), fill=(245, 248, 250, 255), outline=INK, width=4)
    try:
        font = ImageFont.truetype("arialbd.ttf", 250)
    except OSError:
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), "?", font=font)
    tw, th = box[2]-box[0], box[3]-box[1]
    d.text((800-tw/2, 430-th/2-35), "?", font=font, fill=(180, 25, 30, 255))
    im.save(OUT / "09_complex_coupled_network.png")


def icon_assets():
    # Dynamic target: speedometer.
    im = canvas((700, 700), transparent=True); d = ImageDraw.Draw(im)
    d.arc((80, 80, 620, 620), 195, 345, fill=(38, 101, 171, 255), width=38)
    for deg in range(205, 341, 22):
        a = math.radians(deg); x1=350+205*math.cos(a); y1=350+205*math.sin(a); x2=350+250*math.cos(a); y2=350+250*math.sin(a)
        d.line((x1,y1,x2,y2), fill=(38,101,171,255), width=18)
    a=math.radians(315); d.line((350,350,350+190*math.cos(a),350+190*math.sin(a)), fill=INK, width=24)
    d.ellipse((325,325,375,375), fill=INK)
    im.save(OUT / "10_dynamic_target_icon.png")

    # Economic target: green leaf.
    im = canvas((700, 700), transparent=True); d = ImageDraw.Draw(im)
    d.ellipse((115, 105, 590, 570), fill=(65, 177, 72, 255), outline=(26, 130, 61, 255), width=25)
    d.polygon([(115,570),(200,260),(590,105),(455,435)], fill=(28,145,65,255))
    d.line((145,555,500,180), fill=(238,250,231,255), width=28)
    im.save(OUT / "11_economic_target_icon.png")

    # Reliability target: shield.
    im = canvas((700, 700), transparent=True); d = ImageDraw.Draw(im)
    shield=[(350,65),(585,150),(555,410),(470,550),(350,635),(230,550),(145,410),(115,150)]
    d.polygon(shield, fill=(227,240,252,255), outline=(38,101,171,255))
    d.line(shield+[shield[0]], fill=(38,101,171,255), width=32, joint="curve")
    inner=[(350,145),(500,200),(480,385),(415,485),(350,535),(285,485),(220,385),(200,200)]
    d.line(inner+[inner[0]], fill=(86,145,202,255), width=20, joint="curve")
    im.save(OUT / "12_reliability_target_icon.png")


if __name__ == "__main__":
    signal_assets()
    network_asset()
    icon_assets()
    print(f"Generated 12 assets in {OUT}")
