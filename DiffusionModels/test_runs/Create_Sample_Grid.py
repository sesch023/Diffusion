from PIL import Image, ImageFont, ImageDraw, ImageColor
from glob import glob
import random
import re
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import textwrap

def text_to_image(
    text: str,
    font_filepath: str,
    font_size: int,
    color: (int, int, int), #color is in RGB
    size: (int, int),
    font_align="center"
):
    MAX_W, MAX_H = size
    para = textwrap.wrap(text, width=15)
    font = ImageFont.truetype(font_filepath, size=font_size)
    current_h, pad = 30, 10
    out = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for line in para:
        w, h = draw.textsize(line, font=font)
        draw.text(((MAX_W - w) / 2, current_h), line, font=font, fill=color)
        current_h += h + pad
    return out

font = "Arial.ttf"

get_caption = True
target_glob = glob("CosDiffusion_report/*_img_emb/*.png")
cols, rows = 6, 8
num_elements = cols * rows if not get_caption else (cols * rows) // 2

font_size = 20
font_color = (0 , 0, 0)
target_cell_size = (256, 256)


images = []

for g in random.sample(target_glob, num_elements):
    image = cv2.imread(g)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_cell_size, interpolation = cv2.INTER_AREA)

    if get_caption:
        img_num = int(g.split("_")[-1].split(".")[0])
        splitted = re.split("_|/", g)
        cap_file = os.path.dirname(g) + "/" + splitted[-6] + "_" + splitted[-5] + ".txt"
        with open(cap_file, "r") as f:
            captions = f.readlines()
            caption = captions[img_num].strip()
            caption_image = text_to_image(caption, font, font_size, font_color, target_cell_size)
            caption_image = cv2.cvtColor(np.array(caption_image), cv2.COLOR_BGR2RGB)
            images.append(caption_image)

    images.append(image)

plt.xticks([])
plt.yticks([])
fig = plt.figure(figsize=(rows, cols))
grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.03)

for ax, im in zip(grid, images):
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im)

plt.savefig("sample_grid.png", dpi=1200, bbox_inches='tight', pad_inches=0)
        