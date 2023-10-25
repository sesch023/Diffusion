from glob import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import textwrap
from PIL import Image, ImageFont, ImageDraw, ImageColor

target_size = (256, 256)

def text_to_image(
    text: str,
    font_filepath: str,
    font_size: int,
    color: (int, int, int), #color is in RGB
    size: (int, int),
    font_align="center",
    align_down=True,
):
    MAX_W, MAX_H = size
    width_per_row = MAX_W // font_size + 10
    para = textwrap.wrap(text, width=width_per_row)

    font = ImageFont.truetype(font_filepath, size=font_size)
    print(len(para))
    current_h, pad = 10, 10
    current_h += (8 - len(para)) * (pad + font_size) if align_down else 0
    out = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for line in para:
        w, h = draw.textsize(line, font=font)
        draw.text(((MAX_W - w) / 2, current_h), line, font=font, fill=color)
        current_h += h + pad
    return out

font = "../Arial.ttf"

labels = [
    "Eingabe in niedriger Auflösung", 
    "Hochskaliert mit CLIP-Bild-Embeddings", 
    "Hochskaliert mit CLIP-Text-Embeddings", 
    "Hochskaliert mit niedrig aufgelösten CLIP-Bild-Embeddings", 
    "Hochskaliert ohne Embeddings"
]


cols = glob("col_*")
image_cols = []

i = 0
for col in sorted(cols):
    images = sorted(glob(f"{col}/*.png"))
    images = [cv2.imread(img) for img in images]
    images = [cv2.resize(img, target_size, interpolation = cv2.INTER_AREA) for img in images]

    label_image = text_to_image(labels[i], font, 20, (0, 0, 0), target_size) 
    label_image = cv2.cvtColor(np.array(label_image), cv2.COLOR_BGR2RGB)
    images = [label_image] + images
    image_cols.append(images)
    i += 1

max_y = max([len(col) for col in image_cols])

plt.xticks([])
plt.yticks([])
fig = plt.figure(figsize=(max_y, len(image_cols)))
grid = ImageGrid(fig, 111, nrows_ncols=(max_y, len(image_cols)), axes_pad=0.05)

rows_merged = []


for i in range(max_y):
    for k in range(len(image_cols)):
        if i < len(image_cols[k]):
            rows_merged.append(image_cols[k][i])

for ax, im in zip(grid, rows_merged):
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

plt.savefig("merged.png", dpi=600, bbox_inches='tight', pad_inches=0)