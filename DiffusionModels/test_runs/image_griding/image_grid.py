from glob import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

rows = glob("row_*")
image_rows = []

for row in sorted(rows):
    images = sorted(glob(f"{row}/*.png"))
    images = [cv2.imread(img) for img in images]
    image_rows.append(images)

max_x = max([len(row) for row in image_rows])

plt.xticks([])
plt.yticks([])
fig = plt.figure(figsize=(max_x, len(image_rows)))
grid = ImageGrid(fig, 111, nrows_ncols=(max_x, len(image_rows)), axes_pad=0.03)

y_alinged = True

if y_alinged:
    rows_merged = []
    for i in range(max_x):
        for k in range(len(image_rows)):
            if i < len(image_rows[k]):
                rows_merged.append(image_rows[k][i])
else:
    rows_merged = [image for row in image_rows for image in row]

for ax, im in zip(grid, rows_merged):
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


plt.savefig("merged.png", dpi=1200)