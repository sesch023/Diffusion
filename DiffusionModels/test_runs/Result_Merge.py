from glob import glob
import os
import shutil
from tqdm import tqdm

target_folder = "CosDiffusion_report/translator/"
target_glob = glob("CosDiffusion_report/*translator/*.png")

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

i = 0
for g in tqdm(target_glob):
    file_name = os.path.basename(g)
    folder_name = "_".join(os.path.basename(os.path.dirname(g)).split("_", 2)[:2])
    target_name = f"{target_folder}{folder_name}_{file_name}"
    shutil.copy(g, target_name)
    i += 1
