from glob import glob
import os
import shutil

def zero_pad_last_num(s, len=2):
    s_num = s.split("img_")
    s_nums = s_num[-1].split(".")
    s_nums[0] = s_nums[0].zfill(len)
    s_num[-1] = ".".join(s_nums)
    return "img_".join(s_num)

def removed_zero_pad_last_num(s):
    s = s[:-6] + s[-5:] if s[-6] == "0" else s
    return s

vid_length = 16
real_frame_glob = sorted([zero_pad_last_num(frame) for frame in glob("SpatioTemporalDiffusion_report/*_real_frame/*.png")])
fake_frame_glob = sorted([zero_pad_last_num(frame) for frame in glob("SpatioTemporalDiffusion_report/*_fake_frame/*.png")])

target_folder_real = "SpatioTemporalDiffusion_report/real_frame_ds/"
target_folder_fake = "SpatioTemporalDiffusion_report/fake_frame_ds/"

def slice_per(source, step):
    return [source[i*step:(i+1)*step] for i in range(len(source)//step)]

if not os.path.exists(target_folder_real):
    os.makedirs(target_folder_real)
if not os.path.exists(target_folder_fake):
    os.makedirs(target_folder_fake)

sliced_real = slice_per(real_frame_glob, vid_length)
sliced_fake = slice_per(fake_frame_glob, vid_length)

i = 1
for slice_real, slice_fake  in zip(sliced_real, sliced_fake):
    if not os.path.exists(f"{target_folder_real}video_{str(i)}"):
        os.makedirs(f"{target_folder_real}video_{str(i)}")
    if not os.path.exists(f"{target_folder_fake}video_{str(i)}"):
        os.makedirs(f"{target_folder_fake}video_{str(i)}")

    k = 1
    for frame_real, frame_fake in zip(slice_real, slice_fake):
        frame_real = removed_zero_pad_last_num(frame_real)
        frame_fake = removed_zero_pad_last_num(frame_fake)
        shutil.copy(f"{frame_real}", f"{target_folder_real}video_{str(i)}/{str(k).zfill(2)}.png")
        shutil.copy(f"{frame_fake}", f"{target_folder_fake}video_{str(i)}/{str(k).zfill(2)}.png")
        k += 1

    i += 1