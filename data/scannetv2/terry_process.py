import numpy as np
import open3d as o3d
import os
import json
import csv
from scipy.stats import rankdata
import scipy.stats as stats
import configargparse
import torch

def in_box(points, bb_min, bb_max):
    return np.all( points >= bb_min, axis=-1) & np.all( points <= bb_max, axis=-1)

def hom_transfer_coords(ps, Rt):
    ps_ = []
    for p in ps:
        p_ = np.concatenate((p, np.array([1])))
        p_ = Rt@p_
        ps_.append(p_[:3])
    return np.array(ps_)


def compute_normals(scene_name, folder):
    path_ply = os.path.join(folder, scene_name+'_vh_clean_2.ply')
    mesh = o3d.io.read_triangle_mesh(path_ply)
    mesh.compute_vertex_normals()
    mesh.normalize_normals()
    normals = np.asarray(mesh.vertex_normals)
    return normals


def process_one_scene(scene_name, folder):
    file_stuff = os.path.join(folder, scene_name+'_inst_nostuff.pth')
    coords, colors, superpoints, semantic_labels, instance_labels = torch.load(file_stuff)
    normals = compute_normals(scene_name, folder) # compute normals
    assert len(coords) == len(normals)
    return coords, colors, superpoints, semantic_labels, instance_labels, normals


if __name__ == '__main__':
    file = open('scannetv2_val.txt', 'r')
    lines = file.readlines()
    folder = 'val'
    for line in lines:
        scene_name = line[:-1]
        datas = process_one_scene(scene_name, folder)
        torch.save(datas, os.path.join( folder, scene_name + '_infonew.pth'))
        print('saved ', scene_name)
    print('done')



# import shutil
# dir1 = 'scans/'
# dir2 = 'train/'
# for root, ds, files in os.walk(dir1):
#     for file in files:
#         if file.endswith('.txt'):
#             src_file = os.path.join(root, file)
#             dst_file = os.path.join(dir2, file)
#             # Move the file from the source directory to the destination directory
#             shutil.copy2(src_file, dst_file)
#             print(f"copied file: {src_file}")

# ins_wrong = np.count_nonzero(ins_labels_new != instance_labels)
#     sem_wrong = np.count_nonzero(ins_labels_new != instance_labels)
#     print(1 - ins_wrong/len(ins_labels_new))
#     print(1 - sem_wrong/len(sem_labels_new))
#     print(len(a)/len(unique_superpoints))
#     visualise_wrong_label(a,[],[], file[6:18], unique_superpoints, superpoint)
#     visual_boxes_obj(100*box_max_corners, 100*box_min_corners, 'boxes.obj')