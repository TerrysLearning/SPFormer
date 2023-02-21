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


def main():
    folder = 'train'  # len=5
    useless_sem = [-100, 0, 1] # unlabeled, floor, wall
    overlaps_per_scene = {}
    for filename in os.listdir(folder):
        file = os.path.join(folder, filename)
        if file.endswith('_inst_nostuff.pth'):
            coords, colors, superpoint, sem_labels, instance_labels = torch.load(file)
            # do align
            path_txt = file[0:18]+'.txt'
            print(path_txt)
            f_txt = open(path_txt, 'r')
            lines = f_txt.readlines()
            axis_alignment = ''
            for line in lines:
                if line.startswith('axisAlignment'):
                    axis_alignment = line
                    break
            Rt = np.array([float(v) for v in axis_alignment.split('=')[1].strip().split(' ')]).reshape([4, 4])
            coords = hom_transfer_coords(coords, Rt)
            # compute bounding box
            instances = np.unique(instance_labels)
            box_max_corners = []
            box_min_corners = []
            box_semantics = []
            box_instances = []
            for instance_id in instances:
                instance_mask = (instance_id == instance_labels)
                instance_sem = sem_labels[instance_mask][0]
                if instance_sem not in useless_sem:
                    box_semantics.append(instance_sem)
                    box_instances.append(instance_id)
                    instance_point_coords = coords[instance_mask]
                    max_corner = np.max(instance_point_coords, axis=0)
                    min_corner = np.min(instance_point_coords, axis=0)
                    box_max_corners.append(max_corner.tolist())
                    box_min_corners.append(min_corner.tolist())
            if box_max_corners == []:# when no boxes are in the scene
                torch.save((coords, colors, superpoint, sem_labels, instance_labels), file[:18] + '_infonew.pth')
                continue
            box_max_corners = np.array(box_max_corners) + 0.1
            box_min_corners = np.array(box_min_corners) + 0.1
            box_semantics = np.array(box_semantics)
            box_instances = np.array(box_instances)
            # add noise to the bounding box
            noise = 0.05
            rng = np.random.default_rng(seed=3000) 
            box_max_corners += rng.normal(loc=0, scale=noise/2, size= box_max_corners.shape) # scale is std dev
            box_min_corners += rng.normal(loc=0, scale=noise/2, size= box_min_corners.shape) # scale is std dev
            # assign semantic and instance pesudo label based on bounding boxes
            bounds = box_max_corners - box_min_corners
            bb_volume = np.prod(2 * bounds, axis=1)
            unique_superpoints = np.unique(superpoint)
            box_occ = in_box(coords, box_min_corners[:,None], box_max_corners[:,None])
            activations_per_point = [np.argwhere(box_occ[:, i] == 1) for i in range(len(coords))]
            num_BBs_per_point = box_occ.sum(axis=0)
            sem_labels_new = sem_labels.copy()
            ins_labels_new = instance_labels.copy()
            overlaps = 0
            for i, activ in enumerate(activations_per_point):
                if num_BBs_per_point[i] == 1:
                    bb_idx = activ[0,0]
                    ins_labels_new[i] = box_instances[bb_idx]  
                    sem_labels_new[i] = box_semantics[bb_idx] 
                elif num_BBs_per_point[i] > 1: 
                    overlaps += 1 
                    box_ids = activ.reshape(-1)
                    smallest_box_id = np.argmin(bb_volume[box_ids])
                    ins_labels_new[i] = box_instances[box_ids[smallest_box_id]]  
                    sem_labels_new[i] = box_semantics[box_ids[smallest_box_id]] 
            for sp in unique_superpoints:
                sp_mask = superpoint==sp
                ins_labels_new[sp_mask] = stats.mode(ins_labels_new[sp_mask], keepdims = True)[0][0]
                sem_labels_new[sp_mask] = stats.mode(sem_labels_new[sp_mask], keepdims = True)[0][0]
            overlaps_per_scene[file[6:18]] = overlaps
            # torch.save((coords, colors, superpoint, sem_labels_new, ins_labels_new), file[:18] + '_infonew.pth')
    print(overlaps_per_scene)
    torch.save(overlaps_per_scene, '_overlaps_num.pth')


def sort_file():
    dict = torch.load('_overlaps_num.pth')
    values = np.array(list(dict.values()))
    sorted_indices = np.argsort(values)
    big_value_indicts = sorted_indices[values[sorted_indices]>2000]
    small_value_indicts = sorted_indices[values[sorted_indices]<=2000]
    sorted_keys_small = np.array(list(dict.keys()))[small_value_indicts]
    sorted_keys_big = np.array(list(dict.keys()))[big_value_indicts]
    torch.save(sorted_keys_big, 'big_scenes.pth')
    torch.save(sorted_keys_small, 'small_scenes.pth')
    print('done')


def visual_boxes_obj(max_corners, min_corners, output_file):
    'Generate an obj file to visualise the boxes'
    n = len(max_corners)
    f = open(output_file+'.obj', 'w+')
    for i in range(n):
        p1 = 'v '+ str(min_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p2 = 'v '+ str(min_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n'
        p3 = 'v '+ str(min_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p4 = 'v '+ str(min_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n'
        p5 = 'v '+ str(max_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p6 = 'v '+ str(max_corners[i][0])+ ' '+ str(min_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n' 
        p7 = 'v '+ str(max_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(min_corners[i][2])+ '\n'
        p8 = 'v '+ str(max_corners[i][0])+ ' '+ str(max_corners[i][1])+ ' '+ str(max_corners[i][2])+ '\n'
        f.write(p1)            
        f.write(p2)
        f.write(p3)
        f.write(p4)
        f.write(p5)
        f.write(p6)
        f.write(p7)
        f.write(p8)     
    for i in range(n):
        bi = i*8
        f.write('f ' + str(bi+2) + ' '+ str(bi+4) + ' '+ str(bi+1)+ '\n')
        f.write('f ' + str(bi+5) + ' '+ str(bi+2) + ' '+ str(bi+1)+ '\n')
        f.write('f ' + str(bi+1) + ' '+ str(bi+4) + ' '+ str(bi+3)+ '\n')
        f.write('f ' + str(bi+3) + ' '+ str(bi+5) + ' '+ str(bi+1)+ '\n')
        f.write('f ' + str(bi+2) + ' '+ str(bi+8) + ' '+ str(bi+4)+ '\n')
        f.write('f ' + str(bi+6) + ' '+ str(bi+2) + ' '+ str(bi+5)+ '\n')
        f.write('f ' + str(bi+6) + ' '+ str(bi+8) + ' '+ str(bi+2)+ '\n')
        f.write('f ' + str(bi+4) + ' '+ str(bi+8) + ' '+ str(bi+3)+ '\n')
        f.write('f ' + str(bi+7) + ' '+ str(bi+5) + ' '+ str(bi+3)+ '\n')
        f.write('f ' + str(bi+3) + ' '+ str(bi+8) + ' '+ str(bi+7)+ '\n')
        f.write('f ' + str(bi+7) + ' '+ str(bi+6) + ' '+ str(bi+5)+ '\n')
        f.write('f ' + str(bi+8) + ' '+ str(bi+6) + ' '+ str(bi+7)+ '\n')


from plyfile import PlyData

def visualise_wrong_label(w1, w2, w3, scene_name, unique_superpoint, superpoints):
    data_path = 'train/'
    path_ply = os.path.join(data_path, f'{scene_name}_vh_clean_2.ply')
    plydata = PlyData.read(path_ply)
    for id, seg in enumerate(unique_superpoint):
        if id in w1:
            seg_mask = superpoints==seg
            plydata = set_vertex_color(plydata, seg_mask, [255,0,0])  #red
        if id in w2:
            seg_mask = superpoints==seg
            plydata = set_vertex_color(plydata, seg_mask, [0,255,0])  #green
        if id in w3:
            seg_mask = superpoints==seg
            plydata = set_vertex_color(plydata, seg_mask, [0,0,255])  #green
    out_file = scene_name + '_wrong_labels.ply'
    with open(out_file, mode='wb') as f: 
        PlyData(plydata, text=True).write(f)


def set_vertex_color(plydata, mask, color):
    for i, bool in enumerate(mask):
        if bool:
            plydata['vertex'][i][3] = color[0]
            plydata['vertex'][i][4] = color[1]
            plydata['vertex'][i][5] = color[2]
    return plydata


if __name__ == '__main__':
    # main()
    sort_file()



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