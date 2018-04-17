import numpy as np
import csv

path_to_id_dict = {}
prefix = '/dvmm-filer2/users/jason/datasets/CUB_200_2011/images/'
with open('/dvmm-filer2/users/jason/datasets/CUB_200_2011/images.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    split = line.split(' ')
    img_id, img_path = split
    img_path = prefix + img_path
    path_to_id_dict[img_path] = int(img_id)


id_to_bbox_dict = {}
# read bounding_boxes and create a dict that maps image_id -> bbox
with open('/dvmm-filer2/users/jason/datasets/CUB_200_2011/bounding_boxes.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    split = line.split(' ')
    img_id, x, y, width, height = split
    id_to_bbox_dict[int(img_id)] = [float(x), float(y), float(width), float(height)]

id_to_parts_dict = {}
with open('/dvmm-filer2/users/jason/datasets/CUB_200_2011/parts/part_locs.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    split = line.split(' ')
    img_id, part_id, x, y, visible = split
    img_id, part_id = int(img_id), int(part_id)
    if img_id not in id_to_parts_dict:
        id_to_parts_dict[img_id] = [None] * 15
    id_to_parts_dict[img_id][part_id-1] = [float(x), float(y)]

id_to_size_dict = {}
with open('/dvmm-filer2/users/jason/datasets/CUB_200_2011/sizes.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    split = line.split(' ')
    img_id, width, height = split
    img_id, height, width = int(img_id), int(height), int(width)
    if height == 0.0 or width == 0.0:
        print('BIG PROBLEM', 'height:', height, 'width:', width)
    elif height < 100 or width < 100:
        print('height:', height, 'width:', width)
    else:
        print('height:', height, 'width:', width)
    id_to_size_dict[img_id] = [height, width]

train_val_test_splits = {
    'train' : '/home/jason/deep-parts-model/src/cub_fewshot/splits/train_split_few_shot.txt',
    'test'  : '/home/jason/deep-parts-model/src/cub_fewshot/splits/test_split_few_shot.txt',
    'val'   : '/home/jason/deep-parts-model/src/cub_fewshot/splits/val_split_few_shot.txt'
}
# readlines from each file
for split, split_path in train_val_test_splits.items():
    train_val_test_splits[split] = open(split_path, 'r').readlines()
# for each split for each example
for split, lines in train_val_test_splits.items():
    new_csv_lines = []
    # read lines
    for l in lines:
        l = l.strip()
        s = l.split('\t')
        img_path, label = s
        img_id = path_to_id_dict[img_path]
        bbox = id_to_bbox_dict[img_id]
        parts = id_to_parts_dict[img_id]
        parts = np.array(parts) # 15 x 2
        # put parts togeather in order part1, part2, part3, part4, ..., part15
        parts = np.concatenate(parts, axis=0)
        size = id_to_size_dict[img_id]
        full_line = [img_path, label] + size + bbox + parts.tolist()
        new_csv_lines.append(full_line)
    with open(split+'_img_path_label_size_bbox_parts_split.txt', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        for new_line in new_csv_lines:
            writer.writerow(new_line)

