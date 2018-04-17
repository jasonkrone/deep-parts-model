import tensorflow as tf
import os
import glob
import numpy as np
import csv


class FewshotBirdsDataGenerator(object):

        def __init__(self, batch_size=16, episode_length=10, image_dim=(84, 84, 3)):
            self.splits = {
                'train' : '/home/jason/deep-parts-model/src/cub_fewshot/splits/train_img_path_label_size_bbox_parts_split.txt',
                'test'  : '/home/jason/deep-parts-model/src/cub_fewshot/splits/test_img_path_label_size_bbox_parts_split.txt',
                'val'   : '/home/jason/deep-parts-model/src/cub_fewshot/splits/val_img_path_label_size_bbox_parts_split.txt'
            }
            self.batch_size = batch_size
            self.episode_length = episode_length
            self.image_dim = image_dim
            self.num_classes = 200
            self._load_data()

        def _load_data(self):
            self.train_data = self._data_dict_for_split(self.splits['train'])
            print('finished train')
            self.test_data  = self._data_dict_for_split(self.splits['test'])
            print('finished test')
            self.val_data   = self._data_dict_for_split(self.splits['val'])
            print('finished val')

        def _data_dict_for_split(self, split, mode='test'):
            label_to_examples_dict = {}
            with open(split, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # get x, y, bbox, and parts from line
                line = line.strip()
                line = line.split(' ')
                image_path, y, size, bbox, parts = line[0], line[1], line[2:4], line[4:8], line[8:]
                print(size)
                size = [int(s) for s in size]
                y, bbox, parts = int(y), [float(b) for b in bbox], [float(p) for p in parts]
                parts_x, parts_y = parts[0::2], parts[1::2]
                if y not in label_to_examples_dict:
                    label_to_examples_dict[y] = []
                # example is going to be x, p1, p2
                label_to_examples_dict[y].append(self._parser(image_path, size, bbox, parts_x, parts_y, mode))
            return label_to_examples_dict

        def _parser(self, image_path, size, bbox, parts_x, parts_y, mode='test'):
            # decode the image
            image_file    = tf.read_file(image_path)
            image = tf.image.decode_jpeg(image_file, channels=self.image_dim[-1])
            # get height and width of image to normalize the bounding box and part locations
            height, width = size
            # normalize bbox
            x, y, w, h = bbox
            xmin = max(x / width,        0.)
            xmax = min((x + w) / width,  1.)
            ymin = max(y / height,       0.)
            ymax = min((y + h) / height, 1.)
            # normalize parts
            parts_x = [max(px / width, 0) for px in parts_x]
            parts_y = [max(py / height, 0) for py in parts_y]
            # extract parts
            breast_x, breast_y = parts_x[3], parts_y[3]
            crown_x, crown_y = parts_x[4], parts_y[4]
            nape_x, nape_y = parts_x[9], parts_y[9]
            tail_x, tail_y = parts_x[13], parts_y[13]
            leg_x, leg_y = parts_x[7], parts_y[7]
            beak_x, beak_y = parts_x[1], parts_y[1]

            new_height, new_width, new_channels = self.image_dim
            # get crop for body
            bxmin, bxmax = tf.minimum(tail_x, beak_x), tf.maximum(tail_x, beak_x)
            bymin, bymax = tf.minimum(leg_y, nape_y), tf.maximum(leg_y, nape_y)
            boxes = tf.expand_dims(tf.stack([bymin, bxmin, bymax, bxmax], axis=0), 0)
            box_ind = tf.constant([0])
            body_crop = tf.image.crop_and_resize(tf.expand_dims(image, 0), boxes, box_ind, [new_height, new_width], method='bilinear', extrapolation_value=0, name=None)
            body_crop = tf.squeeze(body_crop, [0])

            # get crop for head
            x_len = tf.abs(beak_x - nape_x)
            y_len = tf.abs(crown_x - nape_x)
            bymin, bymax = tf.minimum(nape_y, crown_y), tf.maximum(nape_y, crown_y) + y_len
            bxmin, bxmax = crown_x - x_len, crown_x + x_len
            boxes = tf.expand_dims(tf.stack([bymin, bxmin, bymax, bxmax], axis=0), 0)
            head_crop = tf.image.crop_and_resize(tf.expand_dims(image, 0), boxes, box_ind, [new_height, new_width], method='bilinear', extrapolation_value=0, name=None)
            head_crop = tf.squeeze(head_crop, [0])

            if mode == 'train':
                # resize the image to 256xS where S is max(largest-image-side, 244)
                image = tf.expand_dims(image, 0)
                clipped_height, clipped_width = max(height, 244), max(width, 244)
                if height > width:
                    image = tf.image.resize_bilinear(image, [clipped_height, 256], align_corners=False)
                else:
                    image = tf.image.resize_bilinear(image, [256, clipped_width], align_corners=False)
                # TODO: get rid of this
                image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
                # TODO: ^
                image = tf.squeeze(image, [0])
                # preprocess with random crops and horizontal flipping
                image = tf.random_crop(image, size=[new_height, new_width, new_channels])
                image = tf.image.random_flip_left_right(image)
            else:
                image = tf.image.central_crop(image, central_fraction=0.875)
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [new_height, new_width])
                image = tf.squeeze(image, [0])
            return image, body_crop, head_crop

if __name__ == '__main__':
    data_generator = FewshotBirdsDataGenerator()
