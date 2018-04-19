import tensorflow as tf
import os
import glob
import numpy as np
import csv
import random
from scipy.misc import imread,imresize
from image_utils import *


class FewshotBirdsDataGenerator(object):
    def __init__(self, batch_size=10, episode_length=10, episode_width=5, image_dim=(244, 244, 3)):
        self.splits = {
            'train' : '/home/jason/deep-parts-model/src/cub_fewshot/splits/train_img_path_label_size_bbox_parts_split.txt',
            'test'  : '/home/jason/deep-parts-model/src/cub_fewshot/splits/test_img_path_label_size_bbox_parts_split.txt',
            'val'   : '/home/jason/deep-parts-model/src/cub_fewshot/splits/val_img_path_label_size_bbox_parts_split.txt'
        }
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.episode_width = episode_width
        self.image_dim = image_dim
        self.num_classes = 200
        self._cache = {}
        self._load_data()

    def _load_data(self):
        self.train_data = self._data_dict_for_split(self.splits['train'])
        print('finished train')
        self.test_data  = self._data_dict_for_split(self.splits['test'])
        print('finished test')
        self.val_data   = self._data_dict_for_split(self.splits['val'])
        print('finished val')

    def sample_episode_batch(self, data):
        """Generates a random batch for training or validation.

        Structures each element of the batch as an 'episode'.
        Each episode contains episode_length examples and
        episode_width distinct labels.

        Args:
          data: A dictionary mapping label to list of examples.
          episode_length: Number of examples in each episode.
          episode_width: Distinct number of labels in each episode.
          batch_size: Batch size (number of episodes).

        Returns:
          A tuple (x, y) where x is a list of batches of examples
          with size episode_length and y is a list of batches of labels.
        """
        episodes_x = [[] for _ in xrange(self.episode_length)]
        episodes_y = [[] for _ in xrange(self.episode_length)]
        assert len(data) >= self.episode_width
        keys = data.keys()
        for b in xrange(self.batch_size):
            episode_labels = random.sample(keys, self.episode_width)
            remainder = self.episode_length % self.episode_width
            remainders = [0] * (self.episode_width - remainder) + [1] * remainder
            episode_x = [
              random.sample(data[lab],
                            r + (self.episode_length - remainder) // self.episode_width)
              for lab, r in zip(episode_labels, remainders)]
            episode = sum([[(x, i, ii) for ii, x in enumerate(xx)]
                         for i, xx in enumerate(episode_x)], [])
            random.shuffle(episode)
            # Arrange episode so that each distinct label is seen before moving to
            # 2nd showing
            episode.sort(key=lambda elem: elem[2])
            assert len(episode) == self.episode_length
            for i in xrange(self.episode_length):
                episodes_x[i].append(episode[i][0])
                episodes_y[i].append(episode[i][1] + b * self.episode_width)
        x, p1, p2 = self._get_examples_for_image_configs(episodes_x)
        y = [np.array(yy*2).astype('int32') for yy in episodes_y]
        return (x, p1, p2, y)

    def _data_dict_for_split(self, split, mode='test'):
        label_to_examples_dict = {}
        with open(split, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # get x, y, bbox, and parts from line
            line = line.strip()
            line = line.split(' ')
            image_path, y, size, bbox, parts = line[0], line[1], line[2:4], line[4:8], line[8:]
            size = [int(s) for s in size]
            y, bbox, parts = int(y), [float(b) for b in bbox], [float(p) for p in parts]
            parts_x, parts_y = parts[0::2], parts[1::2]
            if y not in label_to_examples_dict:
                label_to_examples_dict[y] = []
            # example is going to be x, p1, p2
            # instead of storing this store args
            label_to_examples_dict[y].append((image_path, size, bbox, parts_x, parts_y, mode))
        return label_to_examples_dict

    def _get_examples_for_image_configs(self, configs):
        '''
        parses the configs of dim: self.batch_size X self.episode_length X 1
        and returns a np.array of image_and_parts of dim: self.batch_size X 3 X self.episode_length X self.image_dim
        '''
        x  = [[None] * self.batch_size] * self.episode_length
        p1 = [[None] * self.batch_size] * self.episode_length
        p2 = [[None] * self.batch_size] * self.episode_length
        for i, config_batch in enumerate(configs):
            for j, c in enumerate(config_batch):
                image_path, size, bbox, parts_x, parts_y, mode = c
                if image_path in self._cache:
                    image_and_parts = self._cache[image_path]
                else:
                    image_and_parts = self._parser(image_path, size, bbox, parts_x, parts_y, mode)
                    self._cache[image_path] = image_and_parts
                image, body_crop, head_crop = image_and_parts
                x[i][j] = image
                p1[i][j] = body_crop
                p2[i][j] = head_crop
        x  = [np.array(xx).astype('float32') for xx in x]
        p1 = [np.array(p).astype('float32') for p in p1]
        p2 = [np.array(p).astype('float32') for p in p2]
        return (x, p1, p2)

    def _parser(self, image_path, size, bbox, parts_x, parts_y, mode='test'):
        # decode the image
        new_height, new_width, new_channels = self.image_dim
        image = imread(image_path) # imread(image_path)
        # black and white image
        if len(image.shape) == 2 and new_channels == 3:
            # duplicat channels 3 times
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        # get height and width of image to normalize the bounding box and part locations
        height, width = size
        # normalize bbox
        x, y, w, h = [int(b) for b in bbox]
        # extract parts
        breast_x, breast_y = int(parts_x[3]), int(parts_y[3])
        crown_x, crown_y = int(parts_x[4]), int(parts_y[4])
        nape_x, nape_y = int(parts_x[9]), int(parts_y[9])
        tail_x, tail_y = int(parts_x[13]), int(parts_y[13])
        leg_x, leg_y = int(parts_x[7]), int(parts_y[7])
        beak_x, beak_y = int(parts_x[1]), int(parts_y[1])
        try:
            # get crop for body
            bxmin, bxmax = min(tail_x, beak_x), max(tail_x, beak_x)
            bymin, bymax = min(leg_y, nape_y, breast_y), max(leg_y, nape_y, breast_y)
            bymin, bymax, bxmin, bxmax = int(bymin), int(bymax), int(bxmin), int(bxmax)
            body_crop = image[bymin:bymax, bxmin:bxmax, :]
            body_crop = imresize(body_crop, size=(new_height, new_width))
            # get crop for head
            x_len = abs(beak_x - nape_x)
            y_len = abs(crown_x - nape_x)
            bymin, bymax = min(nape_y, crown_y), max(nape_y, crown_y) + y_len
            bxmin, bxmax = max(crown_x - x_len, 0), min(crown_x + x_len, width)
            bymin, bymax, bxmin, bxmax = int(bymin), int(bymax), int(bxmin), int(bxmax)
            head_crop = image[bymin:bymax, bxmin:bxmax, :]
            head_crop = imresize(head_crop, size=(new_height, new_width))
        # one of the parts used in the above calculation was missing
        except:
            image_crop = image[y:y+h, x:x+w, :]
            head_crop = image_crop[int(h/2):, :, :] # top half of crop 
            head_crop = imresize(head_crop, size=(new_height, new_width))
            body_crop = image_crop[0:int(h/2), :, :] # bottom half of crop
            body_crop = imresize(body_crop, size=(new_height, new_width))
            # plot
            #f, ax = plt.subplots(3, figsize=(4, 4))
            #ax[0].imshow(image)
            #ax[1].imshow(head_crop)
            #ax[2].imshow(body_crop)
            #plt.show()
        if mode == 'train':
            # resize the image to 256xS where S is max(largest-image-side, 244)
            # TODO: this seems semi random not sure why STN used this
            clipped_height, clipped_width = max(height, 244), max(width, 244)
            if height > width:
                image = imresize(image, size=(clipped_height, 256))
            else:
                image = imresize(image, size=(256, clipped_width))
            image = random_crop(image, new_height)
            image = horizontal_flip(image)
        else:
            image = central_crop(image, central_fraction=0.875)
            image = imresize(image, size=(new_height, new_width, new_channels))
        image_and_parts = (image, body_crop, head_crop)
        return image_and_parts
    
if __name__ == '__main__':
    data_generator = FewshotBirdsDataGenerator()
    xs, ys = data_generator.sample_episode_batch(data_generator.train_data)
