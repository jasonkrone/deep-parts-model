import tensorflow as tf
import os
import glob
from tf_classification.preprocessing.decode_example import decode_serialized_example


class BirdsDataGenerator(object):
    """
    this class generates supervised batches
    you must use alternate data generator for few-shot batches
    """
    def __init__(self, batch_size=32, episode_length=10, image_dim=(244, 244, 3)):
        self.data_dir = '/dvmm-filer2/users/jason/tensorflow_datasets/cub/with_600_val_split/'
        self.handle_placeholder = tf.placeholder(tf.string, shape=[])
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.image_dim = image_dim
        self.num_classes = 200
        
    def cubirds_sample_episode_batch(self, sess, mode='train'):
        """
        Samples a batch of data from cubirds dataset of the shape: batch-size x episode-length x image_dim
        
        sess: session to run when collecting batches
        mode: specifies the data_set split to use (train, val, or test)
        """
        handle = None
        if mode == 'train':
            handle = sess.run(self.train_handle)
        elif mode == 'val':
            handle = sess.run(self.val_handle)
        elif mode == 'test':
            handle = sess.run(self.test_handle)
        else:
            raise ValueError("mode must be one of: (train, val, or test)")
        
        h, w, c = self.image_dim
        x_shape = [self.batch_size, self.episode_length, h, w, c]
        episodes_x = np.zeros(shape=x_shape)
        # bodys
        episodes_body = np.zeros(shape=x_shape)
        # heads
        episodes_head = np.zeros(shape=x_shape)
        y_shape = [self.batch_size, self.episode_length]
        episodes_y = np.zeros(shape=y_shape)

        feed_dict = {data.handle_placeholder : handle}
        for i in xrange(self.batch_size):
            ep_x, ep_body, ep_head, ep_y = sess.run(data.next_batch, feed_dict=feed_dict)
            episodes_x[i] = ep_x
            episodes_body[i] = ep_body
            episodes_head[i] = ep_head
            episodes_y[i] = ep_y
        return episodes_x, episodes_body, episodes_head, episodes_y
        
    def load_data(self):
        train_path = os.path.join(self.data_dir, 'train*')
        train_data = self.batched_dataset_from_records(train_path, mode='train', batch_size=self.episode_size)
        train_iter = train_data.make_one_shot_iterator()

        val_path = os.path.join(self.data_dir, 'val*')
        val_data = self.batched_dataset_from_records(val_path, mode='train', batch_size=self.episode_size)
        val_iter = val_data.make_one_shot_iterator()

        test_path = os.path.join(self.data_dir, 'test*')
        test_data = self.batched_dataset_from_records(test_path, mode='test', batch_size=self.episode_size)
        test_iter = test_data.make_one_shot_iterator()

        # to switch between train and test, set the handle_placeholder value to train_handle or test_handle
        iterator = tf.data.Iterator.from_string_handle(self.handle_placeholder, train_data.output_types, train_data.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = train_iter.string_handle()
        self.val_handle = val_iter.string_handle()
        self.test_handle = test_iter.string_handle()

    def batched_dataset_from_records(self, records_path, mode='train', batch_size=32):
        files = tf.data.Dataset.list_files(records_path)
        # parallelize creation reading of records
        #files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
        # suffle dataset so batches don't contain single class etc, then repeat for epochs
        #dataset = tf.contrib.data.shuffle_and_repeat(dataset)
        dataset = dataset.shuffle(buffer_size=6000) # next item randomly chosen from buffer of buffer_size
        dataset = dataset.repeat()
        # parse serialized examples in parallel
        dataset = dataset.map(lambda ex : self.parser(ex, mode), num_parallel_calls=8)
        # create batches
        dataset = dataset.batch(batch_size)
        # allow generation of data and consumption of data to occur at the same time
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset

    def parser(self, serialized_example, mode='train'):
        features_to_fetch = [
            ('image/encoded', 'image'), ('image/class/label', 'label'),
            ('image/height', 'height'), ('image/width', 'width'),
            ('image/channels', 'channels'), ('image/object/parts/x', 'part_x'),
            ('image/object/parts/y', 'part_y'), ('image/object/parts/v', 'part_v'),
            ('image/object/bbox/xmin', 'xmin'), ('image/object/bbox/xmax', 'xmax'),
            ('image/object/bbox/ymin', 'ymin'), ('image/object/bbox/ymax', 'ymax')
        ]
        # extract tensor values from serialized example
        example_dict = decode_serialized_example(serialized_example, features_to_fetch)
        label = example_dict['label']
        image = example_dict['image']
        part_x, part_y, part_v = example_dict['part_x'], example_dict['part_y'], example_dict['part_v']
        xmin, xmax, ymin, ymax = example_dict['xmin'], example_dict['xmax'], example_dict['ymin'], example_dict['ymax']

        # set image size
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        height, width, channels = example_dict['height'], example_dict['width'], example_dict['channels']
        height, width, channels = tf.cast(height, tf.int32), tf.cast(width, tf.int32), tf.cast(channels, tf.int32)
        image = tf.reshape(image, ([height, width, channels]))
        new_height, new_width, new_channels = self.image_dim
        
        # extract parts
        part_x, part_y = tf.cast(part_x, tf.float32), tf.cast(part_y, tf.float32)
        part_x, part_y = tf.reshape(part_x, shape=[15]), tf.reshape(part_y, shape=[15])
        breast_x, breast_y = part_x[3], part_y[3]
        crown_x, crown_y = part_x[4], part_y[4]
        nape_x, nape_y = part_x[9], part_y[9]
        tail_x, tail_y = part_x[13], part_y[13]
        leg_x, leg_y = part_x[7], part_y[7]
        beak_x, beak_y = part_x[1], part_y[1]
        
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
            clipped_height, clipped_width = tf.maximum(height, [244]), tf.maximum(width, [244])
            true_fn = lambda : tf.image.resize_bilinear(image, [clipped_height[0], 256], align_corners=False)
            false_fn = lambda : tf.image.resize_bilinear(image, [256, clipped_width[0]], align_corners=False)
            image = tf.cond(tf.greater(height, width), true_fn, false_fn)
            image = tf.squeeze(image, [0])
            # preprocess with random crops and horizontal flipping
            image = tf.random_crop(image, size=[new_height, new_width, new_channels])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.central_crop(image, central_fraction=0.875)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [new_height, new_width])
            image = tf.squeeze(image, [0])
        return image, body_crop, head_crop, label
