import tensorflow as tf
import os
import logging
import librosa
# import numpy as np
from tqdm import tqdm
# from utils import create_dir


class TasNetDataLoader():
    def __init__(self, mode, data_dir, batch_size, sample_rate):
        if mode != "train" and mode != "valid" and mode != "infer":
            raise ValueError("mode: {} while mode should be "
                             "'train', 'valid', or 'infer'".format(mode))
        if not os.path.isdir(data_dir):
            raise ValueError("cannot find data_dir: {}".format(data_dir))

        self.wav_dir = os.path.join(data_dir, mode)
        self.tfr = os.path.join(data_dir, mode + '.tfr')
        self.mode = mode
        self.n_speaker = 2
        self.batch_size = batch_size
        self.sample_rate = sample_rate

        if not os.path.isfile(self.tfr) or os.stat(self.tfr).st_size == 0:
            self._encode()

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def get_next(self):
        logging.info("Loading data from {}".format(self.tfr))
        with tf.name_scope("input"):
            dataset = tf.data.TFRecordDataset(self.tfr).map(self._decode)
            if self.mode == "train":
                dataset = dataset.shuffle(2000 + 3 * self.batch_size)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(self.batch_size * 5)
            self.iterator = dataset.make_initializable_iterator()
            return self.iterator.get_next()

    def _encode(self):
        logging.info("Writing {}".format(self.tfr))
        with tf.python_io.TFRecordWriter(self.tfr) as writer:
            mix_wav_dir = os.path.join(self.wav_dir, "mix")
            s1_wav_dir = os.path.join(self.wav_dir, "s1")
            s2_wav_dir = os.path.join(self.wav_dir, "s2")
            mixfilenames = os.listdir(mix_wav_dir)
            #print(mixfilenames)
            #input()
            s1filenames = os.listdir(s1_wav_dir)
            #print(s1filenames)
            #input()
            s2filenames = os.listdir(s2_wav_dir)
            #print(s2filenames)
            #input()
            
            total_files = len(mixfilenames)
            print(total_files)
            for index in tqdm(range(total_files)):
                
                mix, _ = librosa.load(
                    os.path.join(mix_wav_dir, mixfilenames[index]), self.sample_rate)
                s1, _ = librosa.load(
                    os.path.join(s1_wav_dir, s1filenames[index]), self.sample_rate)
                s2, _ = librosa.load(
                    os.path.join(s2_wav_dir, s2filenames[index]), self.sample_rate)

                # def padding(inputs):
                #     return np.pad(
                #         inputs, (int(2.55 * self.sample_rate), 0),
                #         'constant',
                #         constant_values=(0, 0))

                # mix, s1, s2 = padding(mix), padding(s1), padding(s2)

                def write(l, r):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "mix": self._float_list_feature(mix[l:r]),
                                "s1": self._float_list_feature(s1[l:r]),
                                "s2": self._float_list_feature(s2[l:r])
                            }))
                    writer.write(example.SerializeToString())

                now_length = mix.shape[-1]

                if now_length < int(10 * self.sample_rate):
                    continue
                target_length = int(10 * self.sample_rate)
                stride = int(10 * self.sample_rate)
                
                for i in range(0, now_length - target_length + 1, stride):
                    write(i, i + target_length)
                # if now_length // target_length:
                #     write(now_length - target_length, now_length)

    def _decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                "mix": tf.VarLenFeature(tf.float32),
                "s1": tf.VarLenFeature(tf.float32),
                "s2": tf.VarLenFeature(tf.float32)
            },
        )
        mix = tf.sparse_tensor_to_dense(example["mix"])
        s1 = tf.sparse_tensor_to_dense(example["s1"])
        s2 = tf.sparse_tensor_to_dense(example["s2"])
        audios = tf.stack([mix, s1, s2])
        return audios
