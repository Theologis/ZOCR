import numpy as np
import os
import cv2
from utils import myAtoi
import random





eval_imaga_data = "OCR_ZedBoard/test_data/"

class Feature_Extraction ():
    def __init__(self,proj_directory="./" ,minibatch_size = 128):
        self.cur_valid_index = 0
        self.rng = random.Random(132)
        self.proj_directory = proj_directory
        self.minibatch_size = minibatch_size




    def read_data(self):
        """
            read_data : This function reads preprocessed images(32x32x1 with gray scale) from the test_data/ folder.
            proj_directory (char) : is where the project directory is (args parameter) ${DNNDK_chars74k}/ direcory.
        """
        
        image_dir = self.proj_directory + eval_imaga_data

        images_data = []
        labels = []
        counter = 0
        files = os.listdir(image_dir)
        for each in files:
            # Add images to images_data
            img = cv2.imread((os.path.join(image_dir, each)))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = gray/255.0
            images_data.append(img[:,:,np.newaxis])
            labels.append(int(each[len(each)-6:len(each)-4]))
            counter+=1


        self.test_x = np.array(images_data)
        self.test_y = labels

        if self.test_x[1] is not None  :
            print(counter, 'data successfully loades with shape : ',self.test_x[1].shape)
        else :
            print("Error loading Data!")
    

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
        """
        if partition == 'test':
            cur_index = self.cur_valid_index
            return self.test_x[cur_index:cur_index+self.minibatch_size],self.test_y[cur_index:cur_index+self.minibatch_size]
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")
     

    def next_test(self):
        """ Obtain a batch of test data
        """
        while True:
            X_data, Y_data = self.get_batch('test')
            self.cur_valid_index += self.minibatch_size
            if self.cur_valid_index >= len(self.test_x):
                self.cur_valid_index = 0
                break
            yield X_data, Y_data

    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data
        """
        if partition == 'valid':
            self.val_x,self.val_y,_ = shuffle_data(
                self.val_x,self.val_y)
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")


def shuffle_data(train_x,train_y):
    """ Shuffle the data (called after making a complete pass through
        training or validation data during the training process)
    Params:
        audio_paths (list): Paths to audio clips
    """
    p = np.random.permutation(len(train_x))
    train_x = np.array([train_x[i] for i in p])
    train_y = np.array([train_y[i] for i in p])

    return train_x,train_y,p
