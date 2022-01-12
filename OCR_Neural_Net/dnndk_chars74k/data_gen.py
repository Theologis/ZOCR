import numpy as np
import os
import random
#from tqdm import tqdm
from utils import myAtoi,load_image,save_test_data,scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer





natural_images_BadImag = "dataset/Img/natural_images_BadImag/Bmp/"
natural_images_GoodImg = "dataset/Img/natural_images_GoodImg/Bmp/"
Hnd = "dataset/Img/Hnd/Img/"
Fnt = "dataset/Img/Fnt/"
class Feature_Extraction ():
    def __init__(self,minibatch_size,proj_directory,real_size):

        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0
        self.minibatch_size = minibatch_size
        self.proj_directory = proj_directory
        self.real_size = real_size




    def read_data(self):
        """
            read_data : This function reads raw images of each file in the dataset and returns randomly shuffle the  train,validate and test sets for images and for label(1-64).
            proj_directory (char) : is where the project directory is (args parameter).
        """
        
        image_dir1 = self.proj_directory + natural_images_BadImag
        image_dir2 = self.proj_directory + natural_images_GoodImg
        image_dir3 = self.proj_directory + Hnd
        image_dir4 = self.proj_directory + Fnt

        #image_dirs = [image_dir1,image_dir2,image_dir3,image_dir4]
        image_dirs = [image_dir4]

        images_data = []
        labels = []
        counter = 0

        for image_dir in image_dirs:
            contents = os.listdir(image_dir)
            classes = [each for each in contents if os.path.isdir(image_dir + each)]
            #path = tqdm(classes, total=len(classes), unit='i')
            path = classes
            for each in path:
                class_path = image_dir + each
                files = os.listdir(class_path)
                #while 1 : ##I wont to have equal data (1000) for each letter
                for ii, file in enumerate(files, 1):
                    # Add images to images_data
                    img = load_image(os.path.join(class_path, file),self.real_size)
                    images_data.append(img[:,:,np.newaxis])
                    labels.append(myAtoi(each[len(each)-2:]))
                    counter+=1



        """
        1)Create labels to one hot ancoded vec.
        lb = LabelBinarizer()#               \
        lb.fit(labels)#                       } 1)
        labels_vecs = lb.transform(labels)#  /
        -Not need it any more(I do it in the train_utils)
        2)Shuffle our data so they can split  into training, validation, and test sets contain data from all classes.
        3)Split images and labels.
        """
        
        labels_vecs = labels

        #####2,3)
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)    
        train_idx, val_idx = next(ss.split(images_data, labels_vecs))

        half_val_len = int(len(val_idx)/2)
        val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

        #split to sets
        self.train_x  = (np.array([images_data[i]  for i in train_idx]))                     
        self.train_y = np.array([labels_vecs[i] for i in train_idx])

        self.val_x = (np.array([images_data[i] for i in val_idx]))
        self.val_y = np.array([labels_vecs[i] for i in val_idx])

        self.test_x = (np.array([images_data[i] for i in test_idx]))
        self.test_y = np.array([labels_vecs[i] for i in test_idx])
        
        
        self.label_mask = np.zeros_like(self.train_y)
        self.label_mask[:len(self.train_x)-1] = 1

        if self.train_x[1] is not None  :
            print(counter, 'data successfully loades with shape : ',self.train_x[1].shape)
            return self.train_x, self.train_y,self.val_x, self.val_y,self.test_x, self.test_y
        else :
            print("Error loading Data!")
    

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
        """
        if partition == 'train':
            cur_index = self.cur_train_index
            return self.train_x[cur_index:cur_index+self.minibatch_size],self.train_y[cur_index:cur_index+self.minibatch_size],self.label_mask[cur_index:cur_index+self.minibatch_size]
        elif partition == 'valid':
            cur_index = self.cur_valid_index
            return self.val_x[cur_index:cur_index+self.minibatch_size],self.val_y[cur_index:cur_index+self.minibatch_size]
        elif partition == 'test':
            cur_index = self.cur_test_index
            return self.test_x[cur_index:cur_index+self.minibatch_size],self.test_y[cur_index:cur_index+self.minibatch_size]
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")
     

    def next_train(self):
        """ Obtain a batch of training data
        """
        while True:
            X_data, Y_data,Mask = self.get_batch('train')
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= len(self.train_x):
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
                break
            yield X_data, Y_data,Mask

    def next_valid(self):
        """ Obtain a batch of validation data
        """
        while True:
            X_data, Y_data = self.get_batch('valid')
            self.cur_valid_index += self.minibatch_size
            if self.cur_valid_index >= len(self.val_x):
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
                break
            yield X_data, Y_data

    def next_test(self):
        """ Obtain a batch of test data
        """
        while True:
            X_data, Y_data = self.get_batch('test')
            self.cur_valid_index += self.minibatch_size
            if self.cur_valid_index >= len(self.val_x):
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
                break
            yield X_data, Y_data

    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data
        """
        if partition == 'train':
            self.train_x,self.train_y,p = shuffle_data(
                self.train_x,self.train_y)
            self.label_mask = np.array([self.label_mask[i] for i in p])
        elif partition == 'valid':
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
                    
def load_data(iter):
    proj_directory="./"
    minibatch_size = 128
    real_size =(32,32,1)
    """
        This module loads the data and returns the sets to create the NN .
    """
    images = []
    image = Feature_Extraction(proj_directory = proj_directory,minibatch_size = minibatch_size,real_size = real_size )
    image.read_data()
    for index in range(0,minibatch_size):
        images.append(image.train_x[iter*minibatch_size+index])
    return {"input_real": images}
