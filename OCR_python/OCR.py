import cv2
import os
import tensorflow as tf
import numpy as np
from utils import load_image,sigmoid,BoundingBoxes_from_Points_with_labels
from Data_augmentation import Data_augmentation

from PIL import Image
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

image_folder = "images/"

a='abcdefghijklmnopqrstuvwxyz' 

input_key = 'x_input'
output_key = 'y_output'
export_path =  './savedmodel'
eval_model = './quantize_eval_model.pb' #the frozen graph ganerated by decent_q

load_DNNDK_model = 0
if load_DNNDK_model : from tensorflow.contrib import decent_q

class OCR:
    """
    Optical Characher recognition Class.
    :param start_point: Start point of the Letter
    :param end_point: End point of the Letter.
    :param RGBimag: The image.
    :param input_size = (input_size,input_size,input_size) The input of the NN
    """
    def __init__(self, start_point, end_point, RGBimag,input_size):
        self.start_point = start_point
        self.end_point = end_point
        self.RGBimag = RGBimag
        self.input_size = input_size 
        
    def Letter_recognition(self):
        ##The Neural network Classifier that runs every tracted letter.

        Letters_predection= []
        Letters = []
        count=0
        mapp={}
        ## 62 classes (0-9, A-Z, a-z)
        for x in range(10):
            mapp[x]=count
            count+=1
        for y in a:
            mapp[count]=y.upper()
            count+=1
        for y in a:
            mapp[count]=y
            count+=1

        gray = cv2.cvtColor(self.RGBimag, cv2.COLOR_BGR2GRAY)
            # remove noise
        OCR_input_img = cv2.GaussianBlur(gray,(3,3),0)
        
        count_deleted = 0
        delete_point = [] # j points that was deleted 
        for i in range(len(self.start_point)):
            ##Data preparation for ANN.
            y_p = int((self.end_point[i][1] - self.start_point[i][1])*.15)
            x_p = int((self.end_point[i][0] - self.start_point[i][0])*.15)

            y_start = self.start_point[i][1] - y_p if self.start_point[i][1] - y_p > 0 else 0
            y_end   =  self.end_point[i][1] + y_p  if self.start_point[i][1] - y_p < self.RGBimag.shape[0] else self.RGBimag.shape[0]

            x_start = self.start_point[i][0] - y_p if self.start_point[i][0] - y_p > 0 else 0
            x_end   = self.end_point[i][0] + y_p  if self.start_point[i][0] - y_p < self.RGBimag.shape[1] else self.RGBimag.shape[1]

            Letter_i = OCR_input_img[y_start : y_end , x_start : x_end]
            Letter_i = Letter_i / 255.0

            # load image function
            if(Letter_i.size == 0):
                delete_point.append(i) #save the deleted j
                continue
            Letter_i = cv2.resize(Letter_i, ( self.input_size, self.input_size), interpolation = cv2.INTER_AREA) 
            short_edge = min(Letter_i.shape[:2])
            yy = int((Letter_i.shape[0] - short_edge) / 2)
            xx = int((Letter_i.shape[1] - short_edge) / 2)
            img = Letter_i[yy: yy + short_edge, xx: xx + short_edge]
            img = cv2.resize(img, ( self.input_size, self.input_size), interpolation = cv2.INTER_AREA)     
            Letters.append(img[:,:,np.newaxis]) ##Data to be fed to the NN.

        delete_point.sort()
        count_deleted = 0 # how many points have been deleted 
        for i in delete_point:
            #delete the points
            self.start_point = np.delete(self.start_point, i-count_deleted, 0)
            self.end_point = np.delete(self.end_point, i-count_deleted, 0)
            count_deleted +=1 
            
        ###Run the Classifier.
        if load_DNNDK_model:
            # We use our "load_graph" function
            graph = load_graph(eval_model)

            #Get Tensor
            x = graph.get_tensor_by_name("input_real" +':0')
            y = graph.get_tensor_by_name("discriminator/out" + ':0')
            class_logits =  graph.get_tensor_by_name("discriminator/dense/BiasAdd:0")

            ####Calculate Accuracy
            with tf.Session(graph=graph) as sess:
                Letters_predection,y_class_logits  = sess.run((y,class_logits), {x: Letters})
        else :    
            with tf.Session() as sess:
                signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                
                meta_graph_def = tf.saved_model.loader.load(
                        sess,
                        [tf.saved_model.tag_constants.SERVING],
                        export_path)
                signature = meta_graph_def.signature_def

                x_tensor_name = signature[signature_key].inputs[input_key].name
                y_tensor_name = signature[signature_key].outputs[output_key].name

                x = sess.graph.get_tensor_by_name(x_tensor_name)
                class_logits =  sess.graph.get_tensor_by_name("discriminator/dense/BiasAdd:0")
                y = sess.graph.get_tensor_by_name(y_tensor_name)

                Letters_predection,y_class_logits  = sess.run((y,class_logits), {x: Letters})

        self.predections=[] 
        delete_point = [] # i points that was deleted
        self.P_letter_is_real_letter = [] 
        for i in range(len(y_class_logits)):
            real_class_logits = y_class_logits[i]
            mx = np.amax(real_class_logits )
            stable_real_class_logits = real_class_logits - mx
            gan_logits = np.log(np.amax(np.exp(stable_real_class_logits))) + np.squeeze(mx) 
            self.P_letter_is_real_letter.append(sigmoid(gan_logits)) ##Gives you the propability is a real Letter
            if (int(round(Letters_predection[i].sum(), 0))): ##it means that this is a letter
                y = np.array(Letters_predection[i]) 
                prediction = np.argmax(y[np.newaxis,:],axis=1)
                self.predections.append(mapp.get(prediction[0]-1))
            else :
                delete_point.append(i) #save the deleted j


        BoundingBoxes_from_Points_with_labels(self.start_point,self.end_point,self.RGBimag,self.predections,self.P_letter_is_real_letter)
 
    #def world_detection(self):
        ## Detects the worlds given sp and ep of Letters
        ## Use symmetry in X or Y between the Letters


def do_OCR(start_point,end_point,image_path,input_size):
    RGBimag = cv2.imread(image_path) 
    OCR_inst = OCR(start_point,end_point,RGBimag,input_size)
    OCR_inst.Letter_recognition()



def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph