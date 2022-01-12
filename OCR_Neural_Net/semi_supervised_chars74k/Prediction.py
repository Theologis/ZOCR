import os
import numpy as np
from utils import load_image
import tensorflow as tf



def Run_pred(proj_directory,img_path,real_size,d_size_mult = 64):
    mapp={}
    a='abcdefghijklmnopqrstuvwxyz'
    count=0
    for x in range(10):
        mapp[x]=count
        count+=1
    for y in a:
        mapp[count]=y.upper()
        count+=1
    for y in a:
        mapp[count]=y
        count+=1

    input_key = 'x_input'
    output_key = 'y_output'
    export_path =  './savedmodel'
    ###Returns the prediction of the fitst image.
    img = load_image(img_path,real_size)
    test_img=img[np.newaxis,:,:,np.newaxis]



    ## Run prediction
    with tf.Session() as sess:
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        
        meta_graph_def = tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                export_path)
                 
        signature = meta_graph_def.signature_def

        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        print("Output node/s name :",y_tensor_name)
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        prediction = sess.run(y, {x: test_img})

    prediction=np.argmax(prediction,axis=1)

    print("Prediction of the first image is :",mapp.get(prediction[0]-1))

