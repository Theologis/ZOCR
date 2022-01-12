import tensorflow as tf
from tqdm import tqdm
from model import model_inputs,model_loss,model_opt
import pickle as pkl
import time
import os
import numpy as np

# parameters
n_classes = 62
z_size =100

class GAN:
    """
    A GAN model.
    :param real_size: The shape of the real data.
    :param z_size: The number of entries in the z code vector.
    :param learnin_rate: The learning rate to use for Adam.
    :param num_classes: The number of classes to recognize.
    :param alpha: The slope of the left half of the leaky ReLU activation
    :param beta1: The beta1 parameter for Adam.
    """
    def __init__(self, real_size, learning_rate, num_classes=n_classes, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()
        
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        self.input_real, self.input_z, self.y, self.label_mask = model_inputs(real_size, z_size)
        self.drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")
        
        loss_results = model_loss(self.input_real, self.input_z,
                                              real_size[2], self.y, label_mask=self.label_mask,
                                                                          num_classes = n_classes,
                                                                          alpha=0.2,
                                                           drop_rate=self.drop_rate)
        self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples,self.prediction = loss_results
        
        self.d_opt, self.g_opt, self.shrink_lr,self.d_vars = model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1)
def train(net, dataset, epochs, batch_size,figsize=(5,5)):

    
    
    saver = tf.train.Saver(var_list=net.d_vars)
    sample_z = np.random.normal(0, 1, size=(50))

    samples, train_accuracies, test_accuracies = [], [], []
    steps = 0
    _=dataset.read_data()

    best_val = -1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            print("Epoch",e)
            
            t1e = time.time()
            num_examples = 0
            num_correct = 0
            for x, y, label_mask in dataset.next_train():
                assert 'int' in str(y.dtype)
                steps += 1
                num_examples += label_mask.sum()

                # Sample random noise for G
                batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                # Run optimizers
                t1 = time.time()
                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: x, net.input_z: batch_z,
                                                    net.y : y, net.label_mask : label_mask})
                t2 = time.time()
                num_correct += correct

            sess.run([net.shrink_lr])
            
            
            train_accuracy = num_correct / float(num_examples)
            
            print("\t\tClassifier train accuracy: ", train_accuracy)
            
            num_examples = 0
            num_correct = 0
            for x, y in dataset.next_valid():
                assert 'int' in str(y.dtype)
                num_examples += x.shape[0]
                correct, = sess.run([net.correct], feed_dict={net.input_real: x,
                                                   net.y : y,
                                                   net.drop_rate: 0.})
                num_correct += correct
            
            vali_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier validation accuracy", vali_accuracy)
            print("\t\tStep time: ", t2 - t1)
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)
            
            """"
            gen_samples = sess.run(
                                   net.samples,
                                   feed_dict={net.input_z: sample_z})
            samples.append(gen_samples)
            _ = view_samples(-1, samples, 5, 10, figsize=figsize)
            plt.show()
            """
            #save best valid model
            #if best_val < vali_accuracy :
                
            
            # Save history of accuracies to view after training
            train_accuracies.append(train_accuracy)
            test_accuracies.append(vali_accuracy)
            
        # save the model
        export_path =  './savedmodel'
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(net.input_real)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(net.prediction)

        prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x_input': tensor_info_x},
            outputs={'y_output': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature 
        },
        )
        builder.save()    
        num_examples = 0
        num_correct = 0   
        for x, y in dataset.next_test():
            assert 'int' in str(y.dtype)
            num_examples += x.shape[0]
            correct, = sess.run([net.correct], feed_dict={net.input_real: x,
                                                    net.y : y,
                                                    net.drop_rate: 0.})
            num_correct += correct
            
        test_accuracy = num_correct / float(num_examples)
        print("\nClassifier test accuracy", test_accuracy)
        saver = tf.train.Saver()
        saver.save(sess, './checkpoints/discriminator.ckpt')

        

    return train_accuracies, test_accuracies