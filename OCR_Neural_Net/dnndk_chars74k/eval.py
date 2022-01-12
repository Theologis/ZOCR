import tensorflow as tf
import numpy as np
from eval_data_gen import Feature_Extraction
from tensorflow.contrib import decent_q



eval_model = './quantize_results/quantize_eval_model.pb' #the frozen graph ganerated by decent_q
frozen_model_dnndk = './frozen_model_dnndk.pb' #the frozen graph generated from checkpoint
def evaluate_dnndk_model(input_node,output_node,proj_directory='./'):
    """
    -evaluate_dnndk_model : Perform evaluation for the float and quantized model respectively script
    -proj_directory (char) : is where the project directory is (args parameter) ${DNNDK_chars74k}/ direcory.
    -Note: Users need to ‘import tensorflow.contrib.decent_q’ explicitly in the evaluation python script to
    register the custom quantize operation as tensorflow.contrib is lazy loaded now for xilinx model.
    """
    #print_weights(eval_model)
    #Load test data images
    dataset = Feature_Extraction(proj_directory= proj_directory)
    dataset.read_data()

    # We use our "load_graph" function
    graph = load_graph(eval_model)

    #Get Tensor
    x = graph.get_tensor_by_name(input_node +':0')
    y = graph.get_tensor_by_name(output_node + ':0')

    ####Calculate Accuracy
    with tf.Session(graph=graph) as sess:
        num_examples = 0
        num_correct = 0
        for x_data, y_data in dataset.next_test():
            num_examples += x_data.shape[0]
            
            pred = sess.run(y, feed_dict={x: x_data}
                       )
            pred_class = tf.cast(tf.argmax(pred, 1), tf.int32)
            eq = tf.equal(tf.squeeze(y_data), pred_class)
            correct = tf.reduce_sum(tf.to_float(eq))

            num_correct += correct
            test_accuracy = num_correct / float(num_examples)
        print("\nClassifier test accuracy", test_accuracy.eval())
 
    
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

def print_weights(frozen_graph_filename):
    """
    Load protobuf file(frozen_graph_filename) and print all the nodes and their values.
    """
    from tensorflow.python.framework import tensor_util
    from tensorflow.python.platform import gfile

    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(eval_model,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes=[n for n in graph_def.node]
    wts = [n for n in graph_nodes if n.op=='Const']
    for n in wts:
        print("Name of the node - %s" % n.name)
        print("Value - ")
        print(tensor_util.MakeNdarray(n.attr['value'].tensor))
