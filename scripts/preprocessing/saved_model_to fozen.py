import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config
import numpy as np

def convert_saved_model_to_pb(output_node_names, input_saved_model_dir, output_graph_dir):
    from tensorflow.python.tools import freeze_graph

    output_node_names = ','.join(output_node_names)

    freeze_graph.freeze_graph(input_graph=None, input_saver=None,
                              input_binary=None,
                              input_checkpoint=None,
                              output_node_names=output_node_names,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              output_graph=output_graph_dir,
                              clear_devices=None,
                              initializer_nodes=None,
                              input_saved_model_dir=input_saved_model_dir)


def save_output_tensor_to_pb():
    output_names = ['StatefulPartitionedCall']
    save_pb_model_path = '/home/whoisltd/detect/training/exported-models/my_model/saved_model/saved_model1.pb'
    model_dir = '/home/whoisltd/detect/training/exported-models/my_model/saved_model'
    convert_saved_model_to_pb(output_names, model_dir, save_pb_model_path)


save_output_tensor_to_pb()