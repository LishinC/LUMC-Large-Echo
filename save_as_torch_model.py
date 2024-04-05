from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import random
import sys
import os
sys.path.append('./funcs/')
sys.path.append('./nets/')
from optparse import OptionParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow.contrib.slim as slim
import vgg as network
from pytorch_vgg import VGG
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def LoadTfModel_SaveTorchModel(feature_dim, label_dim, model_name, save_torch_model_path):
    tf.reset_default_graph()
    sess = tf.Session()
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    pytorch_model = VGG(output_dimension=label_dim)

    # Extract the weights
    with tf.Session() as sess:
        saver.restore(sess, model_name)
        # slim.model_analyzer.analyze_vars(tf.global_variables(), print_info=True)

        for var in tf.global_variables():
            if (not "Adam" in var.name) & (('conv' in var.name)|('fc' in var.name)):
                layer_name = var.name.split('/')[1]
                layer_attr = var.name.split('/')[2]
                layer = getattr(pytorch_model, layer_name)
                # print(layer_name, layer_attr, var.eval().shape)
                if 'conv' in layer_name:
                    if 'W:0' in layer_attr:
                        print(layer_name, layer_attr, layer.weight.data.shape, torch.from_numpy(var.eval()).permute(3, 2, 0, 1).shape)
                        assert layer.weight.data.shape == torch.from_numpy(var.eval()).permute(3, 2, 0, 1).shape
                        layer.weight.data = torch.from_numpy(var.eval()).permute(3, 2, 0, 1)
                    elif 'b:0' in layer_attr:
                        print(layer_name, layer_attr, layer.bias.data.shape, var.eval().shape)
                        layer.bias.data = torch.from_numpy(var.eval())
                elif 'fc' in layer_name:
                    if 'W:0' in layer_attr:
                        print(layer_name, layer_attr, layer.weight.data.shape, var.eval().T.shape)
                        assert layer.weight.data.shape == var.eval().T.shape
                        layer.weight.data = torch.from_numpy(var.eval().T)
                    elif 'b:0' in layer_attr:
                        print(layer_name, layer_attr, layer.bias.data.shape, var.eval().shape)
                        layer.bias.data = torch.from_numpy(var.eval())

    torch.save(pytorch_model.state_dict(), save_torch_model_path)


if __name__ == '__main__':
    # The following specifications are according to the original code provided by the author
    model = "view_23_e5_class_11-Mar-2018"
    dicomdir = "dicomsample"
    model_name = './models/' + model

    infile = open("viewclasses_" + model + ".txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)

    # Now load the Tensorflow model and save into a Pytorch model
    save_torch_model_path = 'pytorch_model.pth'
    LoadTfModel_SaveTorchModel(feature_dim, label_dim, model_name, save_torch_model_path)

    # And following is an example code how you load the saved Pytorch model
    pytorch_model = VGG(output_dimension=label_dim)
    pytorch_model.load_state_dict(torch.load(save_torch_model_path))
    pytorch_model.eval()
    print(pytorch_model)
