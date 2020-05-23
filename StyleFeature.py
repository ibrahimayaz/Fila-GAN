# author is He Zhao
# The time to create is 3:40 PM, 23/3/17
import tensorflow as tf
import vgg
import numpy as np
import scipy.io as sio


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = ('relu4_2',)


vgg_path = 'imagenet-vgg-verydeep-19.mat'
data = sio.loadmat(vgg_path)

def get_style_features(image, mask):

    image = tf.mul(image+1, 127.5)
    image = image*((mask+1)/2)
    
    if image._shape_as_list()[1] != 512:
        image = tf.image.resize_images(image, [512,512])

    img_features = {}

    #with tf.device('/cpu:0'):

    img_pre = vgg.preprocess(image)
    net = vgg.net(data, img_pre)

    for layer in STYLE_LAYERS:
        features = net[layer]
        features = tf.reshape(features, shape=[-1, features._shape_as_list()[1]*features._shape_as_list()[2], features._shape_as_list()[3]])[0]
        features_T = tf.transpose(features)
        gram = tf.matmul(features_T, features) / float(features._shape_as_list()[0]*features._shape_as_list()[1])
        img_features[layer] = gram

    return img_features
    
def get_content_features(image,mask):

    image = tf.mul(image + 1, 127.5)
    image = image*((mask+1)/2)

    img_features = {}

    if image._shape_as_list()[1] != 512:
        image = tf.image.resize_images(image, [512,512])
    
    #with tf.device('/cpu:0'):

    img_pre = vgg.preprocess(image)
    net = vgg.net(data, img_pre)

    for layer in CONTENT_LAYER:
        features = net[layer]
        img_features[layer] = features

    return img_features


def get_style_loss(style_features, img, mask):

    img_features = get_style_features(img, mask)      
    

    style_lossE = 0
    for style_layer in STYLE_LAYERS:
        coff = float(1.0 / len(STYLE_LAYERS))
        img_gram = img_features[style_layer]
        style_gram = style_features[style_layer]
        style_lossE += coff * tf.reduce_mean(tf.abs(img_gram - style_gram))

    style_loss = tf.reduce_mean(style_lossE)

    return style_loss
    

def get_content_loss(img, syn, mask):

    img_features = get_content_features(img, mask)
    syn_features = get_content_features(syn, mask)

    content_lossE = 0
    for content_layer in CONTENT_LAYER:
        coff = float(1.0 / len(CONTENT_LAYER))
        img_content = img_features[content_layer]
        syn_content = syn_features[content_layer]
        content_lossE += coff * tf.reduce_mean(tf.abs(img_content - syn_content))

    content_loss = tf.reduce_mean(content_lossE)

    return content_loss


def get_tv_loss(img, mask):
    img = img*((mask+1)/2)
    # x = tf.reduce_sum(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :]))
    # y = tf.reduce_sum(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    x = tf.reduce_mean(tf.abs(img[:, 1:, :, :] - img[:, :-1, :, :]))
    y = tf.reduce_mean(tf.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))

    return x+y
    

