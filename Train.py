# author is He Zhao
# The time to create is 10:23 AM, 29/11/16

import tensorflow as tf
from tensorflow.python.training import training_util
import Net
import numpy as np
import os
import time
import StyleFeature
import scipy.io as sio
import pdb
#import matplotlib.pyplot as plt

from Opts import save_images, matTonpy
# =============================== path set =============================================== #
load_model = None#'initial_model'
save_model = False

# ============================== parameters set ========================================== #
#adversarial
L1 = 1
#style
L2 = 10
#content
L3 = 1
#tv
L4 = 100
#style number
style_flag = 'drive'
styleNum = 0

# ============================== model set ========================================== #
model = 'test'

result_dir = 'Model_and_Result' + '/' + model + ''
sample_directory = result_dir + '/figs'
sample_directory2 = result_dir + '/figs_mask'
# Directory to save sample images from generator in.
model_directory = result_dir + '/models'  # Directory to save trained model to.

if tf.gfile.Exists(result_dir):
    tf.gfile.DeleteRecursively(result_dir)
if not os.path.exists(sample_directory):
    os.makedirs(sample_directory)
if not os.path.exists(sample_directory2):
    os.makedirs(sample_directory2)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

os.system('cp {} {}'.format(__file__, result_dir))
os.system('cp {} {}'.format('Net.py', result_dir))
os.system('cp {} {}'.format('Opts.py', result_dir))
os.system('cp {} {}'.format('StyleFeature.py', result_dir))

with open(model_directory + '/training_log.txt', 'w') as f:
    f.close()
# ============================== parameters set ========================================== #


learning_rate = 0.0002
beta1 = 0.5

batch_size = 1  # Size of image batch to apply at each iteration.
max_epoch = 100

channel = 3
img_size = 512
img_x = 512
img_y = 512
padding_l = 0
padding_r = 0
padding_t = 0
padding_d = 0

style_size = 512

sample_batch = 4
z_size = 400



# =============================== model and data definition ================================ #
generator = Net.generator
discriminator = Net.discriminator

build_data = Net.build_data

tf.reset_default_graph()

gt = tf.placeholder(shape=[None, img_size, img_size, 1], dtype=tf.float32)
img = tf.placeholder(shape=[None, img_size, img_size, channel], dtype=tf.float32)
mask = tf.placeholder(shape=[None, img_size, img_size, 1], dtype=tf.float32)
style = tf.placeholder(shape=[None, style_size, style_size, 3],dtype=tf.float32)
z = tf.placeholder(shape=[None, z_size], dtype=tf.float32)

gt_mask = tf.concat(3, [gt, mask])

syn = generator(gt_mask, z)

real_img_gt = tf.concat(3, [img, gt, mask])
fake_syn_gt = tf.concat(3, [syn, gt, mask])

Dx, Dx_logits = discriminator(real_img_gt)
Dg, Dg_logits = discriminator(fake_syn_gt, reuse=True)

db, mask_s = build_data(batch_size)

# ============================= loss function and optimizer =============================== #
# style_features
import vgg
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
vgg_path = 'imagenet-vgg-verydeep-19.mat'
data = sio.loadmat(vgg_path)

if style_flag == 'drive':
    ss = sio.loadmat('trn20+tst20.mat')['imgAllTrain'][styleNum]
elif style_flag == 'messidor':
    ss = sio.loadmat('style.mat')['imgAll'][styleNum]
elif style_flag == 'stare':
    ss = sio.loadmat('stare_original_mask_binary.mat')['imgAll'][styleNum]
    
ss = (np.reshape(ss, [batch_size, img_size, img_size, 3]) - 0.5) * 2.0
ss = np.lib.pad(ss, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                constant_values=(-1, -1))  # Pad the images so the are 32x32

image = (ss + 1) * 127.5

style_features = {}
style_features_x = {}
style_features_y = {}

with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
    style_image = tf.placeholder(tf.float32, shape=image.shape, name='style_image')
    img_pre = vgg.preprocess(style_image)
    net = vgg.net(data, img_pre)

    for layer in STYLE_LAYERS:
        features = net[layer].eval(feed_dict={style_image:image})
        
        
        features_x = features[:, 1:, :, :] - features[:, :-1, :, :]
        features_y = features[:, :, 1:, :] - features[:, :, :-1, :]

        features_x = np.reshape(features_x, (-1, features_x.shape[1] * features_x.shape[2], features_x.shape[3]))[0]
        features_y = np.reshape(features_y, (-1, features_y.shape[1] * features_y.shape[2], features_y.shape[3]))[0]

        gram_x = np.matmul(features_x.T, features_x) / float(features_x.size)
        gram_y = np.matmul(features_y.T, features_y) / float(features_y.size)
        style_features_x[layer] = gram_x
        style_features_y[layer] = gram_y

        features = np.reshape(features, (-1, features.shape[1] * features.shape[2], features.shape[3]))[0]

        # mean_value = np.mean(features)
        # features = features - mean_value

        gram = np.matmul(features.T, features) / float(features.size)
        style_features[layer] = gram

# ============================================================================================#
# discriminator loss
with tf.name_scope('d_loss'):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dx_logits, tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg_logits, tf.zeros_like(Dg)))
    d_loss = d_loss_real + d_loss_fake

# generator loss
with tf.name_scope('g_loss'):

    g_loss_adversarial = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg_logits, tf.ones_like(Dg)))    

    g_loss_style = StyleFeature.get_style_loss(style_features, syn, mask)      
    g_loss_content = StyleFeature.get_content_loss(img, syn, mask)    
   
    g_loss_tv = StyleFeature.get_tv_loss(syn, mask)
    
    g_loss = L1*g_loss_adversarial + L2*g_loss_style + L3*g_loss_content + L4*g_loss_tv 

    
# split the variable for two differentiable function
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

# optimizer
global_step = tf.Variable(0, trainable=False)
with tf.name_scope('train'):
    d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*0.4).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

# =============================== summary prepare ============================================= #

# write summary
Dx_sum = tf.histogram_summary("Dx", Dx)
Dg_sum = tf.histogram_summary("Dg", Dg)

Dx_sum_scalar = tf.scalar_summary("Dx_value", tf.reduce_mean(Dx))
Dg_sum_scalar = tf.scalar_summary("Dg_value", tf.reduce_mean(Dg))

syn_sum = tf.image_summary("synthesize", syn)

d_loss_real_sum = tf.scalar_summary("d_loss_real", d_loss_real)
d_loss_fake_sum = tf.scalar_summary("d_loss_fake", d_loss_fake)
d_loss_sum = tf.scalar_summary("d_loss", d_loss)
g_loss_sum = tf.scalar_summary("g_loss", g_loss)

# g_sum = tf.merge_summary([Dg_sum, syn_sum, d_loss_fake_sum, g_loss_sum])
# d_sum = tf.merge_summary([Dx_sum, d_loss_real_sum, d_loss_fake_sum, d_loss_sum])

g_sum = tf.merge_summary([Dg_sum_scalar])
d_sum = tf.merge_summary([Dx_sum_scalar])

# =============================== train phase ============================================= #
init = tf.initialize_all_variables()
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=None)

sess.run(init)
#writer = tf.train.SummaryWriter(model_directory, sess.graph)

# ==================================== save initialization ================================ #
if load_model:
    ckpt = tf.train.get_checkpoint_state('Model_and_Result/' + load_model + '/models')
    saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.save(sess, model_directory + '/model-' + str(0) + '.cptk')
    print "load saved model and SAVE"
elif save_model:
    saver.save(sess, model_directory + '/model-' + str(0) + '.cptk')
    print "Saved begining Model "

# ==================================== start training ===================================== #
stime=time.time()
for epoch in xrange(max_epoch):
    batchNum = 1

    for data_train, _ in db:
        for batch in data_train:

            ms = mask_s[batchNum-1]
            ms = (np.reshape(ms, [batch_size, img_size, img_size, 1]) - 0.5) * 2.0
            ms = np.lib.pad(ms, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))

            z_sample = np.random.normal(0, 0.001, size=[batch_size, z_size]).astype(np.float32)#mean
            zs = z_sample

            xs, ys = batch  # Draw a sample batch from MNIST dataset.
            if xs.shape[0] != batch_size:
                continue

            # xs = np.transpose(xs, (0, 2, 3, 1))
            xs = (np.reshape(xs, [batch_size, img_size, img_size, channel]) - 0.5) * 2.0
            xs = np.lib.pad(xs, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))  # Pad the images so the are 32x32

            # ys = np.transpose(ys, (0, 2, 3, 1))
            ys = (np.reshape(ys, [batch_size, img_size, img_size, 1]) - 0.5) * 2.0
            ys = np.lib.pad(ys, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))  # Pad the images so the are 32x32

            ss = sio.loadmat('test_1To4.mat')['imgAllTest'][3]
            ss = (np.reshape(ss, [batch_size, img_size, img_size, 3]) - 0.5) * 2.0
            ss = np.lib.pad(ss, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)), 'constant',
                            constant_values=(-1, -1))  # Pad the images so the are 32x32

            feed_dict = {img: xs, gt: ys, z: zs, mask: ms, style: ss}
            # Update the discriminator
            _, dLoss = sess.run([d_optimizer, d_loss], feed_dict=feed_dict)  

            # Update the generator, twice for good measure.
            _ = sess.run([g_optimizer], feed_dict=feed_dict)
            

            _, gLoss, advL, styleL, contL, tvL = sess.run([g_optimizer, g_loss, g_loss_adversarial, g_loss_style, g_loss_content, g_loss_tv], feed_dict=feed_dict) 
            
            print "[Epoch: %2d / %2d] [%4d]Gen Loss: %.4f Disc Loss: %.4f, style: %.4f, content: %.4f, adv: %.4f, tv: %.4f" \
                  % (epoch, max_epoch, batchNum, gLoss, dLoss, styleL, contL, advL, tvL)
            with open(model_directory + '/training_log.txt', 'a') as text_file:
                text_file.write(
                    "[Epoch: %2d / %2d] [%4d]Gen Loss: %.4f Disc Loss: %.4f, style: %.4f, content: %.4f, adv: %.4f, tv: %.4f \n"
                    % (epoch, max_epoch, batchNum, gLoss, dLoss, styleL, contL, advL, tvL))
            batchNum += 1
            if training_util.global_step(sess, global_step) % 100 == 0:

                img_sample, gt_sample, mask_sample = Net.matTonpy_35()

                z2 = np.random.normal(0, 1.0, size=[batch_size, z_size]).astype(np.float32)                

                # img_sample = img_sample[:, :, :, 1]
                img_sample = (np.reshape(img_sample, [sample_batch, img_x, img_y, channel]) - 0.5) * 2.0
                # Pad the images so the are 32x32
                img_sample = np.lib.pad(img_sample, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)),
                                        'constant', constant_values=(-1, -1))

                gt_sample = (np.reshape(gt_sample, [sample_batch, img_x, img_y, 1]) - 0.5) * 2.0
                # Pad the images so the are 32x32
                gt_sample = np.lib.pad(gt_sample, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)),
                                       'constant', constant_values=(-1, -1))

                mask_sample = (np.reshape(mask_sample, [sample_batch, img_x, img_y, 1]) - 0.5) * 2.0
                # Pad the images so the are 32x32
                mask_sample = np.lib.pad(mask_sample, ((0, 0), (padding_l, padding_r), (padding_t, padding_d), (0, 0)),
                                        'constant', constant_values=(-1, -1))

                z3 = np.random.normal(0, 1.0, size=[batch_size, z_size]).astype(np.float32)
                
                sa = 0
                sb = 1
                sc = 2
                sd = 3

                syn_sample_a, dLreal_val_a, dLfake_val_a = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sa]], gt: [gt_sample[sa]], z: z2, mask:[mask_sample[sa]]})
                syn_sample_b, dLreal_val_b, dLfake_val_b = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sb]], gt: [gt_sample[sb]], z: z3, mask:[mask_sample[sb]]})
                syn_sample_c, dLreal_val_c, dLfake_val_c = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sc]], gt: [gt_sample[sc]], z: z2, mask:[mask_sample[sc]]})
                syn_sample_d, dLreal_val_d, dLfake_val_d = sess.run([syn, Dx, Dg],
                                                                    feed_dict={img: [img_sample[sd]], gt: [gt_sample[sd]], z: zs, mask:[mask_sample[sd]]})

                syn_sample = np.concatenate((syn_sample_a, syn_sample_b, syn_sample_c,syn_sample_d),axis=0)

                syn_sample_am = syn_sample_a * ((mask_sample[sa] + 1) / 2)
                syn_sample_bm = syn_sample_b * ((mask_sample[sb] + 1) / 2)
                syn_sample_cm = syn_sample_c * ((mask_sample[sc] + 1) / 2)
                syn_sample_dm = syn_sample_d * ((mask_sample[sd] + 1) / 2)
                syn_sample_m = np.concatenate((syn_sample_am, syn_sample_bm, syn_sample_cm, syn_sample_dm), axis=0)
                
                dLreal_val = (dLreal_val_a + dLreal_val_b + dLreal_val_c + dLreal_val_d) / 4
                dLfake_val = (dLfake_val_a + dLfake_val_b + dLfake_val_c + dLfake_val_d) / 4

                # Save sample generator images for viewing training progress.
                save_images(np.reshape(syn_sample, [sample_batch, img_x, img_y, channel]),
                            [int(np.sqrt(sample_batch)), int(np.sqrt(sample_batch))],
                            sample_directory + '/fig' + str(training_util.global_step(sess, global_step)) + '.png')
                            
                save_images(np.reshape(syn_sample_m, [sample_batch, img_x, img_y, channel]),
                            [int(np.sqrt(sample_batch)), int(np.sqrt(sample_batch))],
                            sample_directory2 + '/fig' + str(training_util.global_step(sess, global_step)) + '.png')

                print "[Sample (global_step = %d)] real: %.4f fake: %.4f" \
                      % (training_util.global_step(sess, global_step), np.mean(dLreal_val), np.mean(dLfake_val))
                with open(model_directory + '/training_log.txt', 'a') as text_file:
                    text_file.write("[Sample (global_step = %d)] real: %.4f fake: %.4f \n"
                                    % (training_util.global_step(sess, global_step), np.mean(dLreal_val),
                                       np.mean(dLfake_val)))

            if training_util.global_step(sess, global_step) % 1000 == 0:
                saver.save(sess,
                           model_directory + '/model-' + str(training_util.global_step(sess, global_step)) + '.cptk')
                print "Saved Model %d, time: %.4f" % (training_util.global_step(sess, global_step), time.time()-stime)

             

saver.save(sess, model_directory + '/model-' + str(training_util.global_step(sess, global_step)) + '.cptk')
print "Saved Model %d, time: %.4f" % (training_util.global_step(sess, global_step), time.time()-stime)

sess.close()

