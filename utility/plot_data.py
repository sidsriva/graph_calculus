import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime

def plot_images3D_c_mean_var(img_test, img_predict_mean, img_input, img_predict_var):
    img0 = img_test[0]

    # get the input shape
    if (tf.__version__[0:1] == '1'):
        sess = tf.Session()
        with sess.as_default():
            input_shapes = tf.shape(img0).eval()
    elif (tf.__version__[0:1] == '2'):
        input_shapes = tf.shape(img0).numpy()

    i = 1
    print('len of inputs: ', len(input_shapes))
    if (len(input_shapes) == 4):
        the_img_test = tf.reshape(img_test[i], input_shapes[:-1])
        the_img_pre_mean = tf.reshape(img_predict_mean[i], input_shapes[:-1])
        the_img_pre_var = tf.reshape(img_predict_var[i], input_shapes[:-1])
        the_img_input = tf.reshape(img_input[i], input_shapes[:-1])
    elif (len(input_shapes) == 3):
        the_img_test = img_test[i]
        the_img_pre_mean = img_predict_mean[i]
        the_img_pre_var = img_predict_var[i]
        the_img_input = img_input[i]

    if (True):
        the_img_pre_mean = np.ma.masked_where(the_img_test < -0.9, the_img_pre_mean)
        the_img_pre_var = np.ma.masked_where(the_img_test < -0.9, the_img_pre_var)

        tmp_img_test = np.concatenate((the_img_test, the_img_test), axis=2)
        print(np.shape(tmp_img_test))
        the_img_input = np.ma.masked_where(tmp_img_test < -0.9,
                                           the_img_input)

        the_img_test = np.ma.masked_where(the_img_test < -0.9, the_img_test)

        the_img_mark = 1.0e-10 * np.ones(np.shape(the_img_test))

    # display original
    ax = plt.subplot(2, 4, 1)
    c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('label: uc')

    # display reconstruction: mean
    ax = plt.subplot(2, 4, 2)
    c_img = plt.imshow(the_img_pre_mean[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('prediction (mean): uc')

    # display reconstruction: var
    ax = plt.subplot(2, 4, 6)
    c_img = plt.imshow(the_img_pre_var[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('prediction (var): uc')

    # display ground truth
    ax = plt.subplot(2, 4, 3)
    c_img = plt.imshow(the_img_input[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: uc')

    # display error
    ax = plt.subplot(2, 4, 7)
    c_img = plt.imshow((the_img_test[:, :, 0] - the_img_pre_mean[:, :, 0]) / (the_img_test[:, :, 0] + the_img_mark[:, :, 0]))  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('error')

    # display reconstruction
    ax = plt.subplot(2, 4, 4)
    c_img = plt.imshow(the_img_input[:, :, 1])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: hc')


    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig("prediction" + now_str + ".png", format='png')
    plt.show()



def plot_images3D_c(img_test, img_predict, img_input):
    img0 = img_test[0]

    # get the input shape
    if (tf.__version__[0:1] == '1'):
        sess = tf.Session()
        with sess.as_default():
            input_shapes = tf.shape(img0).eval()
    elif (tf.__version__[0:1] == '2'):
        input_shapes = tf.shape(img0).numpy()

    i = 1
    print('len of inputs: ', len(input_shapes))
    if (len(input_shapes) == 4):
        the_img_test = tf.reshape(img_test[i], input_shapes[:-1])
        the_img_pre = tf.reshape(img_predict[i], input_shapes[:-1])
        the_img_input = tf.reshape(img_input[i], input_shapes[:-1])
    elif (len(input_shapes) == 3):
        the_img_test = img_test[i]
        the_img_pre = img_predict[i]
        the_img_input = img_input[i]

    if (True):
        the_img_pre = np.ma.masked_where(the_img_test < -0.9, the_img_pre)

        tmp_img_test = np.concatenate((the_img_test, the_img_test), axis=2)
        print(np.shape(tmp_img_test))
        the_img_input = np.ma.masked_where(tmp_img_test < -0.9,
                                           the_img_input)

        the_img_test = np.ma.masked_where(the_img_test < -0.9, the_img_test)

    # display original
    ax = plt.subplot(2, 4, 1)
    c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('label: uc')

    # display reconstruction
    ax = plt.subplot(2, 4, 2)
    c_img = plt.imshow(the_img_pre[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('prediction: uc')

    # display reconstruction
    ax = plt.subplot(2, 4, 3)
    c_img = plt.imshow(the_img_input[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: uc')

    # display reconstruction
    ax = plt.subplot(2, 4, 4)
    c_img = plt.imshow(the_img_input[:, :, 1])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: hc')


    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig("prediction" + now_str + ".png", format='png')
    plt.show()




def plot_images3D(img_test, img_predict, img_input):
    img0 = img_test[0]

    # get the input shape
    if (tf.__version__[0:1] == '1'):
        sess = tf.Session()
        with sess.as_default():
            input_shapes = tf.shape(img0).eval()
    elif (tf.__version__[0:1] == '2'):
        input_shapes = tf.shape(img0).numpy()

    i = 1
    print('len of inputs: ', len(input_shapes))
    if (len(input_shapes) == 4):
        the_img_test = tf.reshape(img_test[i], input_shapes[:-1])
        the_img_pre = tf.reshape(img_predict[i], input_shapes[:-1])
        the_img_input = tf.reshape(img_input[i], input_shapes[:-1])
    elif (len(input_shapes) == 3):
        the_img_test = img_test[i]
        the_img_pre = img_predict[i]
        the_img_input = img_input[i]

    if (True):
        the_img_pre = np.ma.masked_where(the_img_test < -0.9, the_img_pre)

        tmp_img_test = np.concatenate((the_img_test, the_img_test), axis=2)
        print(np.shape(tmp_img_test))
        the_img_input = np.ma.masked_where(tmp_img_test < -0.9,
                                           the_img_input)

        the_img_test = np.ma.masked_where(the_img_test < -0.9, the_img_test)

    # display original
    ax = plt.subplot(2, 4, 1)
    c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('label: ux')

    # display reconstruction
    ax = plt.subplot(2, 4, 2)
    c_img = plt.imshow(the_img_pre[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('prediction: ux')

    # display reconstruction
    ax = plt.subplot(2, 4, 3)
    c_img = plt.imshow(the_img_input[:, :, 0])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: ux')

    # display reconstruction
    ax = plt.subplot(2, 4, 4)
    c_img = plt.imshow(the_img_input[:, :, 2])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: Tx')

    # display original
    ax = plt.subplot(2, 4, 5)
    c_img = plt.imshow(the_img_test[:, :, 1])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('label: uy')

    # display reconstruction
    ax = plt.subplot(2, 4, 6)
    c_img = plt.imshow(the_img_pre[:, :, 1])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('prediction: uy')

    # display reconstruction
    ax = plt.subplot(2, 4, 7)
    c_img = plt.imshow(the_img_input[:, :, 1])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: uy')

    # display reconstruction
    ax = plt.subplot(2, 4, 8)
    c_img = plt.imshow(the_img_input[:, :, 3])  # tensor
    plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: Ty')

    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig("prediction" + now_str + ".png", format='png')
    plt.show()
