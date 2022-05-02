import numpy as np
import matplotlib.pyplot as plt


def estimate_2D_image_relative_error(img_test, img_pred, nn=1, plot_flag=False):
    """
    Args:
        arg1:   A positional argument.
        arg2:   Another positional argument.

    Kwargs:
        kwarg:  A keyword argument.

    Returns:
        A string holding the result.
    """
    print("in error estimation")
    if np.shape(img_test) != np.shape(img_pred):
        raise ValueError('img_test and img_pred should have the same size! Currently, you have', np.shape(img_test),
                         ' and ', np.shape(img_pred))

    if len(np.shape(img_test)) == 5:
        raise ValueError('5 dimensional data are not supported yet!')

    for i in range(0, nn):
        the_img_pred = img_pred[i]
        the_img_test = img_test[i]
        plt.clf()

        if (True):
            the_img_pred = np.ma.masked_where(the_img_test < -0.9, the_img_pred)
            the_img_test = np.ma.masked_where(the_img_test < -0.9, the_img_test)
            the_img_mark = 1.0e-10 * np.ones(np.shape(the_img_test))

        channel_number = np.shape(the_img_pred)[-1]

        # display original
        ax = plt.subplot(channel_number, 3, 1)
        c_img = plt.imshow(the_img_test[:, :, 0])    # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('True: u(1)')

        # display reconstruction
        ax = plt.subplot(channel_number, 3, 2)
        c_img = plt.imshow(the_img_pred[:, :, 0])    # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('prediction: u(1)')

        # display error
        ax = plt.subplot(channel_number, 3, 3)
        c_img = plt.imshow((the_img_test[:, :, 0] - the_img_pred[:, :, 0]) /
                           (the_img_test[:, :, 0] + the_img_mark[:, :, 0]))    # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('error: u(1)')

        if (channel_number == 2):
            # display original
            ax = plt.subplot(channel_number, 3, 4)
            c_img = plt.imshow(the_img_test[:, :, 1])    # tensor
            plt.gray()
            plt.colorbar(c_img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('True: u(2)')

            # display reconstruction
            ax = plt.subplot(channel_number, 3, 5)
            c_img = plt.imshow(the_img_pred[:, :, 1])    # tensor
            plt.gray()
            plt.colorbar(c_img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('prediction: u(2)')

            # display error
            ax = plt.subplot(channel_number, 3, 6)
            c_img = plt.imshow((the_img_test[:, :, 1] - the_img_pred[:, :, 1]) /
                               (the_img_test[:, :, 1] + the_img_mark[:, :, 1]))    # tensor
            plt.gray()
            plt.colorbar(c_img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('error: u(2)')

        if plot_flag:
            plt.show()
