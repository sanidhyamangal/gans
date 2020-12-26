"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import tensorflow as tf  # for deep learning based models
import matplotlib.pyplot as plt  # for plotting the figs


def generate_and_save_images(model: tf.keras.Model,
                             test_input: tf.Tensor,
                             image_name: str = "generated.png",
                             multi_channel: bool = False,
                             show_image: bool = True) -> None:
    """
    A helper function for generating and saving images during training ops

    :param model: model which needs to be used for generation of images
    :param image_name: name of an image to save as png
    :param test_input: seed value which needs to be used for image generation
    :param multi_channel: multi_channel value for generation and saving of images
    :return: None
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if not multi_channel:
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        else:
            plt.imshow(((predictions[i] * 127.5) + 127.5) / 255.0)
        plt.axis('off')

    plt.savefig(image_name)

    # show image only if flagged to true
    if show_image:
        plt.show()
