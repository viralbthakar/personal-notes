"""
Image Data Pipeline Script

This script defines functions to create a data pipeline for loading and preprocessing images using TensorFlow. It includes functions to preprocess individual images, build a data pipeline, and print the shapes of images in the dataset.

Usage:
    python script.py --path <image_directory> --buffer_size <buffer_size> --batch_size <batch_size>

Options:
    --path (str): Path to image files. Default is "./data/images/*.jpg".
    --buffer_size (int): Buffer size for shuffling. Default is 1000.
    --batch_size (int): Batch size for batching images. Default is 1.
"""
import argparse
import tensorflow as tf


def preprocess_image(image_path):
    """
    Preprocesses an image from the given image file path.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - image (tf.Tensor): Preprocessed image as a tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def build_datapipeline(path, buffer_size=100, batch_size=1):
    """
    Builds a data pipeline for loading and preprocessing images.

    Parameters:
    - path (str): Glob pattern specifying the path to image files.
    - buffer_size (int): Number of elements to buffer in the shuffle queue.
    - batch_size (int): Number of images to include in each batch.

    Returns:
    - image_dataset (tf.data.Dataset): A dataset containing preprocessed images.
    """
    image_dataset = tf.data.Dataset.list_files(path)
    image_dataset = image_dataset.map(preprocess_image)
    image_dataset = image_dataset.shuffle(buffer_size=buffer_size)
    image_dataset = image_dataset.batch(batch_size=batch_size)
    image_dataset = image_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
    return image_dataset


def main(args):
    """
    Main function for creating and using an image data pipeline.

    Parameters:
    - args (argparse.Namespace): Command-line arguments.

    Returns:
    - None
    """
    image_dataset = build_datapipeline(
        path=args.path,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )

    for sample_batch in image_dataset.take(5):
        print(sample_batch.numpy().shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image data pipeline using TensorFlow"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/images/*.jpg",
        help="Path to image files",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000,
        help="Buffer size for shuffling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for batching images",
    )

    args = parser.parse_args()
    main(args)
