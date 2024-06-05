import argparse

import cv2
import numpy as np

from build_model import load_auto_encoder_fake_model


def main(args):
    auto_encoder_fake_model = load_auto_encoder_fake_model(
        args.load_autoencoder_1_path, args.load_autoencoder_2_path, args.offset, args.scale
    )
    image = cv2.imread(args.load_image_path)
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (args.input_width, args.input_height), interpolation=cv2.INTER_LINEAR)
    image = image / args.scale + args.offset
    image = auto_encoder_fake_model.fake(np.expand_dims(image, axis=0))
    cv2.imwrite('./1.jpg', image[0].numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_image_path', type=str, default='./auto_encoder_fake_model/0.jpg', required=True)
    parser.add_argument('--load_autoencoder_1_path', type=str, default='./auto_autoencoder_1', required=True)
    parser.add_argument('--load_autoencoder_2_path', type=str, default='./auto_autoencoder_2', required=True)
    parser.add_argument('--input_width', type=int, default=128, required=True)
    parser.add_argument('--input_height', type=int, default=128, required=True)
    parser.add_argument('--offset', type=float, default=0.0, required=True)
    parser.add_argument('--scale', type=float, default=255.0, required=True)
    args = parser.parse_args()
    main(args)