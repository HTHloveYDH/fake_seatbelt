import argparse

from build_model import build_auto_encoder_fake_model
from datasets import load_dataset
from utils import get_optimizer
from utils import set_callbacks


def main(args):
    trainset, trainset_length = load_dataset(
        './data/anno_train_1.json', './data/anno_train_2.json', args.batch_size, 100, True, 
        args.channel_num, args.input_width, args.input_height, args.scale, args.offset
    )
    validset, validset_length = load_dataset(
        './data/anno_val_1.json', './data/anno_val_2.json', args.batch_size, 100, True, 
        args.channel_num, args.input_width, args.input_height, args.scale, args.offset
    )
    input_shape = (args.input_width, args.input_height, args.channel_num)
    auto_encoder_fake_model = build_auto_encoder_fake_model(
        input_shape, args.filters, args.latent_dim, args.loss_scale, args.offset, 
        args.scale, args.loss_fn_type, args.channel_dim
    )
    optimizer = get_optimizer(args.optimizer_type)
    callbacks = set_callbacks(validset is not None)
    auto_encoder_fake_model.compile(optimizer=optimizer, loss=None)
    auto_encoder_fake_model.fit(
        trainset,
        batch_size=args.batch_size, epochs=args.epochs, verbose='auto', callbacks=callbacks,
        validation_data=validset, steps_per_epoch=trainset_length // args.batch_size, 
        validation_steps=validset_length // args.batch_size if validset is not None else None
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filters', nargs='+', type=int, default=[32, 64, 128, 256, 512], 
        required=True
    )
    parser.add_argument('--input_width', type=int, default=128, required=True)
    parser.add_argument('--input_height', type=int, default=128, required=True)
    parser.add_argument('--channel_num', type=int, default=3, required=True)
    parser.add_argument('--latent_dim', type=int, default=512, required=True)
    parser.add_argument('--loss_scale', type=float, default=1.0, required=True)
    parser.add_argument('--offset', type=float, default=0.0, required=True)
    parser.add_argument('--scale', type=float, default=255.0, required=True)
    parser.add_argument('--batch_size', type=int, default=32, required=True)
    parser.add_argument('--epochs', type=int, default=100, required=True)
    parser.add_argument('--channel_dim', type=int, default=-1, required=True)
    parser.add_argument('--loss_fn_type', type=str, default='MSE', required=False)
    parser.add_argument('--optimizer_type', type=str, default='SGD', required=False)
    args = parser.parse_args()
    main(args)