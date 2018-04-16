from train_helpers import run_train, load_hparams

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hparams', type=str, default="./hparams.json")
    parser.add_argument('--logdir', type=str, required=True)
    
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)'
    )
    
    args = parser.parse_args()

    hparams = load_hparams(args.hparams)
    run_train(hparams, args)
