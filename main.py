
from data.utils import get_data_loader
import argparse
from runner import ModelRunner


def main(args):
    
    model = ModelRunner(args)

    return True 

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='MTL GAN')

    parser.add_argument("--num_workers", type=int, help="Number of Workers", default=2)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=32)
    parser.add_argument("--dataset", type=str, help="Dataset name", default="Cifar100")
    parser.add_argument("--z_dim", type=int, help="Dimension of latent dimension", default=100)
    parser.add_argument("--nn_type", type=str, help="Neural network name", default="DCGAN-SN")
    parser.add_argument("--seed", type=int, help="Seed of randomness", default=4)
    parser.add_argument("--device", type=int, help="Device for running", default=1)


    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())