
from data.utils import get_data_loader
import argparse
from runner import ModelRunner


def main(args) -> None:
    
    model = ModelRunner(args)

    print('Running mode: {}'.format(args['mode']))
    if args['mode'] == 'train':
        model.train()
    elif args['mode'] == 'eval':
        model.eval()
    else:
        raise NotImplementedError('Running mode {} not recognised.'.format(args['mode']))

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='MTL GAN')

    parser.add_argument("--num_workers", type=int, help="Number of Workers", default=0)
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training", default=32)
    parser.add_argument("--dataset", type=str, help="Dataset name", default="Cifar100")
    parser.add_argument("--z_dim", type=int, help="Dimension of latent dimension", default=100)
    parser.add_argument("--nn_type", type=str, help="Neural network name", default="DCGAN-SN")
    #parser.add_argument("--nn_type", type=str, help="Neural network name", default="Test")
    parser.add_argument("--seed", type=int, help="Seed of randomness", default=4)
    parser.add_argument("--device", type=int,  help="Device for running", default=1)
    parser.add_argument("--latent_noise", type=str, help="Distribution of random noise", default="uniform")
    parser.add_argument("--mode", type=str, help="Mode to run", default="train")
    parser.add_argument("--max_train_epochs", type=int, help="Maximum number of epochs to train model", default=10)
    parser.add_argument("--num_classes", type=int, help="Number of classes", default=100)
    parser.add_argument("--tasks", type=list, help="List of tasks", default=['gan','fine','coarse'])
    parser.add_argument("--gan_loss", type=str, help="Loss function for GAN", default='classic')
    parser.add_argument("--task_loss", type=str, help="Loss function for task", default='classic')
    parser.add_argument("--loss_weights", type=dict, help="List of weights for the losses", default={'gan':0.3,'fine':0.3,'coarse':0.3})
    parser.add_argument("--enable_tensorboard", type=bool, help="Flag to enable tensorboard", default=True)


    parser.add_argument('--optimizer', default='Adam', type= str, help='Optimizer for model')
    parser.add_argument('--lr_heads', default=0.0002, type=float, help='learning rate for the heads module')
    parser.add_argument('--lr_generator', default=0.0002, type=float, help='learning rate for the generator')
    parser.add_argument('--sgd_momentum', default=0., type=float, help='momentum parameter for SGD [0.]')
    parser.add_argument('--beta_1', default=0.5, type=float, help='first parameter of Adam optimizer [.5]')
    parser.add_argument('--beta_2', default=0.9, type=float, help='second parameter of Adam optimizer [.9]')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay [0.]')


    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())