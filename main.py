from utils.utils import get_model
from data.utils import get_data_loader


def main(params):
    #code here
    train_loader, test_loader, valid_loader = get_data_loader(params,params['num_workers'])
    model = get_model(params)
    return True 



if __name__ == "__main__":
    params = {'num_workers': 2, 'batch_size':32,'dataset':'Cifar100','z_dim':100,'nn_type':'DCGAN-SN'}
    main(params)
    print('Finished!')