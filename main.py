from utils.utils import get_model


def main(params):
    #code here
    model = get_model(params)
    return True 



if __name__ == "__main__":
    params = {}
    main(params)
    print('Finished!')