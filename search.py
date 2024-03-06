import argparse
import itertools
import segmentation_models_pytorch as smp
from model.densenet import FCDenseNet67
from train import *
from test import  test_net
from util.History import History
# from util import get_parser


def get_space(path):
    """
	Get the search space as the Cartesian product of all the different hyperparams
	Expects epochs, batch size,  learning rates, and scale (IN THAT ORDER) specified as lists in a txt file, formatted with spaces:

	1 10 100 500 1000
	etc		
	"""

    lsts = {'epochs': [], 'batch_size': [], 'lr': [], 'scale': []}
    with open(path, 'r') as f:
        lines = f.readlines()
        for l, k in zip(lines, lsts.keys()):
            for val in l.split():
                try:
                    lsts[k] += [int(val)]
                except ValueError:
                    lsts[k] += [float(val)]

    return list(itertools.product(lsts['epochs'], lsts['batch_size'], lsts['lr'], lsts['scale']))


def heuristic(history: History):
    """
	Heuristic for search evaluation
	
	Computes the average from the last tenth of the training epochs
	"""
    n = len(history.hist)
    tenth = int(0.1 * n)
    tail_end = history.hist[-tenth:]
    return sum(tail_end) / len(tail_end)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Arguments to perform a hyperparameter search')
    # parser.add_argument('--path', '-p', type=str, default=os.getcwd(),
    #                     help='Path to a text file containing lists of values for each hyperparameter to check')
    # parser.add_argument('--net', '-n', type=str, default='unet++', help='Specify model type [unet | unet++ | densenet]')
    parser = get_parser()
    args = parser.parse_args()

    # args = parser.parse_args()

    space = get_space(args.path)  # hyperparam search space
    net_type = args.net.lower()

    # search in space
    # best_score = None
    # best_combo = None
    for i, search in enumerate(space):
        epochs, batch_size, lr, scale = search

        # model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # if net_type == 'unet++':
        #     net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse')
        # elif net_type == 'densenet':
        #     net = FCDenseNet67(1)
        # else:
        #     net = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse')

        try:
            net = get_net(args, device)

            net.to(device=device)

            dataset = FrameDataset('data/inputs', 'data/labels', mode='train',
                                   transform=torchvision.transforms.Compose([ToTensor()]))

            print(f'Net Type: {net_type}')
            print(f'Batch size: {batch_size}')
            print(f'Learning Rate: {lr}')
            print(f'Image Scale: {scale}')

            print(
                '\n****************************************************\n***                  Train                       ***\n****************************************************\n')
            # TRAIN and get score history
            history = train_net(net=net,
                                dataset=dataset,
                                epochs=epochs,
                                batch_size=batch_size,
                                lr=lr,
                                device=device,
                                img_scale=scale,
                                save_frequency=5,
                                save_cp=True,
                                model_name= f'train_{net_type}_bs_{batch_size}_lr_{lr}_scale_{scale}',
                                save_dir=f'results/train_imgs/train_{net_type}_bs_{batch_size}_lr_{lr}_scale_{scale}'
                                )

            # score = heuristic(history)
            # best_combo = search if best_score is None or score > best_score else best_combo
            # best_score = score if best_score is None or best_score < score else best_score


            # TEST
            print('\n****************************************************\n***                Testing                       ***\n****************************************************\n')

            dataset = FrameDataset('data/inputs', 'data/labels', mode='test',
                                   transform=torchvision.transforms.Compose([ToTensor()]))
            test_net(net=net,
                     net_type=net_type,
                     dataset=dataset,
                     batch_size=batch_size,
                     device=device,
                     img_scale=scale,
                     model_name=f'test_{net_type}_bs_{batch_size}_lr_{lr}_scale_{scale}',
                     save_dir=f'results/test_imgs/train_{net_type}_bs_{batch_size}_lr_{lr}_scale_{scale}',
                     smooth=True)
        except Exception as e:
            print(e)
            print('Hyperparams at error: ', search)

        # with open('summaries/search_results.txt', 'a') as f:
        #     f.write('Score = ' + str(best_score) + '\n\n')
        #     epochs, batch_size, lr, scale = best_combo
        #
        #     f.write(f'Epochs: {epochs}')
        #     f.write(f'Net Type: {net_type}')
        #     f.write(f'Batch Size: {batch_size}')
        #     f.write(f'Learning Rate: {lr}')
        #     f.write(f'Image Scale: {scale}')

        del net
        del dataset
        torch.cuda.empty_cache()
