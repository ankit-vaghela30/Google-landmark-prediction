import bulldawg.utils as utils
import numpy as np

def prepare_data(args):
    '''
    Download all data and prepare .npy files 

    Args:
        args: args passed by user
    '''

    # Downloading training data
    train_df = utils.read_csv(args.path, 'train')
    failed_imgs, train_data, labels = utils.download_data(args.path, train_df, 'train')
    print('urls corrupted for '+failed_imgs+ ' images')

    # filtering and saving the data based on number of top frequency classes
    train_data = np.load()
    if args.num_top_classes:
        filtered_imgs, filtered_labels = utils.filter_top_class(args.num_top_classes, train_data, args.path)
    
    # downloading test data
    test_df = utils.read_csv(args.path, 'test')
    failed_imgs, test_data, ids = utils.download_data(args.path, train_df, 'test')
    



    