import bulldawg.cnn as cnn
import bulldawg.resnet as resnet
import bulldawg.utils as utils
import numpy as np

def fit(args):
    '''
    Standard fit method, It loads training data according to specific needs of user, 
    trains specified model and saves it at the path

    Args:
        args: arguments provided by user
    Returns:
        history object of the keras model
    '''
    if args.num_top_classes:
        top_classes = utils.load_data(args.path, 'top_class', True)
        train_data, labels = utils.load_data(args.path, 'training', True)
        labels = utils.process_labels(np.unique(labels).shape[0], train_data.shape[0], labels, top_classes)

    else:
        train_data, labels = utils.load_data(args.path, 'training', False)
        labels = utils.process_labels(np.unique(labels).shape[0], train_data.shape[0], labels, labels)
    
    if args.model is 'cnn':
        model = cnn.build_model(train_data[0].shape, np.unique(labels).shape[0])

    else:
        model = resnet.build_model(50, train_data[0].shape, np.unique(labels).shape[0])
    
    hist = model.fit(train_data, labels, batch_size=128, nb_epoch=20, verbose=1)
    model.save(args.path+ 'model.h5')
    return hist

def predict(args):
    '''
    Predict method: it loads model and test data according to user's arguments and returns 
    predictions and test ids
    
    Args:
        args: arguments specified by user

    Returns:
        returns predictions and test image ids
    '''
    test_data, ids = utils.load_data(args.path, 'testing', False)
    no_classes = 0
    
    if args.num_top_classes:
        no_classes = args.num_top_classes

    else:
        no_classes = 14951

    if args.model is 'cnn':
        model = cnn.build_model(test_data[0].shape, no_classes)

    else:
        model = resnet.build_model(50, test_data[0].shape, no_classes)
    
    model.load_weights(args.path+ 'model.h5')
    predictions = model.predict(test_data)

    return predictions, ids

def run_model(args):
    '''
    This method either trains, predicts or does both and creates a csv in a Kaggle tournament 
    specified submission document format.

    Args:
        args: arguments specified by user
    '''
    if args.process:
        if args.process is 'train':
            fit(args.path)
        else:
            predictions, ids = predict(args)
            if args.num_top_classes:
                utils.prepare_submission(predictions, ids, args.path, True)
            else:
                utils.prepare_submission(predictions, ids, args.path, False)
    else:
        fit(args.path)
        predictions, ids = predict(args)
        if args.num_top_classes:
            utils.prepare_submission(predictions, ids, args.path, True)
        else:
            utils.prepare_submission(predictions, ids, args.path, False)