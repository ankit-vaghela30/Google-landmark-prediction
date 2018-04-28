import pandas as pd
import numpy as np
import bulldawg.preprocess as preprocess
from skimage.io import imsave, imread
from keras.utils import np_utils

def read_csv(path, data_type):
    '''
    This method reads a csv file from the path

    Args:
        path: path to the csv file

        data_type: train or test 

    Returns:
        returns a dataframe representing csv content
    '''
    if data_type is 'train':
        return pd.read_csv(path+'train.csv')
    else:
        return pd.read_csv(path+'test.csv')

def filter_top_class(top_classes, train_data, path):
    '''
    This method is used when you want to get images from top classes where top classes. It means
    that the classes are sorted in descending order of images they contain and then according to
    no_classes paramenter, top data is returned. Additionally, it will save top class indexes and 
    filtered data to disk
    
    Args:
        no_classes: Top number of classes 

        train_data: numpy array having entire training dataset

        path: path where you want to save top class indexes

    Returns:
        Returns numpy array with filtered data

    '''
    # Descending ordered data with given top number of classes
    top_df = pd.DataFrame(train_data.landmark_id.value_counts().head(top_classes))
    top_classes = np.array(top_df.index)
    np.save(path+ 'top_classes_original_idxs.npy', top_classes)

    filtered_imgs = []
    filtered_labels = []
    # Filtering the training data
    for n in range(0,train_data.shape[0]):
        if train_data[n] in top_classes:
            filtered_imgs.append(train_data[n])
            filtered_labels.append(train_data[n])
    
    # converting to numpy array
    filtered_imgs = np.array(filtered_imgs)
    filtered_labels = np.array(filtered_labels)

    # saving filtered data to disk
    np.save(filtered_imgs, path+ 'filtered_train_imgs.npy')
    np.save(filtered_labels, path+ 'filtered_labels.npy')

    return filtered_imgs, filtered_labels

def download_data(path, df_csv, data_type):
    '''
    This method downloads data and stores it as .npy file on disk
    Args:
        path: path to which you want to download .npy file

        df_csv: dataframe we get from the csv file

        data_type: value is 'train' or 'test' suggesting the data to be downloaded 
              is training data or test data
    
    Returns: 
        number of images which were not downloaded due to corrupt url, images and labels/ids
        depending on data_type
    '''
    img_data = []
    labels = []
    failed_images = 0

    for x in range(0, df_csv.shape[0]):
        # read image url
        im = df_csv.iloc[x]['url']

        try:
            image = imread(im)
            img = preprocess.convert_bgr2gray(image)
            img_resize = preprocess.resize_img(img,128)
        
            img_data.append(img_resize)

            if data_type is 'test':
                labels.append(df_csv.iloc[x]['id'])
            else:
                labels.append(df_csv.iloc[x]['landmark_id'])

        except:
            # some images fail to download
            print('Warning: Could not download ',failed_images)
            failed_images = failed_images + 1
    
        if(x%10000 == 0):
            print('finished with images ',x)
    
    img_data = np.array(img_data)
    labels = np.array(labels)

    # some more preprocessing    
    preprocess.modify_imgs(img_data)

    if data_type is 'test':
        np.save('img_data_test.npy',img_data)
        np.save(path+'ids.npy', labels)

    else:
        np.save('img_data_train.npy', img_data)
        np.save(path+'labels.npy', labels)
    
    return failed_images, img_data, labels

def load_data(path, data_type, filtered):
    '''
    Loads training, testing or top class indexes data and returns

    Args:
        path: path where .npy file/s reside

        data_type: data type from 'training', 'testing' and 'top_class'

        filtered: boolean true for filtered data and false for entire dataset
    
    Returns:
        two or one numpy array depending on data type
    '''
    if data_type is 'training':
        if filtered:
            img_data = np.load(path+ 'filtered_train_imgs.npy')
            labels = np.load(path+ 'filtered_labels.npy')    

        else:
            img_data = np.load(path+ 'img_data_train.npy')
            labels = np.load(path+ 'labels.npy')

        return img_data, labels

    elif data_type is 'testing':
        img_data = np.load(path+ 'img_data_test.npy')
        ids = np.load(path+ 'ids.npy')
        return img_data, ids
    
    else:
        top_classes = np.load(path+'top_classes_original_idxs.npy')
        return top_classes

def process_labels(num_classes, num_images, labels, top_classes_np):
    '''
    This method converts labels to on hot encoding format after doing some preprocessing
    Args:
        num_classes: number of classes

        num_images: total number of images

        labels: numpy array consisting of labels

        top_classes_np: numpy array consisting of top frequency classes (labels)

    Return:
        returns processed labels
    '''
    labels_ = np.ones((num_images,), dtype= 'int64')

    for n in range(0, labels.shape[0]):
        i, = np.where(top_classes_np == labels[n])
        labels_[n] = i[0]
    
    Y = np_utils.to_categorical(labels_, num_classes)
    return Y

def prepare_submission(predictions, ids, path, filtered):
    '''
    This method converts predictions into Kaggle tournament's submit data format by 
    creating a csv at the path.

    Args:
        predictions: predictions of the model

        ids: ids of the test data 

        path: path at which submission file gets saved

        filtered: boolean indicating if dataset was filtered i.e. top frequency classes
                  were trained/tested
    '''
    if filtered:
        top_classes = load_data(path, 'top_class', True)
    else:
        x,top_classes = load_data(path, 'training', False)

    submit_score = []
    submit_score.append('id,landmarks')

    for p in range(0,predictions.shape[0]):
        submit_score.append(str(ids[p]) + ',' + str(top_classes[np.where(predictions[p] == np.amax(predictions[p]))[0][0]]) + ' '+"{0:.2f}".format(np.amax(predictions[p])))
    
    np.savetxt('submit_score.csv', submit_score, delimiter=',', fmt='%s')