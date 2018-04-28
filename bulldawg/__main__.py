import argparse
import prepare_data as prd
import fit_predict as fp

def main():
    #getting command line argument
    parser = argparse.ArgumentParser(description='paths')
    parser.add_argument('--model', type=str, help='cnn or resnet')
    parser.add_argument('--process', type=str, help='train or predict?')
    parser.add_argument('--operation', type=str, help='Download and store data(d_data) or use models(u_models)')
    parser.add_argument('--path',type=str,help='path to all npy array')
    parser.add_argument('--num_top_classes',type=int,help='number of top frequency classes to train model on')

    args = parser.parse_args()

    if args.operation is 'd_data':
        print('Downloading and saving the data')
        prd.prepare_data(args)

    if args.model is 'cnn' or args.model is 'resnet':
        print('Model called is ', args.model)
        fp.run_model(args)