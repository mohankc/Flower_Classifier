import argparse
import Classifier as C

def main():
    in_arg = get_input_args()

    print(in_arg)
    output_size = 102
    dropout_p = 0.5
    learn_rate = in_arg.learning_rate
    arch = in_arg.arch
    data_dir = in_arg.dir
    saved_dir = in_arg.save
    epochs = in_arg.epochs
    if in_arg.hidden_units is not None:
        hidden_layers = tuple(int(x) for x in in_arg.hidden_units.split(','))
    else:
        hidden_layers = C.default_hidden_units[arch]
    if in_arg.gpu == 'Y' or in_arg.gpu == 'y':
        enable_gpu = True
    else:
        enable_gpu = False
    print(enable_gpu)
    model, optimizer = C.network(output_size, hidden_layers, dropout_p, learn_rate, arch ,epochs, data_dir,False, enable_gpu, saved_dir)

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type = str, default = 'flowers/', help = 'Path to the dataset folder')
    parser.add_argument('--arch', type = str, default = 'densenet', help = 'CNN model architecture to use for image classification eg vgg, alexnet, resnet, densenet')
    parser.add_argument('--save', type = str, default = None, help = 'Path to save the checkpoint')
    parser.add_argument('--learning_rate', type = float, default = '0.001', help = 'Learning rate')
    parser.add_argument('--hidden_units', type = str,default = None, help = 'list of comma seperated values for each layers')
    parser.add_argument('--epochs', type = int, default = '10', help = 'Numebr of epochs')
    parser.add_argument('--gpu', type = str, default = 'Y', help = 'Whether to use GPU or CPU Y/y or N/n')
    return parser.parse_args()

if __name__ == "__main__":
    main()
