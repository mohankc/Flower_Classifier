import argparse
import Classifier as C

def main():
    in_arg = get_inference_input_args()
    input_path = in_arg.input_path
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    if in_arg.gpu == 'Y' or in_arg.gpu == 'y':
        enable_gpu = True
    else:
        enable_gpu = False
    if input_path is not None and checkpoint is not None:
        C.inference(input_path, checkpoint, top_k, category_names, enable_gpu)
    else:
        print("*****  Please provide [--input_path , --checkpoint] argument values *****")

def get_inference_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, default = None, help = 'Path to input image')
    parser.add_argument('--checkpoint', type = str, default = None, help = 'Checkpoint name')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top k probabilities')
    parser.add_argument('--category_names', type = str, default = None, help = 'Class to categories Json file')
    parser.add_argument('--gpu', type = str, default = 'Y', help = 'Whether to use GPU or CPU Y/y or N/n')
    return parser.parse_args()

if __name__ == "__main__":
    main()
