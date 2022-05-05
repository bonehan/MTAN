import  argparse

"""

    description and super parameter

"""

args = argparse.ArgumentParser()
args.add_argument('--Dataset', default='ADNI') # Preprocessed Data from ADNI
args.add_argument('--TrainSet', default= 0)  # 0: ADNI1, 1: ADNI2
args.add_argument('--Filters', default= [16, 16, 32, 32, 64, 64, 128, 128])
args.add_argument('--Batch_size', default= 6)
args.add_argument('--Latent', default= 64)
args.add_argument('--dropout', default= 0.3)
args.add_argument('--learning_rate', default= 0.001)
args.add_argument('--Epoches', default= 100)
args.add_argument('--num_classes', default= 2)
args.add_argument('--mode', default= 'mean')
args = args.parse_args()
