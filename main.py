from resnet import ResNet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_index', required=True, type=int)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--num_epochs', required=True, type=int)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--learning_rate", required=True, type=float)
parser.add_argument("--verbose", required=True, type=bool)
parser.add_argument("--teacher_logits_path", required=False, default=None)
parser.add_argument("--test_each_epoch", required=False, default=False, type=bool)
args = vars(parser.parse_args())



net = ResNet()
net.train(save_dir=args['save_dir'],
          read_index=args['train_index'],
          num_epochs=args['num_epochs'],
          batch_size=args['batch_size'],
          learning_rate=args['learning_rate'],
          verbose=args['verbose'],
          test_each_epoch=args["test_each_epoch"],
          teacher_logits_path=args['teacher_logits_path'])