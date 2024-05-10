from resnet import ResNet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_index', type=int, default=1)
parser.add_argument('--save_dir', default="models")
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--teacher_logits_path", default="saves/logits/logits_resnet_0.npz")
parser.add_argument("--test_epoch", type=int, default=2)
args = vars(parser.parse_args())



net = ResNet()

# net.train(save_dir=args['save_dir'],
#           read_index=args['train_index'],
#           num_epochs=args['num_epochs'],
#           batch_size=args['batch_size'],
#           learning_rate=args['learning_rate'],
#           verbose=args['verbose'],
#           test_epoch=args["test_epoch"]
#           )


net.train_student(save_dir=args['save_dir'],
          read_index=args['train_index'],
          num_epochs=args['num_epochs'],
          batch_size=args['batch_size'],
          learning_rate=args['learning_rate'],
          verbose=args['verbose'],
          test_epoch=args["test_epoch"],
          teacher_logits_path=args['teacher_logits_path']
          )