import argparse
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torchvision.models

from avalanche.training.strategies import Naive, Replay, JointTraining, BaseStrategy
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
import sys
import dataset.digits as digits
from methods.clue import DistPlugin
from metrics import *

# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='digits', choices=['digits'])
    parser.add_argument('--method', type=str, default='CLUE')
    parser.add_argument('--output', type=str, default='res/output.log')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for a single task')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--buffer_size', type=int, default=200, help='training buffer size for replay methods')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'digits':
        model = torchvision.models.resnet18(pretrained=True)
        train_datasets, val_datasets = digits.get_train_val('data/MyDigits/')

    benchmark = create_multi_dataset_generic_benchmark(train_datasets=train_datasets, test_datasets=val_datasets)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = CrossEntropyLoss()
    loggers = [InteractiveLogger()]
    eval_plugin = EvaluationPlugin(accuracy_metrics(epoch=True, experience=True, stream=True),
                                   loss_metrics(epoch=True),
                                   loggers=loggers, benchmark=benchmark)

    if args.method == 'Naive':
        strategy = Naive(model, optimizer, criterion, train_mb_size=args.batch_size, train_epochs=args.epochs,
                         eval_mb_size=args.batch_size, device=device, evaluator=eval_plugin)
    elif args.method == 'Replay':
        strategy = Replay(model, optimizer, criterion, mem_size=args.buffer_size, train_mb_size=args.batch_size,
                          train_epochs=args.epochs, eval_mb_size=args.batch_size, device=device, evaluator=eval_plugin)
    elif args.method == 'joint_training':
        strategy = JointTraining(model, optimizer, criterion, train_mb_size=args.batch_size, train_epochs=args.epochs,
                         eval_mb_size=args.batch_size, device=device, evaluator=eval_plugin)

    elif args.method == 'CLUE':
        strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.batch_size, train_epochs=args.epochs,
                                eval_mb_size=args.batch_size, device=device,
                                plugins=[DistPlugin(mem_size=args.buffer_size, device=device)], evaluator=eval_plugin)

    # begin training
    results = []
    if args.method == 'joint_training':
        print('start training')
        strategy.train(benchmark.train_stream)
        print('training completed')
        results.append(strategy.eval(benchmark.test_stream))
    else:
        i = 0
        for experience in benchmark.train_stream:
            i += 1
            print('start of experience ', experience.current_experience)
            strategy.train(experience)
            print('training completed')
            results.append(strategy.eval(benchmark.test_stream[:i]))

        calculate_acc(results, args.output)
        calculate_forget(results, args.output)



if __name__ == '__main__':
    main()
