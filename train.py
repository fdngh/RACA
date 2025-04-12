import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import raDataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainers import Trainer
from modules.loss import compute_loss
from models.raca import RACA


def parse_args():
    parser = argparse.ArgumentParser()

    # 保留原有的参数设置
    parser.add_argument('--image_dir', type=str, default='./data/mimic_cxr/images',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='./data/mimic_cxr/output_replaced.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=20, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=0.7, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # 保留原有参数及其默认值
    parser.add_argument('--d_att', type=int, default=32, help='the dimension of Transformer.')
    parser.add_argument('--d_dubi', type=int, default=256, help='the dimension of Transformer.')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--tnum_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--dropout_t', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    #
    parser.add_argument('--lr_ve', type=float, default=4e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=6e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--lt_ed', type=float, default=1e-5, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=8e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')

    parser.add_argument('--seed', type=int, default=42, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--a', type=float, default=0.7, help='.')

    ########RL
    parser.add_argument('--actor_lr', type=float, default=1e-6, help='.')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='.')
    parser.add_argument('--gam', type=float, default=0.98, help='.')
    parser.add_argument('--lmbda', type=float, default=0.98, help='.')
    parser.add_argument('--eps', type=float, default=0.2, help='.')
    parser.add_argument('--b1', type=float, default=0.9, help='.')
    parser.add_argument('--b2', type=float, default=1, help='.')
    parser.add_argument('--b3', type=float, default=1, help='.')
    parser.add_argument('--tt', type=float, default=0.1, help='.')

    parser.add_argument('--step_size', type=int, default=5, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.9, help='the gamma of the learning rate scheduler.')

    parser.add_argument('--step_size_ed', type=int, default=25, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma_ed', type=float, default=1, help='the gamma of the learning rate scheduler.')

    parser.add_argument('--step_size_ted', type=int, default=25, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma_ted', type=float, default=1, help='the gamma of the learning rate scheduler.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('seed', args.seed)
    print('step_size', args.step_size)
    print('dropout_t', args.dropout_t)
    print('tt', args.tt)
    print('a', args.a)
    print('lr_ve', args.lr_ve)
    print('lr_ed', args.lr_ed)
    print('lt_ed', args.lt_ed)
    print('weight_decay', args.weight_decay)

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # 创建tokenizer
    tokenizer = Tokenizer(args)

    # 创建数据加载器
    train_dataloader = raDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = raDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = raDataLoader(args, tokenizer, split='test', shuffle=False)

    # 构建模型
    model = RACA(args, tokenizer)

    # 定义损失函数和评估指标
    criterion = compute_loss
    metrics = compute_scores

    # 构建优化器和学习率调度器
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # 创建trainer并开始训练
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    best_test_metrics = trainer.train()

    # 打印最终的测试指标
    print("Training completed. Best test metrics:")
    for metric_name, metric_value in best_test_metrics.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == '__main__':
    main()