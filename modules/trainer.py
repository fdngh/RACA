import os
import time
import math
import statistics
from abc import abstractmethod

import torch
import pandas as pd
from numpy import inf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # Setup GPU device if available
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        # Setup monitoring for early stopping
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Resume from checkpoint if specified
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        # Initialize best result recorder
        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):

        raise NotImplementedError

    def train(self):

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # Save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # Print logged information to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # Evaluate model performance according to configured metric, save best checkpoint
            best = False
            if self.mnt_mode != 'off':
                try:
                    # Check whether model performance improved or not
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. "
                          "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # Early stopping
                if not_improved_count > self.early_stop:
                    print("Validation performance didn't improve for {} epochs. "
                          "Training stops.".format(self.early_stop))
                    break

            # Save checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        # Print and save best results
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):

        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        # Create directory if it doesn't exist
        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        # Create or update the CSV file
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)

        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):

        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There's no GPU available on this machine, "
                  "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU's configured to use is {}, but only {} are available "
                  "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):

        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):

        # Check if validation results improved
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        # Check if test results improved
        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        """Print the best results for validation and test sets."""
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.args = args

    def calculate_bleu(self, reference, hypothesis):
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference.split()], hypothesis.split(),
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothie)

    def normalize_and_adjust_rewards(self, bleu_sum, t1):
        # Reward adjustment factor
        alpha = 0.4

        # Extract rewards and convert to list of scalar values
        rewards = [r.item() for r in t1['rewards']]

        # Normalize rewards to zero mean and unit variance
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
        normalized_rewards = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]

        # Clip normalized rewards to [-2, 2] range
        normalized_rewards = [max(min(r, 2), -2) for r in normalized_rewards]

        # Calculate loss impact based on the BLEU score
        reward_range = max(normalized_rewards) - min(normalized_rewards)
        loss_impact = math.log(1 + bleu_sum) / (reward_range + 1e-8)
        loss_impact = min(loss_impact, 2)

        # Adjust rewards with the loss impact
        adjusted_rewards = [r + alpha * loss_impact for r in normalized_rewards]

        # Update rewards in the dictionary
        t1['rewards'] = [torch.tensor([r]) for r in adjusted_rewards]

        return t1

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()

        # Training loop
        for batch_idx, (images_id, images, reports_ids, reports_masks, sentence_name, sentence_label) in enumerate(
                self.train_dataloader):
            # Move data to device
            images = images.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)

            # Forward pass
            out = self.model(self.args, images, sentence_name, sentence_label, reports_ids, self.device, mode='train')

            # Process model outputs
            output = out[0]  # Main prediction output
            t1 = out[1]  # Reinforcement learning information
            gc_loss = out[2]  # Global coherence loss
            lc_loss = out[3]  # Local coherence loss

            # Calculate main loss and total loss
            loss = self.criterion(output, reports_ids, reports_masks)
            total_loss = loss + gc_loss + lc_loss
            train_loss += loss.item()

            # Generate reports for BLEU calculation
            generated_reports = self.model.tokenizer.decode_batch(output.argmax(dim=-1).cpu().numpy())
            ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

            # Calculate BLEU scores for the batch
            bleu_scores4 = [self.calculate_bleu(gt, rep) for gt, rep in zip(ground_truths, generated_reports)]

            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            # Update reinforcement learning agent
            t1 = self.normalize_and_adjust_rewards(sum(bleu_scores4), t1)
            self.model.encoder_decoder.model.locate_agent.ss1(t1)

        # Calculate average training loss
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # Validation phase
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, sentence_name, sentence_label) in enumerate(
                    self.val_dataloader):
                # Move data to device
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                # Sample from the model
                output = self.model(self.args, images, sentence_name, sentence_label, reports_ids, self.device,
                                    mode='sample')

                # Decode generated and ground truth reports
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                # Collect results
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            # Compute validation metrics
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        # Test phase
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, sentence_name, sentence_label) in enumerate(
                    self.test_dataloader):
                # Move data to device
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                # Sample from the model
                output = self.model(self.args, images, sentence_name, sentence_label, reports_ids, self.device,
                                    mode='sample')

                # Decode generated and ground truth reports
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                # Collect results
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            # Compute test metrics
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        # Step the learning rate scheduler
        self.lr_scheduler.step()

        return log