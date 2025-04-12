import logging
import os
from abc import abstractmethod

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class BaseTester(object):


    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        # Configure logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Setup GPU device if available
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available on this machine."
                .format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, sentence_label) in tqdm(
                    enumerate(self.test_dataloader)):
                # Move data to device
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                # Generate captions with sampling
                output = self.model(self.args, images, sentence_label, reports_ids, self.device, images_id,
                                    mode='sample')

                # Decode generated and ground truth captions
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                # Collect results
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            # Compute metrics
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            # Save results to CSV files
            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            test_res.to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
            test_gts.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

        return log

    def plot(self):
        # This method is left to be implemented based on specific visualization needs
        pass

    def analyze_alignment_tsne(self):
        self.logger.info('Start to analyze image-text encoder alignment using t-SNE.')
        os.makedirs(os.path.join(self.save_dir, "tsne"), exist_ok=True)

        self.model.eval()
        paired_features = []  # Store paired image-text features

        def get_paired_features(model, images, sentence_label, reports_ids, device):
            img_feats = []
            text_feats = []

            # Register hook for image encoder
            def img_encoder_hook(module, input, output):
                img_feats.append(output.detach().clone())

            # Register hook for text encoder
            def text_encoder_hook(module, input, output):
                text_feats.append(output.detach().clone())

            # Register hooks
            img_handle = model.encoder_decoder.model.encoder.layers[-1].register_forward_hook(img_encoder_hook)
            text_handles = []
            text_handles.append(
                model.encoder_decoder.model.tencoder.encoder.layers[-1].register_forward_hook(text_encoder_hook))

            try:
                # Ensure inputs are on the correct device
                images = images.to(device)
                reports_ids = reports_ids.to(device)

                # Perform forward pass in analysis mode
                with torch.no_grad():
                    _ = model(model.args, images, sentence_label, reports_ids, device, images_id, mode='analyze')

            except Exception as e:
                self.logger.error(f"Forward pass error: {str(e)}")
                raise e
            finally:
                # Remove hooks
                img_handle.remove()
                for handle in text_handles:
                    handle.remove()

            if img_feats and text_feats:
                # Process features
                img_feat = img_feats[0].mean(dim=1)  # [batch_size, dim]
                text_feat = text_feats[0][:, 0, :]  # [batch_size, dim]

                # Calculate cosine similarity
                similarity = torch.cosine_similarity(img_feat, text_feat, dim=1)

                return img_feat, text_feat, similarity

            return None, None, None

        # Extract features from all batches
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks, sentence_label) in tqdm(
                    enumerate(self.test_dataloader)):
                img_feat, text_feat, similarity = get_paired_features(
                    self.model, images, sentence_label, reports_ids, self.device)

                if img_feat is not None:
                    # Store paired features and similarity
                    paired_features.append({
                        'img_feat': img_feat.cpu().numpy(),
                        'text_feat': text_feat.cpu().numpy(),
                        'similarity': similarity.cpu().numpy()
                    })

        if not paired_features:
            self.logger.error("No features were extracted. Please check the model structure and hooks.")
            return

        # Concatenate features from all batches
        img_features = np.concatenate([p['img_feat'] for p in paired_features])
        text_features = np.concatenate([p['text_feat'] for p in paired_features])
        similarities = np.concatenate([p['similarity'] for p in paired_features])

        # Apply t-SNE to combined feature space
        combined_features = np.concatenate([img_features, text_features], axis=0)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_features) - 1))
        features_2d = tsne.fit_transform(combined_features)

        # Split t-SNE results for image and text
        n = len(img_features)
        img_tsne = features_2d[:n]
        text_tsne = features_2d[n:]

        # Create visualization plot
        plt.figure(figsize=(4, 4))

        # Plot image and text points with similarity coloring
        scatter_img = plt.scatter(img_tsne[:, 0], img_tsne[:, 1],
                                  c=similarities, cmap='viridis',
                                  label='Image Features', marker='o', s=3, alpha=0.8)
        scatter_txt = plt.scatter(text_tsne[:, 0], text_tsne[:, 1],
                                  c=similarities, cmap='viridis',
                                  label='Text Features', marker='^', s=3, alpha=0.8)

        # Add colorbar for similarity scale
        cbar = plt.colorbar(scatter_img, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Similarity Score', fontsize=12)

        # Calculate and show statistics
        similarity_mean = np.mean(similarities)
        similarity_std = np.std(similarities)
        title = f'Mean={similarity_mean:.3f}, Std={similarity_std:.3f}'

        plt.title(title, fontsize=12)
        plt.legend(loc='upper right', fontsize=9)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()

        # Save visualization
        plt.savefig(os.path.join(self.save_dir, "tsne", "alignment_tsne.pdf"),
                    dpi=600, bbox_inches='tight')
        plt.close()

        # Calculate and save alignment statistics
        alignment_stats = {
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'std_similarity': np.std(similarities),
            'high_align_pairs': np.sum(similarities > 0.8),  # Number of highly aligned pairs
            'low_align_pairs': np.sum(similarities < 0.2)  # Number of poorly aligned pairs
        }

        with open(os.path.join(self.save_dir, "tsne", "alignment_stats.txt"), 'w') as f:
            for key, value in alignment_stats.items():
                f.write(f"{key}: {value}\n")

        self.logger.info(
            f'Encoder alignment analysis completed. Results saved in {os.path.join(self.save_dir, "tsne")}')