import torch
import torch.nn as nn
import numpy as np
from modules.visual_extractor import VisualExtractor
from modules.visual import EncoderDecoder


class RACA(nn.Module):

    def __init__(self, args, tokenizer):
        super(RACA, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        # Select forward function based on dataset
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, args, images, sentence_label, targets, device, images_id, mode='train'):

        # Extract visual features from paired images
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # Concatenate features from paired images
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(args, fc_feats, att_feats, targets, sentence_label,
                                          device, images_id, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'analyze':
            output = self.encoder_decoder._analyze(args, fc_feats, att_feats, targets,
                                                   sentence_label, device, images_id)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'train', 'sample', or 'analyze'.")

        return output

    def forward_mimic_cxr(self, args, images, sentence_label, targets, device, images_id, mode='train'):
        # Extract visual features from single images
        att_feats, fc_feats = self.visual_extractor(images)

        if mode == 'train':
            # Note: Different parameter list compared to IU X-ray implementation
            output = self.encoder_decoder(args, fc_feats, att_feats, targets, sentence_label,
                                          device, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'analyze':
            output = self.encoder_decoder._analyze(args, fc_feats, att_feats, targets,
                                                   sentence_label, device, images_id)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose from 'train', 'sample', or 'analyze'.")

        return output