import logging
import numpy as np
import pandas as pd
import Vision_Model
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MedCLIP(nn.Module):
    def __init__(self, eval=True, freeze_transformer=False, freeze_CNN=False):
        super().__init__()
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.cnn = Vision_Model.get_biovil_resnet()

        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(url, trust_remote_code=True)
        self.freeze_transformer = freeze_transformer
        self.freeze_CNN = freeze_CNN
        self.freeze_projector = False
        self.train(not eval)

        self.cls_projection_head = self.transformer.cls_projection_head
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sig = nn.Sigmoid()

        for param in self.cnn.parameters():
            param.requires_grad=True
        for param in self.transformer.parameters():
            param.requires_grad = True
        modules = [self.transformer.bert.embeddings, *self.transformer.bert.encoder.layer[:8]]
        for module in modules: # First 8 layers won't be updated
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode=mode)
        self.transformer.train(mode)
        self.cnn.train(mode)
        if self.freeze_transformer:
            self.transformer.train(False)
        if self.freeze_CNN:
            self.cnn.encoder.train(False)
        if self.freeze_projector:
            self.cnn.projector.train(False)

    def get_im_embeddings(self, images, only_patches = False, only_ims = False):
        if not isinstance(images, list):
            images = [images]

        images = [im.to(device) for im in images]
        all_patches, all_im_embs = [], []
        for im in images:
            output = self.cnn(im)
            patch_embs = output.projected_patch_embeddings  # N E P1 P2
            all_patches.append(patch_embs.to(device))
            image_emb = output.projected_global_embedding  # N E
            all_im_embs.append(image_emb.to(device))

        if only_patches:
            return all_patches
        elif only_ims:
            return all_im_embs
        else:
            return all_patches, all_im_embs

    def get_text_embeddings(self, text, only_words = False, only_texts = False):
        token_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                        add_special_tokens=True, truncation=True,
                                                        padding='longest', max_length=256,
                                                        return_tensors='pt').to(device)

        text_output = self.transformer(token_output.input_ids, token_output.attention_mask)
        text_emb = self.transformer.get_projected_text_embeddings(input_ids=token_output.input_ids,
                                                                  attention_mask=token_output.attention_mask).to(device)
        word_embs_projected = [self.cls_projection_head(text_output.last_hidden_state[:, i, :])[:, None, :] for i in
                               np.arange(text_output.last_hidden_state.shape[1])]
        word_embs = torch.cat(word_embs_projected, dim=1).to(device)  # N T E

        if only_words:
            return word_embs
        elif only_texts:
            return text_emb
        else:
            return word_embs, text_emb


    def get_cross_weights(self, all_patches, word_embs):
        cross_weights = []
        N, T, E = word_embs.shape
        word_embs = word_embs / word_embs.norm(dim=2, keepdim=True)
        stack_patches = torch.cat(all_patches, dim=0).to(device)
        stack_patches = stack_patches.reshape(stack_patches.shape[0], E, -1)
        stack_patches = stack_patches / stack_patches.norm(dim=1, keepdim=True)
        cross_weights_text = torch.bmm(word_embs.repeat(len(all_patches), 1, 1), stack_patches)  # NTP
        for i, im_emb in enumerate(all_patches):
            cross_weights.append(cross_weights_text[(i * N):(i * N + N), :, :]) # a cross weight for each patch.
        return cross_weights

    def get_im_text_matrix(self, all_im_embs, text_emb):
        im_logits = []
        for i, im_emb in enumerate(all_im_embs):
            im_logits.append(self.similarity_matrix(im_emb, text_emb))
        return im_logits

    def get_im_im_matrix(self, all_im_embs):
        aug_logits = None
        if len(all_im_embs) > 1:
            aug_logits = []
            for i in np.arange(len(all_im_embs)):
                for j in np.arange(len(all_im_embs)):
                    if i <= j:
                        continue
                    imsims = self.similarity_matrix(all_im_embs[i], all_im_embs[j])
                    aug_logits.append(imsims)
        return aug_logits

    def similarity_matrix(self, emb1, emb2): #N E, N E
        image_features = emb1 / emb1.norm(dim=-1, keepdim=True)  # N E
        text_features = emb2 / emb2.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def forward(self, images, text, graph_data=None):
        if not isinstance(images, list):
            images = [images]

        images = [im[None, :] if im.dim() == 3 else im for im in images]

        word_embs, text_emb = self.get_text_embeddings(text)
        # all_patches, all_im_embs, all_pool_weights = self.get_im_embeddings(images)
        all_patches, all_im_embs = self.get_im_embeddings(images)

        cross_weights = self.get_cross_weights(all_patches, word_embs)
        imtext_logits = self.get_im_text_matrix(all_im_embs, text_emb)
        imim_logits = self.get_im_im_matrix(all_im_embs)

        return imtext_logits, cross_weights, imim_logits, #list per im, #list per im, #list per im-im pair
        # return imtext_logits, imim_logits

    def get_label_embeddings(self, heads):
        '''
        return text embeddings of image labels
        '''
        lab_embeddings = {}
        for head, prompts in heads.items(): 
            head_embeddings = self.get_text_embeddings(prompts, only_texts=True)
            lab_embeddings[head] = head_embeddings
            # lab_embeddings[head] = torch.mean(head_embeddings, dim=0) if avg else head_embeddings

        return lab_embeddings

    # def zero_shot(self, images, label_descriptions):
    #     # Convert label descriptions to text embeddings
    #     if not isinstance(images, list):
    #         images = [images]

    #     images = [im[None, :] if im.dim() == 3 else im for im in images]
    #     text_embeddings = [self.get_text_embeddings(desc, only_texts=True) for desc in label_descriptions]
    #     print("first embedding")
    #     print(text_embeddings[0])
    #     print("second embedding")
    #     print(text_embeddings[1])

    #     # Get image embeddings
    #     image_embeddings = self.get_im_embeddings(images, only_ims=True)

    #     # Calculate similarities and classify
    #     results = []
    #     for img_emb in image_embeddings:
    #         similarities = [self.get_im_text_matrix(img_emb, label_emb) for label_emb in text_embeddings]
    #         # print(similarities)
    #         # predicted_label_index = torch.argmax(torch.cat(similarities, dim=0))
    #         # predicted_label = label_descriptions[predicted_label_index]
    #         # results.append(predicted_label)

    #     return results
