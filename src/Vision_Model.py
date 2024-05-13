from __future__ import annotations

import logging
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, Sequence
from modules import MLP, MultiTaskModel
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TypeImageEncoder = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
MODEL_TYPE = "resnet50"
JOINT_FEATURE_SIZE = 128

BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
REPO_URL = f"https://huggingface.co/{BIOMED_VLP_CXR_BERT_SPECIALIZED}"
CXR_BERT_COMMIT_TAG = "v1.1"

BIOVIL_IMAGE_WEIGHTS_NAME = "biovil_image_resnet50_proj_size_128.pt"
# BIOVIL_IMAGE_WEIGHTS_URL = f"{REPO_URL}/resolve/{CXR_BERT_COMMIT_TAG}/{BIOVIL_IMAGE_WEIGHTS_NAME}"
# BIOVIL_IMAGE_WEIGHTS_MD5 = "02ce6ee460f72efd599295f440dbb453"

logging.getLogger(__name__)

def get_biovil_resnet(pretrained: bool = True, eval=False) -> ImageModel:
    """Download weights from Hugging Face and instantiate the image model."""
    image_weights_checkpoint_path = os.path.join('../models/', f'{BIOVIL_IMAGE_WEIGHTS_NAME}')
    image_model = ImageModel(
        img_model_type=MODEL_TYPE,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=image_weights_checkpoint_path if pretrained else None
    )
    return image_model

def getCNN(pretrained: bool = False, num_heads = 5, loadpath=None, loadmodel='best_model.pt', freeze=False, eval=True, classifier=False) -> ImageModel:
    image_weights_checkpoint_path = f'{BIOVIL_IMAGE_WEIGHTS_NAME}'
    image_model = ImageModel(
        img_model_type=MODEL_TYPE,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=image_weights_checkpoint_path if pretrained else None,
        num_classes= num_heads,
        freeze_encoder=freeze,
        classifier_hidden_dim=128,
        num_tasks=1,
        project= True
    )
    if loadpath:
        if loadmodel == '':
            checkpoint = torch.load(loadpath, map_location = device)
        else:
            checkpoint = torch.load(os.path.join(loadpath, loadmodel), map_location=device)
        image_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    image_model.freeze_encoder = freeze

    image_model.downstream_classifier_kwargs['project'] = True
    if classifier or (not ('cnn' in loadpath) and ('fine' not in loadpath + loadmodel) and ('Fine' not in loadpath + loadmodel)):
        image_model.classifier = image_model.create_downstream_classifier()

    return image_model

class ImageClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, encoder=None, freeze_encoder=False):
        super(ImageClassifier, self).__init__()

        # use resnet by default
        if not encoder: 
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.encoder = encoder
        self.freeze_encoder=freeze_encoder
        self.feature_size = get_encoder_output_dim(self.encoder)
        self.avgpool = F.adaptive_avg_pool2d
        self.fc1 = nn.Linear(self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.3)

    def train(self, mode: bool = True) -> Any:
        """Switch the model between training and evaluation modes."""
        super().train(mode=mode)
        self.encoder.train(not self.freeze_encoder)
        return self

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        outputs = self.forward(x)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        return probs, preds

@dataclass
class ImageClassifierOutput():
    img_embedding: torch.Tensor
    class_logits: torch.Tensor

@enum.unique
class ResnetType(str, enum.Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"

@dataclass
class ImageModelOutput():
    img_embedding: torch.Tensor
    patch_embedding: torch.Tensor
    projected_global_embedding: torch.Tensor
    class_logits: torch.Tensor
    projected_patch_embeddings: torch.Tensor

class ImageModel(nn.Module):
    """Image encoder module"""

    def __init__(self,
                 img_model_type: str,
                 joint_feature_size: int,
                 freeze_encoder: bool = False,
                 pretrained_model_path: Optional[Union[str, Path]] = None,
                 **downstream_classifier_kwargs: Any):
        super().__init__()

        # Initiate encoder, projector, and classifier
        self.encoder = ImageEncoder(img_model_type)
        self.feature_size = get_encoder_output_dim(self.encoder)
        self.projector = MLP(input_dim=self.feature_size, output_dim=joint_feature_size,
                             hidden_dim=joint_feature_size, use_1x1_convs=True)
        self.downstream_classifier_kwargs = downstream_classifier_kwargs
        logging.info(self.downstream_classifier_kwargs)
        self.classifier = self.create_downstream_classifier() if downstream_classifier_kwargs else None

        # Initialise the mode of modules
        self.freeze_encoder = freeze_encoder
        self.dropout = nn.Dropout(0.1)
        self.train()

        if pretrained_model_path is not None:
            if not isinstance(pretrained_model_path, (str, Path)):
                raise TypeError(f"Expected a string or Path, got {type(pretrained_model_path)}")
            state_dict = torch.load(pretrained_model_path, map_location=device)
            try:
                self.load_state_dict(state_dict, strict=True)
                logging.info("Loading everything")
            except:
                self.encoder.load_state_dict(state_dict, strict=False)
                logging.info("Only loading encoder")

    def train(self, mode: bool = True) -> Any:
        """Switch the model between training and evaluation modes."""
        super().train(mode=mode)
        self.encoder.train(mode=not self.freeze_encoder)
        return self

    def forward(self, x: torch.Tensor) -> ImageModelOutput:
        patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        projected_patch_embeddings = self.projector(patch_x)
        projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3)) #All models are doing this

        if 'project' in self.downstream_classifier_kwargs.keys() and self.downstream_classifier_kwargs['project']:
            proj = self.dropout(projected_global_embedding) #Newer version classify from the projected global embedding
        else:
            proj = self.dropout(pooled_x) #Old versions classify from the unprojected global embedding

        logits = self.classifier(proj) if self.classifier else None
        return ImageModelOutput(img_embedding=pooled_x,
                                patch_embedding=patch_x,
                                class_logits=logits,
                                projected_patch_embeddings=projected_patch_embeddings,
                                projected_global_embedding=projected_global_embedding)

    def create_downstream_classifier(self, **kwargs: Any) -> MultiTaskModel:
        """Create the classification module for the downstream task."""
        downstream_classifier_kwargs = kwargs if kwargs else self.downstream_classifier_kwargs
        if 'project' not in downstream_classifier_kwargs.keys() or not downstream_classifier_kwargs['project']:
            return MultiTaskModel(self.feature_size, **downstream_classifier_kwargs)
        elif downstream_classifier_kwargs['project']:
            return MultiTaskModel(JOINT_FEATURE_SIZE, **downstream_classifier_kwargs)

    @torch.no_grad()
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Get patch-wise projected embeddings from the CNN model.
        :param input_img: input tensor image [B, C, H, W].
        :param normalize: If ``True``, the embeddings are L2-normalized.
        :returns projected_embeddings: tensor of embeddings in shape [batch, n_patches_h, n_patches_w, feature_size].
        """
        assert not self.training, "This function is only implemented for evaluation mode"
        outputs = self.forward(input_img)
        projected_embeddings = outputs.projected_patch_embeddings.detach()  # type: ignore
        if normalize:
            projected_embeddings = F.normalize(projected_embeddings, dim=1)
        projected_embeddings = projected_embeddings.permute([0, 2, 3, 1])  # B D H W -> B H W D (D: Features)
        return projected_embeddings


class ImageEncoder(nn.Module):
    """Image encoder trunk module for the ``ImageModel`` class.
    :param img_model_type: Type of image model to use: either ``"resnet18"`` or ``"resnet50"``.
    """

    def __init__(self, img_model_type: str):
        super().__init__()
        self.img_model_type = img_model_type
        self.encoder = self._create_encoder()

    def _create_encoder(self, **kwargs: Any) -> nn.Module:
        supported = ResnetType.RESNET18, ResnetType.RESNET50
        if self.img_model_type not in supported:
            raise NotImplementedError(f"Image model type \"{self.img_model_type}\" must be in {supported}")
        
        if self.img_model_type == ResnetType.RESNET18:
            encoder = resnet18(weights=ResNet18_Weights.DEFAULT, **kwargs)
        else:
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT, **kwargs)

        return encoder

    def forward(self, x: torch.Tensor, return_patch_embeddings: bool = False) -> TypeImageEncoder:
        """Image encoder forward pass."""

        encoder_wo_avg_pool_fc = nn.Sequential(OrderedDict([*(list(self.encoder.named_children())[:-2])]))
        x = encoder_wo_avg_pool_fc(x)
        avg_pooled_emb = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        if return_patch_embeddings:
            return x, avg_pooled_emb

        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        """Workaround for enabling dilated convolutions after model initialization.
        :param replace_stride_with_dilation: for each layer to replace the 2x2 stride with a dilated convolution
        """
        if self.img_model_type == "resnet18":
            # resnet18 uses BasicBlock implementation, which does not support dilated convolutions.
            raise NotImplementedError("resnet18 does not support dilated convolutions")

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = False, False, True

        device = next(self.encoder.parameters()).device
        new_encoder = self._create_encoder(replace_stride_with_dilation=replace_stride_with_dilation).to(device)

        if self.encoder.training:
            new_encoder.train()
        else:
            new_encoder.eval()

        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder


@torch.no_grad()
def get_encoder_output_dim(module: torch.nn.Module) -> int:
    """Calculate the output dimension of ssl encoder by making a single forward pass.
    :param module: Encoder module.
    """
    # Target device
    device = next(module.parameters()).device  # type: ignore
    assert isinstance(device, torch.device)

    x = torch.rand((1, 3, 448, 448)).to(device)

    # Extract the number of output feature dimensions
    representations = module(x)
    return representations.shape[1]

if __name__ == '__main__':
    cnn = get_biovil_resnet()