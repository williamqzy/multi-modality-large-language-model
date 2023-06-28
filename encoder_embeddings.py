import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLanguageEncoder(nn.Module):
    def __init__(self, text_embedding, vision_embedding):
        super().__init__()
        self.text_embedding = text_embedding
        self.vision_embedding = vision_embedding

    def forward(self, textual_tokens, visual_tokens, **kwargs):
        if textual_tokens is None:
            return self.vision_embedding(visual_tokens)

        if visual_tokens is None:
            return self.text_embedding(textual_tokens)

        visual_embed = self.vision_embedding(visual_tokens)
        text_embed = self.text_embedding(textual_tokens)

        return torch.cat([visual_embed, text_embed], dim=1)


class VisionEmbedder(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        image_size=1024,
        patch_size=16,
        in_channels=3,
        embedding_dim=768,
        include_mask_token=False,
        prepend_cls_token=False,
    ):
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

        if include_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.mask_token = None

        if prepend_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.cls_token = None

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, x, masked_position=None, **kwargs):
        batch_size, _, H, W = x.shape
        assert (
            H == self.image_size[0] and W == self.image_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        sequence_length = x.size(1)

        if masked_position is not None:
            assert self.mask_token is not None
            mask_token = self.mask_token.expand(batch_size, sequence_length, -1)
            w = masked_position.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        return x
