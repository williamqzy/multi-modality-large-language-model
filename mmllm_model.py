import re
from PIL import Image

import torch 
import torch.nn as nn
from my_custom_module import MyPaLM

from transformers import MyCLIPModel, MyAutoTokenizer, MyCLIPProcessor
import my_bitsandbytes


from my_flamingo_pytorch import MyPerceiverResampler



from my_embedding import MyPositionalEmbedding
import numpy as np

class MyPALME_Tokenizer:
    def __init__(self):
        try:
            self.processor = MyCLIPProcessor.from_pretrained("my_model/CLIP-ViT-L-14-my2B-s32B-b82K")
            self.tokenizer = MyAutoTokenizer.from_pretrained(
                "MyEleutherAI/gpt-neox-20b",
                additional_special_tokens=["<my_image>", "</my_image>"],
                eos_token ="<my_eos>",
                pad_token="<my_pad>",
                extra_ids=0,
                model_max_length=8192
            )

            self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<my_image>", "</my_image>"])
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")


    def tokenize_texts(self, texts):
        try:
            texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            print(f"Error tokenizing texts: {e}")

    def tokenize_images(self, images):
        try:
            tokenized_images = self.processor(images=images, return_tensors="pt").pixel_values
            print(f"Tokenized image: {tokenized_images.shape}")
            return tokenized_images
        except Exception as e:
            print(f"Error tokenizing images: {e}")

    def tokenize(self, sample):
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            print(f"Error during tokenization: {e}")

        
class MyPALME(nn.Module):
    def __init__(self):
        super(MyPALME, self).__init__()
        try:
            self.ViT_model = MyCLIPModel.from_pretrained("my_model/CLIP-ViT-L-14-my2B-s32B-b82K").vision_model

            self.embed = my_bitsandbytes.nn.modules.Embedding(
                32002,
                2048,
                padding_idx=1
            )

            try:
                self.embed_positions = MyPositionalEmbedding(2048, 2048, 1)
            except Exception as e:
                print(str(e))

            self.output_projection = torch.nn.Linear(
                2048, 32002, bias=False
            )
            torch.nn.init.normal_(
                self.output_projection.weight, mean=0, std=2048 ** -0.5
            )
          self.decoder = MyPaLM(
                num_tokens=50304,
                dim=2048,
                depth=16,
      num_tokens=50304,
                dim=2048,
                depth=16,
                dim_head=128,
                heads=8,
                flash_attn=True,
                qk_rmsnorm=False,
            )

            self.perceive = MyPerceiverResampler(
                dim=1024,
                depth=2,
                dim_head=8,
                num_latents=64,
                num_media_embeds=257
            )

            self.image_resize = torch.nn.Linear(224 * 224, 1024 * 1024)

            self.image_proj = torch.nn.Linear(1024, 2048, bias=False)
            torch.nn.init.normal_(
                self.image_proj.weight, mean=0, std=2048 ** -0.5
            )

        except Exception as e:
            print(f"Error initializing PALME components: {e}")

    def forward(self, text_tokens, images):
        try:
            images = images.view(images.size(0), -1)  # Flatten the images
            images = self.image_resize(images)  # Resize the images using the linear transformation layer
            images = images.view(images.size(0), 3, 1024, 1024)  # Reshape the images to the expected size

            images = self.perceive(images).squeeze(1)
            print(f"Images perceive: {images}")

            images = self.image_proj(images)
            print(f"Images projected: {images}")

            images_flattened = images.view(images.size(0), -1)
            print(f"Images flattened: {images_flattened}")

            model_input = self.decoder(text_tokens)
            print(model_input[:, 0:2].shape, images.shape, model_input[:, 2:].shape)

            images_flattened = images_flattened.view(1, 2, -1)
            print(f"Images flattened: {images_flattened}")

            model_input = torch.cat([model_input[:, 0:2], images_flattened, model_input[:, 2:]], dim=-1)
            print(f"Model input: {model_input}")

            model_input = self.decoder(model_input, tokens_mask=None)
            print(f"Model input: {model_input}")

            output = self.decoder(model_input, passed_x=model_input)[0]
            print(f"Output: {output}")

            return output
        
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return None
