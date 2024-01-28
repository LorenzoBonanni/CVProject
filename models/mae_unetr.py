from torch import nn
from transformers import ViTMAEForPreTraining

from models.unetr_decoder import UNETR_decoder


class MaeUnetr(nn.Module):

    def __init__(self, train_mae, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mae = train_mae
        vitMaemodel = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        vitmae_encoder = vitMaemodel.vit
        vitmae_encoder.embeddings.config.mask_ratio = 0
        vitmae_encoder.config.output_hidden_states = True
        decoder = UNETR_decoder(
            in_channels=3,
            out_channels=2,
            img_size=(224, 224),
            patch_size=16,
            feature_size=16,
            hidden_size=768,
            spatial_dims=2
        )

        self.mae = vitMaemodel
        self.unetrDecoder = decoder

    def freeze_mae(self):
        for param in self.mae.parameters():
            param.requires_grad = False

    def get_parameters(self):
        if not self.train_mae:
            return self.unetrDecoder.parameters()
        else:
            return list(self.mae.parameters()) + list(self.unetrDecoder.parameters())

    def train(self, **kwargs):
        if self.train_mae:
            self.mae.train()
        self.unetrDecoder.train()

    def eval(self, **kwargs):
        self.mae.eval()
        self.unetrDecoder.eval()

    def forward(self, inputs):
        outputs = self.mae(pixel_values=inputs)
        hidden_states = tuple(hs[:, 1:, :] for hs in outputs.hidden_states)
        mask = self.unetrDecoder(
            x=self.mae.unpatchify(outputs.logits),
            x_in=inputs,
            hidden_states_out=hidden_states
        )

        return mask
