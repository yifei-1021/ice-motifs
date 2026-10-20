"""
The multimodal VAE.

The architecture is preserved EXACTLY as published so that existing checkpoints keep
loading. Verified against the shipped `model.last.pth`:

    encoder1 (Steinhardt)  (27,128) (128,64) (64,20)
    encoder2 (SOAP)        (252,512) (512,256) (256,20)
    latent_layer           (40,80) (80,80)          -> dim(Z) = 40
    decoder1               (40,2048) (2048,2048) (2048,279)
    predictor              (40,512) (512,512) (512,2)

Note the two output widths, 279 and 2: SI Section 2 lists the hidden layers only and
stops before them, which makes the decoder look like it never reaches the descriptors.

Changes relative to the original `model_concatdecoder.py`, none of them structural:
  * the unused `from torchvision import ...` is gone (it also breaks - torchvision is
    not a dependency of anything here)
  * `latent` is coerced to int. The original crashed when train.py passed the string
    "40": `self.latent = latent * 2` gives "4040" and nn.Linear raises TypeError.
  * the broken `VAE` helper class (its decoder ended in nn.Linear(input, input), a
    shape error) is gone; only its encoder was ever used, and that is built directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import columns as C


def _encoder(in_dim, hidden, latent, n_layers=2, dropout=0.1):
    """Encoder stack: in_dim -> hidden -> hidden//2 -> ... -> latent*2 (mean, log_var)."""
    layers = []
    cur, h = in_dim, hidden
    for _ in range(n_layers):
        layers += [nn.Linear(cur, h), nn.Tanh(), nn.Dropout(dropout)]
        cur, h = h, h // 2
    layers.append(nn.Linear(cur, latent * 2))
    return nn.Sequential(*layers)


def steinhardt_encoder(latent=10, hidden=128, in_dim=C.N_STEIN):
    return _encoder(in_dim, hidden, latent)


def soap_encoder(latent=10, hidden=512, in_dim=C.N_SOAP):
    return _encoder(in_dim, hidden, latent)


class VAE2(nn.Module):
    """Multimodal VAE over (Steinhardt, SOAP) with a property-prediction head."""

    def __init__(self, encoder1, encoder2, scaler1, scaler2,
                 regression=2, HIDDEN_DIM=512, latent=20,
                 num_decoder_layers=2, num_latent_layers=1,
                 dropout_rate=0.1, verbose=False, decoder_hidden=2048):
        super().__init__()
        self.verbose = verbose

        self.scaler1, self.encoder1 = scaler1, encoder1
        self.input1 = encoder1[0].in_features
        self.hidden1 = decoder_hidden
        self.output1 = encoder1[-1].out_features

        self.scaler2, self.encoder2 = scaler2, encoder2
        self.input2 = encoder2[0].in_features
        self.hidden2 = encoder2[0].out_features
        self.output2 = encoder2[-1].out_features

        self.latent_input = self.output1 + self.output2
        self.latent = int(latent) * 2          # int() - the original accepted a str and crashed
        self.recon1, self.recon2 = self.input1, self.input2
        self.hidden = HIDDEN_DIM
        self.num_decoder_layers = num_decoder_layers
        self.num_latent_layers = num_latent_layers
        self.dropout_rate = dropout_rate
        self.regression_num = regression

        layers, cur = [], self.latent_input
        for _ in range(self.num_latent_layers):
            layers += [nn.Linear(cur, self.latent), nn.Tanh()]
            cur = self.latent
        layers.append(nn.Linear(cur, self.latent))
        self.latent_layer = nn.Sequential(*layers)

        layers, cur = [], self.latent // 2
        for _ in range(self.num_decoder_layers):
            layers += [nn.Linear(cur, self.hidden1), nn.Tanh(), nn.Dropout(self.dropout_rate)]
            cur = self.hidden1
        layers.append(nn.Linear(cur, self.recon1 + self.recon2))
        self.decoder1 = nn.Sequential(*layers)

        layers, cur = [], self.latent // 2
        for _ in range(self.num_decoder_layers):
            layers += [nn.Linear(cur, self.hidden), nn.Tanh(), nn.Dropout(self.dropout_rate)]
            cur = self.hidden
        layers.append(nn.Linear(cur, self.regression_num))
        self.predictor = nn.Sequential(*layers)

    @property
    def dim_z(self):
        return self.latent // 2

    def encode_rep(self, encoder, x):
        h = encoder(x)
        half = h.shape[1] // 2
        return h[:, :half], h[:, half:]

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def forward(self, x1, x2):
        x1 = self.scaler1.transform(x1)
        x2 = self.scaler2.transform(x2)
        z1, z2 = self.encoder1(x1), self.encoder2(x2)
        mean, log_var = self.encode_rep(self.latent_layer, torch.cat((z1, z2), dim=1))
        z = self.reparameterize(mean, log_var)
        recon = self.decoder1(z)
        return ((z1, z2), (mean, log_var),
                (z, recon[:, :self.input1], recon[:, self.input1:], self.predictor(z)))


def build(scaler_stein, scaler_soap, dim_z=40, regression=2,
          stein_latent=10, soap_latent=10, dropout=0.1):
    """Fresh model with the published geometry."""
    return VAE2(steinhardt_encoder(stein_latent), soap_encoder(soap_latent),
                scaler_stein, scaler_soap, regression=regression,
                latent=dim_z, dropout_rate=dropout)


def save(model, path, extra=None):
    """Save as a state_dict plus geometry - never a full pickle."""
    torch.save({"state_dict": model.state_dict(),
                "dim_z": model.dim_z,
                "regression": model.regression_num,
                "input1": model.input1, "input2": model.input2,
                "extra": extra or {}}, path)


def load(path, scaler_stein, scaler_soap, map_location="cpu"):
    """Load a checkpoint written by `save`."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if "state_dict" not in ck:
        raise ValueError(f"{path} is not an icemotifs checkpoint; "
                         "for a legacy full-pickle use icemotifs.compat.load_legacy")
    m = build(scaler_stein, scaler_soap, dim_z=ck["dim_z"], regression=ck["regression"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
