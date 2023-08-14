# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.nn as nn
from alignment_attention_module import AlignmentAttentionModule
from scaling import ScaledLinear


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.label_level_am_attention = AlignmentAttentionModule()
        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim, initial_scale=0.25)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)
        self.enable_attn = False

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        attn_encoder_out: torch.Tensor,
        lengths: torch.Tensor,
        apply_attn: bool = True,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim, (
            encoder_out.shape,
            decoder_out.shape,
        )

        if apply_attn and attn_encoder_out is None:
            if not self.enable_attn:
                self.enable_attn = True
                logging.info("enabling ATTN!")
            attn_encoder_out = self.label_level_am_attention(
                encoder_out, decoder_out, lengths
            )

        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)

        if apply_attn:
            # print(torch.mean(attn_encoder_out, dim=0))
            logit = encoder_out + decoder_out + attn_encoder_out
        else:
            logging.info("disabling cross attn mdl")
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit