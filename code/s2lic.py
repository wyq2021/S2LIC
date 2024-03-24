import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.entropy_models import EntropyParameters
from compressai.layers import RSTB, HyperAnalysis, HyperSynthesis, subpel_conv3x3, EnhModule, ACCTX, AGCTX,ResidualBlockWithStride, ResidualBlockUpsample
from compressai.layers import LatentResidualPrediction
from compressai.layers.ckbd import *
from compressai.ops import ste_round
from compressai.models.utils import update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from timm.models.layers import trunc_normal_
from .utils import conv, deconv

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class S2LIC(nn.Module):
    """
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        input_resolution (int): Just used for window partition decision
    """

    def __init__(self, N=192, M=320):
        super().__init__()

        depths = [1, 2, 3, 1, 1]
        num_heads = [4, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 4.
        qkv_bias = True
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.enh = EnhModule(64)
        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(192,192),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(96,96),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(48,48),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)

        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        depths = depths[::-1]
        num_heads = num_heads[::-1]

        self.g_s0 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s1 = RSTB(dim=N,
                        input_resolution=(48,48),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s3 = RSTB(dim=N,
                        input_resolution=(96,96),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s4 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s5 = RSTB(dim=N,
                        input_resolution=(192,192),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, 
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s6 = subpel_conv3x3(N, 3, 2)

        self.M =  int(M) 
        
        slice_num = 10
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.channel_wise_context = nn.ModuleList(
            ACCTX(dim=slice_ch * i, out_dim=slice_ch*2, num_heads =slice_ch * i // 32) if i else None
            for i in range(slice_num)
        )

        self.global_inter_context = nn.ModuleList(
            AGCTX(dim=slice_ch * i, out_dim=slice_ch, num_heads =slice_ch * i // 32) if i else None
            for i in range(slice_num)
        )

        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch*3, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch*3, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )


        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )  

    def g_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.enh(x)
        x = self.g_a0(x)
        x = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_a2(x)
        x = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_a4(x)
        x = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_a6(x)
        return x

    def g_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.g_s0(x)
        x = self.g_s1(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_s2(x)
        x = self.g_s3(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_s4(x)
        x = self.g_s5(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_s6(x)
        x = self.enh(x,rev=True)
        return x
    
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(10, dim=1)
        y_hat_slices = []
        y_likelihoods = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                ac_ctx = self.channel_wise_context[idx](torch.cat(y_hat_slices, dim=1))
                ag_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([ac_ctx, ag_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor(Use inter, channel context and hyper params)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([ac_ctx, ag_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        } 
    

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(10, dim=1)
        y_hat_slices = []
        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        # pylint: enable=protected-access
        # print(cdf, cdf_lengths, offsets)
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
            else:
                # Anchor
                ac_ctx = self.channel_wise_context[idx](torch.cat(y_hat_slices, dim=1))
                ag_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([ac_ctx, ag_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([ac_ctx, ag_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time
        }

    def decompress(self, strings, shape):
        torch.cuda.synchronize()
        start_time = time.time()
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(10):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                ac_ctx = self.channel_wise_context[idx](torch.cat(y_hat_slices, dim=1))
                ag_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([ac_ctx, ag_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([ac_ctx, ag_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }
