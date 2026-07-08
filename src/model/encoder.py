"""
Lipi v5 MoE Vision Encoder (from scratch — no pretrained backbone).

Backbone (2× width downsample overall, so T = W/4):

    Input: (B, 3, 32, W)
    -> ConvStem:  conv3×3 s(2,2) 3→64 → conv3×3 s(1,1) 64→96   (16, W/2, 96)
    -> ConvA:     3× ConvNeXt (dw7×7 + pw MLP), ch 96          (16, W/2, 96)
    -> BlurPool s(2,1), 96→128                                 ( 8, W/2, 128)
    -> ConvB:     3× ConvNeXt, ch 128                          ( 8, W/2, 128)
    -> BlurPool s(2,2), 128→192  (second width stride)         ( 4, W/4, 192)
    -> proj 192→256
    -> SWA-C:     3× SWA block, dim 256, window 4×32           ( 4, W/4, 256)
    -> merge h=4→2, Linear(512→384)                            ( 2, W/4, 384)
    -> SWA-D:     3× SWA block, dim 384, window 2×64           ( 2, W/4, 384)
    -> LID-1 branch: pool h=2→1, lid1_attn (w=32), group_head
    -> merge h=2→1, Linear(768→384)                            ( 1, W/4, 384)
    -> Group MoE stack: N × (shared attn + 15 routed MLPs + shared MLP)
    -> LID-2 heads (multi-script groups)
    -> Script MoE stack: N × (shared attn + 27 routed MLPs + shared MLP)
    -> Per-script CTC heads (T = W/4)

MoE layers (DeepSeek-style): each layer runs one windowed attention on the
full unpacked frame sequence, then splits into a routed-MLP branch
(per-frame expert lookup by group_id / flat_script_id) and a shared-MLP
branch that runs on every frame, summed. Attention weights are shared —
scripts are all 1D horizontal glyph sequences, so the attention pattern
generalizes; only the channel-wise MLP transform is per-script.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.model.blocks import (
    ConvStem, ConvNeXtBlock, BlurPool2d, SWABlock, MoELayer, GroupCTCModule,
    _patch_merge_h, _per_sample_key_lens,
)
from src.taxonomy import NUM_GROUPS


class LipiMoEEncoder(nn.Module):
    """Lipi v5: ConvStem + ConvNeXt(A,B) + SWA-C/D + LID-1 + MoE stacks.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (15 + blank).
         A dedicated `lid1_attn` SWA block sits between SWA-D (pooled to
         h=1) and the classifier so LID-1 has its own capacity for
         script-family discrimination without forcing the CTC feature
         path into a family/character compromise.
      2. Group MoE stack: N stacked MoELayers with 15 routed MLPs each.
         Every frame passes through the same attention; its MLP is
         picked by group_id, and a shared MLP always runs alongside.
      3. LID-2 classifies each frame into a script within its group
         (multi-script groups only; single-script groups skip it).
      4. Script MoE stack: N stacked MoELayers with 27 routed MLPs each,
         routed by flat script id.
      5. Per-script CTC heads decode characters.

    Time downsampling: overall W is downsampled by 4× (stem s=2 then
    BlurPool s=2 on width) so the output length is T = W/4.  This is
    exposed as the `time_downsample` class attribute so callers can
    compute frame offsets without hard-coding the factor.
    """

    time_downsample = 4  # imgs W / time_downsample = T (output frames)

    def __init__(
        self,
        dim: int = 384,
        stem_mid_ch: int = 64,
        stem_out_ch: int = 96,
        convb_ch: int = 128,
        swac_in_ch: int = 192,
        swac_dim: int = 256,
        num_convA_blocks: int = 3,
        num_convB_blocks: int = 3,
        num_swa_c_blocks: int = 3,
        num_swa_d_blocks: int = 3,
        num_group_layers: int = 3,
        num_script_layers: int = 3,
        swa_c_window_w: int = 32,
        swa_d_window_w: int = 64,
        local_window_w: int = 16,
        wide_window_w: int = 64,
        mlp_ratio: int = 4,
        shared_mlp_ratio: int = 4,
        moe_shared_mlp_ratio: int = 2,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-4,
        num_groups: int = NUM_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
        **unused_kwargs,  # swallow stale kwargs from old checkpoints
    ):
        super().__init__()
        # Silently drop any unknown kwargs — old checkpoint configs may still
        # carry names like num_group_local_blocks / num_super_groups / etc.
        del unused_kwargs
        self.num_groups = num_groups
        self.blank_group_id = num_groups

        if group_script_vocab_sizes is None:
            group_script_vocab_sizes = [[100]] * num_groups
        if group_script_names is None:
            group_script_names = [[f"s{i}" for i in range(len(vs))]
                                  for vs in group_script_vocab_sizes]

        # Build flat script index: (group, local_script) → flat_id
        self.total_scripts = sum(len(vs) for vs in group_script_vocab_sizes)
        self._flat_script_id = {}  # (g, s) → flat
        self._flat_to_group_script = {}  # flat → (g, s)
        self._group_script_counts = [len(vs) for vs in group_script_vocab_sizes]
        flat = 0
        for g, vs in enumerate(group_script_vocab_sizes):
            for s in range(len(vs)):
                self._flat_script_id[(g, s)] = flat
                self._flat_to_group_script[flat] = (g, s)
                flat += 1

        # Which groups are multi-script
        self._multi_script_groups = {
            g for g, vs in enumerate(group_script_vocab_sizes) if len(vs) > 1
        }

        self.config = {
            "dim": dim,
            "stem_mid_ch": stem_mid_ch,
            "stem_out_ch": stem_out_ch,
            "convb_ch": convb_ch,
            "swac_in_ch": swac_in_ch,
            "swac_dim": swac_dim,
            "num_convA_blocks": num_convA_blocks,
            "num_convB_blocks": num_convB_blocks,
            "num_swa_c_blocks": num_swa_c_blocks,
            "num_swa_d_blocks": num_swa_d_blocks,
            "num_group_layers": num_group_layers,
            "num_script_layers": num_script_layers,
            "swa_c_window_w": swa_c_window_w,
            "swa_d_window_w": swa_d_window_w,
            "local_window_w": local_window_w,
            "wide_window_w": wide_window_w,
            "mlp_ratio": mlp_ratio,
            "shared_mlp_ratio": shared_mlp_ratio,
            "moe_shared_mlp_ratio": moe_shared_mlp_ratio,
            "drop_path_rate": drop_path_rate,
            "layer_scale_init": layer_scale_init,
            "num_groups": num_groups,
            "group_script_vocab_sizes": group_script_vocab_sizes,
            "group_script_names": group_script_names,
        }

        # Drop-path schedule: linear ramp from 0 → drop_path_rate across
        # ConvA + ConvB + SWA-C + SWA-D + group MoE stack + script MoE
        # stack. Each MoE layer counts as one stage in the schedule.
        n_stages = (num_convA_blocks + num_convB_blocks
                    + num_swa_c_blocks + num_swa_d_blocks
                    + num_group_layers + num_script_layers)
        dp_schedule = [drop_path_rate * i / max(n_stages - 1, 1)
                       for i in range(n_stages)]
        dp_iter = iter(dp_schedule)

        # ── Stem: (B, 3, 32, W) → (B, stem_out_ch, 16, W/2) ────────────
        self.stem = ConvStem(in_ch=3, mid_ch=stem_mid_ch, out_ch=stem_out_ch)

        # ── ConvA: 3× ConvNeXt @ stem_out_ch at (16, W/2) ─────────────
        self.convA = nn.ModuleList([
            ConvNeXtBlock(dim=stem_out_ch, mlp_ratio=shared_mlp_ratio,
                          drop_path=next(dp_iter),
                          layer_scale_init=1e-4)
            for _ in range(num_convA_blocks)
        ])

        # ── BlurPool s(2,1): 96→128 at (16, W/2) → (8, W/2) ──────────
        self.blur_ab = BlurPool2d(stem_out_ch, convb_ch, stride=(2, 1))

        # ── ConvB: 3× ConvNeXt @ convb_ch at (8, W/2) ─────────────────
        self.convB = nn.ModuleList([
            ConvNeXtBlock(dim=convb_ch, mlp_ratio=shared_mlp_ratio,
                          drop_path=next(dp_iter),
                          layer_scale_init=1e-4)
            for _ in range(num_convB_blocks)
        ])

        # ── BlurPool s(2,2): 128→192 at (8, W/2) → (4, W/4) ──────────
        # Second (and last) width stride happens here.
        self.blur_bc = BlurPool2d(convb_ch, swac_in_ch, stride=(2, 2))

        # ── SWA-C entry projection: swac_in_ch → swac_dim ─────────────
        # Kept as a 1×1 conv (channel-last equivalent) so the SWA-C blocks
        # can run at swac_dim even though BlurPool outputs swac_in_ch.
        self.swac_in_proj = nn.Linear(swac_in_ch, swac_dim)

        # ── SWA-C: 3× SWA @ swac_dim at (4, W/4), window 4×32 ─────────
        self.swa_c = nn.ModuleList([
            SWABlock(dim=swac_dim, num_heads=max(swac_dim // 64, 1),
                     window_h=4, window_w=swa_c_window_w, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_swa_c_blocks)
        ])

        # ── merge h=4→2, dim swac_dim*2 → dim ────────────────────────
        # Concatenates pairs of adjacent rows into a single row and
        # projects to the SWA-D dim. Init averages the two rows into the
        # first swac_dim output channels; extra output channels start at 0.
        self.merge_cd = nn.Linear(swac_dim * 2, dim)
        with torch.no_grad():
            self.merge_cd.weight.zero_()
            d_copy = min(swac_dim, dim)
            self.merge_cd.weight[:d_copy, :d_copy] = 0.5 * torch.eye(d_copy)
            self.merge_cd.weight[:d_copy, swac_dim:swac_dim + d_copy] = \
                0.5 * torch.eye(d_copy)
            self.merge_cd.bias.zero_()

        # ── SWA-D: 3× SWA @ dim at (2, W/4), window 2×64 ─────────────
        self.swa_d = nn.ModuleList([
            SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                     window_h=2, window_w=swa_d_window_w, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_swa_d_blocks)
        ])

        # ── merge h=2→1, dim*2 → dim (CTC path only) ──────────────────
        # Init as average of the two rows so features pass through at
        # step 0.
        self.merge_d1 = nn.Linear(dim * 2, dim)
        with torch.no_grad():
            self.merge_d1.weight.zero_()
            self.merge_d1.weight[:, :dim] = 0.5 * torch.eye(dim)
            self.merge_d1.weight[:, dim:] = 0.5 * torch.eye(dim)
            self.merge_d1.bias.zero_()

        # Each MoE layer consumes one entry from the drop-path schedule.
        # Same order as construction below: group layers first, then script.
        group_dps = [next(dp_iter) for _ in range(num_group_layers)]
        script_dps = [next(dp_iter) for _ in range(num_script_layers)]

        # ── LID-1: per-frame group classification ─────────────────────
        # Branches off SWA-D output (at h=2) before merge_d1. Pool h=2→1,
        # run a dedicated lid1_attn block (window w=32) for LID-1's own
        # horizontal-context capacity, then a small MLP head.
        self.group_h_pool = nn.AdaptiveAvgPool2d((1, None))
        # lid1_attn keeps LayerScale at 1.0: its identity-at-init comes
        # from the zero-init projections below, and a near-zero
        # LayerScale on top would suppress its gradients ~1e4x.
        self.lid1_attn = SWABlock(
            dim=dim, num_heads=max(dim // 64, 1),
            window_h=1, window_w=32, shift=False,
            mlp_ratio=mlp_ratio, drop_path=0.0,
            layer_scale_init=1.0,
        )
        # Zero the output projections so the residual starts as identity.
        nn.init.zeros_(self.lid1_attn.attn.proj.weight)
        nn.init.zeros_(self.lid1_attn.attn.proj.bias)
        nn.init.zeros_(self.lid1_attn.mlp[-1].weight)
        nn.init.zeros_(self.lid1_attn.mlp[-1].bias)
        self.group_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_groups + 1),
        )

        # ── Group MoE stack (routed by group_id) ──────────────────────
        # N stacked MoELayers, alternating local/wide window widths and
        # shifts. Each layer: shared attention + 15 routed MLPs (one per
        # group) + shared MLP (always on).
        self.group_layers = nn.ModuleList([
            MoELayer(
                dim=dim, num_heads=max(dim // 64, 1), num_experts=num_groups,
                window_w=(local_window_w if i % 2 == 0 else wide_window_w),
                shift=(i % 2 == 1),
                routed_mlp_ratio=mlp_ratio,
                shared_mlp_ratio=moe_shared_mlp_ratio,
                drop_path=group_dps[i],
                layer_scale_init=layer_scale_init,
            )
            for i in range(num_group_layers)
        ])

        # ── LID-2: per-frame script classification (multi-script groups)
        self.lid2_heads = nn.ModuleDict()
        for g in range(num_groups):
            n_scripts = len(group_script_vocab_sizes[g])
            if n_scripts > 1:
                self.lid2_heads[str(g)] = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, n_scripts),
                )

        # ── Script MoE stack (routed by flat script_id) ───────────────
        # Same shape as group_layers but with total_scripts routed MLPs
        # per layer (27 for the full taxonomy).
        self.script_layers = nn.ModuleList([
            MoELayer(
                dim=dim, num_heads=max(dim // 64, 1),
                num_experts=self.total_scripts,
                window_w=(local_window_w if i % 2 == 0 else wide_window_w),
                shift=(i % 2 == 1),
                routed_mlp_ratio=mlp_ratio,
                shared_mlp_ratio=moe_shared_mlp_ratio,
                drop_path=script_dps[i],
                layer_scale_init=layer_scale_init,
            )
            for i in range(num_script_layers)
        ])

        # Output
        self.enc_out_dim = dim
        self.norm = nn.LayerNorm(dim)

        # Per-script CTC heads
        self.ctc_modules = nn.ModuleList([
            GroupCTCModule(
                enc_dim=dim,
                script_vocab_sizes=group_script_vocab_sizes[g],
                script_names=group_script_names[g],
            )
            for g in range(num_groups)
        ])

    def _ctc_logits(self, x: Tensor, flat_scripts: Tensor,
                    script_lens_cpu: list) -> Tensor:
        """Dispatch per-script CTC heads over routed frames.

        CTC heads are position-wise Linear(dim → vocab), so we can
        dispatch them by boolean mask on the full sequence — no need
        to pack frames into contiguous segments. Iterate by (group,
        local_script); use script_lens_cpu to skip empty (g, s) without
        a GPU→CPU sync.
        """
        B, T, _ = x.shape
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
        for (g, s), flat_id in self._flat_script_id.items():
            if not any(script_lens_cpu[b][flat_id] > 0 for b in range(B)):
                continue
            mask = (flat_scripts == flat_id)  # (B, T)
            head = self.ctc_modules[g].heads[s]
            vs = head.vocab_size
            head_out = head(x[mask]).to(logits.dtype)  # (K, vs)
            logits[mask, :vs] = head_out
        return logits

    def _get_flat_script_ids(self, group_ids, script_ids):
        """Convert (group_id, local_script_id) pairs to flat script indices.
        Blank/whitespace frames (group_id == blank_group_id) get flat_id = -1.
        """
        flat = torch.full_like(group_ids, -1)
        for (g, s), f in self._flat_script_id.items():
            mask = (group_ids == g) & (script_ids == s)
            flat[mask] = f
        return flat

    def _run_backbone(self, images: Tensor) -> tuple[Tensor, int, int]:
        """Run stem → ConvA → BlurPool → ConvB → BlurPool → proj → SWA-C
        → merge → SWA-D. Returns (x, h, w) with x of shape (B, h*w, dim),
        h=2, w=T = W/4. The h=2 tensor feeds both the LID-1 branch and
        the final merge_d1 → CTC path.
        """
        B = images.shape[0]
        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # Stem: (B, 3, 32, W) → (B, stem_out_ch, 16, W/2)
        x = self.stem(x)

        # ConvA: 3× ConvNeXt in NCHW at (16, W/2)
        for blk in self.convA:
            x = blk(x)

        # BlurPool s(2,1) 96→128: (16, W/2) → (8, W/2)
        x = self.blur_ab(x)

        # ConvB: 3× ConvNeXt at (8, W/2)
        for blk in self.convB:
            x = blk(x)

        # BlurPool s(2,2) 128→192: (8, W/2) → (4, W/4)
        x = self.blur_bc(x)

        # SWA-C: switch to channel-last for SWA. (B, C, H, W) → (B, H*W, C).
        _, C, h, w = x.shape  # h=4, w=W/4
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.swac_in_proj(x)  # C → swac_dim
        for blk in self.swa_c:
            x = blk(x, h, w)

        # merge h=4→2, swac_dim*2 → dim
        x, h = _patch_merge_h(x, h, w, self.merge_cd)  # h=4→2

        # SWA-D at (2, W/4), dim
        for blk in self.swa_d:
            x = blk(x, h, w)

        return x, h, w

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
        compute_ctc: bool = True,
        inter_ctc: bool = False,
        route_sample_p: float = 0.0,
    ) -> dict:
        """Run the encoder forward pass.

        compute_ctc=False skips the script experts + final norm + CTC
        heads. Useful when the CTC loss weight is 0 and we don't need
        character predictions (e.g. LID-only pretraining).

        route_sample_p (scheduled sampling, training only): probability
        per frame of routing by the model's own LID-1/LID-2 predictions
        instead of the provided ground truth. Experts and CTC heads then
        see realistic misroutes during training instead of meeting them
        for the first time at inference. Losses keep using GT labels.

        inter_ctc=True additionally decodes the features after the group
        MoE stack (before script experts) through the same norm + CTC
        heads and returns them as "inter_logits" — an intermediate-CTC
        auxiliary target that regularizes the trunk and forces character
        information to exist before script specialization. Training-only;
        no extra parameters (heads are shared with the final CTC).
        """
        B = images.shape[0]

        x, h, w = self._run_backbone(images)
        # x here is post-SWA-D at (h=2, w=W/4). w is the final T.
        d = x.shape[-1]

        # LID-1 branch: pool h=2→1, run lid1_attn (window w=32), classify.
        # Uses the pre-merge_d1 tensor so merge_d1 only ever sees CTC
        # gradient.
        x_for_group = x.reshape(B, h, w, d).permute(0, 3, 1, 2)  # (B, d, 2, w)
        x_for_group = self.group_h_pool(x_for_group).squeeze(2).permute(0, 2, 1)
        x_for_group = self.lid1_attn(x_for_group, 1, w)
        group_logits = self.group_head(x_for_group)  # (B, w, num_groups+1)

        # Patch-merge 2 → 1, dim*2 → dim (CTC path only).
        x, h = _patch_merge_h(x, h, w, self.merge_d1)

        # Determine per-frame group assignments
        if group_ids is not None:
            if group_ids.dim() == 1:
                frame_groups = group_ids.unsqueeze(1).expand(B, w)
            else:
                frame_groups = group_ids
            # Defensive: clamp invalid IDs (e.g. -100 padding) to blank.
            # Caller should already do this, but bincount/indexing crash otherwise.
            frame_groups = torch.where(
                (frame_groups >= 0) & (frame_groups <= self.blank_group_id),
                frame_groups, torch.full_like(frame_groups, self.blank_group_id))
        else:
            frame_groups = group_logits.argmax(dim=-1)

        # Scheduled sampling: a random subset of frames routes by LID-1's
        # prediction instead of GT (script-stage counterpart below).
        sample_mask = None
        if group_ids is not None and route_sample_p > 0.0 and self.training:
            sample_mask = (torch.rand(B, w, device=images.device)
                           < route_sample_p)
            frame_groups = torch.where(
                sample_mask, group_logits.argmax(dim=-1), frame_groups)

        if detach_for_experts:
            x = x.detach()

        # =====================================================================
        # STAGE 1: Group MoE stack.
        # Each MoELayer runs shared attention on the full sequence, then
        # applies per-frame routed MLPs (indexed by frame_groups) plus a
        # shared MLP. Blank frames (group_id >= num_groups) still get
        # attention + shared MLP; only the routed MLP is skipped.
        # =====================================================================

        # One upfront sync: per-sample per-group frame counts. Reused
        # across every group layer so the sync-free expert skip in
        # MoELayer._routed_mlp costs zero per-layer.
        group_lens_cpu = _per_sample_key_lens(
            frame_groups, self.num_groups).tolist()

        for layer in self.group_layers:
            x = layer(x, frame_groups, group_lens_cpu, w)

        # Features entering the script stack — kept for intermediate CTC.
        x_inter = x if (inter_ctc and compute_ctc) else None

        # =====================================================================
        # LID-2: per-frame script classification
        # =====================================================================

        # Collect LID-2 logits for multi-script groups
        lid2_logits_per_group = {}  # g → (B, T, n_scripts)
        for g_str, head in self.lid2_heads.items():
            g = int(g_str)
            lid2_logits_per_group[g] = head(x)  # (B, T, n_scripts)

        # Determine per-frame script assignments
        # frame_scripts: (B, T) — local script_id within each frame's group
        frame_scripts = torch.zeros(B, w, dtype=torch.long, device=x.device)

        if script_ids is not None:
            # Training: ground truth
            if script_ids.dim() == 1:
                frame_scripts = script_ids.unsqueeze(1).expand(B, w)
            else:
                frame_scripts = script_ids
            # Defensive: clamp invalid script_ids to 0
            frame_scripts = torch.where(
                frame_scripts >= 0, frame_scripts,
                torch.zeros_like(frame_scripts))
        else:
            # Inference: predict from LID-2 for multi-script groups.
            # Reuse the Python-side group counts computed above to skip
            # empty groups without a sync.
            for g, lid2_log in lid2_logits_per_group.items():
                if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                    continue
                g_mask = (frame_groups == g)
                pred = lid2_log.argmax(dim=-1)  # (B, T)
                frame_scripts[g_mask] = pred[g_mask]

        if sample_mask is not None:
            # Scheduled sampling, script stage: sampled frames take the
            # predicted script within their (possibly predicted) group —
            # 0 for single-script groups, LID-2's argmax for multi-script
            # ones — mirroring the inference routing path.
            frame_scripts = torch.where(
                sample_mask, torch.zeros_like(frame_scripts), frame_scripts)
            for g, lid2_log in lid2_logits_per_group.items():
                if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                    continue
                m = (frame_groups == g) & sample_mask
                frame_scripts = torch.where(
                    m, lid2_log.argmax(dim=-1), frame_scripts)

        # Convert to flat script IDs for script expert routing
        flat_scripts = self._get_flat_script_ids(frame_groups, frame_scripts)

        # Skip script experts + final norm + CTC heads when compute_ctc
        # is False (e.g. ctc_weight=0 pretraining). Script experts have
        # no gradient path to any active loss in that case.
        if not compute_ctc:
            T = w
            max_vocab = max(m.max_vocab for m in self.ctc_modules)
            return {
                "logits": torch.zeros(B, T, max_vocab,
                                      device=x.device, dtype=x.dtype),
                "lengths": torch.full((B,), T, dtype=torch.long, device=x.device),
                "group_logits": group_logits,
                "group_ids": frame_groups,
                "lid2_logits_per_group": lid2_logits_per_group,
                "frame_scripts": frame_scripts,
                "flat_scripts": flat_scripts,
            }

        # =====================================================================
        # STAGE 2: Script MoE stack.
        # Same shape as Stage 1, routed by flat script id. Frames whose
        # flat_scripts == -1 (blank / unrouted) skip only the routed MLP.
        # =====================================================================

        # One upfront sync: per-sample per-script frame counts. flat_scripts
        # uses -1 for blank/unrouted, which _per_sample_key_lens filters.
        script_lens_cpu = _per_sample_key_lens(
            flat_scripts, self.total_scripts).tolist()

        for layer in self.script_layers:
            x = layer(x, flat_scripts, script_lens_cpu, w)

        # =====================================================================
        # CTC heads (per-frame routing via boolean mask — no packing)
        # =====================================================================

        x = self.norm(x)
        T = x.shape[1]

        logits = self._ctc_logits(x, flat_scripts, script_lens_cpu)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        out = {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "group_ids": frame_groups,
            "lid2_logits_per_group": lid2_logits_per_group,
            "frame_scripts": frame_scripts,
            "flat_scripts": flat_scripts,
        }

        # Intermediate CTC: same norm + heads applied to the pre-script-
        # stack features, routed identically.
        if x_inter is not None:
            out["inter_logits"] = self._ctc_logits(
                self.norm(x_inter), flat_scripts, script_lens_cpu)

        return out
