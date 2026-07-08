"""
Lipi v5 MoE Vision Encoder (from scratch — no pretrained backbone).

New backbone (2× width downsample overall, so T = W/4):

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
    -> Group experts (local w=16 + wide w=64, per-group, dim=384)
    -> LID-2 heads (multi-script groups)
    -> Script experts (local w=16 + wide w=64, per-script, dim=384)
    -> Per-script CTC heads (T = W/4)
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.model.blocks import (
    ConvStem, ConvNeXtBlock, BlurPool2d, SWABlock, ExpertBlock, GroupCTCModule,
    _patch_merge_h, _per_sample_key_lens,
    _collect_segments, _scatter_segments, _run_expert_block,
)
from src.taxonomy import NUM_GROUPS


def _make_expert_stream(num_blocks: int, dim: int, num_experts: int,
                        window_w: int, drop_path: float, mlp_ratio: int,
                        layer_scale_init: float) -> nn.ModuleList:
    """Build a stack of ExpertBlocks with alternating shift, shared across experts.

    Used four times in LipiMoEEncoder: (group_local, group_wide) at
    num_experts=num_groups and (script_local, script_wide) at
    num_experts=total_scripts. Only window_w differs between local and wide.
    """
    return nn.ModuleList([
        ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_experts,
                    window_h=1, window_w=window_w, shift=(i % 2 == 1),
                    mlp_ratio=mlp_ratio, drop_path=drop_path,
                    layer_scale_init=layer_scale_init)
        for i in range(num_blocks)
    ])


def _zero_init_expert_output_projs(*block_lists: nn.ModuleList) -> None:
    """Zero the attn.proj and mlp.fc2 of every expert in every block.

    Keeps residual paths at identity at init so freshly-added experts pass
    features through untouched until they learn to specialize. Consumes
    no RNG — safe to insert anywhere in __init__ without affecting the
    random-init trajectory of surrounding modules.
    """
    for block_list in block_lists:
        for block in block_list:
            for attn in block.expert_attns:
                nn.init.zeros_(attn.proj.weight)
                nn.init.zeros_(attn.proj.bias)
            for mlp in block.expert_mlps:
                nn.init.zeros_(mlp.fc2.weight)
                nn.init.zeros_(mlp.fc2.bias)


def _make_identity_aggregates(n: int, dim: int) -> nn.ModuleList:
    """Build n Linear(dim*2, dim) aggregators pre-initialized to average the
    two halves of the input (`0.5*I` on the left half, `0.5*I` on the right
    half, zero bias). At init time each aggregator computes (local + wide) / 2.
    """
    aggs = nn.ModuleList()
    for _ in range(n):
        agg = nn.Linear(dim * 2, dim)
        nn.init.zeros_(agg.bias)
        with torch.no_grad():
            agg.weight.zero_()
            agg.weight[:, :dim] = 0.5 * torch.eye(dim)
            agg.weight[:, dim:] = 0.5 * torch.eye(dim)
        aggs.append(agg)
    return aggs


class LipiMoEEncoder(nn.Module):
    """Lipi v5: ConvStem + ConvNeXt(A,B) + SWA-C/D + LID-1 + group/script experts.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (15 + blank). A
         dedicated `lid1_attn` SWA block sits between SWA-D output (pooled
         to h=1) and the classifier so LID-1 has its own capacity for
         script-family discrimination without forcing the CTC feature path
         into a compromise between family and character features.
      2. Group expert blocks process frames per-group (local + wide streams)
      3. LID-2 classifies each frame into a script within its group
      4. Script expert blocks process frames per-script (local + wide streams)
      5. Per-script CTC heads decode characters

    Single-script groups skip LID-2 (only 1 script, trivially assigned).

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
        num_group_local_blocks: int = 1,
        num_group_wide_blocks: int = 1,
        num_script_local_blocks: int = 1,
        num_script_wide_blocks: int = 1,
        swa_c_window_w: int = 32,
        swa_d_window_w: int = 64,
        local_window_w: int = 16,
        wide_window_w: int = 64,
        mlp_ratio: int = 4,
        shared_mlp_ratio: int = 4,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1.0,
        num_groups: int = NUM_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
        **unused_kwargs,  # swallow stale kwargs from old checkpoints
    ):
        super().__init__()
        # Silently drop any unknown kwargs — old checkpoint configs may still
        # carry names like num_shared_a_blocks / num_super_groups / etc.
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
            "num_group_local_blocks": num_group_local_blocks,
            "num_group_wide_blocks": num_group_wide_blocks,
            "num_script_local_blocks": num_script_local_blocks,
            "num_script_wide_blocks": num_script_wide_blocks,
            "swa_c_window_w": swa_c_window_w,
            "swa_d_window_w": swa_d_window_w,
            "local_window_w": local_window_w,
            "wide_window_w": wide_window_w,
            "mlp_ratio": mlp_ratio,
            "shared_mlp_ratio": shared_mlp_ratio,
            "drop_path_rate": drop_path_rate,
            "layer_scale_init": layer_scale_init,
            "num_groups": num_groups,
            "group_script_vocab_sizes": group_script_vocab_sizes,
            "group_script_names": group_script_names,
        }

        # Drop-path schedule: linear ramp from 0 → drop_path_rate across
        # ConvA + ConvB + SWA-C + SWA-D + (group expert stage) + (script
        # expert stage). Parallel local/wide streams share one rate per
        # stage so residual scaling matches.
        n_stages = (num_convA_blocks + num_convB_blocks
                    + num_swa_c_blocks + num_swa_d_blocks
                    + max(num_group_local_blocks, num_group_wide_blocks)
                    + max(num_script_local_blocks, num_script_wide_blocks))
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

        # Parallel expert stages share one drop-path rate per stage.
        group_dp = next(dp_iter) if max(num_group_local_blocks, num_group_wide_blocks) > 0 else 0.0
        script_dp = next(dp_iter) if max(num_script_local_blocks, num_script_wide_blocks) > 0 else 0.0

        # ── LID-1: per-frame group classification ─────────────────────
        # Branches off SWA-D output (at h=2) before merge_d1. Pool h=2→1,
        # run a dedicated lid1_attn block (window w=32) for LID-1's own
        # horizontal-context capacity, then a small MLP head.
        self.group_h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lid1_attn = SWABlock(
            dim=dim, num_heads=max(dim // 64, 1),
            window_h=1, window_w=32, shift=False,
            mlp_ratio=mlp_ratio, drop_path=0.0,
            layer_scale_init=layer_scale_init,
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

        # ── Group expert blocks (routed by group_id) ──────────────────
        # Local vs wide streams differ only in window_w. Output
        # projections zero-init so residuals pass features through at
        # step 0 — experts specialize gradually.
        self.group_local_blocks = _make_expert_stream(
            num_group_local_blocks, dim, num_groups,
            local_window_w, group_dp, mlp_ratio, layer_scale_init)
        self.group_wide_blocks = _make_expert_stream(
            num_group_wide_blocks, dim, num_groups,
            wide_window_w, group_dp, mlp_ratio, layer_scale_init)
        _zero_init_expert_output_projs(
            self.group_local_blocks, self.group_wide_blocks)

        # Group aggregation: concat local + wide → dim, init averaging.
        self.group_aggregates = _make_identity_aggregates(num_groups, dim)

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

        # ── Script expert blocks (routed by flat script_id) ───────────
        self.script_local_blocks = _make_expert_stream(
            num_script_local_blocks, dim, self.total_scripts,
            local_window_w, script_dp, mlp_ratio, layer_scale_init)
        self.script_wide_blocks = _make_expert_stream(
            num_script_wide_blocks, dim, self.total_scripts,
            wide_window_w, script_dp, mlp_ratio, layer_scale_init)
        _zero_init_expert_output_projs(
            self.script_local_blocks, self.script_wide_blocks)

        # Script aggregation.
        self.script_aggregates = _make_identity_aggregates(self.total_scripts, dim)

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

    def _route_expert_stage(
        self,
        x_in: Tensor,
        ids_2d: Tensor,
        ids_lens_cpu: list,
        num_stages: int,
        local_blocks: nn.ModuleList,
        wide_blocks: nn.ModuleList,
        aggregates: nn.ModuleList,
    ) -> Tensor:
        """Route each (b, t) frame to its expert (indexed by ids_2d[b, t]),
        run it through both the local and wide expert-block stacks, then
        aggregate the two streams via the per-expert aggregator Linear.

        Both group-experts and script-experts stages share this exact
        shape — iterate expert-id 0..num_stages, collect its frames, run
        local + wide, concat + aggregate, scatter back.

        Args:
            x_in:         (B, T, d) input features.
            ids_2d:       (B, T) per-frame expert id (-1 or >=num_stages skipped).
            ids_lens_cpu: (B, num_stages) Python list — pre-transferred counts
                          so the outer loop can skip empty experts without
                          a GPU→CPU sync.
            num_stages:   number of experts (num_groups or total_scripts).
        """
        B, w, _ = x_in.shape
        x_out = torch.zeros_like(x_in)
        for s in range(num_stages):
            # Skip without a sync — checks the pre-transferred CPU list.
            if not any(ids_lens_cpu[b][s] > 0 for b in range(B)):
                continue
            mask_s = (ids_2d == s)
            batch_x, batch_info = _collect_segments(x_in, mask_s)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in local_blocks:
                local = _run_expert_block(block, local, s, 1, max_len)

            wide = batch_x
            for block in wide_blocks:
                wide = _run_expert_block(block, wide, s, 1, max_len)

            comb = torch.cat([local, wide], dim=-1)
            agg = aggregates[s](comb)
            _scatter_segments(x_out, agg, mask_s, batch_info)
        return x_out

    def _get_flat_script_ids(self, group_ids, script_ids):
        """Convert (group_id, local_script_id) pairs to flat script indices.
        Blank/whitespace frames (group_id == blank_group_id) get flat_id = -1.
        """
        flat = torch.full_like(group_ids, -1)
        for (g, s), f in self._flat_script_id.items():
            mask = (group_ids == g) & (script_ids == s)
            flat[mask] = f
        return flat

    def _run_backbone(self, images: Tensor) -> tuple[Tensor, Tensor, int]:
        """Run stem → ConvA → BlurPool → ConvB → BlurPool → proj → SWA-C
        → merge → SWA-D. Returns (x_swad_h2, group_logits_input, T).

        x_swad_h2 has shape (B, 2*T, dim) — pre-merge_d1, feeds both the
        LID-1 branch and merge_d1 (CTC path).
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
    ) -> dict:
        """Run the encoder forward pass.

        compute_ctc=False skips the script experts + final norm + CTC
        heads. Useful when the CTC loss weight is 0 and we don't need
        character predictions (e.g. LID-only pretraining).
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

        if detach_for_experts:
            x = x.detach()

        # =====================================================================
        # STAGE 1: Group expert blocks (routed per-segment by group_id)
        # Iterates by group (≤ num_groups = 15) instead of (B, unique_groups)
        # to amortize kernel-launch overhead — all samples with the same
        # group are padded and processed in one batched call per expert.
        # =====================================================================

        # One upfront sync: per-sample per-group frame counts.
        group_lens_cpu = _per_sample_key_lens(
            frame_groups, self.num_groups).tolist()

        x_after_group = self._route_expert_stage(
            x, frame_groups, group_lens_cpu, self.num_groups,
            self.group_local_blocks, self.group_wide_blocks,
            self.group_aggregates)

        # =====================================================================
        # LID-2: per-frame script classification
        # =====================================================================

        # Collect LID-2 logits for multi-script groups
        lid2_logits_per_group = {}  # g → (B, T, n_scripts)
        for g_str, head in self.lid2_heads.items():
            g = int(g_str)
            lid2_logits_per_group[g] = head(x_after_group)  # (B, T, n_scripts)

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
        # STAGE 2: Script expert blocks (routed per-segment by flat script_id)
        # Same routing pattern as STAGE 1 — see _route_expert_stage.
        # =====================================================================

        # One upfront sync: per-sample per-script frame counts. flat_scripts
        # uses -1 for blank/unrouted, which _per_sample_key_lens filters.
        script_lens_cpu = _per_sample_key_lens(
            flat_scripts, self.total_scripts).tolist()

        x_after_script = self._route_expert_stage(
            x_after_group, flat_scripts, script_lens_cpu, self.total_scripts,
            self.script_local_blocks, self.script_wide_blocks,
            self.script_aggregates)

        # =====================================================================
        # CTC heads (per-segment routing)
        # =====================================================================

        x = self.norm(x_after_script)
        T = x.shape[1]

        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)

        # CTC routing: iterate by group. Within each group, per-sample
        # script_ids are passed to the GroupCTCModule which handles
        # per-script head routing internally.
        for g in range(self.num_groups):
            # Skip without a sync using the pre-transferred group counts.
            if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                continue
            mask_g = (frame_groups == g)  # (B, T)
            batch_feats, batch_info = _collect_segments(x, mask_g)
            if batch_feats is None:
                continue

            # One script_id per segment (all frames in a group segment share
            # a script). Build batch_sids on GPU via gather: first-True
            # column per sample, then index frame_scripts. Single H2D
            # transfer for b_tensor; no per-sample .item() sync.
            b_tensor = torch.tensor(
                [b for b, _ in batch_info], device=x.device, dtype=torch.long)
            first_col = mask_g[b_tensor].int().argmax(dim=1)  # (N,)
            batch_sids = frame_scripts[b_tensor, first_col]

            seg_logits, _ = self.ctc_modules[g](batch_feats, script_ids=batch_sids)
            vs = seg_logits.shape[-1]
            for i, (b, sl) in enumerate(batch_info):
                logits[b, mask_g[b], :vs] = seg_logits[i, :sl].to(logits.dtype)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "group_ids": frame_groups,
            "lid2_logits_per_group": lid2_logits_per_group,
            "frame_scripts": frame_scripts,
            "flat_scripts": flat_scripts,
        }
