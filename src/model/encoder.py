"""
Lipi v5 MoE Vision Encoder (from scratch — no pretrained backbone).

Architecture:
    Input: (B, 3, 32, W) — RGB
    -> ConvStem: two plain strided convs (small RF ~5px) → (B, 128, 8, W/2)
    -> Shared SWA-A:   2× blocks at h=8, w=16 (local features)
    -> Patch-merge 8→4
    -> Shared SWA-B: 2× blocks at h=4, w=32 (character-level context)
    -> Patch-merge 4→2, proj 128→256
    -> Shared SWA-C:   2× blocks at h=2, w=64 (multi-char script context)
    -> LID-1 branch: pool h=2→1, lid1_attn (w=32), group classifier
    -> Patch-merge 2→1 (CTC path only)
    -> 1 local + 1 wide group expert block (15 experts)
    -> Per-group aggregation (concat local + wide → dim)
    -> LID-2: per-frame script classification (multi-script groups)
    -> 1 local + 1 wide script expert block (27 experts)
    -> Per-script aggregation (concat local + wide → dim)
    -> Per-script CTC heads (T=W/2)
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.model.blocks import (
    ConvStem, SWABlock, ExpertBlock, GroupCTCModule,
    _patch_merge_h, _per_sample_key_lens,
    _collect_segments, _scatter_segments, _run_expert_block,
)
from src.model.lid import NUM_GROUPS


class LipiMoEEncoder(nn.Module):
    """Lipi v5: ConvStem + shared SWA + LID-1 + group experts + LID-2 + script experts.

    Trained from scratch (no pretrained backbone). Small-RF stem keeps
    boundary contamination minimal before attention layers take over.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (15 + blank). A
         dedicated `lid1_attn` SWA block sits between shared_c and the
         classifier so LID-1 has its own capacity for script-family
         discrimination without forcing shared_c into a compromise
         between family and character features.
      2. Group expert blocks process frames per-group (local + wide streams)
      3. LID-2 classifies each frame into a script within its group
      4. Script expert blocks process frames per-script (local + wide streams)
      5. Per-script CTC heads decode characters

    Single-script groups skip LID-2 (only 1 script, trivially assigned).
    """

    def __init__(
        self,
        dim: int = 256,
        stem_out_ch: int = 128,
        num_shared_a_blocks: int = 2,
        num_shared_b_blocks: int = 2,
        num_shared_c_blocks: int = 2,
        num_group_local_blocks: int = 1,
        num_group_wide_blocks: int = 1,
        num_script_local_blocks: int = 1,
        num_script_wide_blocks: int = 1,
        shared_a_window_w: int = 16,
        shared_b_window_w: int = 32,
        shared_c_window_w: int = 64,
        local_window_w: int = 16,
        wide_window_w: int = 64,
        mlp_ratio: int = 4,
        shared_mlp_ratio: int = 4,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1.0,
        num_groups: int = NUM_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
        # Accepted for backward compat with checkpoints from the LID-0 era;
        # silently ignored when rolled back.
        num_super_groups: int | None = None,
    ):
        super().__init__()
        del num_super_groups  # unused (rolled back); kept as kwarg for ckpt replay
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
            "stem_out_ch": stem_out_ch,
            "num_shared_a_blocks": num_shared_a_blocks,
            "num_shared_b_blocks": num_shared_b_blocks,
            "num_shared_c_blocks": num_shared_c_blocks,
            "num_group_local_blocks": num_group_local_blocks,
            "num_group_wide_blocks": num_group_wide_blocks,
            "num_script_local_blocks": num_script_local_blocks,
            "num_script_wide_blocks": num_script_wide_blocks,
            "shared_a_window_w": shared_a_window_w,
            "shared_b_window_w": shared_b_window_w,
            "shared_c_window_w": shared_c_window_w,
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

        # Drop-path schedule: linearly increase from 0 → drop_path_rate
        # across all residual stages along a sample's path.
        # Stages: shared_a, shared_b, shared_c,
        # group (local/wide parallel), script (local/wide parallel).
        n_stages = (num_shared_a_blocks + num_shared_b_blocks
                    + num_shared_c_blocks
                    + max(num_group_local_blocks, num_group_wide_blocks)
                    + max(num_script_local_blocks, num_script_wide_blocks))
        dp_schedule = [drop_path_rate * i / max(n_stages - 1, 1)
                       for i in range(n_stages)]
        dp_iter = iter(dp_schedule)

        # Convolutional stem: (B, 3, 32, W) → (B, stem_out_ch, 8, W/2)
        self.stem = ConvStem(in_ch=3, out_ch=stem_out_ch)

        # Shared SWA-A at (h=8, w=W/2), dim=stem_out_ch.
        # Window 8×16: full vertical extent × local horizontal context.
        # Small LayerScale init (1e-4) gives near-identity at step 0
        # without zeroing any W_out — all internal weights receive
        # non-zero gradient immediately. Avoids the ReZero chicken-and-egg
        # where zero-init of mlp[-1] blocks gradient to mlp[0].
        self.shared_a = nn.ModuleList([
            SWABlock(dim=stem_out_ch, num_heads=max(stem_out_ch // 64, 1),
                     window_h=8, window_w=shared_a_window_w, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_shared_a_blocks)
        ])

        # Patch-merge (h=8 → 4): concat 2 adjacent rows, project.
        # Init as average of the two rows (near-identity).
        self._post_stem_h = 8
        self.merge_a = nn.Linear(stem_out_ch * 2, stem_out_ch)
        with torch.no_grad():
            self.merge_a.weight.zero_()
            self.merge_a.weight[:, :stem_out_ch] = 0.5 * torch.eye(stem_out_ch)
            self.merge_a.weight[:, stem_out_ch:] = 0.5 * torch.eye(stem_out_ch)
            self.merge_a.bias.zero_()

        # Shared SWA-B at (h=4, w=W/2), dim=stem_out_ch.
        # Window 4×32: full vertical × medium horizontal context.
        # Small LayerScale init (1e-4) → near-identity at step 0 while
        # every internal weight still receives gradient.
        self.shared_b = nn.ModuleList([
            SWABlock(dim=stem_out_ch, num_heads=max(stem_out_ch // 64, 1),
                     window_h=4, window_w=shared_b_window_w, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_shared_b_blocks)
        ])

        # Patch-merge (h=4 → 2): concat 2 adjacent rows, project to dim.
        # Init: each output dim gets average of corresponding dims from
        # the two input rows (zero-pads if out_dim > in_dim).
        self.merge_b = nn.Linear(stem_out_ch * 2, dim)
        with torch.no_grad():
            self.merge_b.weight.zero_()
            d_in = stem_out_ch
            d_out = dim
            d_copy = min(d_in, d_out)
            self.merge_b.weight[:d_copy, :d_copy] = 0.5 * torch.eye(d_copy)
            self.merge_b.weight[:d_copy, d_in:d_in + d_copy] = 0.5 * torch.eye(d_copy)
            self.merge_b.bias.zero_()

        # Shared SWA-C at (h=2, w=W/2), dim.
        # Window 2×64: full vertical × wide horizontal context for
        # multi-character script discrimination before LID-1.
        # Small LayerScale init (1e-4) → near-identity at step 0 while
        # every internal weight still receives gradient.
        self.shared_c = nn.ModuleList([
            SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                     window_h=2, window_w=shared_c_window_w, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_shared_c_blocks)
        ])

        # Patch-merge (h=2 → 1): concat 2 rows → project. Final collapse
        # to frame sequence before experts.
        self.merge_c = nn.Linear(dim * 2, dim)

        # Parallel stages share one drop-path rate per stage so local/wide
        # streams have matched residual scaling.
        group_dp = next(dp_iter) if max(num_group_local_blocks, num_group_wide_blocks) > 0 else 0.0
        script_dp = next(dp_iter) if max(num_script_local_blocks, num_script_wide_blocks) > 0 else 0.0

        # LID-1: per-frame group classification.
        #
        # Dedicated `lid1_attn` block sits between shared_c output (pooled
        # to h=1) and the classifier head. Window w=32 gives LID-1
        # horizontal context for script discrimination. Output projection
        # is zero-init so the residual starts as identity. Can pick up
        # multi-character script signal (e.g. punctuation-only fragments
        # use neighbor context). Output projection is zero-init so the
        # residual starts as identity — from a CTC-seeded checkpoint,
        # LID-1 keeps its existing accuracy and improves from there.
        self.group_h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lid1_attn = SWABlock(
            dim=dim, num_heads=max(dim // 64, 1),
            window_h=1, window_w=32, shift=False,
            mlp_ratio=mlp_ratio, drop_path=0.0,
            layer_scale_init=layer_scale_init,
        )
        # Near-identity init: zero the output projections so lid1_attn
        # starts as x + 0 = x. Preserves pre-rollback LID-1 behavior at
        # migration time.
        nn.init.zeros_(self.lid1_attn.attn.proj.weight)
        nn.init.zeros_(self.lid1_attn.attn.proj.bias)
        nn.init.zeros_(self.lid1_attn.mlp[-1].weight)
        nn.init.zeros_(self.lid1_attn.mlp[-1].bias)
        self.group_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_groups + 1),
        )

        # Group expert blocks (routed by group_id, 15 experts)
        # Init output projections near-zero so residual connections pass
        # features through initially — experts learn to specialize gradually
        # without destroying features that CTC needs.
        # Both streams run at h=1; local/wide differentiate via window_w.
        self.group_local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_groups,
                        window_h=1, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=group_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_group_local_blocks)
        ])
        self.group_wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_groups,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=group_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_group_wide_blocks)
        ])
        for block_list in [self.group_local_blocks, self.group_wide_blocks]:
            for block in block_list:
                for attn in block.expert_attns:
                    nn.init.zeros_(attn.proj.weight)
                    nn.init.zeros_(attn.proj.bias)
                for mlp in block.expert_mlps:
                    nn.init.zeros_(mlp.fc2.weight)
                    nn.init.zeros_(mlp.fc2.bias)

        # Group aggregation: concat local + wide → dim
        # Init as average of local+wide (near-identity)
        self.group_aggregates = nn.ModuleList()
        for _ in range(num_groups):
            agg = nn.Linear(dim * 2, dim)
            nn.init.zeros_(agg.bias)
            with torch.no_grad():
                agg.weight.zero_()
                agg.weight[:, :dim] = 0.5 * torch.eye(dim)
                agg.weight[:, dim:] = 0.5 * torch.eye(dim)
            self.group_aggregates.append(agg)

        # LID-2: per-frame script classification within multi-script groups
        # One head per multi-script group
        self.lid2_heads = nn.ModuleDict()
        for g in range(num_groups):
            n_scripts = len(group_script_vocab_sizes[g])
            if n_scripts > 1:
                self.lid2_heads[str(g)] = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, n_scripts),
                )

        # Script expert blocks (routed by flat script_id, 27 experts)
        # Initialize output projections near-zero so residual connections
        # pass features through initially (prevents randomly initialized
        # script experts from destroying group-expert features)
        self.script_local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64,
                        num_experts=self.total_scripts,
                        window_h=1, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=script_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_script_local_blocks)
        ])
        self.script_wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64,
                        num_experts=self.total_scripts,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=script_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_script_wide_blocks)
        ])
        for block_list in [self.script_local_blocks, self.script_wide_blocks]:
            for block in block_list:
                for attn in block.expert_attns:
                    nn.init.zeros_(attn.proj.weight)
                    nn.init.zeros_(attn.proj.bias)
                for mlp in block.expert_mlps:
                    nn.init.zeros_(mlp.fc2.weight)
                    nn.init.zeros_(mlp.fc2.bias)

        # Script aggregation: concat local + wide → dim
        # Initialize as near-identity (average of local+wide) so untrained
        # script experts pass features through without destroying them
        self.script_aggregates = nn.ModuleList()
        for _ in range(self.total_scripts):
            agg = nn.Linear(dim * 2, dim)
            nn.init.zeros_(agg.bias)
            with torch.no_grad():
                agg.weight.zero_()
                agg.weight[:, :dim] = 0.5 * torch.eye(dim)
                agg.weight[:, dim:] = 0.5 * torch.eye(dim)
            self.script_aggregates.append(agg)

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

    def _get_flat_script_ids(self, group_ids, script_ids):
        """Convert (group_id, local_script_id) pairs to flat script indices.
        Blank/whitespace frames (group_id == blank_group_id) get flat_id = -1.
        """
        flat = torch.full_like(group_ids, -1)
        for (g, s), f in self._flat_script_id.items():
            mask = (group_ids == g) & (script_ids == s)
            flat[mask] = f
        return flat

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

        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # Stem: (B, 3, 32, W) → (B, stem_out_ch, 8, W/2)
        x = self.stem(x)
        _, C, h, w = x.shape  # h=8, w=W/2

        # Shared SWA-A at (h=8, w=W/2)
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        for blk in self.shared_a:
            x = blk(x, h, w)

        # Patch-merge 8 → 4: concat 2 adjacent rows, project.
        assert h == self._post_stem_h, \
            f"expected post-stem height {self._post_stem_h}, got {h}"
        x, h = _patch_merge_h(x, h, w, self.merge_a)  # h=8→4

        # Shared SWA-B at (h=4, w=W/2)
        for blk in self.shared_b:
            x = blk(x, h, w)

        # Patch-merge 4 → 2: concat 2 adjacent rows, project to dim.
        x, h = _patch_merge_h(x, h, w, self.merge_b)  # h=4→2

        d = x.shape[-1]

        # Shared SWA-C at (h=2, w=W/2), dim
        for blk in self.shared_c:
            x = blk(x, h, w)

        # LID-1 branches off BEFORE the final 2→1 merge so LID-1 sees h=2
        # features (top+bottom half of each char) and merge_c only gets
        # CTC gradient. We pool h=2→1 for LID-1, then run `lid1_attn` to
        # give LID-1 its own horizontal-context capacity (window w=32).
        # The post-lid1_attn tensor feeds ONLY the classifier; merge_c /
        # experts / CTC use the pre-lid1_attn pooled features so they
        # aren't pulled toward script-family representation.
        x_for_group = x.reshape(B, h, w, d).permute(0, 3, 1, 2)  # (B, d, 2, w)
        x_for_group = self.group_h_pool(x_for_group).squeeze(2).permute(0, 2, 1)
        # lid1_attn output starts as identity (zero-init proj+mlp); the
        # classifier sees the same features it did pre-rollback, plus room
        # to grow its own context.
        x_for_group = self.lid1_attn(x_for_group, 1, w)
        group_logits = self.group_head(x_for_group)  # (B, W/2, num_groups+1)

        # Patch-merge 2 → 1: concat 2 rows, project to dim. Only touches
        # the CTC path from here on.
        x, h = _patch_merge_h(x, h, w, self.merge_c)

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

        x_after_group = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # One upfront sync: per-sample per-group frame counts.
        group_lens_cpu = _per_sample_key_lens(
            frame_groups, self.num_groups).tolist()

        for g in range(self.num_groups):
            # Skip without a sync: checks a pre-transferred Python list.
            if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                continue
            mask_g = (frame_groups == g)  # (B, w)
            batch_x, batch_info = _collect_segments(x, mask_g)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in self.group_local_blocks:
                local = _run_expert_block(block, local, g, 1, max_len)

            wide = batch_x
            for block in self.group_wide_blocks:
                wide = _run_expert_block(block, wide, g, 1, max_len)

            comb = torch.cat([local, wide], dim=-1)
            agg = self.group_aggregates[g](comb)
            _scatter_segments(x_after_group, agg, mask_g, batch_info)

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
        # =====================================================================

        x_after_script = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # One upfront sync: per-sample per-script frame counts. flat_scripts
        # uses -1 for blank/unrouted, which _per_sample_key_lens filters.
        script_lens_cpu = _per_sample_key_lens(
            flat_scripts, self.total_scripts).tolist()

        # Iterate by flat script-id (≤ total_scripts = 26) instead of
        # (B, unique_scripts). Same batching pattern as group experts.
        for s in range(self.total_scripts):
            if not any(script_lens_cpu[b][s] > 0 for b in range(B)):
                continue
            mask_s = (flat_scripts == s)  # (B, w)
            batch_x, batch_info = _collect_segments(x_after_group, mask_s)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in self.script_local_blocks:
                local = _run_expert_block(block, local, s, 1, max_len)

            wide = batch_x
            for block in self.script_wide_blocks:
                wide = _run_expert_block(block, wide, s, 1, max_len)

            comb = torch.cat([local, wide], dim=-1)
            agg = self.script_aggregates[s](comb)
            _scatter_segments(x_after_script, agg, mask_s, batch_info)

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
