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
    -> LID-1 branch: lid1_merge(2 rows + ConvB texture tap), lid1_attn
       (w=32), group_head
    -> merge h=2→1, Linear(768→384)                            ( 1, W/4, 384)
    -> Group MoE stack: N × (shared attn + 15 routed MLPs + shared MLP)
       (LID-2 heads tap the stack one block before the end)
    -> intermediate CTC + self-conditioning feedback (tied head weights)
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
import torch.nn.functional as F
from torch import Tensor

from src.model.blocks import (
    ConvStem, ConvNeXtBlock, BlurPool2d, LayerScale, SWABlock, MoELayer,
    GroupCTCModule, _patch_merge_h, _per_sample_key_lens, _dynamo_disable,
)
from src.taxonomy import NUM_GROUPS, SCRIPTS, SCRIPT_TO_ID


class LipiMoEEncoder(nn.Module):
    """Lipi v5: ConvStem + ConvNeXt(A,B) + SWA-C/D + LID-1 + MoE stacks.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (15 + blank).
         Its input is a private learned merge (`lid1_merge`) of the two
         SWA-D rows plus a stroke-texture tap from ConvB, and a
         dedicated `lid1_attn` SWA block sits before the classifier so
         LID-1 has its own capacity for script-family discrimination
         without forcing the CTC feature path into a family/character
         compromise.
      2. Group MoE stack: N stacked MoELayers with 15 routed MLPs each.
         Every frame passes through the same attention; its MLP is
         picked by group_id, and a shared MLP always runs alongside.
      3. LID-2 classifies each frame into a script within its group
         (multi-script groups only; single-script groups skip it).
         Its heads tap the group stack one block before the end — two
         expert blocks of family-specialized processing feed the
         fine-grained call, and the last group block plus the script
         stack run after the decision. The heads also read the same
         ConvB texture tap as LID-1 (stroke-level cues for the
         within-group confusable pairs).
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
        names = [name for group_names in group_script_names
                 for name in group_names]
        use_stable_global_ids = (len(names) == len(SCRIPTS)
                                 and set(names) == set(SCRIPTS))
        flat = 0
        for g, vs in enumerate(group_script_vocab_sizes):
            for s in range(len(vs)):
                # Full-taxonomy models use the global taxonomy ID. This keeps
                # every pre-split expert fixed and appends han_dense at 27,
                # even though it is local script 1 inside group 4.
                flat_id = (SCRIPT_TO_ID[group_script_names[g][s]]
                           if use_stable_global_ids else flat)
                self._flat_script_id[(g, s)] = flat_id
                self._flat_to_group_script[flat_id] = (g, s)
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
        # Branches off SWA-D output (at h=2) before merge_d1, plus a
        # texture tap from ConvB. Script ID is texture-like (vertical ink
        # profile, stroke curvature/loop statistics) — cues that are
        # strongest in early conv features and that the CTC-shaped deep
        # trunk is under no pressure to preserve. lid1_merge fuses
        # [row0, row1, ConvB texture] → dim: a learned h-merge private to
        # the LID branch (merge_d1 stays CTC-gradient-only) that keeps
        # the vertical profile a plain h-mean would average away.
        # Init: the two row blocks average the rows — exactly the old
        # h-mean, so step-0 behavior is unchanged — and the ConvB block
        # is zero, so the texture tap is a no-op that grows only if
        # useful. Then a dedicated lid1_attn block (window w=32) for
        # LID-1's own horizontal-context capacity, and a small MLP head.
        # lid1_attn keeps LayerScale at 1.0: its identity-at-init comes
        # from the zero-init projections below, and a near-zero
        # LayerScale on top would suppress its gradients ~1e4x.
        self.lid1_merge = nn.Linear(dim * 2 + convb_ch, dim)
        with torch.no_grad():
            self.lid1_merge.weight.zero_()
            self.lid1_merge.weight[:, :dim] = 0.5 * torch.eye(dim)
            self.lid1_merge.weight[:, dim:2 * dim] = 0.5 * torch.eye(dim)
            self.lid1_merge.bias.zero_()
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
        # Heads read [group-stack tap ‖ ConvB texture] — the within-group
        # calls (telugu/kannada, malayalam/tamil, NE-Indic) ride on
        # stroke-level cues that the CTC-shaped deep features are under
        # no pressure to keep, so the same texture tap that feeds LID-1
        # feeds these heads. Texture columns zero-init: the tap starts
        # silent and grows only if useful.
        self.lid2_heads = nn.ModuleDict()
        for g in range(num_groups):
            n_scripts = len(group_script_vocab_sizes[g])
            if n_scripts > 1:
                head = nn.Sequential(
                    nn.Linear(dim + convb_ch, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, n_scripts),
                )
                with torch.no_grad():
                    head[0].weight[:, dim:].zero_()
                self.lid2_heads[str(g)] = head

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

        # Self-conditioned CTC feedback gate (zero-init: feedback starts
        # as a no-op and grows only if useful; also makes warm-starting
        # from pre-self-cond checkpoints exact).
        self.self_cond_ls = LayerScale(dim, init_value=0.0)

        # Per-script CTC heads
        self.ctc_modules = nn.ModuleList([
            GroupCTCModule(
                enc_dim=dim,
                script_vocab_sizes=group_script_vocab_sizes[g],
                script_names=group_script_names[g],
            )
            for g in range(num_groups)
        ])

    @_dynamo_disable
    def _ctc_logits(self, x: Tensor, flat_scripts: Tensor,
                    script_lens_cpu: list) -> Tensor:
        """Dispatch per-script CTC heads over routed frames.

        Runs eager (torch._dynamo.disable): the loop skips scripts absent
        from the batch, so which heads fire is data-dependent, and each
        head's Linear has a distinct vocab (parameter) shape. Under compile
        that means a fresh recompile per batch composition — dozens, past
        the dynamo cache limit, silently falling back to eager anyway. The
        heads are cheap (one Linear per script at the tail of the forward);
        keeping them eager lets the trunk + MoE compile once and stay
        compiled. Autograd still flows through this region.

        CTC heads are position-wise Linear(dim → vocab), so frame order
        within a script is irrelevant: gather all routed frames once
        (grouped by flat script id), run each head on its contiguous
        slice, and scatter back with a single index_copy. The old
        per-script boolean-mask scatter (``logits[mask] = ...``) ran a
        full-size (B*T, max_vocab) masked_fill in backward once per
        script — 27 giant-tensor ops per call dominated the training
        step. index_copy's backward is one index_select.
        """
        B, T, D = x.shape
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        N = B * T
        logits = torch.zeros(N, max_vocab, device=x.device, dtype=x.dtype)

        # Per-flat-script frame counts (precomputed on CPU — no sync).
        counts = [sum(script_lens_cpu[b][f] for b in range(B))
                  for f in range(self.total_scripts)]
        if sum(counts) > 0:
            ff = flat_scripts.reshape(N)
            pos = ((ff >= 0) & (ff < self.total_scripts)).nonzero(
                as_tuple=True)[0]  # routed frames; must mirror counts' range
            pos = pos.index_select(0, torch.argsort(ff.index_select(0, pos)))
            xg = x.reshape(N, D).index_select(0, pos)  # grouped by script

            outs = []
            start = 0
            for (g, s), flat_id in self._flat_script_id.items():
                k = counts[flat_id]
                if k == 0:
                    continue
                head = self.ctc_modules[g].heads[s]
                y = head(xg[start:start + k]).to(logits.dtype)  # (k, vs)
                outs.append(F.pad(y, (0, max_vocab - head.vocab_size)))
                start += k
            logits = logits.index_copy(
                0, pos, torch.cat(outs) if len(outs) > 1 else outs[0])
        return logits.reshape(B, T, max_vocab)

    @_dynamo_disable
    def _ctc_feedback(self, inter_logits: Tensor, flat_scripts: Tensor,
                      script_lens_cpu: list) -> Tensor:
        """Project the intermediate CTC posterior back to feature space.

        Eager for the same reason as _ctc_logits (data-dependent per-script
        dispatch with distinct vocab shapes). The graph breaks here, before
        the script stack; the trunk and both MoE stacks still compile.

        Tied weights: each script's CTC head is Linear(dim → vocab) with
        weight (vocab, dim), so posterior @ weight maps the per-frame
        token distribution back to dim — per-script, zero new parameters.
        Blank/unrouted frames get zero feedback.

        Same grouped gather/scatter as _ctc_logits: one index_select of
        the routed rows, per-script compact softmax + matmul on contiguous
        slices, one index_copy back — instead of 27 full-size masked
        gathers/scatters whose backward dominated the step.
        """
        B, T, V = inter_logits.shape
        N = B * T
        fb = torch.zeros(N, self.enc_out_dim,
                         device=inter_logits.device, dtype=inter_logits.dtype)

        counts = [sum(script_lens_cpu[b][f] for b in range(B))
                  for f in range(self.total_scripts)]
        if sum(counts) > 0:
            ff = flat_scripts.reshape(N)
            pos = ((ff >= 0) & (ff < self.total_scripts)).nonzero(
                as_tuple=True)[0]
            pos = pos.index_select(0, torch.argsort(ff.index_select(0, pos)))
            lg = inter_logits.reshape(N, V).index_select(0, pos)  # (M, V)

            outs = []
            start = 0
            for (g, s), flat_id in self._flat_script_id.items():
                k = counts[flat_id]
                if k == 0:
                    continue
                head = self.ctc_modules[g].heads[s]
                vs = head.vocab_size
                post = lg[start:start + k, :vs].softmax(dim=-1)  # (k, vs)
                outs.append(
                    (post @ head.proj.weight.to(post.dtype)).to(fb.dtype))
                start += k
            fb = fb.index_copy(
                0, pos, torch.cat(outs) if len(outs) > 1 else outs[0])
        return fb.reshape(B, T, self.enc_out_dim)

    def _get_flat_script_ids(self, group_ids, script_ids):
        """Convert (group_id, local_script_id) pairs to flat script indices.
        Blank/whitespace frames (group_id == blank_group_id) get flat_id = -1.
        """
        flat = torch.full_like(group_ids, -1)
        for (g, s), f in self._flat_script_id.items():
            mask = (group_ids == g) & (script_ids == s)
            flat[mask] = f
        return flat

    def _run_backbone(self, images: Tensor) -> tuple[Tensor, Tensor, int, int]:
        """Run stem → ConvA → BlurPool → ConvB → BlurPool → proj → SWA-C
        → merge → SWA-D. Returns (x, tex, h, w) with x of shape
        (B, h*w, dim), h=2, w=T = W/4. The h=2 tensor feeds both the
        LID-1 branch and the final merge_d1 → CTC path. tex is
        (B, w, convb_ch): ConvB stroke-texture features pooled to the
        final frame grid for the LID-1 texture tap.
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

        # LID-1 texture tap: ConvB features pooled to the final frame
        # grid — h-mean over the 8 rows, width avg-pooled 2×. ceil_mode
        # matches blur_bc's stride-2 conv (k3, p1) width arithmetic
        # (both give ceil(w/2)), so tex width == T for any input width.
        tex = x.mean(dim=2)  # (B, convb_ch, W/2)
        tex = F.avg_pool1d(tex, kernel_size=2, stride=2, ceil_mode=True)
        tex = tex.transpose(1, 2)  # (B, W/4, convb_ch)

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

        return x, tex, h, w

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
        compute_ctc: bool = True,
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

        Self-conditioned CTC (always on, Nozaki & Komatsu 2021): the
        features after the group MoE stack are decoded through the same
        norm + CTC heads, and the resulting posterior is fed back into
        the feature stream through the transposed head weights (tied —
        no extra parameters) behind a zero-init LayerScale. The script
        stack then refines features that already carry the first pass's
        per-frame consensus. The intermediate logits also serve as the
        intermediate-CTC auxiliary target, returned as "inter_logits" in
        training mode only (the tensor is max_vocab-wide; eval skips it
        to keep memory flat).
        """
        B = images.shape[0]

        x, tex, h, w = self._run_backbone(images)
        # x here is post-SWA-D at (h=2, w=W/4). w is the final T.
        d = x.shape[-1]

        # LID-1 branch: lid1_merge fuses the two SWA-D rows (learned
        # h-merge, avg-init — see __init__) with the ConvB texture tap,
        # then lid1_attn (window w=32) and the classifier. Uses the
        # pre-merge_d1 tensor so merge_d1 only ever sees CTC gradient.
        # Row layout matches _patch_merge_h: [row0 chans, row1 chans].
        x_rows = x.reshape(B, h, w, d).permute(0, 2, 1, 3).reshape(B, w, h * d)
        x_for_group = self.lid1_merge(
            torch.cat([x_rows, tex.to(x_rows.dtype)], dim=-1))
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

        # LID-2 taps the group stack one block before the end: two expert
        # blocks of family-specialized processing feed the fine-grained
        # script call, and the last block keeps refining CTC features
        # after the decision is made. Stacks with <3 layers degenerate to
        # tapping the stack output.
        lid2_tap = max(len(self.group_layers) - 2, 0)
        x_lid2 = x
        for i, layer in enumerate(self.group_layers):
            x = layer(x, frame_groups, group_lens_cpu, w)
            if i == lid2_tap:
                x_lid2 = x

        # =====================================================================
        # LID-2: per-frame script classification
        # =====================================================================

        # Collect LID-2 logits for multi-script groups. Heads read the
        # group-stack tap concat the ConvB texture tap (see __init__).
        x_lid2_in = torch.cat([x_lid2, tex.to(x_lid2.dtype)], dim=-1)
        lid2_logits_per_group = {}  # g → (B, T, n_scripts)
        for g_str, head in self.lid2_heads.items():
            g = int(g_str)
            lid2_logits_per_group[g] = head(x_lid2_in)  # (B, T, n_scripts)

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

        # ── Self-conditioned CTC ──────────────────────────────────────
        # First CTC pass on the pre-script-stack features (shared norm +
        # heads), posterior fed back through the tied head weights. The
        # script stack refines features carrying this first-pass
        # consensus; the final CTC below is the second, refined pass.
        inter_logits = self._ctc_logits(
            self.norm(x), flat_scripts, script_lens_cpu)
        x = x + self.self_cond_ls(
            self._ctc_feedback(inter_logits, flat_scripts, script_lens_cpu))

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

        # Intermediate logits are only needed for the auxiliary loss;
        # skip returning the max_vocab-wide tensor at eval to keep
        # inference memory flat.
        if self.training:
            out["inter_logits"] = inter_logits

        return out
