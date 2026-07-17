"""
Lipi v6 MoE Vision Encoder (from scratch — no pretrained backbone).

Backbone (2× width downsample overall, so T = W/4; heights for the 64px
config, 32px runs the same graph at half the row counts):

    Input: (B, 3, 64, W)
    -> ConvStem:  PixelUnshuffle(2) → 2× conv3×3 s1, 12→64→96  (32, W/2, 96)
       (lossless first octave — no fixed low-pass touches raw pixels)
    -> ConvA:     3× ConvNeXt (dw7×7 + pw MLP), ch 96          (32, W/2, 96)
    -> BlurPool s(2,1), 96→160                                 (16, W/2, 160)
    -> ConvB:     3× ConvNeXt, ch 160                          (16, W/2, 160)
       (texture tap: 4 vertical bands × 2 width sub-positions → LID + heads)
    -> BlurPool s(2,2), 160→256  (second width stride)         ( 8, W/4, 256)
    -> proj 256→256
    -> SWA-C:     3× SWA block, dim 256, window 8×16           ( 8, W/4, 256)
       (kept as the re-look grid)
    -> AttentionReadout 8→2 rows, 256→384 (learned queries,
       row embeddings; ≈ mean-pool at step 0)                  ( 2, W/4, 384)
    -> SWA-D:     3× SWA block, dim 384, window 2×64           ( 2, W/4, 384)
    -> LID-1 branch: lid1_merge(2 rows + texture tap), lid1_attn
       (w=32), group_head
    -> Group MoE stack at h=2: N × (shared attn + 14 routed MLPs +
       shared MLP); LID-2 heads tap one block before the end
    -> RoutedReadout 2→1: per-script queries collapse the rows at the
       LID-2 boundary — the first script-conditioned vertical decision
    -> intermediate CTC + self-conditioning feedback (tied head weights)
    -> Script MoE stack: N × (shared attn + 27 routed MLPs + shared MLP),
       with a ReLook back into the SWA-C grid after the first layer
    -> Per-script CTC heads, 2 tokens per frame (output length 2·W/4)

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
    GroupCTCModule, AttentionReadout, RoutedReadout, ReLook,
    _per_sample_key_lens, _dynamo_disable,
)
from src.taxonomy import NUM_GROUPS


class LipiMoEEncoder(nn.Module):
    """Lipi v6: ConvStem + ConvNeXt(A,B) + SWA-C/D + LID-1 + MoE stacks.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (14 + blank).
         Its input is a private learned merge (`lid1_merge`) of the two
         SWA-D rows plus a stroke-texture tap from ConvB, and a
         dedicated `lid1_attn` SWA block sits before the classifier so
         LID-1 has its own capacity for script-family discrimination
         without forcing the CTC feature path into a family/character
         compromise.
      2. Group MoE stack at h=2: N stacked MoELayers with 14 routed MLPs
         each. Every frame passes through the same attention; its MLP is
         picked by group_id, and a shared MLP always runs alongside.
         Rows stay separate so no script-agnostic layer ever makes an
         irreversible vertical decision.
      3. LID-2 classifies each frame into a script within its group
         (multi-script groups only; single-script groups skip it).
         Its heads tap the group stack one block before the end and read
         both rows plus the ConvB texture tap (stroke-level cues for the
         within-group confusable pairs).
      4. RoutedReadout collapses the two rows into one frame with the
         routed script's own learned query — the vertical collapse
         happens exactly where script identity becomes known.
      5. Script MoE stack: N stacked MoELayers with 27 routed MLPs each,
         routed by flat script id, with a ReLook after the first layer
         that retrieves raw stroke rows from the SWA-C grid.
      6. Per-script CTC heads decode characters at 2 tokens per frame.

    Time downsampling: overall W is downsampled by 4× (stem s=2 then
    BlurPool s=2 on width) so there are T = W/4 frames; each frame emits
    `emit_per_frame` CTC tokens, so logits/lengths run at T·emit_per_frame.
    Both are class attributes so callers can compute frame offsets and
    emission lengths without hard-coding factors.
    """

    time_downsample = 4  # imgs W / time_downsample = T (frames)
    emit_per_frame = 2   # CTC emission slots per frame (logits len = 2T)
    tex_bands = 4        # vertical bands in the ConvB texture tap

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
        num_script_layers: int = 4,
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
        in_height: int = 48,
        **unused_kwargs,  # swallow stale kwargs from old checkpoints
    ):
        super().__init__()
        # Silently drop any unknown kwargs — old checkpoint configs may still
        # carry names like num_group_local_blocks / num_super_groups / etc.
        del unused_kwargs
        # The trunk's vertical path is: stem /2 → blur_ab /2 → blur_bc /2 =
        # /8 total, giving h_c = in_height//8 rows at SWA-C, then an
        # AttentionReadout (learned queries) collapses h_c → 2 for any h_c.
        # So the only real constraint is in_height % 8 == 0 (and each strided
        # stage input even, which follows). 48 (h_c=6) tiles fine; the old
        # {32, 64} guard was conservative.
        if in_height % 8 != 0 or in_height < 16:
            raise ValueError(
                f"in_height must be a multiple of 8 (≥16), got {in_height} — "
                "the stem+2 BlurPools downsample by 8 to the SWA-C grid.")
        self.in_height = in_height
        self.num_groups = num_groups
        self.blank_group_id = num_groups

        if group_script_vocab_sizes is None:
            group_script_vocab_sizes = [[100]] * num_groups
        if group_script_names is None:
            group_script_names = [[f"s{i}" for i in range(len(vs))]
                                  for vs in group_script_vocab_sizes]

        # Build flat script index: (group, local_script) → flat_id.
        # Flat IDs are plain group/local flatten order; since taxonomy v3,
        # SCRIPT_TO_ID follows the same order, so full-taxonomy models get
        # the global taxonomy ID for free (tests pin the invariant).
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
        # Grouped CTC dispatch consumes frames sorted ascending by flat id,
        # so heads must be visited in that order. Flatten order already is
        # ascending; the sort is cheap insurance against future reordering.
        self._heads_in_flat_order = sorted(
            self._flat_script_id.items(), key=lambda item: item[1])
        # (group, local) → flat LUT for _get_flat_script_ids. Extra sentinel
        # row (blank group) and column (out-of-range local) hold -1 so any
        # id outside the real pairs routes to "unrouted", matching the old
        # masked-write semantics.
        max_local = max(len(vs) for vs in group_script_vocab_sizes)
        lut = torch.full((len(group_script_vocab_sizes) + 1, max_local + 1),
                         -1, dtype=torch.long)
        for (g, s), f in self._flat_script_id.items():
            lut[g, s] = f
        self.register_buffer("_flat_id_lut", lut, persistent=False)

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
            "in_height": in_height,
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

        # ── BlurPool s(2,2): 160→256 at (16, W/2) → (8, W/4) ─────────
        # Second (and last) width stride happens here.
        self.blur_bc = BlurPool2d(convb_ch, swac_in_ch, stride=(2, 2))

        # Rows on the SWA-C grid: 8 at in_height=64, 4 at 32. The grid is
        # kept alive after SWA-C as the ReLook's key/value source.
        self.h_c = in_height // 8

        # ── SWA-C entry projection: swac_in_ch → swac_dim ─────────────
        # Kept as a 1×1 conv (channel-last equivalent) so the SWA-C blocks
        # can run at swac_dim even though BlurPool outputs swac_in_ch.
        self.swac_in_proj = nn.Linear(swac_in_ch, swac_dim)

        # ── SWA-C: 3× SWA @ swac_dim on the full (h_c, W/4) grid ─────
        # Window area stays 128 tokens across heights: 8×16 at 64px,
        # 4×32 at 32px. The finest post-stride grid gets real processing
        # before any vertical compression — there is no naked resolution
        # level in the trunk.
        swa_c_win_w = swa_c_window_w * 4 // self.h_c
        self.swa_c = nn.ModuleList([
            SWABlock(dim=swac_dim, num_heads=max(swac_dim // 64, 1),
                     window_h=self.h_c, window_w=swa_c_win_w,
                     shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_swa_c_blocks)
        ])

        # ── AttentionReadout h_c→2 rows, swac_dim → dim ───────────────
        # Learned queries + row embeddings replace the fixed pairwise
        # linear merges: the collapse is content-adaptive (it can find
        # the text band under baseline wander) and starts as plain row
        # mean-pooling (near-uniform attention + identity v/out init).
        self.readout_cd = AttentionReadout(
            swac_dim, dim, n_rows=self.h_c, n_queries=2,
            num_heads=max(swac_dim // 64, 1))

        # ── SWA-D: 3× SWA @ dim at (2, W/4), window 2×64 ─────────────
        self.swa_d = nn.ModuleList([
            SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                     window_h=2, window_w=swa_d_window_w, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=1e-4)
            for i in range(num_swa_d_blocks)
        ])

        # Each MoE layer consumes one entry from the drop-path schedule.
        # Same order as construction below: group layers first, then script.
        group_dps = [next(dp_iter) for _ in range(num_group_layers)]
        script_dps = [next(dp_iter) for _ in range(num_script_layers)]

        # ── LID-1: per-frame group classification ─────────────────────
        # Branches off SWA-D output (at h=2), plus a texture tap from
        # ConvB. Script ID is texture-like (vertical ink profile, stroke
        # curvature/loop statistics) — cues that are strongest in early
        # conv features and that the CTC-shaped deep trunk is under no
        # pressure to preserve. The tap keeps 4 vertical bands × 2 width
        # sub-positions per frame (a plain h-mean would average away the
        # vertical profile the tap exists to carry; the width pair holds
        # sub-frame stroke order for the 2-token emission slots).
        # lid1_merge fuses [row0, row1, tex] → dim. Init: the two row
        # blocks average the rows and the tex block is zero, so the tap
        # starts silent and grows only if useful. Then a dedicated
        # lid1_attn block (window w=32) for LID-1's own horizontal-context
        # capacity, and a small MLP head. lid1_attn keeps LayerScale at
        # 1.0: its identity-at-init comes from the zero-init projections
        # below, and a near-zero LayerScale on top would suppress its
        # gradients ~1e4x.
        self.tex_dim = 2 * self.tex_bands * convb_ch
        self.lid1_merge = nn.Linear(dim * 2 + self.tex_dim, dim)
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

        # ── Group MoE stack at h=2 (routed by group_id) ───────────────
        # N stacked MoELayers, alternating local/wide window widths and
        # shifts. Each layer: shared attention + 14 routed MLPs (one per
        # group) + shared MLP (always on). Runs on both rows (window_h=2,
        # per-column ids repeated across rows) so vertical structure
        # survives until script identity is known.
        self.group_layers = nn.ModuleList([
            MoELayer(
                dim=dim, num_heads=max(dim // 64, 1), num_experts=num_groups,
                window_w=(local_window_w if i % 2 == 0 else wide_window_w),
                shift=(i % 2 == 1), window_h=2,
                routed_mlp_ratio=mlp_ratio,
                shared_mlp_ratio=moe_shared_mlp_ratio,
                drop_path=group_dps[i],
                layer_scale_init=layer_scale_init,
            )
            for i in range(num_group_layers)
        ])

        # ── LID-2: per-frame script classification (multi-script groups)
        # Heads read [both group-stack rows ‖ ConvB texture] — the
        # within-group calls (telugu/kannada, malayalam/tamil, NE-Indic)
        # ride on vertical-position and stroke-level cues that the
        # CTC-shaped deep features are under no pressure to keep.
        # Texture columns zero-init: the tap starts silent and grows
        # only if useful.
        self.lid2_heads = nn.ModuleDict()
        for g in range(num_groups):
            n_scripts = len(group_script_vocab_sizes[g])
            if n_scripts > 1:
                head = nn.Sequential(
                    nn.Linear(dim * 2 + self.tex_dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, n_scripts),
                )
                with torch.no_grad():
                    head[0].weight[:, dim * 2:].zero_()
                self.lid2_heads[str(g)] = head

        # ── RoutedReadout: script-conditioned collapse h=2 → 1 ────────
        # The final vertical decision, made by the routed script's own
        # query at the LID-2 boundary. Blank/unrouted frames use the
        # shared default query. Starts as the row mean (≡ old merge_d1).
        self.routed_collapse = RoutedReadout(
            dim, self.total_scripts, num_heads=max(dim // 64, 1))

        # ── Script MoE stack (routed by flat script_id) ───────────────
        # Same shape as group_layers but with total_scripts routed MLPs
        # per layer (27 for the full taxonomy), at h=1 post-collapse.
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

        # ReLook: after the first script layer, each (routed) frame
        # cross-attends back into its column's SWA-C rows — a second look
        # at near-raw stroke evidence taken after the model knows what
        # script it's reading. Zero-init output proj → exact no-op at
        # step 0.
        self.relook = ReLook(dim, swac_dim, n_rows=self.h_c,
                             num_heads=max(dim // 64, 1))

        # Output
        self.enc_out_dim = dim
        self.norm = nn.LayerNorm(dim)

        # Self-conditioned CTC feedback gate (zero-init: feedback starts
        # as a no-op and grows only if useful).
        self.self_cond_ls = LayerScale(dim, init_value=0.0)

        # Per-script CTC heads (emit_per_frame tokens per frame each)
        self.ctc_modules = nn.ModuleList([
            GroupCTCModule(
                enc_dim=dim,
                script_vocab_sizes=group_script_vocab_sizes[g],
                script_names=group_script_names[g],
                emit_per_frame=self.emit_per_frame,
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

        CTC heads are position-wise Linear(dim → E·vocab), so frame order
        within a script is irrelevant: gather all routed frames once
        (grouped by flat script id), run each head on its contiguous
        slice, and scatter back with a single index_copy. The old
        per-script boolean-mask scatter (``logits[mask] = ...``) ran a
        full-size (B*T, max_vocab) masked_fill in backward once per
        script — 27 giant-tensor ops per call dominated the training
        step. index_copy's backward is one index_select.

        Returns (B, T·emit_per_frame, max_vocab): each frame's E emission
        slots are interleaved along time (frame t owns slots E·t…E·t+E-1).
        """
        B, T, D = x.shape
        E = self.emit_per_frame
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        N = B * T
        logits = torch.zeros(N, E, max_vocab, device=x.device, dtype=x.dtype)

        # Per-flat-script frame counts (precomputed on CPU — no sync).
        counts = [sum(script_lens_cpu[b][f] for b in range(B))
                  for f in range(self.total_scripts)]
        M = sum(counts)
        if M > 0:
            pos = self._routed_positions(flat_scripts.reshape(N), M)
            xg = x.reshape(N, D).index_select(0, pos)  # grouped by script

            # Write each head's slice into one preallocated buffer —
            # avoids a per-script F.pad alloc plus the final cat.
            yg = torch.zeros(M, E, max_vocab,
                             device=x.device, dtype=logits.dtype)
            start = 0
            for (g, s), flat_id in self._heads_in_flat_order:
                k = counts[flat_id]
                if k == 0:
                    continue
                head = self.ctc_modules[g].heads[s]
                yg[start:start + k, :, :head.vocab_size] = \
                    head(xg[start:start + k]).to(logits.dtype)
                start += k
            logits = logits.index_copy(0, pos, yg)
        return logits.reshape(B, T * E, max_vocab)

    def _routed_positions(self, ff: Tensor, M: int) -> Tensor:
        """Flat positions of routed frames, grouped by flat script id.

        M (the routed-frame count) is known from CPU-side counts, so
        invalid frames are sorted past position M instead of being
        selected with nonzero() — which forces a GPU→CPU sync per call.
        """
        key = torch.where((ff >= 0) & (ff < self.total_scripts),
                          ff, ff.new_full((), self.total_scripts))
        return torch.argsort(key, stable=True)[:M]

    @_dynamo_disable
    def _ctc_feedback(self, inter_logits: Tensor, flat_scripts: Tensor,
                      script_lens_cpu: list) -> Tensor:
        """Project the intermediate CTC posterior back to feature space.

        Eager for the same reason as _ctc_logits (data-dependent per-script
        dispatch with distinct vocab shapes). The graph breaks here, before
        the script stack; the trunk and both MoE stacks still compile.

        Tied weights: each script's CTC head is Linear(dim → E·vocab)
        with weight (E·vocab, dim); viewing it (E, vocab, dim) and summing
        each emission slot's posterior through its own weight block maps
        the per-frame token distributions back to dim — position-aware
        feedback, per-script, zero new parameters. Blank/unrouted frames
        get zero feedback.

        Same grouped gather/scatter as _ctc_logits: one index_select of
        the routed rows, per-script compact softmax + matmul on contiguous
        slices, one index_copy back — instead of 27 full-size masked
        gathers/scatters whose backward dominated the step.
        """
        B, TE, V = inter_logits.shape
        E = self.emit_per_frame
        T = TE // E
        N = B * T
        fb = torch.zeros(N, self.enc_out_dim,
                         device=inter_logits.device, dtype=inter_logits.dtype)

        counts = [sum(script_lens_cpu[b][f] for b in range(B))
                  for f in range(self.total_scripts)]
        M = sum(counts)
        if M > 0:
            pos = self._routed_positions(flat_scripts.reshape(N), M)
            lg = inter_logits.reshape(N, E, V).index_select(0, pos)

            outs = []
            start = 0
            for (g, s), flat_id in self._heads_in_flat_order:
                k = counts[flat_id]
                if k == 0:
                    continue
                head = self.ctc_modules[g].heads[s]
                vs = head.vocab_size
                post = lg[start:start + k, :, :vs].softmax(dim=-1)  # (k,E,vs)
                w = head.proj.weight.view(E, vs, -1).to(post.dtype)
                outs.append(
                    torch.einsum("kev,evd->kd", post, w).to(fb.dtype))
                start += k
            fb = fb.index_copy(
                0, pos, torch.cat(outs) if len(outs) > 1 else outs[0])
        return fb.reshape(B, T, self.enc_out_dim)

    def _smooth_routing(self, fg: Tensor) -> Tensor:
        """Inference-time routing diffusion (eval/deploy only, never
        training): absorb short spurious runs into identical flanks.

        Cost asymmetry motivates this: a text frame misrouted to blank
        gets all-zero logits → argmax=0 → forced CTC blank → silent
        character deletion, while a genuine space frame routed to a
        script is harmless (that script's head can still emit blank).
        Rules, two passes so shrinking runs collapse fully:
          - any length-1 island between identical flanks → absorbed
            (han han BLANK han → han; also fixes wrong-group islands)
          - length-2 blank runs between identical non-blank flanks →
            absorbed
        """
        for _ in range(2):
            if fg.shape[1] < 3:
                break
            left, mid, right = fg[:, :-2], fg[:, 1:-1], fg[:, 2:]
            iso = (mid != left) & (left == right)
            out = fg.clone()
            out[:, 1:-1] = torch.where(iso, left, mid)
            fg = out
            if fg.shape[1] >= 4:
                a = fg[:, 1:-2]
                b = fg[:, 2:-1]
                l2, r2 = fg[:, :-3], fg[:, 3:]
                pair = ((a == self.blank_group_id)
                        & (b == self.blank_group_id)
                        & (l2 == r2) & (l2 != self.blank_group_id))
                out = fg.clone()
                out[:, 1:-2] = torch.where(pair, l2, a)
                out[:, 2:-1] = torch.where(pair, l2, b)
                fg = out
        return fg

    def _get_flat_script_ids(self, group_ids, script_ids):
        """Convert (group_id, local_script_id) pairs to flat script indices.
        Blank/whitespace frames (group_id == blank_group_id) get flat_id = -1.

        One LUT gather instead of a masked write per (group, script) pair
        (~26 pairs × 3 kernels each per call, twice per step).
        """
        lut = self._flat_id_lut
        g = group_ids.clamp(0, lut.shape[0] - 1)
        s = script_ids.clamp(0, lut.shape[1] - 1)
        return torch.where(group_ids >= 0, lut[g, s],
                           group_ids.new_full((), -1))

    def _run_backbone(
            self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Run stem → ConvA → BlurPool → ConvB → BlurPool → proj → SWA-C
        → AttentionReadout → SWA-D. Returns (x, tex, grid, h, w):
          x:    (B, 2*w, dim) h-major, h=2, w=T = W/4 — feeds the LID-1
                branch and the group stack.
          tex:  (B, w, tex_dim) ConvB stroke texture, 4 vertical bands ×
                2 width sub-positions per frame (sub-position-major).
          grid: (B, h_c*w, swac_dim) the post-SWA-C rows, kept alive as
                the ReLook key/value source.
        """
        B = images.shape[0]
        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # Stem (lossless): (B, 3, H, W) → (B, stem_out_ch, H/2, W/2)
        x = self.stem(x)

        # ConvA: 3× ConvNeXt in NCHW at (H/2, W/2)
        for blk in self.convA:
            x = blk(x)

        # BlurPool s(2,1): (H/2, W/2) → (H/4, W/2)
        x = self.blur_ab(x)

        # ConvB: 3× ConvNeXt at (H/4, W/2)
        for blk in self.convB:
            x = blk(x)

        # Texture tap: pool ConvB rows into tex_bands vertical bands and
        # pair adjacent columns (the two width sub-positions of each
        # output frame) — vertical position and sub-frame stroke order
        # both survive, unlike the old h-mean + width average. Column
        # pairing matches blur_bc's stride-2 width arithmetic (ceil(Wc/2))
        # via replicate-pad when Wc is odd.
        Bc, Cc, hb, Wc = x.shape
        tex = F.avg_pool2d(x, kernel_size=(hb // self.tex_bands, 1))
        if Wc % 2:
            tex = F.pad(tex, (0, 1), mode="replicate")
        tex = tex.reshape(Bc, Cc, self.tex_bands, -1, 2)   # (B,C,4,T,2)
        tex = tex.permute(0, 3, 4, 2, 1).reshape(Bc, -1, self.tex_dim)

        # BlurPool s(2,2): (H/4, W/2) → (h_c, W/4)
        x = self.blur_bc(x)

        # SWA-C: switch to channel-last. (B, C, h_c, w) → (B, h_c*w, C).
        _, C, h, w = x.shape  # h = h_c (8 at 64px, 4 at 32px), w = W/4
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)

        x = self.swac_in_proj(x)  # C → swac_dim
        for blk in self.swa_c:
            x = blk(x, h, w)
        grid = x  # ReLook keys/values: the finest post-stride rows

        # AttentionReadout: (h_c, w) → (2, w), swac_dim → dim
        x = self.readout_cd(x, h, w)
        h = 2

        # SWA-D at (2, W/4), dim
        for blk in self.swa_d:
            x = blk(x, h, w)

        return x, tex, grid, h, w

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
        compute_ctc: bool = True,
        route_sample_p: float = 0.0,
        route_smooth: bool = False,
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
        if images.shape[2] != self.in_height:
            raise ValueError(
                f"Input height {images.shape[2]} != model in_height "
                f"{self.in_height} — the data and model heights must match "
                "(regenerate shards or rebuild the model).")

        x, tex, grid, h, w = self._run_backbone(images)
        # x here is post-SWA-D at (h=2, w=W/4). w is the final T.
        d = x.shape[-1]

        # LID-1 branch: lid1_merge fuses the two SWA-D rows (learned
        # h-merge, avg-init — see __init__) with the ConvB texture tap,
        # then lid1_attn (window w=32) and the classifier.
        # Row layout: h-major, so [row0 chans, row1 chans] per column.
        x_rows = x.reshape(B, h, w, d).permute(0, 2, 1, 3).reshape(B, w, h * d)
        x_for_group = self.lid1_merge(
            torch.cat([x_rows, tex.to(x_rows.dtype)], dim=-1))
        x_for_group = self.lid1_attn(x_for_group, 1, w)
        group_logits = self.group_head(x_for_group)  # (B, w, num_groups+1)

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
            if route_smooth:
                frame_groups = self._smooth_routing(frame_groups)

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
        # STAGE 1: Group MoE stack at h=2.
        # Each MoELayer runs shared attention on both rows, then applies
        # per-frame routed MLPs (indexed by frame_groups, repeated across
        # rows) plus a shared MLP. Blank frames (group_id >= num_groups)
        # still get attention + shared MLP; only the routed MLP is skipped.
        # =====================================================================

        # Per-column ids repeated across both rows (h-major layout).
        groups_h2 = torch.cat([frame_groups, frame_groups], dim=1)

        # One upfront sync: per-sample per-group frame counts (at h=2
        # granularity — every count is 2× the column count). Reused
        # across every group layer so the sync-free expert skip in
        # MoELayer._routed_mlp costs zero per-layer.
        group_lens_cpu = _per_sample_key_lens(
            groups_h2, self.num_groups).tolist()

        # LID-2 taps the group stack one block before the end: two expert
        # blocks of family-specialized processing feed the fine-grained
        # script call, and the last block keeps refining CTC features
        # after the decision is made. Stacks with <3 layers degenerate to
        # tapping the stack output.
        lid2_tap = max(len(self.group_layers) - 2, 0)
        x_lid2 = x
        for i, layer in enumerate(self.group_layers):
            x = layer(x, groups_h2, group_lens_cpu, w, h=2)
            if i == lid2_tap:
                x_lid2 = x

        # =====================================================================
        # LID-2: per-frame script classification
        # =====================================================================

        # Collect LID-2 logits for multi-script groups. Heads read both
        # rows of the group-stack tap concat the ConvB texture tap.
        x_lid2_rows = x_lid2.reshape(B, 2, w, d).permute(
            0, 2, 1, 3).reshape(B, w, 2 * d)
        x_lid2_in = torch.cat([x_lid2_rows, tex.to(x_lid2_rows.dtype)], dim=-1)
        lid2_logits_per_group = {}  # g → (B, T, n_scripts)
        for g_str, head in self.lid2_heads.items():
            g = int(g_str)
            lid2_logits_per_group[g] = head(x_lid2_in)  # (B, T, n_scripts)

        # Determine per-frame script assignments
        # frame_scripts: (B, T) — local script_id within each frame's group
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
            frame_scripts = torch.zeros(B, w, dtype=torch.long,
                                        device=x.device)
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
            T_e = w * self.emit_per_frame
            max_vocab = max(m.max_vocab for m in self.ctc_modules)
            return {
                "logits": torch.zeros(B, T_e, max_vocab,
                                      device=x.device, dtype=x.dtype),
                "lengths": torch.full((B,), T_e, dtype=torch.long,
                                      device=x.device),
                "group_logits": group_logits,
                "group_ids": frame_groups,
                "lid2_logits_per_group": lid2_logits_per_group,
                "frame_scripts": frame_scripts,
                "flat_scripts": flat_scripts,
            }

        # =====================================================================
        # RoutedReadout: script-conditioned vertical collapse h=2 → 1.
        # The rows survive every script-agnostic stage; the collapse
        # happens here, where the routed script's own query decides how
        # its frames weight the rows. Blank/unrouted frames use the
        # shared default query (≈ row mean at init either way).
        # =====================================================================
        x = self.routed_collapse(x, flat_scripts, 2, w)

        # =====================================================================
        # STAGE 2: Script MoE stack (h=1).
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

        for i, layer in enumerate(self.script_layers):
            x = layer(x, flat_scripts, script_lens_cpu, w)
            if i == 0:
                # ReLook: routed frames take a second look at the raw
                # stroke rows of their own column in the SWA-C grid —
                # retrieval after routing, zero-init (no-op at step 0).
                x = x + self.relook(x, grid.to(x.dtype), self.h_c, w)

        # =====================================================================
        # CTC heads (per-frame routing via grouped gather/scatter)
        # =====================================================================

        x = self.norm(x)

        logits = self._ctc_logits(x, flat_scripts, script_lens_cpu)

        lengths = torch.full((B,), w * self.emit_per_frame,
                             dtype=torch.long, device=x.device)

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
