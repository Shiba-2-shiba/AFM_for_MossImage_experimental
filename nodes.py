import json
import logging

from comfy_api.latest import ComfyExtension, io

from .afm.adapter import install_afm
from .afm.config import AFMConfig
from .afm.residual import ResidualConfig, ResidualTestConfig, install_residual
from .afm.k_balance import install_k_balance, parse_k_blocks
from .afm.k_balance_math import KBalanceConfig, STRENGTH_LIMIT, MAX_CHANGE_LIMIT


class MossImageAFM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImageAFM", display_name="MOSS Image AFM",
            category="model/patches/moss_image",
            description=("Native MOSS-Image向けAFM。0-2: 全体の改善。"
                         "24-26: 感情表現などの強調。30-32: ポーズなどの変化。"
                         "各区間で一つを選択します。2区間モードは重複時に第1区間を優先します。"),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=["off", "observe", "edit"], default="observe"),
                io.Float.Input("strength", default=0.05, min=-10, max=10, step=0.01),
                io.Combo.Input("block_scope", options=["0-2", "24-26", "30-32"], default="0-2",
                               display_name="効果",
                               tooltip="0-2: 全体の改善。\n24-26: 感情表現などの強調。\n"
                                       "30-32: ポーズなどの変化。\nいずれか一つを選択します。"),
                io.Combo.Input("branch_mode", options=["positive_only", "negative_only", "both"],
                               default="positive_only"),
                io.Float.Input("start_percent", default=0, min=0, max=1, step=0.01,
                               advanced=True),
                io.Float.Input("end_percent", default=1, min=0, max=1, step=0.01,
                               advanced=True),
                io.Float.Input("cutoff", default=0.25, min=0.001, max=0.999, step=0.01,
                               advanced=True),
                io.Boolean.Input("soft_mask", default=True, advanced=True),
                io.Float.Input("mask_width", default=0.05, min=0.001, max=1, step=0.01,
                               advanced=True),
                io.Boolean.Input("preserve_dc", default=True, advanced=True),
                io.Int.Input("head_chunk_size", default=1, min=1, max=256, advanced=True),
                io.Int.Input("query_chunk_size", default=128, min=1, max=8192, advanced=True),
                io.Float.Input("max_workspace_mib", default=512, min=1, max=32768,
                               advanced=True),
                io.Combo.Input("fail_mode", options=["raise", "fallback_original"],
                               default="raise", advanced=True),
                io.Combo.Input("logging", options=["off", "summary"], default="summary",
                               advanced=True),
                io.Boolean.Input("two_interval_mode", default=False, optional=True,
                                 display_name="2区間モード",
                                 tooltip="重複時は第1区間を優先します。第1区間が0–1なら"
                                         "第2区間は動作しません。"),
                io.Combo.Input("second_block_scope", options=["0-2", "24-26", "30-32"],
                               default="0-2", optional=True, advanced=True,
                               display_name="第2区間の効果",
                               tooltip="2区間モードON時のみ使用します。"),
                io.Float.Input("second_start_percent", default=0.5, min=0, max=1, step=0.01,
                               optional=True, advanced=True, display_name="第2区間の開始"),
                io.Float.Input("second_end_percent", default=1.0, min=0, max=1, step=0.01,
                               optional=True, advanced=True, display_name="第2区間の終了"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode="observe", strength=0.05, block_scope="0-2",
                branch_mode="positive_only", start_percent=0.0, end_percent=1.0,
                cutoff=0.25, soft_mask=True, mask_width=0.05, preserve_dc=True,
                head_chunk_size=1, query_chunk_size=128, max_workspace_mib=512.0,
                fail_mode="raise", logging="summary", two_interval_mode=False,
                second_block_scope="0-2", second_start_percent=0.5, second_end_percent=1.0):
        config = AFMConfig(
            mode=mode, strength=strength, block_scope=block_scope, branch_mode=branch_mode,
            start_percent=start_percent, end_percent=end_percent, cutoff=cutoff,
            soft_mask=soft_mask, mask_width=mask_width, preserve_dc=preserve_dc,
            head_chunk_size=head_chunk_size, query_chunk_size=query_chunk_size,
            max_workspace_mib=max_workspace_mib, fail_mode=fail_mode, logging=logging,
            two_interval_mode=two_interval_mode, second_block_scope=second_block_scope,
            second_start_percent=second_start_percent, second_end_percent=second_end_percent)
        return io.NodeOutput(install_afm(model, config))


class MossImageResidual(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImageResidual", display_name="MOSS Image Residual",
            category="model/patches/moss_image",
            description=("Native MOSSのブロック9-25と34-41のAttention/MLP残差を倍率調整。"
                         "LoRA適用後、MOSS Image AFMの前へ接続します。"),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("mid_multiplier", default=1.0, min=0, max=2, step=0.01,
                               display_name="9-25の倍率",
                               tooltip="0始まり・両端を含む。1.0で変更なし。LoRAを含むAttention＋MLP出力を、両CFG・全期間で調整。"),
                io.Float.Input("late_multiplier", default=1.0, min=0, max=2, step=0.01,
                               display_name="34-41の倍率",
                               tooltip="0始まり・両端を含む。1.0で変更なし。LoRAを含むAttention＋MLP出力を、両CFG・全期間で調整。"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def validate_inputs(cls, mid_multiplier=None, late_multiplier=None):
        # Linked inputs are unavailable during prompt validation; execute validates them.
        # Literal FLOAT inputs have already been normalized by ComfyUI.
        try:
            ResidualConfig(1.0 if mid_multiplier is None else mid_multiplier,
                           1.0 if late_multiplier is None else late_multiplier)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, model, mid_multiplier=1.0, late_multiplier=1.0):
        return io.NodeOutput(install_residual(model, ResidualConfig(mid_multiplier, late_multiplier)))


class MossImageResidualTest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImageResidualTest", display_name="MOSS Image Residual Test",
            category="model/patches/moss_image",
            description=("Native MOSSの全42ブロックを3ブロックずつ倍率調整するtestノード。"
                         "全項目1.0で変更なし。変更したい区間だけ調整してください。"
                         "LoRA適用後、MOSS Image AFMの前へ接続します。"),
            inputs=[
                io.Model.Input("model"),
                *[io.Float.Input(
                    f"blocks_{start:02d}_{start + 2:02d}",
                    default=1.0, min=0, max=2, step=0.01,
                    display_name=f"{start}-{start + 2}の倍率",
                    tooltip="0始まり・両端を含む3ブロック。1.0で変更なし。"
                            "LoRAを含むAttention＋MLP出力を、画像・テキスト、両CFG・全期間で調整。")
                  for start in range(0, 42, 3)],
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def validate_inputs(cls, **multipliers):
        # ComfyUI supplies available literals; linked inputs are validated in execute.
        try:
            ResidualTestConfig(**{name: 1.0 if value is None else value
                                  for name, value in multipliers.items() if name != "model"})
        except (TypeError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, model, **multipliers):
        return io.NodeOutput(install_residual(model, ResidualTestConfig(**multipliers)))


class MossImageKRowBalance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MossImageKRowBalance", display_name="MOSS Image K Row Balance (Experimental)",
            category="model/patches/moss_image",
            description=("選択したblockのK重みの行ノルムをhead内で調整する実験ノード。"
                         "native MOSSの融合INT8 ConvRot限定。画質改善は未確認。"
                         "最初はResidual/AFM/LoRAを併用せず単独で比較してください。"),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=0.0, min=-STRENGTH_LIMIT, max=STRENGTH_LIMIT,
                               step=0.05, tooltip="0はcloneのみ。1まではhead内の偏りを緩和。"
                               "1を超えると過補正も起こり得ます。負は逆方向。"),
                io.Float.Input("max_change", default=0.03, min=0.0, max=MAX_CHANGE_LIMIT, step=0.01,
                               advanced=True, tooltip="各K行の倍率の変化上限。0.03なら0.97～1.03。"
                               "strengthが非0で上限0なら、数値を変えない再構成対照。"),
                io.String.Input("blocks", default="41", optional=True,
                                tooltip="0始まり。例: 0,1,2,22,24,26,30,31,40,41。"
                                "0-2などの範囲とallにも対応。指定した全blockに共通設定を適用。"),
            ],
            outputs=[io.Model.Output(), io.String.Output("report", display_name="K balance report")],
        )

    @classmethod
    def validate_inputs(cls, strength=None, max_change=None, blocks=None):
        try:
            KBalanceConfig(0.0 if strength is None else strength,
                           0.03 if max_change is None else max_change)
            if blocks is not None:
                parse_k_blocks(blocks)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, model, strength=0.0, max_change=0.03, blocks="41"):
        result, report = install_k_balance(model, KBalanceConfig(strength, max_change), blocks)
        logging.info("MOSS K Row Balance v2 prepared: blocks=%s strength=%s max_change=%s "
                     "changed_rows=%s clipped_rows=%s delegate=%s patcher=%s -> %s",
                     report["block_indices"], strength, max_change, report["changed_rows"],
                     report.get("clipped_rows", 0), report["dynamic_delegate_used"],
                     report["input_patcher_type"], report["effective_patcher_type"])
        return io.NodeOutput(result, json.dumps(report, ensure_ascii=False, allow_nan=False))


class MossAFMExtension(ComfyExtension):
    async def get_node_list(self):
        return [MossImageAFM, MossImageResidual, MossImageResidualTest, MossImageKRowBalance]
