from comfy_api.latest import ComfyExtension, io

from .afm.adapter import install_afm
from .afm.config import AFMConfig


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


class MossAFMExtension(ComfyExtension):
    async def get_node_list(self):
        return [MossImageAFM]
