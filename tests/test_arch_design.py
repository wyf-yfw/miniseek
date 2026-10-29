from __future__ import annotations

from api.arch_inspect import inspect_design


def test_design_miniseek_m_fig3():
    out = inspect_design("miniseek-m")
    assert out["config"]["arch"] == "miniseek-m"
    assert out["meta"]["diagram"] == "fig3" and out["meta"]["trainable"] is True
    assert out["total_params"] <= 200_000_000
    fig = out["fig3"]

    def sig(groups):
        return [(g["rows"][0]["mode"], g["repeat"], (g.get("nested") or {}).get("repeat")) for g in groups]

    assert sig(fig["encoder"]["groups"]) == [("SWA", 2, None), ("Full", 1, 1)]
    assert sig(fig["decoder"]["groups"]) == [("Full", 1, None), ("Reuse", 1, None), ("Reindex", 1, 1)]


def test_design_m_ref():
    out = inspect_design("m-ref")
    assert out["meta"]["source"] == "reference"
    assert out["meta"]["release_name"] == "DeepSeek-V4.1-Flash"
