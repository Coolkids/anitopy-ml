"""本地模型实际验证页面，仅监听回环地址。"""

from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from anitopy_ml.api import MediaParser
from anitopy_ml.errors import AnitopyMlError


LABEL_NAMES = {
    "TITLE": "作品主标题",
    "TITLE_ALIAS": "作品别名",
    "SEASON_EXPR": "季数表达",
    "EPISODE_EXPR": "集数表达",
    "EPISODE_COUNT_EXPR": "总集数表达",
    "RELEASE_GROUP": "发布组",
    "RELEASE_VERSION": "发布版本",
    "SOURCE": "片源",
    "RESOLUTION": "分辨率",
    "VIDEO_TERM": "视频术语",
    "AUDIO_TERM": "音频术语",
    "SUBTITLE_LANGUAGE": "字幕语言",
    "SUBTITLE_MODE": "字幕方式",
}


def label_name(label: object) -> str:
    """返回验证页面使用的中文标签名称。"""
    code = str(label)
    return LABEL_NAMES.get(code, f"其他字段（{code}）")


def page(title: str, body: str) -> bytes:
    """生成统一的中文HTML页面。"""
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
textarea{{width:100%;min-height:7rem;font-family:ui-monospace,monospace}}button{{margin-top:.6rem;padding:.5rem 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ddd;padding:.55rem;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}}code,pre{{white-space:pre-wrap;overflow-wrap:anywhere}}pre{{background:#f5f5f5;padding:1rem}}
.hint{{color:#555}}.error{{color:#a00}}.empty{{color:#777}}</style></head><body>
<h1>{html.escape(title)}</h1>{body}</body></html>"""
    return content.encode("utf-8")


def _episode_text(extracted: dict[str, object]) -> str:
    """以可核对的中文形式展示单集、范围与声明总集数。"""
    values: list[str] = []
    for item in extracted.get("episodes", []):
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw", ""))
        value = str(item.get("value", ""))
        values.append(value if raw == value else f"{value}（原文：{raw}）")
    for item in extracted.get("episode_ranges", []):
        if isinstance(item, dict):
            values.append(f"{item.get('start')}–{item.get('end')}")
    declared = extracted.get("declared_episode_count")
    if declared is not None:
        values.append(f"声明总集数：{declared}")
    return "、".join(values) if values else "未识别"


def _text_list(value: object) -> str:
    """展示结构化字符串列表。"""
    if not isinstance(value, list) or not value:
        return "未识别"
    return "、".join(str(item) for item in value)


def _model_spans(payload: dict[str, object]) -> list[dict[str, object]]:
    """按原文顺序收集模型证据片段。"""
    spans: list[dict[str, object]] = []
    evidence = payload.get("evidence", {})
    if not isinstance(evidence, dict):
        return spans
    for item in evidence.values():
        if not isinstance(item, dict) or item.get("source") != "model":
            continue
        for span in item.get("spans", []):
            if isinstance(span, dict):
                spans.append(span)
    return sorted(spans, key=lambda item: (int(item["start"]), int(item["end"])))


def render_verification_page(
    title: str = "",
    result: object | None = None,
    error: str = "",
) -> bytes:
    """渲染标题输入与模型结果，不保存任何输入或验证结论。"""
    form = f"""<p class="hint">输入一个实际媒体文件名，页面只在本机运行模型，不保存标题。</p>
<form method="post"><label for="title">待验证名称</label><br>
<textarea id="title" name="title" required autofocus>{html.escape(title)}</textarea><br>
<button type="submit">开始识别</button></form>"""
    if error:
        return page("模型实际验证", form + f'<p class="error">{html.escape(error)}</p>')
    if result is None:
        return page("模型实际验证", form + '<p class="hint">核心核对项：主标题、别名、季数、集数。发布组和技术字段仅作辅助参考。</p>')

    payload = result.model_dump()
    extracted = payload["extracted"]
    if not isinstance(extracted, dict):
        return page("模型实际验证", form + '<p class="error">模型返回的结构化结果无效。</p>')
    rows = (
        ("主标题", extracted.get("title") or "未识别"),
        ("作品别名", _text_list(extracted.get("title_aliases"))),
        ("季数", _text_list(extracted.get("seasons"))),
        ("集数", _episode_text(extracted)),
    )
    core_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(str(value))}</td></tr>"
        for name, value in rows
    )
    supplementary = (
        ("发布组", _text_list(extracted.get("release_groups"))),
        ("发布版本", extracted.get("release_version") or "未识别"),
        ("片源", _text_list(extracted.get("source"))),
        ("分辨率", _text_list(extracted.get("resolution"))),
        ("视频编码", _text_list(extracted.get("video_codecs"))),
        ("音频编码", _text_list(extracted.get("audio_codecs"))),
        ("字幕语言", _text_list(extracted.get("subtitle_languages"))),
        ("字幕方式", extracted.get("subtitle_mode") or "未识别"),
    )
    supplementary_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(str(value))}</td></tr>"
        for name, value in supplementary
    )
    span_rows = "".join(
        "<tr><td>{label}</td><td>[{start}, {end})</td><td>{text}</td></tr>".format(
            label=html.escape(label_name(span.get("label"))),
            start=html.escape(str(span.get("start"))),
            end=html.escape(str(span.get("end"))),
            text=html.escape(str(span.get("text"))),
        )
        for span in _model_spans(payload)
    )
    warnings = payload.get("warnings", [])
    warning_text = "；".join(str(item) for item in warnings) if warnings else "无"
    result_body = f"""<hr><h2>核心识别结果</h2>
<p class="hint">请优先人工核对下列四项；主标题与别名相连时，只要两者的整体边界正确即可接受。</p>
<table>{core_rows}</table>
<details><summary>辅助字段</summary><table>{supplementary_rows}</table></details>
<h2>模型原文证据</h2><table><tr><th>标签</th><th>位置</th><th>原文</th></tr>
{span_rows or '<tr><td colspan="3" class="empty">模型未给出证据片段。</td></tr>'}</table>
<p class="hint">解析状态：{html.escape(str(payload.get('status')))}。模型版本：{html.escape(str(payload.get('model_version') or '未标记'))}。</p>
<p class="hint">提示：{html.escape(warning_text)}</p>
<details><summary>完整 JSON 结果</summary><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></details>"""
    return page("模型实际验证", form + result_body)


def main() -> None:
    """启动仅用于实际标题验证的本地网页。"""
    parser = argparse.ArgumentParser(description="启动本地模型实际验证页面。")
    parser.add_argument(
        "--model",
        default="artifacts/runs/V9_alias_boundary_noise_seed2",
        help="本地模型目录，默认使用V9种子2。",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，仅允许127.0.0.1。")
    parser.add_argument("--port", type=int, default=8765, help="监听端口。")
    arguments = parser.parse_args()
    if arguments.host != "127.0.0.1":
        parser.error("模型验证页面只允许监听127.0.0.1。")
    if not 1 <= arguments.port <= 65535:
        parser.error("监听端口必须在1到65535之间。")
    model_path = Path(arguments.model)
    try:
        parser_instance = MediaParser.from_pretrained(model_path, device=arguments.device)
    except AnitopyMlError as error:
        parser.error(str(error))

    class VerificationHandler(BaseHTTPRequestHandler):
        """处理本地标题提交并返回模型结果。"""

        def send_html(self, content: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            self.send_html(render_verification_page())

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                self.send_html(render_verification_page(error="提交内容长度必须在1到16384字节之间。"), 400)
                return
            form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            title = form.get("title", [""])[0].strip()
            try:
                result = parser_instance.parse(title)
            except AnitopyMlError as error:
                self.send_html(render_verification_page(title, error=str(error)), 400)
                return
            self.send_html(render_verification_page(title, result))

        def log_message(self, _format: str, *_args: object) -> None:
            """避免将用户提交的标题写入终端访问日志。"""

    server = ThreadingHTTPServer((arguments.host, arguments.port), VerificationHandler)
    print(f"模型验证页面已启动：http://{arguments.host}:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("模型验证页面已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
