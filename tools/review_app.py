"""本地中文标注审核页面，仅监听回环地址。"""

from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

from anitopy_ml.annotation.review import mark_unresolved, review_sample, save_manual_revision
from anitopy_ml.annotation.store import AnnotationStore
from anitopy_ml.data.grouping import review_candidate_key
from anitopy_ml.errors import AnitopyMlError
from anitopy_ml.schemas import VALID_SPAN_LABELS

LABEL_NAMES = {
    "TITLE": "作品主标题",
    "TITLE_ALIAS": "作品别名",
    "EPISODE_TITLE": "单集标题",
    "SEASON_EXPR": "季数表达",
    "EPISODE_EXPR": "集数表达",
    "YEAR": "年份",
    "EPISODE_COUNT_EXPR": "总集数表达",
    "MEDIA_TYPE_HINT": "媒体类型提示",
    "SPECIAL_TYPE": "特别篇类型",
    "RELEASE_GROUP": "发布组",
    "SOURCE": "片源",
    "PLATFORM": "播出平台",
    "RESOLUTION": "分辨率",
    "VIDEO_TERM": "视频术语",
    "BIT_DEPTH": "位深",
    "HDR": "高动态范围",
    "AUDIO_TERM": "音频术语",
    "AUDIO_LANGUAGE": "音频语言",
    "SUBTITLE_LANGUAGE": "字幕语言",
    "SUBTITLE_MODE": "字幕方式",
    "RELEASE_VERSION": "发布版本",
    "CHECKSUM": "校验和",
    "FILE_EXTENSION": "文件扩展名",
    "VOLUME_EXPR": "卷数表达",
    "NOISE": "噪声文本",
}


def label_name(label: object) -> str:
    """返回审核界面使用的中文标签名称。"""
    code = str(label)
    return LABEL_NAMES.get(code, f"未知标签（{code}）")


def page(title: str, body: str) -> bytes:
    """生成统一的中文HTML页面。"""
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
textarea{{width:100%;min-height:26rem;font-family:ui-monospace,monospace}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}
button{{margin:.25rem;padding:.4rem .8rem}}.error{{color:#a00}}.hint{{color:#555}}</style></head><body>
<h1>{html.escape(title)}</h1>{body}</body></html>"""
    return content.encode("utf-8")


def select_group_representatives(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """每个宽松标题组保留一个待审代表，不修改任何标注状态。"""
    groups: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        key = review_candidate_key(str(entry["raw_text"])) or str(entry["sample_id"])
        groups.setdefault(key, []).append(entry)
    representatives: list[dict[str, object]] = []
    for key, items in groups.items():
        selected = min(items, key=lambda item: (len(str(item["raw_text"])), str(item["sample_id"])))
        representative = dict(selected)
        representative["相似待审数"] = len(items)
        representative["相似标题键"] = key
        representatives.append(representative)
    return sorted(representatives, key=lambda item: (str(item["review_status"]), str(item["sample_id"])))


def render_index(
    entries: list[dict[str, object]],
    active_status: str,
    active_view: str,
) -> bytes:
    """渲染只包含最新版本的审核队列。"""
    filters = (
        ("all", "全部待处理"),
        ("pending", "待审核"),
        ("needs_review", "待复核"),
    )
    filter_links = " ".join(
        f'<a href="/?status={name}&view={active_view}">{"【" if name == active_status else ""}{label}'
        f'{"】" if name == active_status else ""}</a>'
        for name, label in filters
    )
    links = "".join(
        "<li><a href=\"/sample?id={sample_id}\">{sample_id}</a> "
        "— 第{version}版，{tier}/{status}，本组待审{similar_count}条：{preview}</li>".format(
            sample_id=quote(str(item["sample_id"])),
            version=html.escape(str(item["version"])),
            tier=html.escape(str(item["tier"])),
            status=html.escape(str(item["review_status"])),
            similar_count=html.escape(str(item.get("相似待审数", 1))),
            preview=html.escape(str(item["raw_text"])[:80]),
        )
        for item in entries
    )
    body = (
        "<p>队列筛选：" + filter_links + "</p>"
        + (
            '<p>视图：【每组一个代表】 <a href="/?status=' + active_status + '&view=all">查看全部待审样本</a></p>'
            if active_view == "representative"
            else '<p>视图：全部待审样本 <a href="/?status=' + active_status + '&view=representative">每组仅显示一个代表</a></p>'
        )
        + "<p>以下记录的最新版本仍需人工处理：</p><ul>"
        + (links or "<li>当前筛选条件下没有待处理样本。</li>")
        + "</ul>"
    )
    return page("待审核标注", body)


def build_model_suggestion(result: object) -> dict[str, object]:
    """将模型解析结果转换为审核页面可只读展示的建议数据。"""
    payload = result.model_dump()
    spans: list[dict[str, object]] = []
    for evidence in payload.get("evidence", {}).values():
        if evidence.get("source") != "model":
            continue
        for span in evidence.get("spans", []):
            spans.append(dict(span))
    spans.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
    return {
        "spans": spans,
        "extracted": payload["extracted"],
        "status": payload["status"],
        "warnings": payload["warnings"],
        "model_version": payload.get("model_version"),
    }


def render_model_suggestion(suggestion: dict[str, object] | None, message: str = "") -> str:
    """渲染只读模型建议，并明确其不会自动写入标注库。"""
    if suggestion is None:
        detail = html.escape(message or "未加载可用的本地模型。")
        return f'<h2>模型建议</h2><p class="hint">当前无模型建议：{detail}</p>'
    rows = "".join(
        "<li>{label}：[{start}, {end})「{text}」</li>".format(
            label=html.escape(label_name(span["label"])),
            start=span["start"],
            end=span["end"],
            text=html.escape(str(span["text"])),
        )
        for span in suggestion["spans"]
    )
    status = "完整建议" if suggestion["status"] == "ok" else "部分建议"
    warning_text = "；".join(str(item) for item in suggestion["warnings"])
    safe_json = json.dumps(
        {"spans": suggestion["spans"], "extracted": suggestion["extracted"]},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<h2>模型建议</h2>
<p class="hint">状态：{html.escape(status)}。模型版本：{html.escape(str(suggestion.get('model_version') or '未标记'))}。模型建议不会自动保存，也不会改变审核状态。</p>
<ul>{rows or '<li>模型没有给出可载入的片段。</li>'}</ul>
<p class="hint">{html.escape(warning_text)}</p>
<script id="model-suggestion-json" type="application/json">{safe_json}</script>
<button type="button" id="apply-model-suggestion">将模型建议载入编辑草稿</button>"""


def render_sample(
    payload: dict[str, object],
    message: str = "",
    work_group_id: str | None = None,
    model_suggestion: dict[str, object] | None = None,
    model_message: str = "",
) -> bytes:
    """渲染一条记录、片段编辑器和审核操作。"""
    sample_id = str(payload["sample_id"])
    raw_text = str(payload["raw_text"])
    spans = payload.get("spans", [])
    span_rows = "".join(
        "<li>{label}：[{start}, {end})「{text}」</li>".format(
            label=html.escape(label_name(span["label"])),
            start=span["start"],
            end=span["end"],
            text=html.escape(str(span["text"])),
        )
        for span in spans
    )
    escaped_json = html.escape(json.dumps(payload, ensure_ascii=False, indent=2))
    label_options = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label_name(label))}</option>'
        for label in sorted(VALID_SPAN_LABELS)
    )
    notice = f'<p class="hint">{html.escape(message)}</p>' if message else ""
    editor_script = """<script>
const titleElement = document.getElementById("raw-title");
const payloadElement = document.getElementById("payload-json");
const rawText = titleElement.textContent;
const initialPayloadText = payloadElement.value;

function readPayload() {
  try { return JSON.parse(payloadElement.value); }
  catch (error) { alert("标注JSON格式无效，无法继续编辑。"); return null; }
}
function writePayload(payload) {
  payloadElement.value = JSON.stringify(payload, null, 2);
  renderSpanEditor(payload);
  syncFieldInputs(payload);
}
function codePointOffset(utf16Offset) {
  return Array.from(rawText.slice(0, utf16Offset)).length;
}
function absoluteUtf16Offset(container, node, offset) {
  let total = 0;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let current;
  while ((current = walker.nextNode())) {
    if (current === node) return total + offset;
    total += current.nodeValue.length;
  }
  throw new Error("选择范围不在原始标题内。");
}
function renderSpanEditor(payload) {
  const target = document.getElementById("span-editor");
  target.replaceChildren();
  payload.spans.forEach((span, index) => {
    const row = document.createElement("p");
    const select = document.createElement("select");
    const labels = Array.from(document.getElementById("label-choice").options);
    labels.forEach((option) => {
      const item = option.cloneNode(true);
      item.selected = item.value === span.label;
      select.appendChild(item);
    });
    select.addEventListener("change", () => {
      const latest = readPayload();
      if (!latest) return;
      latest.spans[index].label = select.value;
      writePayload(latest);
    });
    const text = document.createTextNode(" [" + span.start + ", " + span.end + ")「" + span.text + "」 ");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "删除片段";
    remove.addEventListener("click", () => {
      const latest = readPayload();
      if (!latest) return;
      latest.spans.splice(index, 1);
      writePayload(latest);
    });
    row.append(select, text, remove);
    target.appendChild(row);
  });
}
function addSelectedSpan() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) {
    alert("请先在原始标题中拖选一段文字。");
    return;
  }
  const range = selection.getRangeAt(0);
  if (!titleElement.contains(range.startContainer) || !titleElement.contains(range.endContainer)) {
    alert("只能选择原始标题中的文字。");
    return;
  }
  let start, end;
  try {
    start = codePointOffset(absoluteUtf16Offset(titleElement, range.startContainer, range.startOffset));
    end = codePointOffset(absoluteUtf16Offset(titleElement, range.endContainer, range.endOffset));
  } catch (error) {
    alert("无法读取所选片段的位置。");
    return;
  }
  const payload = readPayload();
  if (!payload) return;
  const overlaps = payload.spans.some((span) => start < span.end && end > span.start);
  if (start >= end || overlaps) {
    alert("片段不能为空，且不能与已有片段重叠。");
    return;
  }
  payload.spans.push({
    start: start, end: end, text: Array.from(rawText).slice(start, end).join(""),
    label: document.getElementById("label-choice").value, source: "human"
  });
  payload.spans.sort((left, right) => left.start - right.start || left.end - right.end);
  writePayload(payload);
  selection.removeAllRanges();
}
function syncFieldInputs(payload) {
  document.getElementById("season-input").value = (payload.extracted.seasons || []).join(", ");
  document.getElementById("episode-input").value =
    (payload.extracted.episodes || []).map((item) => item.raw || item.value).join(", ");
}
function updateStructuredFields() {
  const payload = readPayload();
  if (!payload) return;
  const seasons = document.getElementById("season-input").value.match(/\\d+/g) || [];
  const episodes = document.getElementById("episode-input").value.split(",")
    .map((item) => item.trim()).filter(Boolean);
  payload.extracted.seasons = seasons.map((item) => Number(item));
  payload.extracted.episodes = episodes.map((item) => ({
    raw: item, value: item.replace(/^0+(?=\\d)/, ""), numbering: "unknown"
  }));
  payloadElement.value = JSON.stringify(payload, null, 2);
}
function resetDraft() {
  payloadElement.value = initialPayloadText;
  const payload = readPayload();
  if (payload) { renderSpanEditor(payload); syncFieldInputs(payload); }
}
function applyModelSuggestion() {
  const element = document.getElementById("model-suggestion-json");
  if (!element) return;
  const suggestion = JSON.parse(element.textContent);
  if (!confirm("将用模型建议覆盖当前未保存的片段和结构字段，是否继续？")) return;
  const payload = readPayload();
  if (!payload) return;
  payload.spans = suggestion.spans;
  payload.extracted = suggestion.extracted;
  writePayload(payload);
}
document.getElementById("add-span").addEventListener("click", addSelectedSpan);
document.getElementById("reset-draft").addEventListener("click", resetDraft);
document.getElementById("season-input").addEventListener("change", updateStructuredFields);
document.getElementById("episode-input").addEventListener("change", updateStructuredFields);
const applyModelButton = document.getElementById("apply-model-suggestion");
if (applyModelButton) applyModelButton.addEventListener("click", applyModelSuggestion);
const firstPayload = readPayload();
if (firstPayload) { renderSpanEditor(firstPayload); syncFieldInputs(firstPayload); }
</script>"""
    body = f"""{notice}<p><a href="/">返回待审核列表</a></p>
<h2>原始标题</h2><pre id="raw-title">{html.escape(raw_text)}</pre>
<h2>当前片段</h2><ul>{span_rows or '<li>暂无片段建议</li>'}</ul>
{render_model_suggestion(model_suggestion, model_message)}
<h2>片段编辑</h2>
<p class="hint">在原始标题中拖选文字，选择标签后添加。浏览器UTF-16偏移会转换为后端使用的Unicode码点偏移。</p>
标签：<select id="label-choice">{label_options}</select>
<button type="button" id="add-span">将所选文字添加为片段</button>
<div id="span-editor"></div>
<p>季数（逗号分隔）：<input id="season-input"></p>
<p>集数（逗号分隔）：<input id="episode-input"></p>
<p class="hint">字段编辑与片段编辑会同步到下方JSON；JSON中的英文标签代码用于保持数据兼容，样本ID和原始标题不可修改。</p>
<form method="post" action="/save"><input type="hidden" name="sample_id" value="{html.escape(sample_id)}">
审核人员：<input name="actor" required><br><textarea id="payload-json" name="payload_json">{escaped_json}</textarea><br>
<button type="submit">保存为待复核版本</button>
<button type="button" id="reset-draft">撤销未保存编辑</button></form>
<form method="post" action="/decision"><input type="hidden" name="sample_id" value="{html.escape(sample_id)}">
审核人员：<input name="actor" required>
<button name="decision" value="approve">审核通过（金标）</button>
<button name="decision" value="reject">审核拒绝</button></form>
<form method="post" action="/unresolved"><input type="hidden" name="sample_id" value="{html.escape(sample_id)}">
审核人员：<input name="actor" required>
<button type="submit">标记为无法判断，进入复核队列</button></form>
<h2>作品组</h2>
<p class="hint">作品组用于后续按作品划分训练、验证和测试集。此处仅保存本地人工覆盖值，不会访问外部作品库。</p>
<form method="post" action="/group"><input type="hidden" name="sample_id" value="{html.escape(sample_id)}">
作品组ID：<input id="work-group-id" name="work_group_id" value="{html.escape(work_group_id or '')}" required>
审核人员：<input name="actor" required>
<button type="submit">保存作品组</button></form>
<h3>候选面板（模拟）</h3>
<p>尚未接入外部作品库。可先输入稳定的人工组ID；步骤10完成后将在这里显示可审核的作品库候选。</p>
<button type="button" onclick="document.getElementById('work-group-id').value='人工组-'">填入人工组前缀</button>
{editor_script}"""
    return page(f"审核样本 {sample_id}", body)


def build_handler(database: str, model_parser: object | None = None, model_message: str = ""):
    """为指定SQLite数据库创建HTTP处理器类型。"""
    class ReviewHandler(BaseHTTPRequestHandler):
        def send_html(self, content: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            with AnnotationStore(database) as store:
                if parsed.path == "/":
                    requested_status = parse_qs(parsed.query).get("status", ["all"])[0]
                    statuses = {
                        "pending": ("pending",),
                        "needs_review": ("needs_review",),
                        "all": ("pending", "needs_review"),
                    }.get(requested_status, ("pending", "needs_review"))
                    active_status = requested_status if requested_status in {"pending", "needs_review", "all"} else "all"
                    requested_view = parse_qs(parsed.query).get("view", ["representative"])[0]
                    active_view = requested_view if requested_view in {"representative", "all"} else "representative"
                    entries = store.review_queue(statuses=statuses, limit=10000)
                    if active_view == "representative":
                        entries = select_group_representatives(entries)
                    self.send_html(render_index(entries, active_status, active_view))
                    return
                if parsed.path == "/sample":
                    sample_id = parse_qs(parsed.query).get("id", [""])[0]
                    payload = store.latest_payload(sample_id)
                    if payload:
                        suggestion = None
                        suggestion_message = model_message
                        if model_parser is not None:
                            try:
                                suggestion = build_model_suggestion(model_parser.parse(str(payload["raw_text"])))
                            except AnitopyMlError as error:
                                suggestion_message = f"模型推断失败：{error}"
                        self.send_html(
                            render_sample(
                                payload,
                                work_group_id=store.work_group_id(sample_id),
                                model_suggestion=suggestion,
                                model_message=suggestion_message,
                            )
                        )
                    else:
                        self.send_html(page("未找到样本", "<p class=\"error\">未找到该审核样本。</p>"), 404)
                    return
            self.send_html(page("未找到页面", "<p class=\"error\">页面不存在。</p>"), 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            sample_id = form.get("sample_id", [""])[0]
            actor = form.get("actor", [""])[0]
            try:
                with AnnotationStore(database) as store:
                    if self.path == "/decision":
                        decision = form.get("decision", [""])[0]
                        record = review_sample(store, sample_id, actor=actor, decision=decision)
                        self.send_html(
                            render_sample(
                                record.model_dump(),
                                "审核决定已保存。",
                                store.work_group_id(sample_id),
                            )
                        )
                        return
                    if self.path == "/save":
                        payload = json.loads(form.get("payload_json", [""])[0])
                        record = save_manual_revision(store, sample_id, payload, actor=actor)
                        self.send_html(
                            render_sample(
                                record.model_dump(),
                                "编辑版本已保存，等待复核。",
                                store.work_group_id(sample_id),
                            )
                        )
                        return
                    if self.path == "/unresolved":
                        record = mark_unresolved(store, sample_id, actor=actor)
                        self.send_html(
                            render_sample(
                                record.model_dump(),
                                "已标记为无法判断，等待复核。",
                                store.work_group_id(sample_id),
                            )
                        )
                        return
                    if self.path == "/group":
                        store.set_work_group(
                            sample_id,
                            form.get("work_group_id", [""])[0],
                            actor=actor,
                        )
                        payload = store.latest_payload(sample_id)
                        if payload is None:
                            raise ValueError("未找到需要设置作品组的样本。")
                        self.send_html(
                            render_sample(
                                payload,
                                "作品组已保存。",
                                store.work_group_id(sample_id),
                            )
                        )
                        return
                self.send_html(page("操作失败", "<p class=\"error\">不支持的操作。</p>"), 400)
            except (AnitopyMlError, json.JSONDecodeError, ValueError) as error:
                self.send_html(page("操作失败", f"<p class=\"error\">{html.escape(str(error))}</p>"), 400)

        def log_message(self, *_: object) -> None:
            """避免将标题内容写入默认HTTP访问日志。"""

    return ReviewHandler


def main() -> None:
    """启动本地审核页面。"""
    parser = argparse.ArgumentParser(description="启动本地中文标注审核页面。")
    parser.add_argument("--database", required=True, help="SQLite标注库路径。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机。")
    parser.add_argument("--port", type=int, default=8765, help="监听端口。")
    parser.add_argument(
        "--model",
        default="artifacts/runs/当前金标连通性",
        help="本地模型目录；不存在时页面仅保留人工标注功能。",
    )
    arguments = parser.parse_args()
    if arguments.host != "127.0.0.1":
        parser.error("第一版审核页面只允许监听127.0.0.1。")
    model_parser = None
    model_message = ""
    if arguments.model:
        try:
            from anitopy_ml.api import MediaParser

            model_parser = MediaParser.from_pretrained(arguments.model)
        except AnitopyMlError as error:
            model_message = f"未加载模型建议：{error}"
    server = ThreadingHTTPServer(
        (arguments.host, arguments.port),
        build_handler(arguments.database, model_parser=model_parser, model_message=model_message),
    )
    print(f"审核页面已启动：http://{arguments.host}:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("审核页面已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
