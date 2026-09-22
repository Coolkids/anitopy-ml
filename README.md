# Anitopy-ML

Anitopy-ML 是一个本地运行的多语言媒体标题语义解析工具。它使用 XLM-R 字符级 BIO 抽取模型，识别中日英混排文件名中的作品名称、别名、季数、集数、发布组、片源、编码、字幕等信息，并返回原文证据和置信度状态。

项目的训练数据由仓库根目录的 `data.json` 模板确定性生成；模型推理默认离线完成，不依赖作品数据库或网络服务。

## 当前模型状态

当前冻结候选为 `V11_template_coverage_warm_start_seed1`。它已在当前模板版本的作品隔离、全模板覆盖冻结分区上完成一次评测：

| 指标 | 结果 |
| --- | ---: |
| 实体 F1 | 99.78% |
| 名称与别名联合正确率 | 100.00% |
| 季数正确率 | 100.00% |
| 集数正确率 | 99.45% |
| 核心字段整条正确率 | 99.45% |

上述结果只适用于本项目 `data.json` 所定义的模板体系，不等同于真实发布标题的准确率。完整依据见 [冻结发布判定](reports/V11当前模板冻结发布判定.md)。

## 功能

- 本地解析单条或批量媒体标题。
- 提取主标题、别名、季数、集数、集数范围与总集数。
- 识别发布组、发布版本、片源、分辨率、视频编码、位深、音频、字幕和文件扩展名。
- 返回原文片段、Unicode 偏移、模型置信度、校准状态和中文警告。
- 提供命令行、Python API 和本地模型验证页面。
- 使用模板自动生成字符级 BIO 训练数据，并按作品键隔离训练、验证和冻结测试分区。

## 安装

需要 Python 3.12 或更高版本，以及 [uv](https://docs.astral.sh/uv/)。推理 XLM-R 模型需要安装 `train` 可选依赖。

```powershell
git clone https://github.com/Coolkids/anitopy-ml.git
cd anitopy-ml
uv sync --extra train
uv run anitopy-ml doctor
```

若使用 NVIDIA GPU，确认本机 PyTorch 能识别 CUDA 后，将命令中的 `--device auto` 改为 `--device cuda`。仅验证加载或没有 CUDA 时可使用 `--device cpu`。

## 安装模型包

模型权重不随源码仓库提交。下载对应版本的模型包并解压到例如：

```text
artifacts/
└── releases/
    └── anitopy-ml-v11/
        ├── best_model.pt
        ├── calibration.json
        ├── acceptance_policy.json
        ├── checkpoint_metadata.json
        ├── 训练报告.json
        └── tokenizer/
```

`best_model.pt`、`tokenizer/`、`calibration.json` 和 `acceptance_policy.json` 必须来自同一个模型包。不要将训练用的 `training_state.pt` 放入发布包。

## 快速开始

### 命令行

```powershell
uv run anitopy-ml parse `
  "[绿茶字幕组&LoliHouse] 穹庐下的魔女 / Tenmaku no Jaadugar - 09v2 [WebRip 1080p HEVC-10bit AAC][简繁日内封字幕]" `
  --model artifacts/releases/anitopy-ml-v11 `
  --device auto
```

输出为 JSON，核心字段位于 `extracted`，原文证据位于 `evidence`。集数使用规范化数值字符串，例如原文 `09` 对应 `"value": "9"`，发布版本则保留为 `v2`。

### Python API

```python
from anitopy_ml.api import MediaParser

parser = MediaParser.from_pretrained(
    "artifacts/releases/anitopy-ml-v11",
    device="auto",
)

result = parser.parse(
    "[绿茶字幕组&LoliHouse] 穹庐下的魔女 / "
    "Tenmaku no Jaadugar - 09v2 [WebRip 1080p HEVC-10bit AAC][简繁日内封字幕]"
)

print(result.extracted.title)
print(result.extracted.title_aliases)
print(result.extracted.episodes)
print(result.model_dump())
```

批量调用会保持输入顺序；单条失败时可使用 `on_error="record"` 记录错误并继续处理：

```python
results = parser.parse_batch(["示例作品 S01E02", ""], on_error="record")
```

### 批量解析 CSV

```powershell
uv run anitopy-ml batch `
  --input data.csv `
  --column record_title `
  --model artifacts/releases/anitopy-ml-v11 `
  --output reports/批量解析.jsonl `
  --on-error record
```

### 本地模型验证页面

```powershell
uv run python tools/model_verify_app.py `
  --model artifacts/releases/anitopy-ml-v11 `
  --device auto
```

浏览器访问 `http://127.0.0.1:8765`。页面不保存输入标题，优先显示标题、别名、季数和集数。

## 输出说明

- `extracted`：规范化后的结构化字段。
- `evidence`：字段对应的原文片段、偏移、来源和置信度。
- `status`：`ok` 表示获得主标题；`partial` 表示仅获得部分可确定字段。
- `calibrated`：`true` 表示该模型包的验证集校准器已为字段置信度提供经验校准值。
- `warnings`：中文警告，例如超长输入或约束层发现的无效范围。

发布组和技术字段是辅助信息。当前版本的发布判断重点是名称与别名、季数和集数。

## 开发与验证

```powershell
uv run python -m unittest discover -s tests/unit -p "test_*.py"
```

当前单元测试共 104 项。训练数据生成、训练、校准、冻结评测及模型调用的详细命令见以下文档：

- [开发环境](docs/开发环境.md)
- [训练数据导出](docs/训练数据导出.md)
- [调用指南](docs/调用指南.md)
- [字段契约](docs/字段契约.md)
- [模型验证页面](docs/模型验证页面.md)

## 数据与模型更新

修改 `data.json` 的标题词表或模板后，必须重新生成数据、去重、执行 BIO 审计、重新划分作品隔离分区，并使用新的验证集校准。已读取的冻结测试分区不能用于后续模型选择或模板调整；下一版必须建立新的未读冻结分区。

训练、验证和冻结评测均使用模板合成数据。真实标题只应用于本地模型验证和后续人工观察，不能直接混入训练或评测分区。
