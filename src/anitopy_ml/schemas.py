"""解析、标注和来源记录使用的数据结构与校验函数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from anitopy_ml.errors import InputValidationError, SchemaValidationError

SpanLabel = Literal[
    "TITLE",
    "TITLE_ALIAS",
    "EPISODE_TITLE",
    "SEASON_EXPR",
    "EPISODE_EXPR",
    "YEAR",
    "EPISODE_COUNT_EXPR",
    "MEDIA_TYPE_HINT",
    "SPECIAL_TYPE",
    "RELEASE_GROUP",
    "SOURCE",
    "PLATFORM",
    "RESOLUTION",
    "VIDEO_TERM",
    "BIT_DEPTH",
    "HDR",
    "AUDIO_TERM",
    "AUDIO_LANGUAGE",
    "SUBTITLE_LANGUAGE",
    "SUBTITLE_MODE",
    "RELEASE_VERSION",
    "CHECKSUM",
    "FILE_EXTENSION",
    "VOLUME_EXPR",
    "NOISE",
]

VALID_SPAN_LABELS = set(SpanLabel.__args__)
BIO_LABELS = ("O",) + tuple(
    label
    for span_label in SpanLabel.__args__
    for label in (f"B-{span_label}", f"I-{span_label}")
)


@dataclass(frozen=True, slots=True)
class Span:
    """原始标题中的半开区间片段。"""

    start: int
    end: int
    text: str
    label: SpanLabel
    source: str = "human"

    def validate(self, raw_text: str) -> None:
        """校验片段边界、文本和标签是否与原文一致。"""
        if not (0 <= self.start < self.end <= len(raw_text)):
            raise SchemaValidationError("片段范围超出原始标题边界。")
        if raw_text[self.start : self.end] != self.text:
            raise SchemaValidationError("片段文本与原始标题中的位置不一致。")
        if self.label not in VALID_SPAN_LABELS:
            raise SchemaValidationError("片段标签不在允许范围内。")


@dataclass(frozen=True, slots=True)
class Evidence:
    """字段结果所依据的原文证据和生成来源。"""

    spans: tuple[Span, ...] = ()
    source: str = "unknown"
    version: str | None = None
    confidence: float | None = None
    calibrated: bool = False

    def validate(self, raw_text: str) -> None:
        """校验证据中的片段及置信度范围。"""
        for span in self.spans:
            span.validate(raw_text)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise SchemaValidationError("置信度必须在0到1之间。")


@dataclass(slots=True)
class ExtractedFields:
    """仅根据标题原文提取或规范化的字段。"""

    title: str | None = None
    title_aliases: list[str] = field(default_factory=list)
    media_type: Literal["movie", "tv", "unknown"] = "unknown"
    release_kind: Literal[
        "episode", "season_pack", "movie", "special", "collection", "unknown"
    ] = "unknown"
    seasons: list[int] = field(default_factory=list)
    episodes: list[dict[str, str]] = field(default_factory=list)
    episode_ranges: list[dict[str, str]] = field(default_factory=list)
    declared_episode_count: int | None = None
    special_type: str | None = None
    year: int | None = None
    source: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    resolution: list[str] = field(default_factory=list)
    video_codecs: list[str] = field(default_factory=list)
    video_encoders: list[str] = field(default_factory=list)
    audio_codecs: list[str] = field(default_factory=list)
    subtitle_languages: list[str] = field(default_factory=list)
    subtitle_mode: str | None = None
    release_groups: list[str] = field(default_factory=list)
    release_version: str | None = None
    file_extension: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    """外部作品库提供的候选或已确认关联，永不替代原文提取字段。"""

    provider: str
    media_type: Literal["movie", "tv"]
    item_id: int
    title: str
    score: float | None = None
    status: Literal["matched", "ambiguous", "unmatched", "unavailable"] = "unmatched"

    def validate(self) -> None:
        """校验外部目录关联的基本字段。"""
        if self.item_id <= 0:
            raise SchemaValidationError("作品库条目ID必须为正整数。")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise SchemaValidationError("候选分数必须在0到1之间。")


@dataclass(slots=True)
class ParseResult:
    """解析接口返回的完整结果。"""

    raw_text: str
    extracted: ExtractedFields = field(default_factory=ExtractedFields)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    catalog: CatalogMatch | None = None
    schema_version: str = "1.0"
    model_version: str | None = None
    normalizer_version: str = "1.0"
    status: Literal["ok", "partial", "error"] = "ok"
    link_status: Literal[
        "disabled", "matched", "ambiguous", "unmatched", "unavailable"
    ] = "disabled"
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """校验结果内部引用及输入标题。"""
        if not isinstance(self.raw_text, str) or not self.raw_text.strip():
            raise InputValidationError("标题不能为空。")
        for item in self.evidence.values():
            item.validate(self.raw_text)
        if self.catalog is not None:
            self.catalog.validate()

    def model_dump(self) -> dict[str, Any]:
        """转换为仅由JSON兼容类型组成的字典。"""
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DataLineage:
    """记录样本来源与训练用途约束。"""

    sources: tuple[str, ...]
    training_allowed: bool
    authorization_ref: str | None = None

    def validate(self) -> None:
        """校验来源记录至少包含一个可追溯来源。"""
        if not self.sources or any(not source.strip() for source in self.sources):
            raise SchemaValidationError("数据来源不能为空。")

    def validate_for_training(self) -> None:
        """拒绝导出未获准用于训练的样本。"""
        self.validate()
        if not self.training_allowed:
            raise SchemaValidationError("当前数据来源未获准用于训练。")


@dataclass(frozen=True, slots=True)
class AnnotationMetadata:
    """标注层级、审核状态和版本信息。"""

    tier: Literal["gold", "silver", "weak", "review", "unmatched"]
    review_status: Literal["pending", "accepted", "rejected", "needs_review"]
    reviewer_id: str | None = None
    version: int = 1
    unlabeled_regions: tuple[tuple[int, int], ...] = ()

    def validate(self, raw_text: str) -> None:
        """校验审核版本和未标注区域位置。"""
        if self.version <= 0:
            raise SchemaValidationError("标注版本必须为正整数。")
        for start, end in self.unlabeled_regions:
            if not 0 <= start < end <= len(raw_text):
                raise SchemaValidationError("未标注区域超出标题边界。")


@dataclass(slots=True)
class AnnotationRecord:
    """一条可审核、可训练且可追溯的标题标注记录。"""

    sample_id: str
    raw_text: str
    spans: tuple[Span, ...]
    extracted: ExtractedFields
    annotation: AnnotationMetadata
    lineage: DataLineage
    catalog: CatalogMatch | None = None
    schema_version: str = "1.0"

    def validate(self) -> None:
        """校验标注边界、扁平BIO约束和来源用途。"""
        if not self.sample_id.strip():
            raise SchemaValidationError("样本ID不能为空。")
        if not self.raw_text.strip():
            raise InputValidationError("标题不能为空。")
        ordered_spans = sorted(self.spans, key=lambda item: (item.start, item.end))
        previous_end = -1
        for span in ordered_spans:
            span.validate(self.raw_text)
            if span.start < previous_end:
                raise SchemaValidationError("扁平BIO标注不允许片段重叠。")
            previous_end = span.end
        self.annotation.validate(self.raw_text)
        self.lineage.validate()
        if self.catalog is not None:
            self.catalog.validate()

    def model_dump(self) -> dict[str, Any]:
        """转换为可JSON序列化的标注记录。"""
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """发布模型的推理依赖、标签表和可复现信息。"""

    model_name: str
    model_version: str
    model_revision: str
    tokenizer_revision: str
    schema_version: str
    normalizer_version: str
    labels: tuple[str, ...]
    training_manifest_sha256: str

    def validate(self) -> None:
        """校验模型清单能唯一、完整地描述推理标签。"""
        required = (
            self.model_name,
            self.model_version,
            self.model_revision,
            self.tokenizer_revision,
            self.schema_version,
            self.normalizer_version,
            self.training_manifest_sha256,
        )
        if any(not item.strip() for item in required):
            raise SchemaValidationError("模型清单包含空的必要版本字段。")
        if not self.labels or self.labels[0] != "O" or len(self.labels) != len(set(self.labels)):
            raise SchemaValidationError("模型标签必须以唯一的O标签开始。")
        if any(label not in BIO_LABELS for label in self.labels):
            raise SchemaValidationError("模型清单包含未知BIO标签。")

    def label_to_id(self) -> dict[str, int]:
        """按清单顺序生成稳定的标签编号。"""
        self.validate()
        return {label: index for index, label in enumerate(self.labels)}
