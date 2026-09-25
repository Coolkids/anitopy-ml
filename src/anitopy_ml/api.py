"""面向调用方的离线媒体标题解析接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from anitopy_ml.errors import ConfigurationError, InputValidationError
from anitopy_ml.inference.character import CharacterTagger
from anitopy_ml.inference.calibration import (
    AcceptancePolicy,
    ConfidenceCalibrator,
    load_acceptance_policy,
    load_calibrator,
)
from anitopy_ml.inference.decoding import build_parse_result
from anitopy_ml.modeling.decoder import constrained_bio_decode
from anitopy_ml.modeling.runtime import require_training_dependencies
from anitopy_ml.schemas import BIO_LABELS, ParseResult


class MediaParser:
    """加载本地字符连通性模型并提供单条、批量解析。"""

    def __init__(
        self,
        *,
        model: Any,
        vocabulary: dict[str, int] | None,
        tokenizer: Any | None,
        labels: Sequence[str],
        device: Any,
        model_version: str,
        character_boundary: bool = False,
        calibrator: ConfidenceCalibrator | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
    ) -> None:
        self._model = model
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._labels = tuple(labels)
        self._device = device
        self._model_version = model_version
        self._character_boundary = character_boundary
        self._calibrator = calibrator
        self._acceptance_policy = acceptance_policy

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        offline: bool = True,
        use_calibration: bool = True,
    ) -> "MediaParser":
        """从本地字符或XLM-R检查点加载模型；当前不支持联网下载。"""
        if not offline:
            raise ConfigurationError("当前加载器只支持离线模型目录，不会联网下载模型。")
        directory = Path(model_dir)
        if directory.is_dir() and (directory / "best_model.pt").is_file():
            return cls._load_extractor(directory, device=device, use_calibration=use_calibration)
        target = directory / "character_tagger.pt" if directory.is_dir() else directory
        if not target.is_file():
            raise ConfigurationError("未找到字符模型文件 character_tagger.pt。")
        torch, _ = require_training_dependencies()
        if device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("已请求CUDA推理，但当前Torch未检测到可用CUDA设备。")
        selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        try:
            payload = torch.load(target, map_location=selected_device, weights_only=True)
            vocabulary = payload["词表"]
            labels = payload["标签"]
            state = payload["模型"]
        except (KeyError, TypeError, RuntimeError) as error:
            raise ConfigurationError("字符模型文件结构无效，无法安全加载。") from error
        if (
            not isinstance(vocabulary, dict)
            or not vocabulary
            or not isinstance(labels, list)
            or not labels
            or any(label not in BIO_LABELS for label in labels)
        ):
            raise ConfigurationError("字符模型的词表或BIO标签表无效。")
        model = CharacterTagger(len(vocabulary), len(labels)).to(selected_device)
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise ConfigurationError("字符模型权重与词表或标签表不匹配。") from error
        model.eval()
        return cls(
            model=model,
            vocabulary={str(key): int(value) for key, value in vocabulary.items()},
            tokenizer=None,
            labels=labels,
            device=selected_device,
            model_version="字符级连通性模型",
        )

    @classmethod
    def _load_extractor(
        cls,
        directory: Path,
        *,
        device: Literal["auto", "cpu", "cuda"],
        use_calibration: bool,
    ) -> "MediaParser":
        """加载训练产物中的XLM-R权重和同目录分词器。"""
        from anitopy_ml.modeling.config import ExtractorModelConfig
        from anitopy_ml.modeling.model import create_extractor_model

        torch, transformers = require_training_dependencies()
        if device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("已请求CUDA推理，但当前Torch未检测到可用CUDA设备。")
        selected_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        previous_verbosity = transformers.logging.get_verbosity()
        progress_enabled = transformers.logging.is_progress_bar_enabled()
        try:
            # CLI 的标准输出必须只保留解析结果，屏蔽第三方权重加载报告。
            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
            metadata = __import__("json").loads((directory / "checkpoint_metadata.json").read_text(encoding="utf-8"))
            labels = tuple(metadata["模型配置"]["标签"])
            config = ExtractorModelConfig(
                model_name=str(metadata["模型配置"]["模型名称"]),
                model_revision=str(metadata["模型配置"]["模型版本"]),
                labels=labels,
            )
            config.validate()
            tokenizer = transformers.AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, use_fast=True)
            state = torch.load(directory / "best_model.pt", map_location=selected_device, weights_only=True)
            model = create_extractor_model(config, local_files_only=True).to(selected_device)
            character_boundary = any(str(key).startswith("character_") for key in state)
            model.load_state_dict(state, strict=character_boundary)
        except (OSError, KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ConfigurationError("XLM-R检查点不完整、权重不匹配或本地基础模型缓存缺失。") from error
        finally:
            transformers.logging.set_verbosity(previous_verbosity)
            if progress_enabled:
                transformers.logging.enable_progress_bar()
        if not getattr(tokenizer, "is_fast", False):
            raise ConfigurationError("XLM-R检查点缺少支持偏移映射的快速分词器。")
        model.eval()
        calibrator = None
        acceptance_policy = None
        calibration_path = directory / "calibration.json"
        if use_calibration and calibration_path.is_file():
            calibrator = load_calibrator(calibration_path, model_directory=directory)
            policy_path = directory / "acceptance_policy.json"
            if policy_path.is_file():
                acceptance_policy = load_acceptance_policy(policy_path, model_directory=directory)
        return cls(
            model=model,
            vocabulary=None,
            tokenizer=tokenizer,
            labels=labels,
            device=selected_device,
            model_version=f"XLM-R抽取模型：{directory.name}",
            character_boundary=character_boundary,
            calibrator=calibrator,
            acceptance_policy=acceptance_policy,
        )

    def parse(self, title: str) -> ParseResult:
        """离线解析单条标题并返回模型片段及确定性格式字段。"""
        if not isinstance(title, str) or not title.strip():
            raise InputValidationError("标题不能为空。")
        if self._tokenizer is not None:
            return self._parse_extractor_batch([title])[0]
        torch, _ = require_training_dependencies()
        if self._vocabulary is None:
            raise ConfigurationError("字符模型词表缺失。")
        unknown_id = self._vocabulary.get("<未知>")
        if unknown_id is None:
            raise ConfigurationError("字符模型词表缺少未知字符标记。")
        input_ids = [self._vocabulary.get(character, unknown_id) for character in title]
        inputs = torch.tensor([input_ids], device=self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(inputs)[0]
            probabilities = torch.softmax(logits, dim=-1)
        decoded = constrained_bio_decode(logits.detach().cpu().tolist(), self._labels)
        label_to_id = {label: index for index, label in enumerate(self._labels)}
        confidences = [
            float(probabilities[index, label_to_id[label]].item())
            for index, label in enumerate(decoded)
        ]
        return build_parse_result(
            title,
            decoded,
            confidences=confidences,
            model_version=self._model_version,
        )

    def _parse_extractor(self, title: str) -> ParseResult:
        """以XLM-R词元预测回填字符级BIO证据。"""
        torch, _ = require_training_dependencies()
        if self._tokenizer is None:
            raise ConfigurationError("XLM-R分词器缺失。")
        encoded = self._tokenizer(
            title,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"][0].tolist()]
        inputs = encoded["input_ids"].to(self._device)
        attention = encoded["attention_mask"].to(self._device)
        self._model.eval()
        with torch.no_grad():
            if self._character_boundary:
                owners = [0] * len(title)
                positions = [0] * len(title)
                for token_index, (start, end) in enumerate(offsets):
                    if start != end:
                        for character_index in range(start, end):
                            owners[character_index] = token_index
                            positions[character_index] = min(character_index - start, 31)
                logits = self._model(
                    inputs,
                    attention,
                    torch.tensor([owners], device=self._device),
                    torch.tensor([[ord(character) for character in title]], device=self._device),
                    torch.tensor([positions], device=self._device),
                )["character_logits"][0]
                probabilities = torch.softmax(logits, dim=-1)
                decoded = constrained_bio_decode(logits.detach().cpu().tolist(), self._labels)
                label_to_id = {label: index for index, label in enumerate(self._labels)}
                confidences = [float(probabilities[index, label_to_id[label]].item()) for index, label in enumerate(decoded)]
                result = build_parse_result(
                    title,
                    decoded,
                    confidences=confidences,
                    model_version=self._model_version,
                    confidence_transform=self._calibrator.calibrate if self._calibrator else None,
                )
                return self._apply_acceptance_policy(result)
            logits = self._model(inputs, attention)["token_logits"][0]
            probabilities = torch.softmax(logits, dim=-1)
        active = [index for index, (start, end) in enumerate(offsets) if start != end]
        decoded = constrained_bio_decode([logits[index].detach().cpu().tolist() for index in active], self._labels)
        label_to_id = {label: index for index, label in enumerate(self._labels)}
        character_labels = ["O"] * len(title)
        confidences = [0.0] * len(title)
        for token_index, label in zip(active, decoded, strict=True):
            start, end = offsets[token_index]
            confidence = float(probabilities[token_index, label_to_id[label]].item())
            if label == "O":
                continue
            prefix, entity_type = label.split("-", maxsplit=1)
            for index in range(start, end):
                character_labels[index] = f"B-{entity_type}" if prefix == "B" and index == start else f"I-{entity_type}"
                confidences[index] = confidence
        result = build_parse_result(
            title,
            character_labels,
            confidences=confidences,
            model_version=self._model_version,
            confidence_transform=self._calibrator.calibrate if self._calibrator else None,
        )
        if len(active) >= 254:
            result.warnings.append("标题超过当前XLM-R单窗口长度，尾部文本尚未参与模型预测。")
        return self._apply_acceptance_policy(result)

    def _parse_extractor_batch(self, titles: Sequence[str]) -> list[ParseResult]:
        """批量执行字符边界 XLM-R 推理，保留单条解析的原文证据语义。"""
        if not titles:
            return []
        if not self._character_boundary:
            return [self._parse_extractor(title) for title in titles]
        torch, _ = require_training_dependencies()
        if self._tokenizer is None:
            raise ConfigurationError("XLM-R分词器缺失。")
        encoded = self._tokenizer(
            list(titles),
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        offset_batches = [
            [tuple(map(int, item)) for item in offsets]
            for offsets in encoded["offset_mapping"].tolist()
        ]
        max_characters = max(len(title) for title in titles)
        owners_rows: list[list[int]] = []
        character_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        for title, offsets in zip(titles, offset_batches, strict=True):
            owners = [0] * max_characters
            positions = [0] * max_characters
            for token_index, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                for character_index in range(start, min(end, len(title))):
                    owners[character_index] = token_index
                    positions[character_index] = min(character_index - start, 31)
            owners_rows.append(owners)
            character_rows.append([ord(character) for character in title] + [0] * (max_characters - len(title)))
            position_rows.append(positions)
        inputs = encoded["input_ids"].to(self._device)
        attention = encoded["attention_mask"].to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(
                inputs,
                attention,
                torch.tensor(owners_rows, device=self._device),
                torch.tensor(character_rows, device=self._device),
                torch.tensor(position_rows, device=self._device),
            )["character_logits"]
            probabilities = torch.softmax(logits, dim=-1)
        label_to_id = {label: index for index, label in enumerate(self._labels)}
        results: list[ParseResult] = []
        for row_index, title in enumerate(titles):
            title_logits = logits[row_index, : len(title)]
            decoded = constrained_bio_decode(title_logits.detach().cpu().tolist(), self._labels)
            confidences = [
                float(probabilities[row_index, character_index, label_to_id[label]].item())
                for character_index, label in enumerate(decoded)
            ]
            result = build_parse_result(
                title,
                decoded,
                confidences=confidences,
                model_version=self._model_version,
                confidence_transform=self._calibrator.calibrate if self._calibrator else None,
            )
            results.append(self._apply_acceptance_policy(result))
        return results

    def _apply_acceptance_policy(self, result: ParseResult) -> ParseResult:
        """保留低置信度原文证据，同时明确哪些模型字段需要人工复核。"""
        if self._acceptance_policy is None:
            return result
        review_fields = [
            label
            for label, evidence in result.evidence.items()
            if evidence.source == "model"
            and self._acceptance_policy.decide(label, evidence.confidence, evidence.calibrated) != "自动接收"
        ]
        if review_fields:
            result.warnings.append(f"以下模型字段未达到自动接收条件，建议人工复核：{', '.join(review_fields)}。")
            title = result.evidence.get("title")
            if title is not None and self._acceptance_policy.decide("title", title.confidence, title.calibrated) != "自动接收":
                result.status = "partial"
        return result

    def acceptance_decision(self, label: str, confidence: float | None, calibrated: bool) -> str:
        """返回当前模型包对单个字段的接受建议，供离线评测复用同一策略。"""
        if self._acceptance_policy is None:
            return "未校准" if not calibrated else "需要复核"
        return self._acceptance_policy.decide(label, confidence, calibrated)

    @property
    def model_version(self) -> str:
        """返回当前加载模型的稳定版本说明。"""
        return self._model_version

    def parse_batch(
        self,
        titles: Sequence[str],
        *,
        on_error: Literal["raise", "record"] = "record",
    ) -> list[ParseResult | dict[str, object]]:
        """按输入顺序解析，选择记录错误时不会中断其余标题。"""
        if on_error not in {"raise", "record"}:
            raise ConfigurationError("批量错误策略只能是raise或record。")
        if self._tokenizer is not None:
            results: list[ParseResult | dict[str, object] | None] = [None] * len(titles)
            valid: list[tuple[int, str]] = []
            for index, title in enumerate(titles):
                if isinstance(title, str) and title.strip():
                    valid.append((index, title))
                    continue
                error = InputValidationError("标题不能为空。")
                if on_error == "raise":
                    raise error
                results[index] = {"raw_text": title, "status": "error", "warnings": [str(error)]}
            for offset in range(0, len(valid), 32):
                batch = valid[offset : offset + 32]
                parsed = self._parse_extractor_batch([title for _, title in batch])
                for (index, _), result in zip(batch, parsed, strict=True):
                    results[index] = result
            return [result for result in results if result is not None]
        results: list[ParseResult | dict[str, object]] = []
        for title in titles:
            try:
                results.append(self.parse(title))
            except (ConfigurationError, InputValidationError) as error:
                if on_error == "raise":
                    raise
                results.append({"raw_text": title, "status": "error", "warnings": [str(error)]})
        return results
