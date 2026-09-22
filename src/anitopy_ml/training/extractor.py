"""XLM-R 抽取模型的离线训练入口。"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from anitopy_ml import __version__
from anitopy_ml.errors import ConfigurationError, InputValidationError, SchemaValidationError
from anitopy_ml.modeling.config import ExtractorModelConfig
from anitopy_ml.modeling.decoder import constrained_bio_decode
from anitopy_ml.modeling.metrics import entity_metrics
from anitopy_ml.modeling.model import create_extractor_model
from anitopy_ml.modeling.runtime import require_training_dependencies
from anitopy_ml.training.control import EarlyStopping, TrainingProgress
from anitopy_ml.training.objectives import masked_multitask_loss
from anitopy_ml.training.checkpoint import read_checkpoint_metadata, write_checkpoint_metadata
from anitopy_ml.training.trainer import save_torch_training_state
from anitopy_ml.training.augment import augment_train_records


@dataclass(frozen=True, slots=True)
class ExtractorTrainingConfig:
    """正式抽取训练的可复现参数。"""

    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5

    def validate(self) -> None:
        """拒绝会导致训练失真的无效参数。"""
        if self.learning_rate <= 0 or self.gradient_accumulation_steps <= 0:
            raise SchemaValidationError("学习率和梯度累积步数必须大于0。")
        if self.gradient_clip_norm <= 0 or self.early_stopping_patience <= 0:
            raise SchemaValidationError("梯度裁剪阈值和早停耐心值必须大于0。")


def read_extractor_training_config(path: str | Path) -> ExtractorTrainingConfig:
    """读取中文键名的训练配置文件。"""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = ExtractorTrainingConfig(
            learning_rate=float(payload["学习率"]),
            gradient_accumulation_steps=int(payload["梯度累积步数"]),
            gradient_clip_norm=float(payload["梯度裁剪阈值"]),
            early_stopping_patience=int(payload["早停耐心值"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConfigurationError("抽取训练配置无效或缺少必要字段。") from error
    config.validate()
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取训练分区，并拒绝空分区。"""
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise InputValidationError(f"无法读取训练分区：{path}。") from error
    if not records:
        raise InputValidationError(f"训练分区为空：{path}。")
    return records


def _prepare_records(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """将字符级BIO记录转换为字符边界头输入，不丢弃跨词元边界监督。"""
    prepared: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for record in records:
        text = str(record.get("text", ""))
        characters = record.get("characters")
        labels = record.get("labels")
        if not text or not isinstance(characters, list) or "".join(map(str, characters)) != text:
            raise SchemaValidationError("训练记录的原文与字符序列不一致。")
        if not isinstance(labels, list):
            raise SchemaValidationError("训练记录缺少字符级BIO标签。")
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        offsets = [tuple(map(int, offset)) for offset in encoded["offset_mapping"]]
        owners = [0] * len(text)
        positions = [0] * len(text)
        for token_index, (start, end) in enumerate(offsets):
            if start == end:
                continue
            for character_index in range(start, end):
                owners[character_index] = token_index
                positions[character_index] = min(character_index - start, 31)
        prepared.append(
            {
                "input_ids": list(map(int, encoded["input_ids"])),
                "attention_mask": list(map(int, encoded["attention_mask"])),
                "character_token_indices": owners,
                "character_ids": [ord(character) for character in text],
                "character_positions": positions,
                "labels": [label_to_id[label] for label in labels],
                "loss_mask": [True] * len(labels),
            }
        )
    return prepared, issues


def _batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """按稳定顺序分批，确保同一输入得到可复现的训练顺序。"""
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def load_initial_model_weights(
    model: Any,
    *,
    source_directory: str | Path,
    output_directory: str | Path,
    model_config: ExtractorModelConfig,
) -> dict[str, object]:
    """只加载兼容检查点权重，为新数据版本重新开始优化。"""
    source = Path(source_directory)
    output = Path(output_directory)
    if source.resolve() == output.resolve():
        raise InputValidationError("权重初始化来源目录不能与新的训练输出目录相同。")
    metadata = read_checkpoint_metadata(source)
    source_config = metadata["模型配置"]
    expected = {
        "模型名称": model_config.model_name,
        "模型版本": model_config.model_revision,
        "标签": list(model_config.labels),
    }
    if any(source_config.get(key) != value for key, value in expected.items()):
        raise ConfigurationError("初始化检查点的模型名称、版本或BIO标签与当前训练不兼容。")
    weights_path = source / "best_model.pt"
    if not weights_path.is_file():
        raise InputValidationError(f"初始化目录缺少best_model.pt：{source}。")
    torch, _ = require_training_dependencies()
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ConfigurationError("初始化检查点权重无效或与当前抽取模型结构不兼容。") from error
    return {
        "方式": "仅加载best_model.pt；优化器、随机状态和校准文件均不继承",
        "来源目录": str(source),
        "来源权重SHA256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "来源训练数据清单SHA256": metadata["训练进度"]["training_manifest_sha256"],
    }


def _pad_character_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """分别填充分词器序列与字符序列，保持字符标签完整。"""
    token_length = max(len(item["input_ids"]) for item in batch)
    character_length = max(len(item["labels"]) for item in batch)
    result = {name: [] for name in ("input_ids", "attention_mask", "character_token_indices", "character_ids", "character_positions", "labels", "loss_mask")}
    for item in batch:
        token_padding = token_length - len(item["input_ids"])
        character_padding = character_length - len(item["labels"])
        result["input_ids"].append(item["input_ids"] + [pad_token_id] * token_padding)
        result["attention_mask"].append(item["attention_mask"] + [0] * token_padding)
        for name in ("character_token_indices", "character_ids", "character_positions"):
            result[name].append(item[name] + [0] * character_padding)
        result["labels"].append(item["labels"] + [-100] * character_padding)
        result["loss_mask"].append(item["loss_mask"] + [0] * character_padding)
    return result


def _evaluate(model: Any, records: list[dict[str, Any]], *, labels: tuple[str, ...], device: Any, pad_token_id: int) -> dict[str, float | int]:
    """计算对齐后词元序列的严格实体指标。"""
    torch, _ = require_training_dependencies()
    predicted_entities = 0
    expected_entities = 0
    correct_entities = 0
    model.eval()
    with torch.no_grad():
        for batch in _batches(records, 32):
            padded = _pad_character_batch(batch, pad_token_id)
            inputs = torch.tensor(padded["input_ids"], device=device)
            attention = torch.tensor(padded["attention_mask"], device=device)
            token_indices = torch.tensor(padded["character_token_indices"], device=device)
            character_ids = torch.tensor(padded["character_ids"], device=device)
            character_positions = torch.tensor(padded["character_positions"], device=device)
            logits = model(inputs, attention, token_indices, character_ids, character_positions)["character_logits"].detach().cpu().tolist()
            for row_logits, row_labels, row_mask in zip(logits, padded["labels"], padded["loss_mask"], strict=True):
                active_logits = [scores for scores, enabled in zip(row_logits, row_mask, strict=True) if enabled]
                active_expected = [labels[label_id] for label_id, enabled in zip(row_labels, row_mask, strict=True) if enabled]
                if active_logits:
                    metrics = entity_metrics(constrained_bio_decode(active_logits, labels), active_expected)
                    predicted_entities += int(metrics["预测实体数"])
                    expected_entities += int(metrics["真实实体数"])
                    correct_entities += int(metrics["正确实体数"])
    precision = correct_entities / predicted_entities if predicted_entities else 0.0
    recall = correct_entities / expected_entities if expected_entities else 0.0
    return {
        "预测实体数": predicted_entities,
        "真实实体数": expected_entities,
        "正确实体数": correct_entities,
        "精确率": precision,
        "召回率": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def train_extractor(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    training_config_path: str | Path,
    epochs: int,
    batch_size: int,
    max_length: int,
    device_name: str = "auto",
    augment_whitespace: bool = False,
    initial_model_dir: str | Path | None = None,
    seed: int = 20260918,
) -> dict[str, object]:
    """训练离线 XLM-R 抽取模型并返回中文实验报告。"""
    if epochs <= 0 or batch_size <= 0 or max_length < 8:
        raise InputValidationError("训练轮数、批大小必须大于0，最大长度不得小于8。")
    torch, transformers = require_training_dependencies()
    if seed < 0:
        raise InputValidationError("随机种子不能为负数。")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("训练设备只能是auto、cpu或cuda。")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("已指定CUDA训练，但当前PyTorch无法使用CUDA。")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    directory = Path(data_dir)
    manifest_path = directory / "training_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        labels = tuple(manifest["标签"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise InputValidationError("训练数据清单缺少完整BIO标签表。") from error
    model_config = ExtractorModelConfig(labels=labels)
    model_config.validate()
    config = read_extractor_training_config(training_config_path)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config.model_name,
            revision=model_config.tokenizer_revision,
            local_files_only=True,
            use_fast=True,
        )
    except OSError as error:
        raise ConfigurationError("本地缺少XLM-R分词器文件；请先下载并缓存模型后再离线训练。") from error
    if not getattr(tokenizer, "is_fast", False):
        raise ConfigurationError("XLM-R训练必须使用支持偏移映射的快速分词器。")
    output = Path(output_dir)
    try:
        model = create_extractor_model(model_config, local_files_only=True).to(device)
    except OSError as error:
        raise ConfigurationError("本地缺少XLM-R模型权重；请先下载并缓存模型后再离线训练。") from error
    weight_initialization = (
        load_initial_model_weights(
            model,
            source_directory=initial_model_dir,
            output_directory=output,
            model_config=model_config,
        )
        if initial_model_dir is not None
        else None
    )
    raw_train = _read_jsonl(directory / "train.jsonl")
    augmented_train = augment_train_records(raw_train) if augment_whitespace else []
    train, train_issues = _prepare_records(
        [*raw_train, *augmented_train], tokenizer=tokenizer, label_to_id=model_config.label_to_id(), max_length=max_length
    )
    validation, validation_issues = _prepare_records(
        _read_jsonl(directory / "validation.jsonl"), tokenizer=tokenizer, label_to_id=model_config.label_to_id(), max_length=max_length
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper = EarlyStopping(config.early_stopping_patience)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    global_step = 0
    started_at = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batch_groups = _batches(train, batch_size)
        for batch_index, batch in enumerate(batch_groups, start=1):
            padded = _pad_character_batch(batch, int(tokenizer.pad_token_id))
            inputs = torch.tensor(padded["input_ids"], device=device)
            attention = torch.tensor(padded["attention_mask"], device=device)
            target = torch.tensor(padded["labels"], device=device)
            loss_mask = torch.tensor(padded["loss_mask"], device=device)
            token_indices = torch.tensor(padded["character_token_indices"], device=device)
            character_ids = torch.tensor(padded["character_ids"], device=device)
            character_positions = torch.tensor(padded["character_positions"], device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output_values = model(inputs, attention, token_indices, character_ids, character_positions)
                loss = masked_multitask_loss(output_values["character_logits"], target, loss_mask)["loss"]
                scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            total_loss += float(loss.detach().item())
            if batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(batch_groups):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        metrics = _evaluate(model, validation, labels=labels, device=device, pad_token_id=int(tokenizer.pad_token_id))
        validation_f1 = float(metrics["F1"])
        history.append({"轮次": epoch, "训练损失": total_loss / len(batch_groups), "验证指标": metrics})
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            torch.save(model.state_dict(), output / "best_model.pt")
            tokenizer.save_pretrained(output / "tokenizer")
        should_stop = stopper.update(validation_f1)
        save_torch_training_state(output, model=model, optimizer=optimizer)
        write_checkpoint_metadata(
            output,
            progress=TrainingProgress(
                epoch=epoch,
                global_step=global_step,
                best_validation_f1=best_f1,
                early_stopping={"patience": config.early_stopping_patience, "bad_epochs": stopper.bad_epochs},
                training_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            ),
            model_config={"模型名称": model_config.model_name, "模型版本": model_config.model_revision, "标签": list(labels)},
            weight_initialization=weight_initialization,
        )
        if should_stop:
            break
    report = {
        "说明": "这是模板分布内的XLM-R实体级评测，不能代表真实发布标题准确率。",
        "设备": str(device),
        "训练样本数": len(train),
        "原始训练样本数": len(raw_train),
        "空白增强样本数": len(augmented_train),
        "验证样本数": len(validation),
        "训练轮次": len(history),
        "最佳验证实体F1": best_f1,
        "耗时秒": round(perf_counter() - started_at, 2),
        "训练数据清单SHA256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "权重初始化": weight_initialization,
        "随机种子": seed,
        "训练配置": {
            "学习率": config.learning_rate,
            "梯度累积步数": config.gradient_accumulation_steps,
            "梯度裁剪阈值": config.gradient_clip_norm,
            "早停耐心值": config.early_stopping_patience,
            "最大长度": max_length,
            "批大小": batch_size,
        },
        "环境": {"项目版本": __version__, "Torch版本": torch.__version__, "Transformers版本": transformers.__version__},
        "对齐问题": {"训练": train_issues, "验证": validation_issues},
        "训练历史": history,
    }
    (output / "训练报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
