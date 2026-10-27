"""XLM-R 抽取模型的离线训练入口。"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from itertools import batched, islice
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator

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
from anitopy_ml.training.trainer import load_torch_training_state, save_torch_training_state
from anitopy_ml.training.streaming import iter_jsonl, scan_partition, training_records


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


def _prepare_records(
    records: Iterable[dict[str, Any]],
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
        if not isinstance(labels, list) or len(labels) != len(text) or any(label not in label_to_id for label in labels):
            raise SchemaValidationError("训练记录缺少字符级BIO标签。")
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        offsets = [tuple(map(int, offset)) for offset in encoded["offset_mapping"]]
        if max((end for _, end in offsets), default=0) < len(text.rstrip()):
            raise SchemaValidationError("训练标题超过分词最大长度，请增大--max-length，避免对截断字符提供错误监督。")
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


def _batches(records: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """按稳定顺序分批，确保同一输入得到可复现的训练顺序。"""
    for group in batched(records, batch_size):
        yield list(group)


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


def _evaluate(model: Any, records: Iterable[dict[str, Any]], *, labels: tuple[str, ...], device: Any, pad_token_id: int, batch_size: int = 32) -> dict[str, float | int]:
    """计算对齐后词元序列的严格实体指标。"""
    torch, _ = require_training_dependencies()
    predicted_entities = 0
    expected_entities = 0
    correct_entities = 0
    model.eval()
    with torch.no_grad():
        for batch in _batches(records, batch_size):
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
    resume: bool = False,
    max_steps: int | None = None,
    checkpoint_steps: int = 1000,
    shuffle_buffer_size: int = 1024,
    gradient_checkpointing: bool = False,
) -> dict[str, object]:
    """训练离线 XLM-R 抽取模型并返回中文实验报告。"""
    if epochs <= 0 or batch_size <= 0 or max_length < 8:
        raise InputValidationError("训练轮数、批大小必须大于0，最大长度不得小于8。")
    if checkpoint_steps <= 0 or shuffle_buffer_size <= 0 or (max_steps is not None and max_steps <= 0):
        raise InputValidationError("检查点间隔、洗牌缓冲区和本次最大优化器步数必须大于0。")
    if resume and initial_model_dir is not None:
        raise InputValidationError("--resume与--init-model不能同时使用。")
    output = Path(output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise InputValidationError("新的训练输出目录必须为空；继续同一训练请使用--resume。")
    if resume and not (output / "training_state.pt").is_file():
        raise InputValidationError("恢复目录缺少training_state.pt。")
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
    try:
        model = create_extractor_model(model_config, local_files_only=True).to(device)
    except OSError as error:
        raise ConfigurationError("本地缺少XLM-R模型权重；请先下载并缓存模型后再离线训练。") from error
    if gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
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
    train_stats = scan_partition(directory / "train.jsonl", augment_whitespace=augment_whitespace)
    validation_stats = scan_partition(directory / "validation.jsonl")
    for name, stats in (("train", train_stats), ("validation", validation_stats)):
        declared = manifest.get("分区", {}).get(name, {})
        if declared.get("SHA256", stats["SHA256"]) != stats["SHA256"] or declared.get("样本数", stats["样本数"]) != stats["样本数"]:
            raise InputValidationError(f"{name}分区实际内容与训练清单不一致。")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    settings = {
        "version": 1,
        "manifest_sha256": manifest_hash,
        "train": train_stats,
        "validation": validation_stats,
        "model_config": asdict(model_config),
        "config": asdict(config),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "augment_whitespace": augment_whitespace,
        "shuffle_buffer_size": shuffle_buffer_size,
        "gradient_checkpointing": gradient_checkpointing,
        "device_type": device.type,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    train_count = train_stats["样本数"] + train_stats["增强样本数"]
    total_batches = (train_count + batch_size - 1) // batch_size

    def prepare_stream(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for group in _batches(records, batch_size):
            prepared, _ = _prepare_records(group, tokenizer=tokenizer, label_to_id=model_config.label_to_id(), max_length=max_length)
            yield from prepared

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper = EarlyStopping(config.early_stopping_patience)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    global_step = 0
    batch_cursor = 0
    total_loss = 0.0
    elapsed = 0.0
    if resume:
        restored = load_torch_training_state(output, model=model, optimizer=optimizer, scaler=scaler, expected_settings=settings)
        history = restored["history"]
        global_step = restored["global_step"]
        batch_cursor = restored["batch_cursor"]
        total_loss = restored["total_loss"]
        best_f1 = restored["best_f1"]
        stopper = EarlyStopping(**restored["stopper"])
        weight_initialization = restored["weight_initialization"]
        elapsed = restored["elapsed"]
        # 原子状态文件是恢复进度的依据；旁路元数据可能在断电时尚未更新。
        if len(history) > epochs or not 0 <= batch_cursor < total_batches:
            raise SchemaValidationError("检查点中的轮次或批次进度无效。")
        if batch_cursor and batch_cursor % config.gradient_accumulation_steps:
            raise SchemaValidationError("检查点不在完整梯度累积边界，无法安全恢复。")
    started_at = perf_counter()
    run_steps = 0

    def save_checkpoint() -> None:
        # 只在优化器更新后保存；恢复时无需保存未提交的梯度。
        extra = {
            "settings": settings, "history": history, "global_step": global_step,
            "batch_cursor": batch_cursor, "total_loss": total_loss, "best_f1": best_f1,
            "stopper": asdict(stopper), "weight_initialization": weight_initialization,
            "elapsed": elapsed + perf_counter() - started_at,
        }
        save_torch_training_state(output, model=model, optimizer=optimizer, scaler=scaler, extra_state=extra)
        write_checkpoint_metadata(
            output,
            progress=TrainingProgress(
                epoch=len(history), global_step=global_step,
                best_validation_f1=best_f1 if best_f1 >= 0 else None,
                early_stopping=asdict(stopper), training_manifest_sha256=manifest_hash,
            ),
            model_config={"模型名称": model_config.model_name, "模型版本": model_config.model_revision, "标签": list(labels)},
            weight_initialization=weight_initialization,
        )

    while len(history) < epochs and stopper.bad_epochs < stopper.patience:
        epoch = len(history) + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw = training_records(
            directory / "train.jsonl", augment_whitespace=augment_whitespace,
            buffer_size=shuffle_buffer_size, seed=seed + epoch,
        )
        # 跳过已提交的原始记录，避免恢复时重复分词和反向传播。
        remaining = prepare_stream(islice(raw, batch_cursor * batch_size, None))
        for accumulation_group in batched(_batches(remaining, batch_size), config.gradient_accumulation_steps):
            for batch in accumulation_group:
                padded = _pad_character_batch(batch, int(tokenizer.pad_token_id))
                tensors = {name: torch.tensor(value, device=device) for name, value in padded.items()}
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output_values = model(*(tensors[name] for name in ("input_ids", "attention_mask", "character_token_indices", "character_ids", "character_positions")))
                    loss = masked_multitask_loss(output_values["character_logits"], tensors["labels"], tensors["loss_mask"])["loss"]
                    scaled_loss = loss / len(accumulation_group)
                scaler.scale(scaled_loss).backward()
                total_loss += float(loss.detach().item())
                batch_cursor += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            run_steps += 1
            epoch_finished = batch_cursor == total_batches
            if epoch_finished:
                metrics = _evaluate(model, prepare_stream(iter_jsonl(directory / "validation.jsonl")), labels=labels, device=device, pad_token_id=int(tokenizer.pad_token_id), batch_size=batch_size)
                validation_f1 = float(metrics["F1"])
                history.append({"轮次": epoch, "训练损失": total_loss / total_batches, "验证指标": metrics})
                if validation_f1 > best_f1:
                    best_f1 = validation_f1
                    temporary_weights = output / "best_model.pt.tmp"
                    torch.save(model.state_dict(), temporary_weights)
                    temporary_weights.replace(output / "best_model.pt")
                    tokenizer.save_pretrained(output / "tokenizer")
                stopper.update(validation_f1)
                batch_cursor = 0
                total_loss = 0.0
            stage_finished = max_steps is not None and run_steps >= max_steps
            if epoch_finished or stage_finished or global_step % checkpoint_steps == 0:
                save_checkpoint()
            if global_step % 100 == 0 or stage_finished or epoch_finished:
                print(f"训练进度：已完成{len(history)}轮，全局优化器步数{global_step}，当前轮批次{batch_cursor}/{total_batches}。", flush=True)
            if stage_finished:
                break
        if max_steps is not None and run_steps >= max_steps:
            break
    finished = len(history) >= epochs or stopper.bad_epochs >= stopper.patience
    report = {
        "说明": "这是模板分布内的XLM-R实体级评测，不能代表真实发布标题准确率。",
        "设备": str(device),
        "状态": "训练完成" if finished else "分段暂停",
        "全局优化器步数": global_step,
        "本次优化器步数": run_steps,
        "当前轮已完成批次": batch_cursor,
        "每轮批次数": total_batches,
        "当前轮训练损失": total_loss / batch_cursor if batch_cursor else None,
        "训练样本数": train_count,
        "原始训练样本数": train_stats["样本数"],
        "空白增强样本数": train_stats["增强样本数"],
        "验证样本数": validation_stats["样本数"],
        "训练轮次": len(history),
        "最佳验证实体F1": best_f1 if best_f1 >= 0 else None,
        "耗时秒": round(elapsed + perf_counter() - started_at, 2),
        "训练数据清单SHA256": manifest_hash,
        "权重初始化": weight_initialization,
        "随机种子": seed,
        "训练配置": {
            "学习率": config.learning_rate,
            "梯度累积步数": config.gradient_accumulation_steps,
            "梯度裁剪阈值": config.gradient_clip_norm,
            "早停耐心值": config.early_stopping_patience,
            "最大长度": max_length,
            "批大小": batch_size,
            "洗牌缓冲区大小": shuffle_buffer_size,
            "梯度检查点": gradient_checkpointing,
        },
        "环境": {"项目版本": __version__, "Torch版本": torch.__version__, "Transformers版本": transformers.__version__},
        "对齐问题": {"训练": [], "验证": []},
        "训练历史": history,
    }
    temporary_report = output / "训练报告.json.tmp"
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(output / "训练报告.json")
    return report
