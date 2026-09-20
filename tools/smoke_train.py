"""基于当前人工金标的轻量字符标注器连通性训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

# 允许直接执行本脚本，无需预先以可编辑模式安装项目。
项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

import torch

from anitopy_ml.training.checkpoint import read_checkpoint_metadata, write_checkpoint_metadata
from anitopy_ml.training.collator import dynamic_pad
from anitopy_ml.training.control import TrainingProgress
from anitopy_ml.training.trainer import load_torch_training_state, save_torch_training_state
from anitopy_ml.inference.character import CharacterTagger


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取字符级BIO训练数据。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_vocabulary(records: list[dict[str, Any]]) -> dict[str, int]:
    """从训练分区构建稳定字符词表。"""
    characters = sorted({character for record in records for character in record["characters"]})
    return {"<填充>": 0, "<未知>": 1, **{character: index + 2 for index, character in enumerate(characters)}}


def encode_records(
    records: list[dict[str, Any]],
    vocabulary: dict[str, int],
    label_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    """将字符与标签转换为可动态填充的整数序列。"""
    return [
        {
            "input_ids": [vocabulary.get(character, vocabulary["<未知>"]) for character in record["characters"]],
            "attention_mask": [1] * len(record["characters"]),
            "labels": [label_to_id[label] for label in record["labels"]],
            "loss_mask": [True] * len(record["labels"]),
        }
        for record in records
    ]


def batches(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """按固定顺序分批，保证连通性训练可重复。"""
    return [records[index : index + size] for index in range(0, len(records), size)]


def evaluate(model: CharacterTagger, records: list[dict[str, Any]], device: torch.device) -> float:
    """返回全部非填充字符上的标签准确率。"""
    if not records:
        return 0.0
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in batches(records, 8):
            padded = dynamic_pad(batch, pad_token_id=0)
            inputs = torch.tensor(padded["input_ids"], device=device)
            labels = torch.tensor(padded["labels"], device=device)
            mask = torch.tensor(padded["loss_mask"], dtype=torch.bool, device=device)
            predicted = model(inputs).argmax(dim=-1)
            correct += int((predicted[mask] == labels[mask]).sum().item())
            total += int(mask.sum().item())
    return correct / total if total else 0.0


def verify_checkpoint_restore(
    *,
    output_dir: Path,
    model: CharacterTagger,
    optimizer: torch.optim.Optimizer,
    vocabulary: dict[str, int],
    labels: list[str],
    reference_record: dict[str, Any],
    device: torch.device,
    manifest_sha256: str,
    epoch: int,
    global_step: int,
) -> bool:
    """保存训练状态，恢复到新模型并验证同一输入的输出完全一致。"""
    model.eval()
    padded = dynamic_pad([reference_record], pad_token_id=0)
    inputs = torch.tensor(padded["input_ids"], device=device)
    with torch.no_grad():
        expected = model(inputs).detach().cpu()

    save_torch_training_state(output_dir, model=model, optimizer=optimizer)
    write_checkpoint_metadata(
        output_dir,
        progress=TrainingProgress(
            epoch=epoch,
            global_step=global_step,
            best_validation_f1=None,
            early_stopping={},
            training_manifest_sha256=manifest_sha256,
        ),
        model_config={
            "类型": "字符级连通性检查模型",
            "词表大小": len(vocabulary),
            "标签数": len(labels),
        },
    )

    restored = CharacterTagger(len(vocabulary), len(labels)).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    load_torch_training_state(
        output_dir,
        model=restored,
        optimizer=restored_optimizer,
        map_location=str(device),
    )
    restored.eval()
    with torch.no_grad():
        actual = restored(inputs).detach().cpu()
    metadata = read_checkpoint_metadata(output_dir)
    consistent = bool(torch.equal(expected, actual)) and metadata["训练进度"]["epoch"] == epoch
    if not consistent:
        raise RuntimeError("检查点恢复后的模型输出或训练进度与保存前不一致。")
    return consistent


def main() -> None:
    """执行小样本连通性训练并保存结果。"""
    parser = argparse.ArgumentParser(description="使用当前人工金标执行轻量连通性训练。")
    parser.add_argument("--data-dir", required=True, help="字符级BIO数据目录。")
    parser.add_argument("--output-dir", required=True, help="训练结果目录。")
    parser.add_argument("--epochs", type=int, default=8, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小。")
    arguments = parser.parse_args()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        parser.error("训练轮数和批大小必须大于0。")

    random.seed(20260917)
    torch.manual_seed(20260917)
    data_dir = Path(arguments.data_dir)
    manifest_path = data_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = list(manifest["标签"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    train = read_jsonl(data_dir / "train.jsonl")
    validation = read_jsonl(data_dir / "validation.jsonl")
    if not train:
        parser.error("训练分区没有合格样本，不能执行连通性训练。")
    vocabulary = make_vocabulary(train)
    encoded_train = encode_records(train, vocabulary, label_to_id)
    encoded_validation = encode_records(validation, vocabulary, label_to_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterTagger(len(vocabulary), len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in batches(encoded_train, arguments.batch_size):
            padded = dynamic_pad(batch, pad_token_id=0)
            inputs = torch.tensor(padded["input_ids"], device=device)
            targets = torch.tensor(padded["labels"], device=device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, len(labels)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append(
            {
                "轮次": epoch,
                "训练损失": sum(losses) / len(losses),
                "验证字符准确率": evaluate(model, encoded_validation, device),
            }
        )
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"模型": model.state_dict(), "词表": vocabulary, "标签": labels}, output_dir / "character_tagger.pt")
    restore_verified = verify_checkpoint_restore(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        vocabulary=vocabulary,
        labels=labels,
        reference_record=encoded_validation[0] if encoded_validation else encoded_train[0],
        device=device,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        epoch=arguments.epochs,
        global_step=arguments.epochs * len(batches(encoded_train, arguments.batch_size)),
    )
    report = {
        "说明": "这是字符级连通性训练，不是正式XLM-R训练或效果评估。",
        "设备": str(device),
        "训练样本数": len(train),
        "验证样本数": len(validation),
        "训练历史": history,
        "检查点恢复自检通过": restore_verified,
    }
    (output_dir / "训练报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"连通性训练完成：设备 {device}，结果已写入 {output_dir}。")


if __name__ == "__main__":
    main()
