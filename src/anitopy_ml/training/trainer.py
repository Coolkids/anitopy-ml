"""可恢复的抽取模型训练循环。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.modeling.runtime import require_training_dependencies
from anitopy_ml.training.objectives import masked_multitask_loss


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """训练循环的最小可复现参数。"""

    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0

    def validate(self) -> None:
        """校验不会形成无效的优化器步数或裁剪参数。"""
        if self.learning_rate <= 0 or self.gradient_accumulation_steps <= 0:
            raise SchemaValidationError("学习率和梯度累积步数必须大于0。")
        if self.max_gradient_norm <= 0:
            raise SchemaValidationError("梯度裁剪阈值必须大于0。")


def train_one_epoch(
    model: Any,
    batches: Iterable[dict[str, Any]],
    optimizer: Any,
    config: TrainerConfig,
) -> dict[str, float | int]:
    """执行一个 epoch，支持梯度累积并返回可记录的平均损失。"""
    config.validate()
    torch, _ = require_training_dependencies()
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    batch_count = 0
    optimizer_steps = 0
    for batch_count, batch in enumerate(batches, start=1):
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        has_media_labels = batch.get("media_labels") is not None
        losses = masked_multitask_loss(
            output["token_logits"],
            batch["labels"],
            batch["loss_mask"],
            media_logits=output.get("media_logits") if has_media_labels else None,
            media_labels=batch.get("media_labels") if has_media_labels else None,
        )
        loss = losses["loss"] / config.gradient_accumulation_steps
        loss.backward()
        total_loss += float(losses["loss"].detach().item())
        if batch_count % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
    if batch_count and batch_count % config.gradient_accumulation_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
        optimizer.step()
        optimizer.zero_grad()
        optimizer_steps += 1
    return {
        "批次数": batch_count,
        "优化器步数": optimizer_steps,
        "平均损失": total_loss / batch_count if batch_count else 0.0,
    }


def save_torch_training_state(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
) -> Path:
    """保存模型、优化器、调度器和 Torch 随机状态，供中断后恢复。"""
    torch, _ = require_training_dependencies()
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "training_state.pt"
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, target)
    return target


def load_torch_training_state(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    map_location: str = "cpu",
) -> None:
    """恢复模型、优化器、调度器与随机状态。"""
    torch, _ = require_training_dependencies()
    target = Path(directory) / "training_state.pt"
    if not target.is_file():
        raise SchemaValidationError("未找到Torch训练状态文件。")
    state = torch.load(target, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    # map_location 为 CUDA 时，torch.load 会一并迁移随机数状态；
    # 但 torch.set_rng_state 只能接收 CPU 上的 uint8 张量。
    cpu_rng_state = state["torch_rng_state"]
    if not isinstance(cpu_rng_state, torch.Tensor):
        raise SchemaValidationError("训练状态中的CPU随机数状态无效。")
    torch.set_rng_state(cpu_rng_state.detach().to(device="cpu", dtype=torch.uint8).contiguous())
    if state["cuda_rng_state"] is not None and torch.cuda.is_available():
        cuda_rng_states = state["cuda_rng_state"]
        if not isinstance(cuda_rng_states, (list, tuple)) or not all(
            isinstance(item, torch.Tensor) for item in cuda_rng_states
        ):
            raise SchemaValidationError("训练状态中的CUDA随机数状态无效。")
        torch.cuda.set_rng_state_all(
            item.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            for item in cuda_rng_states
        )
