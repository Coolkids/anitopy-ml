"""训练早停与可恢复进度控制。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from anitopy_ml.errors import SchemaValidationError


@dataclass(slots=True)
class EarlyStopping:
    """按验证指标最大化策略维护早停状态。"""

    patience: int
    min_delta: float = 0.0
    best_score: float | None = None
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        """记录一个验证分数；返回是否应停止训练。"""
        if self.patience <= 0 or self.min_delta < 0:
            raise SchemaValidationError("早停参数无效。")
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score, self.bad_epochs = score, 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """可写入检查点的最小训练恢复状态。"""

    epoch: int
    global_step: int
    best_validation_f1: float | None
    early_stopping: dict[str, object]
    training_manifest_sha256: str

    def validate(self) -> None:
        """拒绝缺失数据版本或负训练进度的恢复状态。"""
        if self.epoch < 0 or self.global_step < 0:
            raise SchemaValidationError("训练进度不能为负数。")
        if len(self.training_manifest_sha256) != 64:
            raise SchemaValidationError("训练数据清单哈希必须为SHA256。")

    def model_dump(self) -> dict[str, object]:
        """返回JSON兼容的进度记录。"""
        self.validate()
        return asdict(self)
