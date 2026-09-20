"""抽取与媒体类型任务的掩码多任务损失。"""

from __future__ import annotations

from typing import Any

from anitopy_ml.errors import SchemaValidationError
from anitopy_ml.modeling.runtime import require_training_dependencies
from anitopy_ml.training.alignment import IGNORE_LABEL_ID


def masked_multitask_loss(
    token_logits: Any,
    token_labels: Any,
    loss_mask: Any,
    *,
    media_logits: Any | None = None,
    media_labels: Any | None = None,
    token_weight: float = 1.0,
    media_weight: float = 0.2,
) -> dict[str, Any]:
    """计算仅覆盖有效标签的BIO损失和可选媒体类型损失。"""
    if token_weight < 0 or media_weight < 0:
        raise SchemaValidationError("多任务损失权重不能为负数。")
    torch, _ = require_training_dependencies()
    functional = torch.nn.functional
    active = loss_mask.bool() & token_labels.ne(IGNORE_LABEL_ID)
    if active.any():
        token_loss = functional.cross_entropy(token_logits[active], token_labels[active])
    else:
        token_loss = token_logits.sum() * 0
    media_loss = token_loss * 0
    if media_logits is not None or media_labels is not None:
        if media_logits is None or media_labels is None:
            raise SchemaValidationError("媒体类型logits和标签必须同时提供。")
        media_active = media_labels.ge(0)
        if media_active.any():
            media_loss = functional.cross_entropy(media_logits[media_active], media_labels[media_active])
    total = token_weight * token_loss + media_weight * media_loss
    return {"loss": total, "token_loss": token_loss, "media_loss": media_loss}
