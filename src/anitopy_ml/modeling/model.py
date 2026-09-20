"""默认离线加载的 XLM-R 抽取与媒体类型双头模型。"""

from __future__ import annotations

from typing import Any

from anitopy_ml.modeling.config import ExtractorModelConfig
from anitopy_ml.modeling.runtime import require_training_dependencies


def create_extractor_model(
    config: ExtractorModelConfig,
    *,
    local_files_only: bool = True,
) -> Any:
    """创建词元、字符 BIO 头和媒体类型头，默认禁止下载模型文件。"""
    config.validate()
    torch, transformers = require_training_dependencies()
    encoder = transformers.AutoModel.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        local_files_only=local_files_only,
    )
    hidden_size = int(encoder.config.hidden_size)
    nn = torch.nn

    class MultiTaskExtractor(nn.Module):
        """共享 XLM-R 编码器并直接预测原始字符边界的抽取模型。"""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(config.dropout)
            self.token_classifier = nn.Linear(hidden_size, len(config.labels))
            self.character_embedding = nn.Embedding(8192, 64)
            self.character_position_embedding = nn.Embedding(32, 16)
            self.character_encoder = nn.GRU(
                hidden_size + 80,
                hidden_size // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.character_classifier = nn.Linear(hidden_size, len(config.labels))
            self.media_classifier = nn.Linear(hidden_size, 3)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            character_token_indices: Any | None = None,
            character_ids: Any | None = None,
            character_positions: Any | None = None,
        ) -> dict[str, Any]:
            """返回词元、字符 BIO logits及池化后的三类媒体 logits。"""
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_hidden = self.dropout(output.last_hidden_state)
            denominator = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (token_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / denominator
            result = {
                "token_logits": self.token_classifier(token_hidden),
                "media_logits": self.media_classifier(self.dropout(pooled)),
            }
            if character_token_indices is not None or character_ids is not None or character_positions is not None:
                if character_token_indices is None or character_ids is None or character_positions is None:
                    raise ValueError("字符边界头需要完整的词元索引、字符和位置输入。")
                indices = character_token_indices.clamp(min=0, max=token_hidden.shape[1] - 1)
                gathered = token_hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden_size))
                features = torch.cat(
                    (
                        gathered,
                        self.character_embedding(character_ids.remainder(8192)),
                        self.character_position_embedding(character_positions.clamp(min=0, max=31)),
                    ),
                    dim=-1,
                )
                character_hidden, _ = self.character_encoder(self.dropout(features))
                result["character_logits"] = self.character_classifier(self.dropout(character_hidden))
            return result

    return MultiTaskExtractor()
