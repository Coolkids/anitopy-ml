"""集中管理面向用户的中文提示。"""

MESSAGES = {
    "INPUT_EMPTY": "标题不能为空。",
    "INPUT_INVALID": "输入内容不符合要求。",
    "SCHEMA_INVALID": "结构化数据不符合字段契约。",
    "CONFIG_INVALID": "配置无效或缺少必要配置。",
    "UNKNOWN_ERROR": "发生未预期错误，请查看调试信息。",
    "MODEL_MISSING": "未找到模型文件，请先完成模型导出或指定正确的模型目录。",
    "OFFLINE_LINK": "离线模式下不能执行需要网络的作品关联。",
}


def message_for(code: str) -> str:
    """根据错误码获取稳定的中文提示。"""
    return MESSAGES.get(code, MESSAGES["UNKNOWN_ERROR"])
