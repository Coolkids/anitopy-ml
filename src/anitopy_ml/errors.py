"""项目的可预期异常类型。"""


class AnitopyMlError(Exception):
    """所有可转为中文业务提示的基础异常。"""

    code = "UNKNOWN_ERROR"


class InputValidationError(AnitopyMlError):
    """输入内容不符合接口约束。"""

    code = "INPUT_INVALID"


class PreprocessingError(InputValidationError):
    """标题预处理失败并保留稳定业务错误码。"""

    VALID_CODES = {
        "PREPROCESSING_EMPTY",
        "PREPROCESSING_LIMIT",
        "PREPROCESSING_ZERO_WIDTH",
    }

    def __init__(self, message: str, *, code: str) -> None:
        if code not in self.VALID_CODES:
            raise ValueError("未知预处理错误码。")
        super().__init__(message)
        self.code = code


class SchemaValidationError(AnitopyMlError):
    """结构化数据不符合数据契约。"""

    code = "SCHEMA_INVALID"


class ConfigurationError(AnitopyMlError):
    """配置缺失或组合冲突。"""

    code = "CONFIG_INVALID"
