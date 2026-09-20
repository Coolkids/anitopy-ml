"""项目的可预期异常类型。"""


class AnitopyMlError(Exception):
    """所有可转为中文业务提示的基础异常。"""

    code = "UNKNOWN_ERROR"


class InputValidationError(AnitopyMlError):
    """输入内容不符合接口约束。"""

    code = "INPUT_INVALID"


class SchemaValidationError(AnitopyMlError):
    """结构化数据不符合数据契约。"""

    code = "SCHEMA_INVALID"


class ConfigurationError(AnitopyMlError):
    """配置缺失或组合冲突。"""

    code = "CONFIG_INVALID"
