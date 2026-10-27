# Docker 容器部署

镜像只包含 Web API 和推理依赖，不包含模型权重或 Hugging Face 基础编码器，因此体积较小。首次启动时，入口程序会下载模型包和 `FacebookAI/xlm-roberta-base` 到持久化卷；同一卷中的后续容器会直接复用它们，并以离线方式加载模型。

接口说明见 [接口服务](接口服务.md) 和 [OpenAPI 定义](接口定义.openapi.yaml)。

## 前置条件

- Docker Engine 与 Docker Compose v2。
- 容器首次启动需要访问模型发布地址及 Hugging Face；完成后重启容器不需要网络。
- 使用 NVIDIA GPU 时，Docker 主机还需要 NVIDIA 驱动和 NVIDIA Container Toolkit。

## 构建或拉取镜像

在仓库根目录构建运行镜像：

```powershell
docker build -t anitopy-ml:v11 .
```

GitHub Actions 发布镜像后，也可直接拉取：

```powershell
docker pull ghcr.io/coolkids/anitopy-ml:v0.3.0
```

构建过程不会下载任何模型文件。镜像内默认值如下，运行时都可以用环境变量覆盖：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ANITOPY_MODEL_DIR` | `/opt/anitopy-ml/models/anitopy-ml-v11` | 持久化卷内的模型目录 |
| `ANITOPY_MODEL_URL` | V11 GitHub Release 地址 | 待下载模型 ZIP 的地址 |
| `ANITOPY_MODEL_UPDATE` | `missing` | 模型下载或更新策略 |
| `ANITOPY_MODEL_VERSION` | 未设置 | 目标模型发布版本；设置后要求模型包声明完全相同的版本 |
| `ANITOPY_MODEL_SHA256` | 未设置 | 可选的下载 ZIP 整包 SHA256 |
| `ANITOPY_BASE_MODEL` | `FacebookAI/xlm-roberta-base` | 与模型权重匹配的基础编码器 |
| `ANITOPY_PROCESSING_CONFIG` | 未设置 | 可选的只读 TOML 名称预处理配置路径，默认关闭 |
| `HF_HOME` | `/opt/anitopy-ml/huggingface` | Hugging Face 持久化缓存目录 |

模型 ZIP 解压后必须在根目录包含 `best_model.pt`、`checkpoint_metadata.json` 和 `tokenizer/`。新发布包另含 `model_release.json`，记录独立的模型版本号及逐文件 SHA256。下载会先写入临时目录，检查目标版本、文件完整性和哈希后再切换模型目录。

## 使用 docker run

以下命令创建两个具名卷，分别保存模型包和基础编码器缓存。服务仅绑定到本机回环地址：

如需启用名称预处理，可把配置文件以只读方式挂载进容器，并设置 `ANITOPY_PROCESSING_CONFIG=/app/config/processing.toml`。修改规则后需要重新创建容器；结果集处理目前只预留透传入口。

```powershell
docker volume create anitopy-ml-models
docker volume create anitopy-ml-huggingface

docker run --rm --name anitopy-ml-api `
  -p 127.0.0.1:8000:8000 `
  -v anitopy-ml-models:/opt/anitopy-ml/models `
  -v anitopy-ml-huggingface:/opt/anitopy-ml/huggingface `
  -e ANITOPY_DEVICE=cpu `
  -e ANITOPY_ALLOWED_HOSTS=localhost,127.0.0.1 `
  ghcr.io/coolkids/anitopy-ml:v0.3.0
```

首次启动会在日志中显示下载与缓存准备过程。使用本地构建镜像时，把末尾名称替换为 `anitopy-ml:v11`。

GPU 主机可使用：

```powershell
docker run --rm --name anitopy-ml-api `
  --gpus all `
  -p 127.0.0.1:8000:8000 `
  -v anitopy-ml-models:/opt/anitopy-ml/models `
  -v anitopy-ml-huggingface:/opt/anitopy-ml/huggingface `
  -e ANITOPY_DEVICE=cuda `
  -e ANITOPY_ALLOWED_HOSTS=localhost,127.0.0.1 `
  anitopy-ml:v11
```

容器启动后检查服务和调用接口：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz

$body = @{ title = "[LoliHouse] Tenmaku no Jaadugar - 09 [WebRip 1080p HEVC AAC]" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/parse -ContentType application/json -Body $body
```

## 使用 Docker Compose

在部署目录新建 `compose.yaml`：

```yaml
services:
  anitopy-ml:
    image: ghcr.io/coolkids/anitopy-ml:v0.3.0
    container_name: anitopy-ml-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      ANITOPY_DEVICE: cpu
      ANITOPY_ALLOWED_HOSTS: localhost,127.0.0.1
      ANITOPY_MAX_BATCH_SIZE: "100"
      ANITOPY_MAX_TITLE_LENGTH: "4096"
    volumes:
      - anitopy-ml-models:/opt/anitopy-ml/models
      - anitopy-ml-huggingface:/opt/anitopy-ml/huggingface

volumes:
  anitopy-ml-models:
  anitopy-ml-huggingface:
```

启动、查看状态和停止服务：

```powershell
docker compose up -d
docker compose ps
docker compose logs -f anitopy-ml
docker compose down
```

如果要从当前源码构建，替换服务的 `image` 配置：

```yaml
    build:
      context: .
    image: anitopy-ml:v11
```

之后执行：

```powershell
docker compose up --build -d
```

### Compose 使用 GPU

在服务配置中将设备与资源保留配置改为：

```yaml
    environment:
      ANITOPY_DEVICE: cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

然后重新创建容器：

```powershell
docker compose up -d --force-recreate
```

## 更新模型

更新功能在容器启动阶段执行；模型在服务进程内缓存，更新后需要重建或重启服务容器。先用当前源码构建包含新功能的镜像：

```powershell
docker build -t anitopy-ml:local .
```

默认 `missing` 策略只在模型目录缺失或不完整时下载。推荐使用 `version_changed`，让目标版本决定是否更新；同一地址发布新版本时也能触发下载。版本相同时复用持久化模型。

| `ANITOPY_MODEL_UPDATE` | 行为 |
| --- | --- |
| `missing` | 默认值；仅在模型缺失或不完整时下载。 |
| `always` | 每次容器启动都重新下载并替换模型。 |
| `url_changed` | 仅在 `ANITOPY_MODEL_URL` 与上次成功下载地址不同，或模型缺失时更新。 |
| `version_changed` | 模型缺失、未声明版本或实际版本与 `ANITOPY_MODEL_VERSION` 不同就更新；必须设置目标版本。 |
| `never` | 禁止下载；要求挂载卷中已有完整模型。 |

### 为训练结果生成模型版本

软件包与镜像版本、模型发布版本、基础编码器版本分别管理。例如，软件可以是 `0.3.0`，模型是 `v13.0.0`，XLM-R 基础编码器修订仍是 `main`。不要修改 `checkpoint_metadata.json` 中的 `模型配置.模型版本` 来设置发布版本，该字段属于基础编码器。

为已完成的 V13 训练结果生成带版本号的模型 ZIP：

```powershell
uv run anitopy-ml model package `
  --model artifacts/runs/V13_large_seed1 `
  --version v13.0.0 `
  --output dist/anitopy-ml-v13.0.0.zip
```

命令流式读取权重，保留训练目录原样。ZIP 包含推理权重、检查点元数据、完整分词器、已存在的校准及接受策略文件，以及自动生成的 `model_release.json`；不会包含优化器或训练状态。命令输出 ZIP 的 SHA256，可复制到 `ANITOPY_MODEL_SHA256`。同一版本应对应固定内容；权重或推理附件变更后分配新的模型版本号。

将 ZIP 上传到自己的可访问发布地址，再把下载 URL 配置到容器。下面的 `example.com` 是占位地址，需要替换；本命令不会替你上传或发布模型。

```json
{
  "schema_version": 1,
  "model_version": "v13.0.0",
  "files": {
    "best_model.pt": "实际的64位SHA256",
    "checkpoint_metadata.json": "实际的64位SHA256",
    "tokenizer/tokenizer.json": "实际的64位SHA256",
    "tokenizer/tokenizer_config.json": "实际的64位SHA256"
  }
}
```

以上是发布清单结构示意，由打包命令自动生成，不能直接把示意哈希作为实际发布文件使用。旧 ZIP 没有此文件时，仍支持原来的下载策略与目录名称标识；如果设置了 `ANITOPY_MODEL_VERSION`，则必须使用带发布清单的模型包，目标版本不能仅靠环境变量给旧包改名。

### 按版本自动更新

停止并移除旧容器后，使用相同持久化卷启动新容器：

```powershell
docker run --rm --name anitopy-ml-api `
  -p 127.0.0.1:8000:8000 `
  -v anitopy-ml-models:/opt/anitopy-ml/models `
  -v anitopy-ml-huggingface:/opt/anitopy-ml/huggingface `
  -e ANITOPY_MODEL_URL="https://example.com/releases/anitopy-ml-v13.0.0.zip" `
  -e ANITOPY_MODEL_VERSION=v13.0.0 `
  -e ANITOPY_MODEL_UPDATE=version_changed `
  anitopy-ml:local
```

需要固定 ZIP 内容时，再增加 `-e ANITOPY_MODEL_SHA256="打包命令输出的64位SHA256"`。目标版本或哈希不符、包不完整、下载失败时会拒绝切换并退出，原模型文件保持不变。若目录切换失败会尝试恢复旧目录；下次启动也会恢复异常退出留下的旧目录备份。同一卷的更新操作使用进程锁串行化。

Compose 部署时，将服务改为当前镜像，并加入以下环境变量：

```yaml
    image: anitopy-ml:local
    environment:
      ANITOPY_MODEL_URL: https://example.com/releases/anitopy-ml-v13.0.0.zip
      ANITOPY_MODEL_VERSION: v13.0.0
      ANITOPY_MODEL_UPDATE: version_changed
      # 可选：填入打包命令实际输出的 ZIP SHA256。
      # ANITOPY_MODEL_SHA256: ...
```

执行 `docker compose up -d --force-recreate anitopy-ml` 后，启动程序会安装目标版本，再启动 API。回退时将目标版本、URL 和可选哈希一起改为旧发布版本后重新创建容器；版本判断使用完全相等比较。

### 手动更新与版本查询

想先完成下载校验、再启动服务时，可以手动执行 `model-update`。该命令强制下载当前 URL，即使版本与上次相同也会重新安装；`never` 策略仍禁止下载。它只准备模型包，不加载基础编码器或启动 API。

先修改 Compose 中的目标版本、URL 和哈希，再执行：

```powershell
docker compose stop anitopy-ml
docker compose run --rm --no-deps anitopy-ml model-update
if ($LASTEXITCODE -ne 0) { throw "模型更新失败；原模型已保留，请修复配置或改回旧版本后再启动。" }
docker compose up -d --force-recreate anitopy-ml
```

更新共享模型卷前先停止使用该目录的服务容器，避免工作进程继续使用旧的内存权重。模型安装后仍应通过健康检查确认实际模型可加载。

查询持久化卷里的模型版本，不下载或加载模型：

```powershell
docker compose run --rm --no-deps anitopy-ml model-info
```

本地目录也可查询：

```powershell
uv run anitopy-ml model info --model artifacts/releases/anitopy-ml-v13
```

服务运行后查询实际已加载版本：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

新模型包的响应中 `model_version` 为 `v13.0.0`，解析接口各条结果中的 `model_version` 使用相同来源。该值取自已加载模型的发布文件；改变目标环境变量不会把仍在运行的旧模型报告成新版本。`model-info` 查询磁盘状态，`/healthz` 查询当前进程，两者在尚未重启时可能不同。

新模型应使用与原模型兼容的 `ANITOPY_BASE_MODEL`；若更换了基础编码器，也应一并设置该变量，入口程序会在缓存中补齐它。

若管理员已经把模型文件和基础编码器缓存预先写入卷，可设置 `ANITOPY_MODEL_UPDATE=never`，让启动过程完全离线。

## 运行限制

- 服务默认不提供认证；绑定到非本机地址或通过反向代理公开访问前，应自行配置网络访问控制与认证。
- 每个 Gunicorn 工作进程都会加载一份模型。默认镜像固定为一个工作进程，避免重复占用 CPU 内存或显存。
- 健康检查会加载模型，首次成功响应通常慢于后续请求。
- `ANITOPY_MAX_BATCH_SIZE` 和 `ANITOPY_MAX_TITLE_LENGTH` 用于控制单个请求的资源占用。
