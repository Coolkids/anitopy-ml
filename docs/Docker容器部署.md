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
docker pull ghcr.io/coolkids/anitopy-ml:v0.1.0
```

构建过程不会下载任何模型文件。镜像内默认值如下，运行时都可以用环境变量覆盖：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ANITOPY_MODEL_DIR` | `/opt/anitopy-ml/models/anitopy-ml-v11` | 持久化卷内的模型目录 |
| `ANITOPY_MODEL_URL` | V11 GitHub Release 地址 | 待下载模型 ZIP 的地址 |
| `ANITOPY_MODEL_UPDATE` | `missing` | 模型下载或更新策略 |
| `ANITOPY_BASE_MODEL` | `FacebookAI/xlm-roberta-base` | 与模型权重匹配的基础编码器 |
| `HF_HOME` | `/opt/anitopy-ml/huggingface` | Hugging Face 持久化缓存目录 |

模型 ZIP 解压后必须在根目录包含 `best_model.pt`、`checkpoint_metadata.json` 和 `tokenizer/`。下载会先写入临时目录，确认文件完整后才替换当前模型目录。

## 使用 docker run

以下命令创建两个具名卷，分别保存模型包和基础编码器缓存。服务仅绑定到本机回环地址：

```powershell
docker volume create anitopy-ml-models
docker volume create anitopy-ml-huggingface

docker run --rm --name anitopy-ml-api `
  -p 127.0.0.1:8000:8000 `
  -v anitopy-ml-models:/opt/anitopy-ml/models `
  -v anitopy-ml-huggingface:/opt/anitopy-ml/huggingface `
  -e ANITOPY_DEVICE=cpu `
  -e ANITOPY_ALLOWED_HOSTS=localhost,127.0.0.1 `
  ghcr.io/coolkids/anitopy-ml:v0.1.0
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
    image: ghcr.io/coolkids/anitopy-ml:v0.1.0
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

默认 `missing` 策略只在模型目录缺失或不完整时下载，适合稳定运行。更新时，指定新模型 ZIP 地址并设置更新策略，然后重新创建容器。原有持久化卷会被新模型原子替换。

| `ANITOPY_MODEL_UPDATE` | 行为 |
| --- | --- |
| `missing` | 默认值；仅在模型缺失或不完整时下载。 |
| `always` | 每次容器启动都重新下载并替换模型。 |
| `url_changed` | 仅在 `ANITOPY_MODEL_URL` 与上次成功下载地址不同，或模型缺失时更新。 |
| `never` | 禁止下载；要求挂载卷中已有完整模型。 |

例如升级到用户自己发布的 V12 模型：

```powershell
docker run --rm --name anitopy-ml-api `
  -p 127.0.0.1:8000:8000 `
  -v anitopy-ml-models:/opt/anitopy-ml/models `
  -v anitopy-ml-huggingface:/opt/anitopy-ml/huggingface `
  -e ANITOPY_MODEL_URL="https://example.com/releases/anitopy-ml-v12.zip" `
  -e ANITOPY_MODEL_UPDATE=url_changed `
  anitopy-ml:v11
```

Compose 部署时，在 `environment` 中加入同名变量，执行 `docker compose up -d --force-recreate` 即可。新模型应使用与原模型兼容的 `ANITOPY_BASE_MODEL`；若更换了基础编码器，也应一并设置该变量，入口程序会在缓存中补齐它。

若管理员已经把模型文件和基础编码器缓存预先写入卷，可设置 `ANITOPY_MODEL_UPDATE=never`，让启动过程完全离线。

## 运行限制

- 服务默认不提供认证；绑定到非本机地址或通过反向代理公开访问前，应自行配置网络访问控制与认证。
- 每个 Gunicorn 工作进程都会加载一份模型。默认镜像固定为一个工作进程，避免重复占用 CPU 内存或显存。
- 健康检查会加载模型，首次成功响应通常慢于后续请求。
- `ANITOPY_MAX_BATCH_SIZE` 和 `ANITOPY_MAX_TITLE_LENGTH` 用于控制单个请求的资源占用。
