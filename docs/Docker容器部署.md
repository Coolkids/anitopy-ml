# Docker 容器部署

镜像默认启动 Django Web API，监听容器内 `8000` 端口。模型会在首个请求或健康检查时加载；默认镜像在构建阶段已下载 V11 模型包及 `FacebookAI/xlm-roberta-base` 基础权重，运行时不访问网络。

接口说明见 [接口服务](接口服务.md) 和 [OpenAPI 定义](接口定义.openapi.yaml)。

## 前置条件

- Docker Engine 及 Docker Compose v2。
- 若使用 NVIDIA GPU，Docker 主机需要安装 NVIDIA 驱动和 NVIDIA Container Toolkit。
- 模型镜像较大，因为其中包含模型权重与基础编码器；首次拉取或构建需要较长时间。

## 从源码构建

在仓库根目录执行：

```powershell
docker build -t anitopy-ml:v11 .
```

默认构建参数如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DOWNLOAD_MODEL` | `1` | 下载模型 ZIP 并预取基础模型 |
| `MODEL_URL` | V11 GitHub Release 地址 | `anitopy-ml-v11.zip` 下载地址 |
| `BASE_MODEL` | `FacebookAI/xlm-roberta-base` | 需要写入镜像缓存的基础模型 |
| `MODEL_DIR` | `/opt/anitopy-ml/models/anitopy-ml-v11` | 容器内模型目录 |

构建其他发布版本时，显式指定模型地址：

```powershell
docker build `
  --build-arg MODEL_URL="https://github.com/Coolkids/anitopy-ml/releases/download/v0.1.0/anitopy-ml-v11.zip" `
  -t anitopy-ml:v0.1.0 .
```

GitHub Actions 发布完成后，也可直接拉取镜像：

```powershell
docker pull ghcr.io/coolkids/anitopy-ml:v0.1.0
```

## 使用 docker run

以下命令将服务仅绑定到本机回环地址，适合本机程序调用：

```powershell
docker run --rm --name anitopy-ml-api `
  -p 127.0.0.1:8000:8000 `
  -e ANITOPY_DEVICE=cpu `
  -e ANITOPY_ALLOWED_HOSTS=localhost,127.0.0.1 `
  ghcr.io/coolkids/anitopy-ml:v0.1.0
```

使用本地构建镜像时，将末尾镜像名称替换为 `anitopy-ml:v11`。

GPU 主机可使用：

```powershell
docker run --rm --name anitopy-ml-api `
  --gpus all `
  -p 127.0.0.1:8000:8000 `
  -e ANITOPY_DEVICE=cuda `
  -e ANITOPY_ALLOWED_HOSTS=localhost,127.0.0.1 `
  anitopy-ml:v11
```

容器启动后检查模型是否可用：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

收到 `status: ok` 后即可调用解析接口：

```powershell
$body = @{ title = "[LoliHouse] Tenmaku no Jaadugar - 09 [WebRip 1080p HEVC AAC]" } | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/parse `
  -ContentType application/json `
  -Body $body
```

## 使用 docker compose

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
```

启动、查看状态和停止服务：

```powershell
docker compose up -d
docker compose ps
docker compose logs -f anitopy-ml
docker compose down
```

若希望 Compose 从当前源码构建镜像，将 `image` 改为本地标签并添加 `build: .`：

```yaml
services:
  anitopy-ml:
    build:
      context: .
      args:
        DOWNLOAD_MODEL: "1"
    image: anitopy-ml:v11
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      ANITOPY_DEVICE: cpu
```

之后执行：

```powershell
docker compose up --build -d
```

### Compose 使用 GPU

在支持 NVIDIA GPU 的 Docker 主机上，将以下内容加入服务配置：

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

## 外部挂载模型

默认镜像已包含模型，不需要挂载卷。若自行构建 `DOWNLOAD_MODEL=0` 的镜像，则必须同时提供完整模型目录和匹配的 Hugging Face 基础模型缓存：

```powershell
docker run --rm `
  -p 127.0.0.1:8000:8000 `
  -e ANITOPY_MODEL_DIR=/models/anitopy-ml-v11 `
  -v "${PWD}\models\anitopy-ml-v11:/models/anitopy-ml-v11:ro" `
  -v "${PWD}\huggingface:/opt/anitopy-ml/huggingface:ro" `
  anitopy-ml:without-model
```

模型目录至少包含 `best_model.pt`、`tokenizer/`、`checkpoint_metadata.json`、`calibration.json` 和 `acceptance_policy.json`。模型权重、分词器、校准器和接受策略必须来自同一个发布包。

## 运行限制

- 服务默认不提供认证；绑定到非本机地址或通过反向代理公开访问前，应自行配置网络访问控制与认证。
- 每个 Gunicorn 工作进程都会加载一份模型。默认镜像固定为一个工作进程，避免重复占用 CPU 内存或显存。
- 健康检查会加载模型，首次成功响应的时间通常长于后续请求。
- `ANITOPY_MAX_BATCH_SIZE` 和 `ANITOPY_MAX_TITLE_LENGTH` 用于控制单个请求的资源占用。
