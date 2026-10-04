# Paper RAG：IoT 设备识别方案调研助手

一个面向 PDF 论文的多模态 RAG（Retrieval-Augmented Generation）应用。项目提供 Streamlit 页面和 FastAPI 接口，通过阿里云百炼调用千问模型；既支持 Milvus Lite + SQLite 的轻量本地模式，也支持 Milvus Server + PostgreSQL + Redis + MinIO 的企业开发模式。

应用场景是帮助物联网安全研发团队阅读论文、比较候选识别方案，并核对原文依据。首页提供方案对比工作台，也保留连续论文问答。

## 功能

- 选择 2–3 篇论文，填写部署需求，生成五个维度的带引用对比表
- 将论文事实、实验可比性和需求适配分析分开展示，资料不足时明确提示
- 核对引用的片段 ID、论文来源和原文摘录，支持保存与下载 Markdown 对比报告
- 批量上传 PDF 论文并提取逐页文本
- 按 Token 滑动窗口切分论文内容
- 使用 `text-embedding-v4` 生成文本向量
- 使用 `qwen3-vl-embedding` 建立论文图片的跨模态索引
- 无法直接抽取矢量图表时，将对应 PDF 页面渲染为图片作为兜底
- 支持 Milvus Lite 与 Milvus Server 两种向量存储模式
- 支持向量检索与 BM25 混合检索
- 使用 `qwen3-rerank` 对粗筛候选进行可选精排
- 使用 `qwen-plus` 生成带页码引用的回答
- 用户明确询问图表或图片时，使用 `qwen3-vl-flash` 分析视觉信息
- 支持多会话、新建会话和删除会话
- 支持 SQLite 与 PostgreSQL 两种聊天持久化模式
- 提供 FastAPI 会话管理、聊天接口及 Swagger 文档
- 提供 Docker Compose 基础设施和数据迁移脚本
- 根据最近对话改写追问，支持“它”“这个方法”等上下文指代
- 应用重启后恢复聊天历史

## 方案对比原型

先上传至少两篇论文并构建索引，启动 FastAPI 和 Streamlit，在“方案对比”模式选择论文并填写需求，例如“部署在交换机上，关注高吞吐、处理延迟和硬件资源限制”。点击“生成方案对比”后，报告会展示数据要求、识别方法、部署条件、实验依据、适用范围与局限，并单列实验可比性和适配分析。适配分析属于模型推断，需由研发人员核对。

对比会固定当前物理文本索引，在每篇所选论文内分别召回候选，统一 rerank 后每篇最多保留 5 个不同页优先的片段。它使用每篇最多 `RERANK_CANDIDATE_K` 个候选，因此所选 2–3 篇论文的 rerank 总候选量最多为该参数的 2–3 倍。每次提交会调用一次文本 embedding、启用时一次 rerank，以及有文本证据时一次对话模型。该入口使用 FastAPI 进程中的检索配置，页面的问答检索开关只作用于原有本地问答路径。

模型按 JSON 输出事实和引用，后端检查每个引用是否对应本次检索结果、是否属于该论文、摘录是否出现在原文中。无法核对的事实项会替换成“资料不足”；实验可比性需有所有所选论文的实验事实与引用，否则显示证据不足。引用检查不等同于验证事实推断正确，也不能保证五个维度的证据都被检索到，仍需人工核对。本次比较使用文本证据；查看架构图或实验图可切换到“论文问答”继续询问。

对比表及完整文本引用快照保存到现有会话历史，重启后可查看并下载 Markdown；API 当次响应另含结构化 `report`，目前不另建结构化报告数据库表。生成耗时较长时，可配置 `COMPARISON_REQUEST_TIMEOUT`（默认 180 秒）；超时后先刷新会话记录，确认结果再重试。

接口：`POST /api/v1/comparisons`，请求示例：

```json
{
  "conversation_id": "已有会话 ID",
  "sources": ["DeviceRadar.pdf", "ProfilIoT.pdf"],
  "requirements": "部署在交换机上，关注高吞吐与延迟"
}
```

未建索引或所选论文不在当前索引中返回 409；会话不存在返回 404；模型生成或解析失败返回 502，不保存虚构的助手回答。结构化生成采用千问的 [JSON Object 模式](https://www.alibabacloud.com/help/zh/model-studio/qwen-structured-output)并由 Pydantic 验证字段。该原型尚未完成真实选型题集的在线质量验收。

## 工作流程

```text
用户问题
  ↓
读取当前会话最近几轮消息
  ↓
千问将追问改写为独立检索问题
  ↓
Milvus 向量检索 + BM25 关键词检索
  ↓
千问根据论文片段生成带引用的回答
  ↓
SQLite 或 PostgreSQL 保存问题、回答和引用快照
```

## 项目结构

```text
paper_rag/
├── backend/                    # FastAPI 应用、路由和 Schema
├── scripts/                    # 迁移脚本与 worker 启动入口
├── tests/                      # 自动化测试
├── app.py                      # Streamlit 多会话页面
├── api_client.py               # Streamlit 调用 FastAPI 的 HTTP 客户端
├── chat_service.py             # 对话、检索、生成和持久化编排
├── comparison_models.py        # 部署需求、对比维度与结构化报告
├── comparison_service.py       # 方案对比、引用核对和报告渲染
├── graph_store.py              # PostgreSQL 图节点、关系、证据与版本门控
├── graph_builder.py            # 关系抽取和版本化图构建
├── graph_retriever.py          # 图关系证据回查与文本片段核对
├── ingestion_service.py        # 文本与图片索引重建编排
├── job_queue.py                # Redis/RQ 任务入队与状态查询
├── worker_tasks.py             # 后台 worker 执行的任务
├── config.py                   # 模型、路径和检索参数
├── database.py                 # SQLite/PostgreSQL 动态分派
├── postgres_database.py        # PostgreSQL 连接池与数据访问
├── milvus_store.py             # Lite/Server 统一 Milvus 连接
├── generator.py                # 追问改写与答案生成
├── image_ingest.py             # PDF 图片抽取、向量化和入库
├── image_retriever.py          # 文本到图片的跨模态检索
├── ingest.py                   # PDF 解析、分块、向量化和入库
├── retriever.py                # Milvus + BM25 混合检索
├── compose.yaml                # 企业开发基础设施
├── env.example                 # 环境变量示例
├── requirements.txt            # 生产依赖
├── requirements-dev.txt        # 测试依赖
└── data/                       # 本地论文、索引和聊天库（不会提交）
```

## 环境要求

- Windows 10/11 + WSL2，或原生 Linux
- Python 3.11 或 3.12
- 阿里云百炼 API Key
- Docker Desktop（仅企业开发模式需要）

Milvus Lite 主要面向 Linux/macOS。Windows 用户建议在 WSL2 中运行，不建议直接使用 Windows Python。

## 1. 安装 WSL2

在管理员 PowerShell 中执行：

```powershell
wsl --install -d Ubuntu
```

安装完成并重启后，打开 Ubuntu 终端。

## 2. 创建 Python 环境

项目推荐使用 `uv` 管理独立的 Python 3.12 环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
```

进入项目目录并创建环境：

```bash
cd /mnt/c/Users/<你的用户名>/path/to/paper_rag
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv --python 3.12 ~/.venvs/paper-rag
~/.local/bin/uv pip install \
  --python ~/.venvs/paper-rag/bin/python \
  --link-mode=copy \
  -r requirements.txt
```

虚拟环境放在 WSL 的 Linux 主目录中，可以避免 `/mnt/c` 上的文件权限和链接兼容问题。

## 3. 配置阿里云百炼

复制环境变量模板：

```bash
cp env.example .env
```

编辑 `.env`：

```dotenv
OPENAI_API_KEY=sk-你的百炼APIKey
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

CHAT_MODEL=qwen-plus
VISION_MODEL=qwen3-vl-flash
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
IMAGE_EMBEDDING_MODEL=qwen3-vl-embedding
IMAGE_EMBEDDING_DIM=1024
IMAGE_TOP_K=2
```

如果百炼控制台提供了包含 Workspace ID 的专属兼容地址，应使用控制台给出的地址替换 `OPENAI_BASE_URL`。

请确保 embedding 模型输出维度和 `EMBEDDING_DIM` 一致。更换维度后必须重新构建知识库索引。

## 4. 选择存储模式

不配置下面的变量时，应用默认使用 `data/paper_rag.db`（Milvus Lite）和 `data/chat_history.db`（SQLite），适合单机试用。

企业开发模式先启动基础设施：

```bash
docker compose up -d
docker compose ps
```

所有长期服务应显示为 `healthy`。然后在 `.env` 中加入：

```dotenv
MILVUS_URI=http://127.0.0.1:19530
MILVUS_DATABASE=default
DATABASE_URL=postgresql://paper_rag:paper_rag_dev@127.0.0.1:5432/paper_rag
```

本地服务端口：

| 服务 | 地址或端口 |
|---|---|
| PostgreSQL | `127.0.0.1:5432` |
| Redis | `127.0.0.1:6379` |
| MinIO API | `http://127.0.0.1:9000` |
| MinIO Console | `http://127.0.0.1:9001` |
| Milvus | `http://127.0.0.1:19530` |
| Milvus WebUI | `http://127.0.0.1:9091/webui/` |

`env.example` 中的密码只适用于本地开发，正式环境必须替换并通过密钥管理系统注入。

## 5. 启动应用

企业模式需要在三个 Ubuntu 终端中分别运行 Streamlit、FastAPI 和 worker。

终端 1：启动 FastAPI。

```bash
cd /mnt/c/Users/<你的用户名>/path/to/paper_rag
source ~/.venvs/paper-rag/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

终端 2：启动知识库构建 worker。

```bash
cd /mnt/c/Users/<你的用户名>/path/to/paper_rag
source ~/.venvs/paper-rag/bin/activate
python scripts/run_worker.py
```

终端 3：启动 Streamlit 页面。

```bash
cd /mnt/c/Users/<你的用户名>/path/to/paper_rag
source ~/.venvs/paper-rag/bin/activate
streamlit run app.py
```

API 接收页面请求并把耗时的索引构建任务放入 Redis；worker 在后台完成 PDF 解析、向量生成和 Milvus 写入；Streamlit 每 2 秒查询一次任务进度。如果 API 不在本机 `8000` 端口，在 `.env` 中设置 `API_BASE_URL`。

需要只检查 worker 能否连接 Redis、但不持续驻留时，可运行：

```bash
python scripts/run_worker.py --burst
```

API 文档地址：

```text
http://localhost:8000/docs
```

浏览器通常访问：

```text
http://localhost:8501
```

如果 WSL 没有将端口转发到 localhost，可在 Ubuntu 中执行以下命令获取 IP：

```bash
hostname -I
```

然后访问 `http://<WSL-IP>:8501`。

## 6. 构建论文知识库

1. 在左侧上传一篇或多篇 PDF。
2. 点击“重新构建知识库索引”。
3. 在侧边栏查看排队、PDF 解析、文本向量、图片向量和写入进度。
4. 在页面底部输入论文相关问题。

重建时会先在独立的版本 collection 中写入数据、校验条数并加载索引，完成后才切换查询别名。写入、校验或加载失败时，原索引仍可继续使用；文本完成但图片构建失败时，保留原图片索引并在页面提示。

新建文本索引时，片段 ID 由 PDF 内容、页码、片段顺序和文本确定；同一 PDF 在分块设置不变时重建会得到相同 ID。旧索引中的随机 ID 不会自动变化，只有下次重建才会应用新规则。图关系必须关联当前激活的文本索引版本，避免引用旧版片段。

第一次重建会把原同名 collection 重命名为备份，再创建同名查询别名；这一首次转换有短暂的名称切换窗口，创建别名失败时会尝试恢复原名称。后续重建直接切换别名。旧版本保留供回滚，会占用额外存储，当前不会自动删除。聊天历史保留，旧回答引用的是当时的片段快照。

也可以通过异步 API 发起重建：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingestion/jobs
```

接口会返回 `job_id`。使用该 ID 查询进度：

```bash
curl http://127.0.0.1:8000/api/v1/ingestion/jobs/<job_id>
```

任务状态包括 `queued`、`started`、`finished` 和 `failed`；`stage` 会显示 `parsing`、`embedding_text`、`writing_text`、启用图构建时的 `building_graph`、`embedding_images` 或 `completed`。同一时间只允许一个知识库重建任务，重复提交会返回 HTTP 409。

任务记录默认保留 24 小时（`INGESTION_RESULT_TTL`）。查询返回 404 时，页面会提示记录已过期或不存在，并恢复重建按钮；这不代表原任务失败。临时网络错误或 Redis 故障仍会保留任务跟踪并继续查询。

## 从 Lite 迁移到企业模式

迁移现有向量数据不会重新调用 embedding API：

```bash
python scripts/migrate_milvus.py \
  --target-uri http://127.0.0.1:19530
```

迁移 SQLite 聊天历史：

```bash
python scripts/migrate_chat_history.py
```

迁移脚本会核对源端和目标端记录数。目标已有同名数据时默认停止；只有确认需要覆盖时才使用 `--replace`。

完成并验证后，再将 `MILVUS_URI` 和 `DATABASE_URL` 写入 `.env`。原 Lite/SQLite 文件不会被迁移脚本删除，可用于回滚。

## 对话历史

轻量模式的聊天数据保存在：

```text
data/chat_history.db
```

企业模式由 `DATABASE_URL` 指向 PostgreSQL。两种模式都包含：

- `conversations`：会话标题与时间
- `messages`：用户和助手消息
- `citations`：回答对应的论文、页码、分数和文本快照
- `image_citations`：回答对应的图片路径、论文、页码和分数

默认向模型发送最近 6 条消息（约 3 轮问答）。可以在 `config.py` 中调整：

```python
MAX_HISTORY_MESSAGES = 6
```

## 检索质量评测

现有单页问答使用 `eval/my_qa_set.json`。为评估是否需要图结构索引，`eval/relationship_qa_set.json` 另提供需要两处原文证据的关联题；其中的 `reference_answer` 供人工核对，不参与自动评分。

在文本索引及其依赖服务可用时运行：

```bash
python eval/evaluate.py --qa-set eval/relationship_qa_set.json --top-k 5
```

报告中的“Hit Rate（证据找齐）”要求一道题的所有标注页都进入 Top-K；“证据覆盖率”显示部分找到了多少。MRR 按找齐全部证据时的最后一个排名计算。这个评测只检查检索，不检查生成回答是否正确；比较图索引前后，应使用相同题集和 Top-K。评测会调用现有 embedding，启用 rerank 时也会调用 rerank API。

加 `--show-trace` 可显示每题在向量、BM25、融合候选和最终结果各阶段的来源与页码，不输出论文原文；它仍会调用上述模型 API。

实验性地测试页码多样性可加 `--diverse-rerank`：rerank 返回全部候选排名，再优先选择不同论文页的片段。默认关闭；它不能保证覆盖问题的每个事实，需用相同评测集验证后再决定是否在 `.env` 中启用 `ENABLE_DIVERSE_RERANK=true`。该模式仍只向 rerank 发送原有候选，但会请求返回更多排名结果。

另一项默认关闭的实验是 `ENABLE_REQUIREMENT_SELECTION=true`：在一次 rerank 的候选结果中，先覆盖题目点名的论文（或恰好两篇论文的“分别/各自”问题），再按相关性与互补性填满 Top-K。它不调用额外模型，也不能凭空找回候选中没有的证据；与页码多样性同时开启时优先使用证据需求选择。只用本地 BM25 做无模型对照：`python eval/evaluate_selector_local.py --quiet`。在线配对对照可用 `python eval/evaluate.py --compare-selection --show-trace`：每题只检索并完整 rerank 一次，再在同一份排名上比较原始 Top-K、按页多样性和需求选择；如果 rerank 结果不完整，脚本会报错而不是给出误导性分数。**该命令仍会对每道题调用 embedding 和 rerank API**，需确认论文允许发送且接受费用后再运行。评测中的来源与页码 trace 不含论文原文，完整候选文本仅在进程内存中供选择器使用。

在当前两篇论文、8 道关联题的本地 BM25 代理评测中，Top-20 候选仅 4/8 题找齐证据；压缩到 Top-5 后，原始顺序为 0/8，页码多样性和需求选择均为 1/8。需求选择尚未表现出优于页码多样性的收益，因此保持默认关闭；这组结果不能推断在线向量加 rerank 的效果。

只检查本地 BM25 候选、完全不调用模型 API 时，可运行 `python eval/evaluate_bm25.py --top-k 20`；它不是完整混合检索的分数。

## 图索引开发状态

图索引已接入**可选的构建与查询实验链路**，经过小样本评测但默认关闭。仅在 PostgreSQL 模式下，将 `.env` 中 `ENABLE_GRAPH_BUILD=true` 和 `ENABLE_GRAPH_RETRIEVAL=true`，再通过页面重新构建知识库。图构建会对**每个文本片段额外调用一次对话模型**，产生额外费用和构建时间；关闭这两个开关则不产生图相关调用。`graph_builder.py` 只接受指定实体/关系类型、且证据引文必须原样出现在对应片段中；这能排除无出处的引文，但不能保证模型抽取的关系语义正确。

如果当前文本索引已经存在，可以先运行 `python scripts/build_graph.py` 只读查看当前物理版本和片段数；确认论文可发送到所配置的外部模型服务并接受费用后，再运行 `python scripts/build_graph.py --execute`，直接为该版本建图，无需重新计算 embedding 或图片索引。`--probe` 只试抽取一个片段，同样会发送原文并产生一次模型调用。不要对敏感或无权发送的论文使用这两个参数。

`graph_store.py` 定义构建批次、实体节点、关系、节点出现位置和关系证据表；每个批次绑定 Milvus 的物理文本 collection，而不是可切换的公共别名。全部片段抽取成功、节点和关系计数校验通过、每条关系都有来源证据后，批次才会标记为 `ready`。构建失败会在任务结果的 `graph_error` 中报告，文本索引仍可用；Lite/SQLite 模式不能构建图，继续使用原有检索。

查询侧会从问题中出现的实体名称及常规检索命中的片段出发，查找相关关系证据，并用当前物理 Milvus collection 的片段 ID、来源和页码再次核对。只有核对成功、且来源页不与常规结果重复的原文片段，才可能占用最终 Top-K 中的少量位置。设置 `ENABLE_GRAPH_RETRIEVAL=true` 后启用，`GRAPH_EVIDENCE_K` 控制最多占用的位置数（默认 2）；没有可用图或图查询失败时保持原有文本结果。此开关不会创建图。

目前名称归一化仅统一大小写和空白，仍可能把同一实体的不同别名拆成多个节点。下一步需要改进别名归一化、缺失页的关系抽取与最终证据选择，并人工核对关系正确性。当前实验结果不足以将图检索标为已上线功能。

有 `ready` 图后，可用同一组关联题比较图检索开关前后的证据覆盖率：

```bash
python eval/evaluate.py --qa-set eval/relationship_qa_set.json --compare-graph --show-trace
```

脚本会先检查当前文本索引版本是否有 `ready` 图；没有则报错，不会给出误导性的“图检索无变化”。两组评测都会调用 embedding，并可能调用 rerank API。`graph` 阶段只表示新增的图证据页，最终结果看 `final` 阶段。

如果暂时不想产生评测模型费用，可运行 `python eval/evaluate_graph_local.py --top-k 5 --graph-slots 2`，只用现有文本索引、PostgreSQL 图和本地 BM25 比较证据页覆盖率。它是**离线代理评测**，不能代替在线向量+rerank的结论。图证据选择仍属实验功能；在证据缺失或实际评测退化时，保持 `ENABLE_GRAPH_RETRIEVAL=false`。

当前两篇论文的[离线与在线对照结果](eval/graph_local_baseline.md)：在线 Top-5 的找齐率均为 3/8，加入图证据后标注页覆盖从 9/16 到 11/16。ProfilIoT 的关系证据仍明显不足，图检索继续默认关闭。

## 本地数据与隐私

以下内容已被 `.gitignore` 排除，不会提交到 Git：

- `.env` 和 API Key
- 上传的 PDF
- Milvus 向量数据库
- SQLite 聊天历史
- Streamlit 日志
- Python 缓存和虚拟环境

Docker 命名卷也不会提交到 Git。普通停止不会删除数据：

```bash
docker compose stop
```

恢复运行：

```bash
docker compose start
```

`docker compose down -v` 会永久删除 PostgreSQL、Milvus、MinIO 和 Redis 的本地卷，除非明确需要清空全部企业数据，否则不要执行。

需要注意：构建索引时，论文分块会发送到百炼 embedding 接口；回答问题时，检索到的论文片段会发送到百炼对话接口。请勿上传或处理不允许发送给外部模型服务的敏感文档。

启用图片索引后，抽取出的论文图片会发送到百炼多模态 embedding 接口；图片被检索命中时，还会发送给视觉模型用于生成回答。

## 常见问题

### `ImportError: cannot import name 'rewrite_query'`

通常是 Streamlit 热重载缓存了旧模块。完整停止旧进程后重新启动：

```bash
pkill -f "streamlit run app.py"
streamlit run app.py
```

### `No module named 'milvus_lite'`

确认正在 WSL/Linux 中运行，并重新安装依赖：

```bash
~/.local/bin/uv pip install \
  --python ~/.venvs/paper-rag/bin/python \
  --link-mode=copy \
    -r requirements.txt
```

### Ubuntu 中找不到 `docker`

在 Docker Desktop 的 `Settings → Resources → WSL Integration` 中启用 Ubuntu。不要在 Ubuntu 中额外安装一套 Docker Engine，以免与 Docker Desktop 冲突。

### Docker 镜像拉取超时

先确认 Docker Desktop 的代理设置可访问 `auth.docker.io` 和 `registry-1.docker.io`。使用 Windows localhost 代理时，可在 `%USERPROFILE%\.wslconfig` 中启用：

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
dnsTunneling=true
```

修改后执行 `wsl --shutdown` 并重启 Docker Desktop。

### 企业模式启动后连接失败

依次检查：

```bash
docker compose ps
curl http://127.0.0.1:9091/healthz
docker compose exec postgres pg_isready -U paper_rag -d paper_rag
docker compose exec redis redis-cli ping
```

预期分别看到 Milvus HTTP 200、PostgreSQL accepting connections 和 Redis `PONG`。

### API 返回认证或模型错误

检查：

- `.env` 中的 API Key 是否正确
- API Key 与百炼地域是否一致
- `OPENAI_BASE_URL` 是否为控制台提供的兼容接口
- `qwen-plus` 和 `text-embedding-v4` 是否已经开通

### 更换 embedding 模型后检索失败

模型输出维度必须和 `EMBEDDING_DIM` 一致。修改后在页面中重新构建整个知识库索引。

## 当前默认配置

| 项目 | 默认值 |
|---|---|
| 对话模型 | `qwen-plus` |
| 视觉模型 | `qwen3-vl-flash` |
| Embedding 模型 | `text-embedding-v4` |
| 向量维度 | `1024` |
| 图片 Embedding 模型 | `qwen3-vl-embedding` |
| 图片向量维度 | `1024` |
| 图片检索数量 | `2` |
| 文本块大小 | `400 tokens` |
| 重叠大小 | `80 tokens` |
| 检索数量 | `5` |
| 向量检索权重 | `0.7` |
| Rerank 模型 | `qwen3-rerank` |
| Rerank 候选数量 | `20` |
| 历史上下文 | 最近 `6` 条消息 |

## License

当前仓库尚未添加开源许可证。未经许可，请勿将代码或论文数据用于公开分发。
