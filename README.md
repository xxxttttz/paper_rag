# Paper RAG：论文检索与连续问答助手

一个面向本地 PDF 论文的 RAG（Retrieval-Augmented Generation）应用。项目使用 Streamlit 提供聊天界面，使用 Milvus Lite 保存论文向量，使用 SQLite 持久化对话历史，并通过阿里云百炼的 OpenAI 兼容接口调用千问模型。

## 功能

- 批量上传 PDF 论文并提取逐页文本
- 按 Token 滑动窗口切分论文内容
- 使用 `text-embedding-v4` 生成文本向量
- 使用 Milvus Lite 保存并检索论文片段
- 支持向量检索与 BM25 混合检索
- 使用 `qwen-plus` 生成带页码引用的回答
- 支持多会话、新建会话和删除会话
- 使用 SQLite 持久保存问题、回答及引用片段
- 根据最近对话改写追问，支持“它”“这个方法”等上下文指代
- 应用重启后恢复聊天历史

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
SQLite 保存问题、回答和引用快照
```

## 项目结构

```text
paper_rag/
├── app.py              # Streamlit 多会话页面
├── chat_service.py     # 对话、检索、生成和持久化编排
├── config.py           # 模型、路径和检索参数
├── database.py         # SQLite 会话历史存储
├── generator.py        # 追问改写与答案生成
├── ingest.py           # PDF 解析、分块、向量化和入库
├── retriever.py        # Milvus + BM25 混合检索
├── env.example         # 环境变量示例
├── requirements.txt    # Python 依赖
└── data/               # 本地论文、向量库和聊天库（不会提交）
```

## 环境要求

- Windows 10/11 + WSL2，或原生 Linux
- Python 3.11 或 3.12
- 阿里云百炼 API Key

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
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
```

如果百炼控制台提供了包含 Workspace ID 的专属兼容地址，应使用控制台给出的地址替换 `OPENAI_BASE_URL`。

请确保 embedding 模型输出维度和 `EMBEDDING_DIM` 一致。更换维度后必须重新构建知识库索引。

## 4. 启动应用

```bash
cd /mnt/c/Users/<你的用户名>/path/to/paper_rag
source ~/.venvs/paper-rag/bin/activate
streamlit run app.py
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

## 5. 构建论文知识库

1. 在左侧上传一篇或多篇 PDF。
2. 点击“重新构建知识库索引”。
3. 等待 PDF 解析、向量生成和 Milvus 写入完成。
4. 在页面底部输入论文相关问题。

每次重新构建都会替换现有 Milvus collection。聊天历史不会因此删除，但旧回答保存的是当时引用片段的文本快照。

## 对话历史

聊天数据保存在：

```text
data/chat_history.db
```

SQLite 中包含：

- `conversations`：会话标题与时间
- `messages`：用户和助手消息
- `citations`：回答对应的论文、页码、分数和文本快照

默认向模型发送最近 6 条消息（约 3 轮问答）。可以在 `config.py` 中调整：

```python
MAX_HISTORY_MESSAGES = 6
```

## 本地数据与隐私

以下内容已被 `.gitignore` 排除，不会提交到 Git：

- `.env` 和 API Key
- 上传的 PDF
- Milvus 向量数据库
- SQLite 聊天历史
- Streamlit 日志
- Python 缓存和虚拟环境

需要注意：构建索引时，论文分块会发送到百炼 embedding 接口；回答问题时，检索到的论文片段会发送到百炼对话接口。请勿上传或处理不允许发送给外部模型服务的敏感文档。

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
| Embedding 模型 | `text-embedding-v4` |
| 向量维度 | `1024` |
| 文本块大小 | `400 tokens` |
| 重叠大小 | `80 tokens` |
| 检索数量 | `5` |
| 向量检索权重 | `0.7` |
| 历史上下文 | 最近 `6` 条消息 |

## License

当前仓库尚未添加开源许可证。未经许可，请勿将代码或论文数据用于公开分发。
