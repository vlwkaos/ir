# ir

[ENG](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

面向 Markdown 知识库和混合代码仓库的本地语义搜索引擎。[qmd](https://github.com/tobi/qmd) 的 Rust 移植版，具有三项核心差异：

- **按集合独立 SQLite** — 每个集合是独立文件，无共享全局索引
- **持久化守护进程** — 模型在查询间保持加载；首次搜索自动启动
  冷启动的首次查询可在守护进程后台预热期间先返回 BM25 结果。
- **双重 LLM 缓存** — 扩展器输出和重排序分数持久化；重复查询即时返回
- **链接检索** — 可选代码单元与显式 Markdown/注释链接，便于代理获取上下文

已在 4 个 BEIR 数据集上测量检索质量；重排序相对纯向量最高提升 +14.5% nDCG@10。

<details>
<summary><strong>功能特性</strong></summary>

- **混合检索** — BM25 探测 → 分数融合 (0.80·向量 + 0.20·BM25) → LLM 重排序
- **查询扩展** — 存在扩展器模型时生成 lex/vec/hyde 类型子查询
- **强信号快捷** — BM25 最高分 ≥ 0.75 且差距 ≥ 0.10 时直接返回，跳过所有 LLM 计算
- **守护进程模式** — 模型在查询间保持热启动；首次搜索自动启动
  冷启动不阻塞首条有效 BM25 结果的返回。
- **双重 LLM 缓存** — 扩展器输出全局缓存；重排序分数按集合缓存
- **按集合独立 SQLite** — 独立 WAL 日志，隔离备份，集合间零竞争
- **内容寻址存储** — 相同文件在集合内通过 SHA-256 去重
- **FTS5 注入安全** — 所有用户输入在构造 FTS5 查询前经过转义处理
- **GPU 加速** — macOS 上默认使用 Metal，Linux 上可通过 feature 标志启用 CUDA/ROCm/Vulkan；通过 `IR_GPU_LAYERS=N` 调整
- **自动下载** — 首次使用时从 HuggingFace Hub 自动获取模型；`HF_HUB_OFFLINE=1` 可禁用

</details>

## 安装

**Homebrew (macOS)：**

```bash
brew install vlwkaos/tap/ir
```

**从源码构建：**

```bash
cargo install --path .
```

需要 Rust 1.80 及以上版本。macOS 上会自动与 Metal 链接 llama.cpp。Linux 上可通过 `--features llama-cuda`、`llama-rocm` 或 `llama-vulkan` 启用 GPU 加速。

## 快速开始

```bash
ir collection add notes ~/notes   # 注册集合
ir update notes                   # 扫描文件 → 提取文本 → 构建 FTS5 索引（BM25）
ir embed notes                    # 文本分块 → 运行嵌入模型 → 存储向量（启用向量和混合检索）
ir search "rust 内存安全"          # 搜索（守护进程自动启动）
```

**中文集合：**

```bash
ir preprocessor install zh        # 下载 lindera CLI + jieba 词典，注册为 "zh"
                                  # 安装后显示集合绑定选择器
ir collection add wiki ~/wiki     # 添加集合
ir preprocessor bind zh wiki      # 将 "zh" 绑定到集合并重新索引
ir search "机器学习" -c wiki
```

不使用预处理器时，"검색엔진"、"機械学習" 等黏着语词形会被当作单个 FTS 令牌处理，无法匹配词素级查询。中文同理——"机器学习"若不分词，则无法匹配"机器"或"学习"。

`ir update` 速度快（无需模型，纯文本处理）。`ir embed` 首次运行较慢（逐块模型推理），后续仅对变更内容重新嵌入。BM25 检索仅需 `update`；向量和混合检索需要 `embed`。

## 代码 + 知识链接检索

默认 Markdown 行为保持不变。若要索引混合代码/知识库，请使用 mixed 预设：

```bash
ir collection add project . --preset mixed
ir update project
ir embed project
ir search "为什么跳过 reranking" -c project --chunk --related 3 --json
```

`--preset mixed` 包含 Markdown 与主流代码扩展名（Rust、Python、JS/TS、Go、Java、C/C++、C#、Ruby、PHP、Swift、Kotlin、Scala、shell、Lua、Dart、Elixir、Erlang、F#、Clojure），并排除常见 build/vendor 目录。代码符号提取是 best-effort；无法识别的代码形态仍会按文件级单元索引。

相关链接只来自显式结构，不做静默推断：`[[wikilink]]`、本地 Markdown 链接、frontmatter `related:`，以及正文/注释中的 `[concept-slug]`。行号只是显示提示；结果会返回索引时的单元文本和表示单元文本哈希的 `indexed_hash`。代理使用规范见 [docs/linked-retrieval-agent.md](docs/linked-retrieval-agent.md)。

<details>
<summary><strong>模型</strong></summary>

模型在首次使用时从 HuggingFace Hub 自动下载并缓存到 `~/.cache/huggingface/`，无需手动配置。

| 模型 | HF 仓库 | 用途 |
|---|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | `ggml-org/embeddinggemma-300M-GGUF` | `ir embed`、向量检索、混合检索 |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | `unsloth/Qwen3.5-0.8B-GGUF` | 统一扩展+重排序（可选） |
| [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | `unsloth/Qwen3.5-2B-GGUF` | 统一扩展+重排序（可选） |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` | 仅重排序（可选） |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | `tobil/qmd-query-expansion-1.7B` | 仅查询扩展（可选） |
| [BGE-M3 568M](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) | `ggml-org/bge-m3-Q8_0-GGUF` | 韩语嵌入替代方案（可选） |

BM25 检索无需任何模型。默认 tier-2 路径为专用扩展器 + 重排序器。`IR_COMBINED_MODEL` 仅用于显式联合模型实验或测试。

**本地模型：**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

联合模式仅显式使用：

```bash
export IR_COMBINED_MODEL="$HOME/local-models/Qwen3.5-2B-Q4_K_M.gguf"   # 测试/实验专用
```

搜索顺序：环境变量 → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/` → HF Hub 自动下载。

`IR_*_MODEL` 环境变量接受 `.gguf` 文件路径、包含已知模型文件的目录路径或 HuggingFace 仓库 ID（`owner/name`）。无法识别的值会立即报错，而非静默加载默认模型。

**配置目录：**

```bash
export IR_CONFIG_DIR="~/vault/.config/ir"   # 可跨设备使用
```

`IR_CONFIG_DIR` 指定存放配置文件、集合数据库和守护进程文件的目录。支持 `~` 和 `$VAR` 展开，可安全用于跨设备同步的 MCP 配置。优先级：`IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir`（已废弃） → `~/.config/ir`。

**GPU：**

```bash
IR_GPU_LAYERS=0 ir search "查询"    # 强制 CPU
IR_GPU_LAYERS=32 ir search "查询"   # 部分卸载
```

</details>

<details>
<summary><strong>使用方法</strong></summary>

**集合管理：**

```bash
ir collection add notes ~/notes
ir collection add code  ~/code
ir collection ls
ir collection rm notes
ir status                    # 各集合索引状态
```

**索引与嵌入：**

```bash
ir update                    # 索引所有集合
ir update notes              # 索引指定集合
ir update notes --force      # 从头完整重建索引

ir embed                     # 嵌入所有未嵌入文档
ir embed notes --force       # 重新嵌入全部内容
```

**搜索：**

```bash
ir search "rust 内存安全"
ir search "sqlite 架构"        --mode bm25
ir search "异步模式"           --mode vector
ir search "错误处理"           --mode hybrid -c notes --min-score 0.4

# 输出格式
ir search "所有权" --json
ir search "所有权" --md
ir search "所有权" --files       # 仅路径
ir search "所有权" --full        # 结果中包含完整文档内容
ir search "所有权" --chunk       # 包含最匹配的分块/单元文本
ir search "所有权" --related 3   # 包含显式一跳关联上下文（最多 20）
ir search "所有权" --quiet       # 抑制 stderr（进度、日志）— 用于脚本

# 字段过滤（-f/--filter，可重复；所有条件 AND 连接）
ir search "设计" -f "modified_at>=2026-01-01"
ir search "设计" -f "meta.tags=rust"
ir search "设计" -f "path~notes/"
ir search "设计" -f "modified_at>=2025-01-01" -f "meta.author=vlwkaos"
```

**文档检索：**

```bash
ir get "2026/Daily/04/2026-04-07.md"            # 集合相对路径
ir get "Notes/2026/Daily/04/2026-04-07.md"      # 库根路径（自动去除集合目录前缀）
ir get "2026-04-07" -c periodic                  # 子串匹配，限定集合
ir get "some/path.md" --json                     # 以 JSON 输出完整元数据
ir get "some/path.md" --section "安装"           # 仅提取指定标题章节（不区分大小写）
ir get "some/path.md" --max-chars 3000           # 前 3000 个字符
ir get "some/path.md" --offset 1000 --max-chars 2000  # 第 1000~3000 个字符

ir multi-get "file1.md" "file2.md" "file3.md"   # 批量检索
ir multi-get "file1.md" "file2.md" --json        # {found: [...], not_found: [...]}
ir multi-get "file1.md" "file2.md" --files       # 仅输出找到的路径
ir multi-get "file1.md" "file2.md" --max-chars 2000  # 截断每份文档
```

路径匹配顺序：精确匹配 → 后缀匹配（`%/path`）→ 子串匹配。库根路径（首个路径组件与集合目录名匹配时）在常规匹配前优先处理。

**过滤语法（`-f/--filter`）：**

每个条件格式为 `字段 运算符 值`。多个 `-f` 条件以 AND 连接。

| 字段 | 说明 |
|------|------|
| `path` | 文档路径（相对于集合根目录） |
| `modified_at` | 文件修改时间（UTC RFC3339） |
| `created_at` | 文件创建时间（UTC RFC3339） |
| `meta.<name>` | 前言字段（如 `meta.tags`、`meta.author`） |

| 运算符 | 含义 |
|--------|------|
| `=` / `!=` | 等于 / 不等于（区分大小写） |
| `>` / `>=` / `<` / `<=` | 字典序比较（日期规范化为 UTC RFC3339） |
| `~` / `!~` | 包含 / 不包含（不区分大小写） |

`modified_at`、`created_at` 和 `meta.date` 的日期值规范化为 UTC RFC3339（`YYYY-MM-DD` → `YYYY-MM-DDT00:00:00Z`）。多值前言字段（如标签数组）只要**任意**一个元素满足条件即视为匹配——包括 `!=`。标注为 `["rust", "go"]` 的文档满足 `meta.tags!=rust`，因为 `"go"` 满足条件。无元数据行的文档对 `meta.*` 条件始终不匹配。

> **注意：** 本次发布后首次使用时集合数据库将升级至 schema v2。一次性回填（从现有前言填充 `document_metadata`）速度很快（1 万文档以内 <1 秒）。

**守护进程：**

```bash
ir daemon start              # 启动（首次搜索时自动启动）
ir daemon stop
ir daemon status
```

守护进程将模型保持在内存中。后续查询通过 Unix 套接字完全跳过模型加载（~30ms 往返）。

</details>

<details>
<summary><strong>增量索引</strong></summary>

ir 通过 SHA-256 哈希内容寻址存储高效处理更新，仅重新处理变更文件。

**工作原理：**

- **变更检测**：对文件进行哈希（SHA-256）并与存储的哈希比较
- **智能更新**：仅重新处理已修改或新增的文件
- **删除处理**：已移除的文件被标记为非活跃（软删除）
- **去重**：集合内相同内容共享存储

**索引操作：**

```bash
# 常规增量更新（默认）
ir update                    # 所有集合
ir update notes              # 指定集合

# 强制从头完整重建索引
ir update notes --force      # 重建整个索引

# 查看变更摘要
ir update notes
# 输出："2 added, 1 updated, 0 deactivated"
```

**嵌入操作：**

```bash
# 增量嵌入（仅处理新增/变更文档）
ir embed                     # 嵌入未处理内容
ir embed notes               # 指定集合

# 强制重新嵌入全部内容
ir embed notes --force       # 重新计算所有向量
```

**性能特征：**

- 初始索引：快（无需模型，纯文本提取）
- 增量更新：仅处理变更文件
- 哈希比较：即使数千个文件也能即时完成
- 嵌入：首次较慢，增量更新快

**示例工作流程：**

```bash
# 周一：初始设置
ir collection add notes ~/notes
ir update notes              # 索引 500 个文件
ir embed notes               # 计算 500 个嵌入（慢）

# 周二：新增 3 个文件，修改 2 个
ir update notes              # 输出："3 added, 2 updated, 0 deactivated"
ir embed notes               # 仅嵌入 5 个文档（快）

# 周三：删除 1 个文件
ir update notes              # 输出："0 added, 0 updated, 1 deactivated"
# 删除无需重新嵌入
```

增量方式使得可以频繁运行 `ir update` 而不影响性能——仅处理变更内容。

</details>

<details>
<summary><strong>MCP 服务器 — Claude Desktop / Claude Code</strong></summary>

`ir mcp` 运行 Model Context Protocol 服务器，让 Claude 可以直接搜索已索引的文档。

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`)：

```json
{
  "mcpServers": {
    "ir": {
      "command": "ir",
      "args": ["mcp"]
    }
  }
}
```

**Claude Code**（项目根目录的 `.mcp.json` 或 `~/.claude/mcp.json`）：

```json
{
  "mcpServers": {
    "ir": {
      "command": "ir",
      "args": ["mcp"]
    }
  }
}
```

提供五个工具：

| 工具 | 说明 |
|------|------|
| `search` | 混合 BM25+向量检索。返回路径、标题、分数、摘要。参数：`mode`、`limit`、`min_score`、`collections`、`full`（包含完整文档文本）、`include_chunk`（包含最匹配分块/单元文本）、`include_related`、`related_limit`（最多 20）、`filter`（`{field, op, value}` 对象数组，AND 连接）。 |
| `get` | 按路径检索文档文本（精确 → 后缀 → 子串匹配）。参数：`collections`、`section`（标题文本，不区分大小写）、`offset`（字符偏移）、`max_chars`（截断）。 |
| `multi_get` | 批量文档检索。参数：`paths[]`、`collections`、`max_chars`（截断每份文档）。返回 `found` 和 `not_found`。 |
| `status` | 索引状态——集合名称、文档数、数据库大小、守护进程状态。 |
| `update` | 文件变更后重新索引集合。接受 `collection` 和 `force` 参数。 |

`filter` 数组接受结构化条件：`{"field": "modified_at", "op": ">=", "value": "2024-01-01"}`。字段：`path`、`modified_at`、`created_at`、`meta.<名称>`。运算符：`=`、`!=`、`>`、`>=`、`<`、`<=`、`~`（包含）、`!~`（不包含）。

**HTTP 模式**（用于远程访问或多客户端场景）：

```bash
ir mcp --http 3620                              # 监听所有接口，端口 3620
ir mcp --http 3620 --cors '*'                   # 允许任意浏览器源（通配符）
ir mcp --http 3620 --cors 'https://app.example.com'  # 仅允许指定源
```

将客户端配置指向 `http://<host>:3620/mcp`。首次调用搜索工具时守护进程自动启动。

`--cors` 设置 `Access-Control-Allow-Origin`，允许浏览器客户端（Web 应用、Claude.ai 网页端）连接。`--cors '*'` 同时禁用 rmcp 的 DNS 重绑定主机检查，仅在可信网络中使用。不指定 `--cors` 时不发送 CORS 头（不影响 curl/CLI 客户端）。

> **安全说明：** HTTP 模式无身份验证，绑定所有接口。仅在可信网络中暴露。`update` 工具可触发重新索引，应视为本地写入访问服务。

</details>

<details>
<summary><strong>预处理器 — 中文 / 韩文 / 日文</strong></summary>

预处理器在 BM25 索引前对文本进行分词。不使用预处理器时，黏着语词形（"이스탄불의"、"東京都"、"机器学习"）会被当作单个 FTS 令牌处理，无法匹配词素或词级查询。索引时和查询时使用相同的预处理器。

**中文（lindera + jieba，Mode::Decompose）：**

```bash
ir preprocessor install zh          # 下载 lindera CLI + jieba 词典，注册为 "zh"
                                    # 安装后显示集合绑定选择器
ir collection add wiki ~/wiki       # 添加集合（若尚未添加）
ir preprocessor bind zh wiki        # 将 "zh" 绑定到集合并重新索引
ir search "机器学习" -c wiki
```

`ir preprocessor install zh` 从 lindera 的 GitHub 官方发布页下载 lindera CLI 二进制文件和 jieba 分词词典（与 `ko`/`ja` 使用相同的二进制文件，不同词典）。支持平台：**macOS**（arm64、x86\_64）和 **Linux**（x86\_64、aarch64）。无需系统依赖或 Rust 工具链。

与 `ko` 不同，绑定 `zh` **不**自动写入 routing 覆盖值——使用全局强信号阈值。如果在您的语料库上观察到过多或过少的 tier-1 升级，可添加集合级 `routing:` 块。

**限制：** 无停用词过滤。功能词（的、了、在、是……）会作为索引项保留。对功能词密集的查询，BM25 精度较低；hybrid+rerank 可补偿这一不足。

**韩文（lindera，Mode::Decompose）：**

```bash
ir preprocessor install ko          # 下载官方 lindera CLI + ko-dic，注册为 "ko"
ir collection add wiki ~/wiki
ir preprocessor bind ko wiki        # 绑定并重新索引
ir search "서울 지하철" -c wiki
```

绑定内置 `ko` alias 时会同时为该集合写入韩文 routing 默认值：

```yaml
routing:
  fused_strong_product: 0.05
```

这是绑定时写入的默认值，不是隐藏的运行时特殊逻辑。若已手动设置 `routing:` 块，则以显式配置为准。

**日文：**

```bash
ir preprocessor install ja    # 日文（lindera + ipadic）
```

**集合级 routing 覆盖**（`config.yml`，可选）：

```yaml
collections:
  - name: wiki-zh
    path: ~/wiki
    preprocessor: [zh]
    routing:
      fused_strong_product: 0.06   # 根据基准测试结果调整
```

支持字段：`fused_strong_floor`、`fused_strong_product`、`bm25_strong_floor`、`bm25_strong_gap`。覆盖仅在所有检索集合使用相同值时生效。混合不同覆盖的多集合检索会回退到全局默认阈值。

**管理：**

```bash
ir preprocessor list          # 显示已注册和可用的内置预处理器
ir preprocessor remove zh     # 取消注册（保留二进制文件）
ir preprocessor remove zh -d  # 取消注册并删除二进制文件
```

协议为 stdin/stdout 逐行处理：输入一行 UTF-8 文本，输出零或一行分词结果（若所有令牌被过滤则输出零行），进程在行间保持存活。子进程必须原样传递仅含 ASCII 的单词行——`ir` 使用内部哨兵令牌检测无输出的行。任何遵循此协议的可执行文件均可注册。

lindera 处理速度：M 系列 Mac 上约 5,600 韩文文档/秒 · 1.8 MB/秒。冷启动时间几乎为零（Rust 二进制，内嵌词典）。

**韩文 BM25 基准测试**（MIRACL-Korean，213 个查询）：

| 预处理器 | nDCG@10 | 备注 |
|---|---|---|
| 无 | 0.0009 | 黏着语令牌从不匹配 |
| lindera | 0.0460 | 形态分析带来 50 倍提升 |
| lindera hybrid+rerank | **0.8411** | 2,835 个段落上接近天花板 |

复合词分解基准测试（50 个针对复合词子成分的查询）：

| 预处理器 | nDCG@10 | 备注 |
|---|---|---|
| 无 | 0.0000 | FTS 索引中无子成分 |
| lindera | **0.6326** | Mode::Decompose 分解复合词 |

中文基准测试（MIRACL-Chinese）：待校准——运行 `ir preprocessor install zh && scripts/calibrate-fixtures.sh synthetic-zh` 后更新。

详细结果与原理：[research/experiment.md](research/experiment.md)

</details>

<details>
<summary><strong>检索管线</strong></summary>

```
查询
  │
  ├─ BM25 探测 ──► 分数 ≥ 0.75 AND 差距 ≥ 0.10？──► 直接返回
  │
  ├─ 有扩展器：  扩展 → lex/vec/hyde 子查询 → RRF 融合
  ├─ 无扩展器：  BM25 + 向量 → 分数融合（0.80·向量 + 0.20·BM25）
  │
  └─ 重排序器：最终得分 = 0.40·融合 + 0.60·P(相关)
```

扩展器和重排序器输出缓存在 SQLite 中。重复查询跳过 LLM 推理。

详见 [research/pipeline.md](research/pipeline.md) 中的分阶段异步守护进程设计。

</details>

<details>
<summary><strong>基准测试 — BEIR（4 个数据集，nDCG@10）</strong></summary>

EmbeddingGemma 300M 嵌入 + qmd-expander-1.7B + Qwen3-Reranker-0.6B。

| 数据集 | BM25 | 向量 | 混合 | +重排序 | LLM 增益 |
|---|---|---|---|---|---|
| NFCorpus（323q） | 0.2046 | 0.3898 | 0.3954 | **0.4001** | +1.2% |
| SciFact（300q） | 0.0500 | 0.7847 | 0.7873 | **0.7797** | −1.0% |
| FiQA（648q） | 0.0298 | 0.4324 | 0.4266 | **0.4567** | +7.1% |
| ArguAna（1406q） | 0.0012 | 0.4264 | 0.4263 | **0.4879** | +14.5% |

BM25 融合相对纯向量无统计显著提升（配对 t 检验）。重排序增益在对话型/论证型检索任务上最大。

复现方法：[research/experiment.md](research/experiment.md)

`scripts/bench.sh <dataset>` 输出各模式表格（`bm25`、`vector`、`hybrid`）并将完整 JSON 结果缓存到 `logs/results/<dataset>/`。中文基准：`scripts/bench.sh miracl-zh`（需先运行 `scripts/download-miracl-zh.sh`）。

</details>

<details>
<summary><strong>与 qmd 的对比</strong></summary>

ir 是 [qmd](https://github.com/tobi/qmd) 的 Rust 移植版，具有不同的存储模型和持久化守护进程。

| | qmd | ir |
|---|---|---|
| 存储 | 所有集合共用单个 SQLite | 按集合独立 SQLite — `rm name.sqlite` 即可删除 |
| 并发写入 | 共享 WAL 日志 | 每个集合独立 WAL |
| sqlite-vec | 动态加载 `.so` | 静态编译 |
| 进程模型 | 每次查询重新启动 | 守护进程保持模型热启动 |
| LLM 缓存 | 重排序分数（按集合） | 重排序分数 + 扩展器输出（全局） |
| 质量（NFCorpus nDCG@10） | 未公开数据 | 0.4001 |

**性能**（macOS M4 Max，相同模型和查询）：

| | ir | qmd | 倍率 |
|---|---:|---:|---|
| **冷启动**（无缓存） | 3.0s | 9.5s | **3×** |
| **热启动**（守护进程 + 缓存就绪） | 30ms | 840ms | **28×** |

冷启动差异：ir 将重排序候选上限设为 20 个，qmd 为 40 个。热启动差异：qmd 每次查询需 ~800ms 进程启动 + JS 运行时开销；ir 守护进程往返为 30ms（嵌入 + kNN）。

</details>

<details>
<summary><strong>开发</strong></summary>

```bash
cargo build                  # 调试构建
cargo build --release        # 发布构建
cargo test                   # 单元测试（无需模型）
cargo test -- --ignored      # 模型依赖测试（需要模型）
cargo run --bin eval -- --data test-data/nfcorpus --mode all
```

</details>

<details>
<summary><strong>数据库 Schema</strong></summary>

每个集合数据库（`~/.config/ir/collections/<name>.sqlite`）：

```
content          — 哈希 → 完整文本（内容寻址）
documents        — 路径、标题、哈希、活跃标志
documents_fts    — FTS5 虚拟表（porter 分词器）
vectors_vec      — sqlite-vec kNN（768 维余弦，EmbeddingGemma 格式）
content_vectors  — 分块元数据（哈希、序号、位置、模型）
llm_cache        — 重排序分数缓存（sha256(模型+查询+文档) → 分数）
meta             — 集合元数据（名称、schema 版本）
```

全局缓存（`~/.config/ir/expander_cache.sqlite`）：

```
expander_cache   — sha256(模型+查询) → JSON Vec<SubQuery>
```

触发器在 insert/update/delete 时保持 `documents_fts` 与 `documents` 同步。

</details>
