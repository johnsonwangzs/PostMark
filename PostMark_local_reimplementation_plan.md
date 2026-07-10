# PostMark 本地开源版改造方案（修订版）

> 目标：在官方仓库 `lilakk/PostMark` 的基础上，改造成一个完全离线、只依赖本地模型与固定资源的 PostMark baseline。
>
> 本方案不追求复现论文表格中的原始数值，但必须保留 PostMark 的核心算法结构：
> **文本语义编码 → 随机 anchor 预选 `3k` 个候选词 → 使用候选词真实 embedding 语义重排到 `k` → 本地 LLM 改写并插入水印词 → keyed blind detection。**
>
> 最终实现统一命名为 **PostMark-Local**。selector/table 的 `implementation_profile` 与 presence detector 的 `detector_profile` 独立配置：`compat` 尽量对齐论文和官方代码，`portable` 允许使用重建词表、exact-lemma、Nomic fuzzy 等本地替代组件。只要任一核心组件为 portable，整体结果就必须分开报告，不能视为论文原始 PostMark 数值。

## 1. 改造目标

必须实现以下要求：

1. 删除所有 OpenAI、Together.ai 和其他在线 API 相关代码、依赖、参数和密钥文件。
2. base LLM 和 inserter 均通过本地 HuggingFace 模型加载，例如本地的 Llama-3.1-8B-Instruct/Chat。
3. embedder 使用本地 `nomic-ai/nomic-embed-text-v1` 或用户指定的本地 Nomic 模型路径。
4. 不使用作者缺失的 `filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl`，而是在本地自行构建等价的 Nomic anchor pool。
5. 删除 paraphraser、`text3`、`list3` 和 paraphrase attack 相关逻辑。
6. 不依赖缺失的自定义序列化文件 `paragram_xxl.pkl`；优先支持从用户预先准备的本地 Paragram 原始向量构建兼容检测资源，同时提供明确标注的 portable detector。
7. 支持用户自己的通用 JSONL 输入，而不是写死 OpenGen、LFQA 和 FActScore。
8. 保证 watermark 和 detection 使用同一份固定资源、同一选择配置和经过指纹校验的模型/tokenizer，结果可复现。
9. 保留原始 PostMark 的 iterative insertion 主线，默认使用 `v2`：每次插入一小组水印词。
10. 提供最小测试、兼容性 fixture、命令示例、公平评测协议和验收标准。

## 2. 非目标

本次改造暂不实现：

- GPT-4、GPT-4o、GPT-3.5-Turbo 或任何在线模型；
- Together.ai；
- paraphrase attack；
- 论文中的人工评估和 GPT judge；
- 对论文所有表格数值的严格复现，以及把本地替代 detector/inserter 的结果直接与论文数字横向比较；
- 多语言统一支持；
- 大规模并行生成；
- 复杂攻击鲁棒性评估。

第一阶段应优先保证：

```text
构建资源 → 嵌入水印 → 盲检测 → 输出分数
```

能够稳定跑通。

本方案中的“完全离线”指：模型、tokenizer、spaCy 数据和可选 Paragram 原始向量完成离线部署后，资源构建、watermark、detect、test 全流程不发起网络请求。模型和语料的预先获取不属于运行阶段，但必须在文档中单独说明。


## 3. 需要保留的 PostMark 核心机制

原始 PostMark 的关键不是 Nomic 模型本身，而是以下三部分共同工作：

1. **文本编码器**：将待处理文本编码为一个语义向量。
2. **固定的候选水印词—语义锚点映射表**：每个候选水印词随机绑定到一个预先计算好的文本片段 embedding。给定输入文本后，根据输入文本 embedding 与所有 anchor embeddings 的 cosine similarity，选出最相近的若干 anchor，再取出这些 anchor 对应的候选词。
3. **候选词真实语义重排**：官方代码不是直接使用 anchor top-k，而是先按随机 anchor 取 `3k` 个候选词，再编码这些候选词本身，按候选词真实 embedding 与输入文本 embedding 的 cosine similarity 取最终 `k` 个词。该步骤属于论文明确描述的 semantic similarity filtering，不能作为“无效二次筛选”删除。

因此，缺失的：

```text
filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl
```

不是 Nomic 模型权重，而是一批作者预先计算好的文本片段 embedding，即语义 anchor pool。

本方案默认自行构建一个等价资源，并保存为：

```text
resources/postmark_nomic_table.pt
```

文件中应直接保存最终固定映射，而不是每次运行时重新随机分配：

```python
{
    "version": 2,
    "selection_mode": "official_two_stage",
    "embedder": {
        "path": "...",
        "revision": "...",
        "fingerprint": "sha256:...",
        "tokenizer_path": "...",
        "tokenizer_fingerprint": "sha256:...",
        "pooling": "mean",
        "normalization": "l2",
        "max_length": 512,
        "task_prefix": ""
    },
    "candidate_words": list[str],
    "anchor_embeddings": torch.Tensor,   # [num_words, embedding_dim]
    "candidate_word_embeddings": torch.Tensor,  # [num_words, embedding_dim]
    "seed": 42,
    "prefilter_multiplier": 3,
    "candidate_words_sha256": "...",
    "selected_chunks_sha256": "...",
    "source": {
        "corpus_path": "...",
        "corpus_revision": "...",
        "corpus_manifest_sha256": "...",
        "text_field": "text",
        "chunk_words": 250,
        "num_anchor_chunks": 100000,
        "chunking_algorithm": "sentbound_v1"
    }
}
```

`candidate_words[i]`、`anchor_embeddings[i]` 和 `candidate_word_embeddings[i]` 必须始终一一对应。运行时不得重新采样或 shuffle。

table 的 `content_sha256` 必须存放在同名 sidecar manifest 中，避免文件自哈希问题。它应对“移除 content hash 字段后的规范 metadata、candidate words 规范 JSON，以及包含 tensor 名称、dtype、shape、byte order 和连续原始字节的规范序列”计算。模型路径本身不能作为一致性依据；路径不同但权重相同应允许，权重或 snapshot 指纹不同则默认硬失败，除非用户显式使用仅用于调试的 `--allow_resource_mismatch`。

## 4. 建议的最终文件结构

```text
PostMark/
├── README.md
├── requirements.txt
├── prompts/
│   └── insert.txt
├── configs/
│   └── postmark_compat.json
├── postmark/
│   ├── __init__.py
│   ├── hf_llm.py
│   ├── nomic_embedder.py
│   ├── resources.py
│   ├── watermark.py
│   ├── detect.py
│   ├── build_candidate_words.py
│   ├── build_nomic_anchor_pool.py
│   ├── build_paragram_table.py
│   └── common.py
├── resources/
│   ├── candidate_words.json
│   ├── postmark_nomic_table.pt
│   ├── postmark_nomic_table.manifest.json
│   ├── paragram_table.pt
│   └── paragram_table.manifest.json
├── tests/
│   ├── test_resource_roundtrip.py
│   ├── test_word_selection.py
│   ├── test_presence.py
│   ├── test_jsonl_pipeline.py
│   ├── test_offline_mode.py
│   ├── test_resume_by_id.py
│   ├── test_retry_policy.py
│   ├── test_metrics.py
│   ├── test_quality_metrics.py
│   └── fixtures/
│       └── official_selector_fixture.json
└── examples/
    └── input.jsonl
```

可以保留原仓库中的 `outputs/`、`annotations/`、`parse_*.py`，但它们不应成为本地 baseline 的运行依赖。


## 5. 删除所有在线 API 相关内容

### 5.1 修改 `requirements.txt`

删除：

```text
openai
together
```

保留或补充：

```text
torch
transformers
accelerate
safetensors
numpy
scikit-learn
spacy
tqdm
sentencepiece
protobuf
pytest  # 或放入 requirements-dev.txt
```

不再被主流程使用的 `nltk`、`tiktoken`、`scipy`、`torchaudio`、`torchvision` 不应继续保留。明确要求 Python 3.10+，因为接口使用 `str | None` 等类型语法。不要在 requirements 中固定安装 CUDA runtime、cuDNN 等大型 NVIDIA wheel；让 PyTorch/CUDA 由用户环境管理，并单独提供经过验证的版本清单或 lock file。

### 5.2 删除在线模型实现

从原 `postmark/models.py` 中移除：

- `from openai import OpenAI`
- `from together import Together`
- `ChatGPT`
- `OpenAIEmb`
- `Llama3_70B_Chat`
- `openai_key.txt`
- `together_key.txt`
- 所有 `"gpt"` 分支
- 所有 API retry/sleep 逻辑

建议不要继续维护原来按模型名称写死的 `LLM` 分发器，而是改成通用本地 HuggingFace wrapper。

## 6. 本地 HuggingFace LLM 接口

新增：

```text
postmark/hf_llm.py
```

实现统一类：

```python
class LocalHFLLM:
    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        local_files_only: bool = True,
    ):
        ...

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        do_sample: bool | None = None,
    ) -> str:
        ...
```

### 6.1 加载要求

使用：

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
```

必须支持：

- 已存在于本地缓存中的 HuggingFace model ID；
- 本地模型目录；
- 与模型不同路径的本地 tokenizer；
- `device_map="auto"`；
- `float16`、`bfloat16`、`float32`；
- `trust_remote_code`；
- 默认 `local_files_only=True`，不得隐式回退到 Hub；
- tokenizer 没有 pad token 时回退到 EOS token。

模型和 tokenizer 加载后必须计算或读取稳定 fingerprint。路径只用于定位资源，fingerprint 才用于判断两次运行是否使用同一快照。

### 6.2 Chat 模型

默认使用：

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
)
```

必须允许通过 CLI 关闭 chat template，以兼容 base 模型。

### 6.3 生成策略

Base LLM 默认：

```text
temperature = 1.0
do_sample = true
top_p = 0.9
```

Inserter 默认：

```text
temperature = 0.0
do_sample = false
```

禁止在 `do_sample=false` 时传递无效的 temperature/top_p 参数。

### 6.4 模型复用

允许 base LLM 和 inserter 使用相同模型路径。若两者路径和加载参数相同，复用同一 tokenizer/model 实例，避免同一个 8B 模型在显存中加载两次。

### 6.5 随机性与离线约束

- prompt 模式下 base LLM 默认采样。每个样本的生成 seed 必须由全局 seed 和稳定 sample id 派生，不能只依赖进程级 RNG 状态，否则断点续跑会改变后续样本。
- `do_sample=false` 时不得传 `temperature`、`top_p`、`top_k` 等采样参数。
- 所有 `from_pretrained()` 调用都必须显式传入 `local_files_only=True`；远程 ID 未缓存或本地文件缺失时快速失败并给出预部署提示。
- 正式离线 smoke test 在 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 且网络被禁用的环境中运行。


## 7. 通用输入数据格式

删除原 `--dataset opengen/factscore/lfqa` 的写死逻辑。

### 7.1 已有原始文本

输入：

```json
{"id": "0", "text": "Existing LLM-generated response."}
```

命令：

```bash
python -m postmark.watermark   --input_path data/input.jsonl   --output_path runs/postmark/watermarked.jsonl   --text_field text   ...
```

此时不需要 `--base_llm_path`。

### 7.2 只有 prompt

输入：

```json
{"id": "0", "prompt": "Explain the collapse of the Soviet Union."}
```

命令：

```bash
python -m postmark.watermark   --input_path data/input.jsonl   --output_path runs/postmark/watermarked.jsonl   --prompt_field prompt   --base_llm_path /path/to/local/llama   ...
```

此时先生成 `text1`，再嵌入水印。

### 7.3 参数约束

必须满足以下之一：

```text
text_field 非空
```

或者：

```text
prompt_field 非空 且 base_llm_path 非空
```

两种模式不能同时启用。

### 7.4 baseline 公平性与样本标识

正式 baseline 对比应优先使用“已有原始文本”模式：先生成并冻结一份所有方法共享的 `text1` JSONL，再分别运行各 watermark。不得让不同方法各自从 prompt 重新采样 base answer。

- `id` 必须唯一且稳定；没有 `id_field` 时，从规范化后的输入内容生成 SHA-256 id。
- 每条输出必须保存 `id`、`input_sha256`、`selection_config_sha256` 和 `run_config_sha256`。
- prompt 模式只作为数据准备或便利模式；生成出的 `text1` 应先固化，再进入正式横向评测。

## 8. 构建候选水印词表

新增：

```text
postmark/build_candidate_words.py
```

提供 `compat` 和 `portable` 两种构建方式。默认 `compat` 不重新做 POS tagging，而是将仓库已有的 `valid_wtmk_words_in_wiki_base-only-f1000.pkl` 保序转换为版本化 JSON。该 pickle 只在受信任仓库的构建阶段读取，正式运行不加载 pickle。

### 8.1 `compat` 候选词表

官方论文报告最终词表为 3,266 个词，官方仓库也已包含相应 pickle。转换时必须：

- 保留原列表顺序，不重新排序或去重；
- 记录源 pickle 的 SHA-256；
- 验证元素均为非空字符串且没有重复项；
- 默认校验词数为 3,266，若仓库资源不同则明确失败并允许用户显式进入 portable 模式；
- 输出规范 JSON，并记录最终 words 数组的 SHA-256。

输出格式：

```json
{
  "version": 2,
  "profile": "compat",
  "source": "valid_wtmk_words_in_wiki_base-only-f1000.pkl",
  "source_sha256": "...",
  "words_sha256": "...",
  "words": ["ability", "able"]
}
```

### 8.2 `portable` 重建模式

当官方候选词 pickle 不可用时，允许从 `wikitext_freq.json` 或用户语料重建。该模式的结果必须标记为 portable，不能冒充官方词表。默认过滤规则为：

- lowercase；
- 只包含英文字母；
- 长度至少 3；
- 频次不低于 `1000`；
- 词性为 noun、verb、adjective 或 adverb；
- 排除 proper noun；
- 去重并稳定排序。

isolated-word spaCy POS tagging 与官方 NLTK `NN/VB/JJ/RB` 规则并不等价，因此必须记录 spaCy 模型路径、版本和 fingerprint。若需要最大程度接近官方但没有 pickle，可提供精确复刻 NLTK 规则的离线构建选项。

如果 `wikitext_freq.json` 不存在，允许从用户 JSONL 语料统计：

```bash
python -m postmark.build_candidate_words   --implementation_profile portable   --input_path data/corpus.jsonl   --text_field text   --output_path resources/candidate_words.json   --min_frequency 10   --max_words 20000
```

默认不要直接从最终测试集、calibration 集或其近重复文本构建候选词表。优先使用独立语料、训练集或公开语料，避免 evaluation leakage。


## 9. 自建 Nomic anchor pool

新增：

```text
postmark/build_nomic_anchor_pool.py
```

这是替代作者缺失 `.pkl` 的核心脚本。

### 9.1 输入

用户提供一个独立英文语料 JSONL：

```json
{"text": "A sufficiently long English document ..."}
```

参数：

```text
--input_path
--text_field
--candidate_words_path
--implementation_profile
--selection_mode
--embedder_path
--tokenizer_path
--output_path
--chunk_words
--num_anchor_chunks
--batch_size
--seed
--max_length
--corpus_revision
--corpus_manifest_sha256
--chunking_algorithm
--mapping_algorithm_version
--pooling
--normalization
--task_prefix
--local_files_only
```

建议默认值：

```text
chunk_words = 250
num_anchor_chunks = 100000
batch_size = 32
seed = 42
max_length = 512
```

首次调试可使用：

```text
num_anchor_chunks = 10000
```

但必须保证：

```text
num_anchor_chunks >= len(candidate_words)
```

### 9.2 文本切分

`compat` 模式优先使用固定 revision 和 manifest 的 RedPajama English split，以接近论文中的随机 250-word snippet 设置。允许使用其他独立英文语料，但生成的资源必须获得不同的 resource id，不能与其他 anchor table 混合汇总。

按空格词数生成约 `chunk_words` 长度的片段，尽量在句子边界结束。`compat` 模式固定并版本化切分算法；简单非重叠分块仅属于 portable 模式。

过滤：

- 太短片段；
- 空文本；
- 重复片段；
- 非英文占比过高的片段。

构建器应先稳定枚举、过滤并去重 `num_anchor_chunks` 个片段，再根据 seed 采样 `len(candidate_words)` 个索引。只需编码被选中的片段，无需先编码全部 100,000 个片段后再丢弃绝大部分。必须记录完整语料 manifest、选中索引和选中片段内容的哈希。

### 9.3 Nomic 编码

使用用户指定的本地路径：

```python
AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
AutoModel.from_pretrained(embedder_path, trust_remote_code=True, local_files_only=True)
```

实现统一 mean pooling：

```python
token_embeddings = model_output[0]
mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
embedding = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
embedding = F.normalize(embedding, p=2, dim=1)
```

必须保证：

- 构建资源、watermark 和 detector 使用完全相同的 pooling；
- embedding 统一 L2 normalize；
- 资源记录模型/tokenizer revision、fingerprint、维度、max length 和 task prefix；
- batch 推理；
- 使用 `torch.inference_mode()`；
- 保存前移到 CPU，并统一保存为 float32；
- 除选中 anchor 外，同时编码全部候选词并保存 `candidate_word_embeddings`，供官方二阶段选择重排使用。

### 9.4 固定词—anchor 映射

使用固定随机种子：

```python
rng = random.Random(seed)
```

步骤：

1. 稳定枚举并去重 `num_anchor_chunks` 个候选片段；
2. 使用固定 seed 和版本化的采样算法选出 `len(candidate_words)` 个索引；
3. 编码选中片段，并按官方逻辑对选出的 embedding 顺序做固定 shuffle；
4. 与候选词保序建立一一对应；
5. 编码候选词本身，得到对齐的 `candidate_word_embeddings`；
6. 保存最终映射、构建 manifest 和内容哈希。

不要在运行时的 `NomicPostMarkEmbedder.__init__()` 中再次随机采样或 shuffle。

最终资源：

```text
resources/postmark_nomic_table.pt
```

必须供 watermark 和 detector 共同加载。

### 9.5 构建 profile 与消融模式

默认资源必须支持：

```text
--implementation_profile compat
--selection_mode official_two_stage
```

可以额外实现：

```text
--selection_mode anchor_only
--selection_mode direct_word
```

但二者都只能作为消融。官方算法并不是在 random anchor 和 direct word 之间二选一，而是先通过 random anchor 预选，再使用 direct word embedding 重排。


## 10. Nomic embedder 实现

新增：

```text
postmark/nomic_embedder.py
```

接口：

```python
class NomicPostMarkEmbedder:
    def __init__(
        self,
        embedder_path: str,
        table_path: str,
        *,
        tokenizer_path: str | None = None,
        implementation_profile: str = "compat",
        selection_mode: str = "official_two_stage",
        ratio: float = 0.12,
        max_length: int = 512,
        device: str | None = None,
        local_files_only: bool = True,
    ):
        ...

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        ...

    def select_words(
        self,
        text: str,
        *,
        ratio: float | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        ...
```

### 10.1 选词数量

`compat` 默认严格使用官方代码口径：

```python
k = int(len(text.split()) * ratio)
```

即向下取整、使用 whitespace word count、不强制最少 1 个词，也不默认设置 64 上限。选择前必须满足 `0 < k <= len(candidate_words)`：

- `k == 0` 时不调用 selector/inserter，保留 `text2=text1`、令 detection score 为 `0.0`，记录终态失败 `failure_reason=k_zero`，并作为自然 false negative 保留在 intention-to-treat 指标中；
- `k > len(candidate_words)` 时不得静默 clamp。保留 `text2=text1`、score `0.0`，记录 `failure_reason=k_exceeds_vocabulary`；如果实验希望预先排除超长文本，资格规则必须在查看 watermark 结果前预注册，并对所有 baseline 使用同一 eligible id 集合。

`portable` 模式才允许：

```text
--min_watermark_words
--max_watermark_words
```

以及：

```python
k = max(min_words, round(stable_word_count(text) * ratio))
k = min(k, max_words) if max_words is not None else k
```

profile、word-count 规则、取整规则和 min/max 必须进入 `selection_config_sha256`。

### 10.2 选词逻辑

默认 `official_two_stage`：

```python
text_embedding = encode_texts([text])[0]  # [d]
assert 0 < k <= len(candidate_words)
m = min(3 * k, len(candidate_words))

anchor_scores = anchor_embeddings @ text_embedding
candidate_indices = stable_topk(anchor_scores, m)

word_scores = candidate_word_embeddings[candidate_indices] @ text_embedding
reranked = stable_topk(word_scores, k)
indices = candidate_indices[reranked]
words = sorted({candidate_words[i] for i in indices})
```

这里的第二阶段是官方论文和代码中的 semantic similarity filtering，必须作为 compat 默认路径。`anchor_only` 和 `direct_word` 仅用于显式消融，输出 manifest 必须标记为不可与 compat 混比。`stable_topk` 应对分数相同的情况使用候选索引作为固定 tie-breaker。

### 10.3 资源一致性检查

加载资源时验证：

- `len(candidate_words) == anchor_embeddings.shape[0] == candidate_word_embeddings.shape[0]`；
- 两种 embedding 的维度与当前 Nomic 模型一致；
- 两种 embedding 均无 NaN/Inf 且近似单位范数；
- candidate words、table、corpus、selected chunks 的哈希一致；
- 模型/tokenizer fingerprint、pooling、normalization、max length、task prefix、selection mode 和算法版本一致；
- watermark 与 detector 的 ratio、profile、word-count 和 k 规则一致；
- 路径不同但 fingerprint 相同可以继续；fingerprint 或配置不同默认硬失败；
- 表文件不存在时给出资源构建命令提示。

仅允许通过 `--allow_resource_mismatch` 或 `--allow_config_mismatch` 进入诊断模式；此时输出必须标记 `config_consistent=false`、`eligible_for_aggregate=false`，且正式指标汇总器拒绝读取。另需有正向测试证明“路径不同但 fingerprint 相同”可以正常通过。


## 11. Watermarker 改造

重写：

```text
postmark/watermark.py
```

核心类：

```python
class PostMarkWatermarker:
    def __init__(
        self,
        inserter: LocalHFLLM,
        selector: NomicPostMarkEmbedder,
        *,
        prompt_path: str,
        iterate: str = "v2",
        group_size: int = 10,
        min_group_presence: float = 0.5,
        max_insert_attempts: int = 1,
        retry_strategy: str = "missing_words",
    ):
        ...

    def insert_watermark(self, text: str) -> dict:
        ...
```

### 11.1 保留官方 insertion prompt

继续使用：

```text
prompts/insert.txt
```

baseline 第一版不要改写 prompt。

### 11.2 迭代插入

默认 `v2`：

1. 对原始文本 `text1` 计算 `list1`；
2. 按 `group_size=10` 切分；
3. 将当前文本和一组 watermark words 填入 prompt；
4. inserter 使用 greedy decoding 改写；
5. `compat` 使用官方大小写不敏感 substring 规则检查该组词；`portable` 可选择 exact-token/lemma；
6. `max_insert_attempts` 表示每个 group 的总尝试次数，最小值为 1；greedy decoding 对相同 prompt 只执行一次，后续 prompt 必须基于当前最佳文本且只请求缺失词，或显式使用带独立 attempt seed 的 sampling retry；
7. 下一组基于当前 group 的最佳候选继续；
8. 得到 `text2`；
9. 对 `text2` 重新独立计算 `list2`。

输出：

```json
{
  "id": "0",
  "status": "completed",
  "text1": "...",
  "list1": ["..."],
  "text2": "...",
  "list2": ["..."],
  "diagnostics": {
    "list_overlap": 0.73,
    "requested_word_presence": 0.81,
    "num_groups": 3,
    "num_attempts": 4,
    "insertion_failed": false,
    "embedding_input_truncated": false,
    "generation_output_truncated": false,
    "length_delta_words": 24,
    "stop_reason": "all_groups_processed"
  },
  "input_sha256": "...",
  "selection_config_sha256": "...",
  "run_config_sha256": "...",
  "selector_resource_sha256": "..."
}
```

`requested_word_presence` 是 `list1` 在 `text2` 中的出现率，只用于调试嵌入阶段。

多次尝试时必须始终保留 requested-word presence 最高的候选，后续更差结果不得覆盖。每次尝试记录 prompt hash、缺失词、presence、是否被选为最佳候选和 stop reason；连续无提升时提前停止。

若所有生成尝试均为空或抛出可恢复错误，使用 `text1` 作为 fallback `text2`，写入终态 `status=failed` 和明确 failure reason，并照常进入 detector/正式指标。只要存在非空候选，即使用最佳候选，不因插入率、截断或质量不足而丢弃。`status=completed` 和 `status=failed` 都是已终结记录。

### 11.3 断点续跑

- 输出已存在时按稳定 sample id 建立索引，不能按已有记录数或行号恢复；
- 仅当 `id + input_sha256 + selection_config_sha256 + run_config_sha256` 完全匹配，且记录为终态 `completed` 或 `failed` 时跳过；
- 相同 id 对应不同输入、资源、模型、prompt 或配置时硬失败，并提示使用 `--overwrite` 或新输出目录；
- 输入/输出重复 id、manifest 不匹配和损坏的非尾部 JSONL 行均硬失败；
- 若且仅若最后一行是不完整 JSON，先将原文件备份为带时间戳的 `.corrupt.bak`，再截断到最后一个完整换行并重跑该 id；中间坏行禁止自动修复；
- 非终态记录视为中断并重跑。若用户需要重新尝试终态 failed 记录，必须通过显式 `--retry_failed` 创建新的 run config/hash 和新的输出路径，禁止在原 JSONL 中原地混写不同配置；
- append 继续；
- 每条写入后 `flush()` 并 `os.fsync()`，保证掉电后最多损坏最后一行；
- 支持 `--overwrite`；
- 删除原来补 `text3` 的逻辑。

### 11.4 CLI

至少支持：

```text
--input_path
--output_path
--text_field
--prompt_field
--id_field
--base_llm_path
--base_tokenizer_path
--inserter_path
--inserter_tokenizer_path
--embedder_path
--embedder_tokenizer_path
--table_path
--prompt_path
--implementation_profile
--selection_mode
--ratio
--min_watermark_words
--max_watermark_words
--iterate
--group_size
--min_group_presence
--max_insert_attempts
--retry_strategy
--insertion_presence_mode
--max_new_tokens
--torch_dtype
--device_map
--use_chat_template
--local_files_only
--run_manifest_path
--allow_resource_mismatch
--allow_config_mismatch
--retry_failed
--limit
--seed
--overwrite
```


## 12. 删除 paraphraser

彻底删除或不再导入：

- `paraphrase_sent_init.txt`
- `paraphrase_sent.txt`
- `paraphrase()`
- `--para`
- `--paraphraser`
- `text3`
- `list3`
- `score3`
- paraphrase attack 评估分支

原 prompt 文件可以作为历史文件保留，但主脚本不能依赖。

## 13. 重建 detector

重写：

```text
postmark/detect.py
```

detector 必须是**盲检测**，不能依赖嵌入阶段保存的 `list1` 或 `list2` 才能工作。

这里的“盲检测”准确含义是：检测时不需要原始未加水印文本，也不读取嵌入阶段保存的 `list1/list2`，但需要持有相同的 secret/keyed table 和完整 selection config。它不是无密钥检测。

对每个候选文本：

1. 使用相同 Nomic 模型；
2. 使用相同 `postmark_nomic_table.pt`；
3. 根据候选文本自身语义重新选出期望水印词；
4. 计算这些期望词在候选文本中的出现率；
5. 将出现率作为 watermark score。

```python
expected_words = selector.select_words(candidate_text)
score = presence(candidate_text, expected_words)
```

### 13.1 Presence 模式 A：`paragram_fuzzy`（compat 默认）

官方检测口径为 exact token 加 Paragram fuzzy matching，默认 cosine threshold 为 `0.7`：

1. 使用 spaCy 或等价稳定 tokenizer 对候选文本分词；
2. 去除标点和空白，token lowercase，但不以 lemma 替代 token；
3. 期望词 exact token 命中时直接标记 present；
4. 否则查询期望词和文本 token 的本地 Paragram 向量；
5. 任意 token 与期望词的 cosine similarity `>= 0.7` 时标记 present；
6. OOV 词只能通过 exact token 命中，不能使用零向量参与 fuzzy 匹配。

```python
score = present_count / len(expected_words)
```

新增：

```text
postmark/build_paragram_table.py
```

该脚本从用户预先准备的、受信任的本地 Paragram 原始向量文件构建 `resources/paragram_table.pt`，并使用仓库现有的 `paragram_xxl_words.json` 校验词到索引的对齐关系，而不是依赖缺失的 `paragram_xxl.pkl`。资源必须记录原始文件 hash、词表 hash、维度、normalization 和构建器版本。正式运行不下载向量，也不加载不受信任的 pickle。

若用户没有本地 Paragram 原始向量，可以在同一个 compat selector/table 上运行 portable detector。此时保持 `implementation_profile=compat`，另设 `detector_profile=portable`；整体输出标记 `paper_method_compatible=false`、`exact_paper_reproduction=false`。

### 13.2 Presence 模式 B：`exact_lemma`（portable）

1. spaCy tokenize；
2. lowercase 并 lemmatize；
3. 若期望词本身或 lemma 出现在文本 token/lemma 集合中，则认为 present。

该模式不提供官方的 synonym fuzzy matching，只用于完全自包含的 portable baseline 或消融。
运行时保持与 table 一致的 selector `implementation_profile`，并设置 `detector_profile=portable`。

### 13.3 Presence 模式 C：`nomic_fuzzy`（portable/消融）

1. 先 exact lemma；
2. 对未命中的期望词用 Nomic 编码；
3. 对候选文本 content-word token/lemma 编码；
4. 任意 cosine similarity 不低于阈值则认为 present。

参数：

```text
--presence_mode nomic_fuzzy
--similarity_threshold 0.75
```

要求：

- batch encode；
- 去 stop words、标点和过短 token；
- 缓存候选词 embedding；
- 限制单个文本最大 token 数；
- threshold 必须在独立 dev/calibration 数据上预先固定，不能沿用未经验证的 `0.75` 后直接与论文比较；
- 结果必须标记为 portable/消融，不得与 `paragram_fuzzy` 混合汇总。
- 运行时保持与 table 一致的 selector `implementation_profile`，并设置 `detector_profile=portable`。

### 13.4 通用候选文本输入

```json
{"id": "x", "text": "..."}
```

命令：

```bash
python -m postmark.detect   --input_path data/candidates.jsonl   --text_field text   --output_path runs/postmark/detected.jsonl   ...
```

输出：

```json
{
  "id": "x",
  "text": "...",
  "expected_words": ["..."],
  "watermark_score": 0.43,
  "implementation_profile": "compat",
  "detector_profile": "compat",
  "presence_mode": "paragram_fuzzy",
  "selection_config_sha256": "...",
  "detector_config_sha256": "...",
  "selector_resource_sha256": "...",
  "presence_resource_sha256": "...",
  "config_consistent": true,
  "eligible_for_aggregate": true,
  "paper_method_compatible": true,
  "exact_paper_reproduction": false
}
```

### 13.5 成对评估模式

读取 `text1/text2`，分别重算：

- `score1`：原始文本；
- `score2`：水印文本。

计算：

- ROC-AUC；
- 论文兼容的同集 ROC 插值 TPR@1% FPR，仅用于复核旧口径；
- 独立 calibration negatives 定阈值后的 held-out TPR@1% FPR 和实际 held-out FPR；
- 按 sample id bootstrap 的 95% confidence interval；
- 平均 `score1`；
- 平均 `score2`；
- 平均分数差。

```bash
python -m postmark.detect   --input_path runs/postmark/watermarked.jsonl   --paired   --negative_field text1   --positive_field text2   --calibration_path data/calibration.jsonl   --output_path runs/postmark/scored.jsonl   ...
```

### 13.6 正式检测不得使用保存的 list

`list1/list2` 只用于调试、分析稳定性和验证重算结果。正式 detector 必须重新调用：

```python
selector.select_words(text)
```

加载时 detector 必须从 table/manifest 继承或核对 `implementation_profile`、`selection_mode`、`ratio`、word-count/k 规则、embedder/tokenizer fingerprint、pooling、max length 和 task prefix。任一 selection contract 不一致均默认硬失败。`detector_profile` 独立控制 presence 实现；只要 selector contract 一致，就允许 compat table 搭配 portable detector，但两者必须使用不同 `detector_config_sha256` 并分开汇总。


## 14. 实验配置与公平评测

### 14.1 配置与 manifest

每次资源构建和实验运行都生成规范 JSON manifest：

- `selection_config_sha256` 覆盖 table hash、embedder/tokenizer fingerprint、pooling、normalization、max length、task prefix、profile、selection mode、ratio/top-k、word-count/k 规则及算法版本；
- `run_config_sha256` 在 selection config 基础上继续覆盖 prompt hash、inserter/base LLM fingerprint、解码参数、group/retry 策略和 seed；
- `detector_config_sha256` 覆盖 selection config hash、detector profile、presence mode、Paragram/Nomic presence resource hash、similarity threshold、spaCy/tokenizer fingerprint、token filter 和 OOV 规则；
- `evaluation_config_sha256` 覆盖 detector config hash、calibration/evaluation split hash、阈值规则、bootstrap 次数/方法/seed 和质量评估器配置；
- 每条输出保存适用的 hash，汇总器拒绝混合不一致配置；
- 诊断性 mismatch override 的输出必须标记 `config_consistent=false`、`eligible_for_aggregate=false`，不得进入正式指标；
- `paper_method_compatible` 表示 selector/detector 是否保持论文方法口径；`exact_paper_reproduction` 在本项目中始终为 false，因为 anchor/inserter 已改变。这两个字段不得与配置一致性混为一谈。

### 14.2 数据隔离与共享原文

- anchor corpus、候选词构建语料、dev/calibration 和 held-out evaluation 必须互相隔离，并尽可能做 exact/near-duplicate 检查；
- 所有 baseline 使用完全相同、预先冻结的 `text1`、sample id 和 input hash；
- ratio、group size、presence threshold、fuzzy threshold、table seed 和 retry 策略只能在 dev 集确定；
- 不得按插入成功率、list overlap、截断状态或输出质量过滤测试样本，正式汇总采用 intention-to-treat，覆盖全部输入 id；
- 建议至少运行 3 个预先指定的 table seed，报告均值和标准差，避免挑选有利映射。

### 14.3 TPR@1% FPR 协议

正式评测优先使用独立 calibration split：

1. `--calibration_path` 是只包含 negatives 的 JSONL，每行至少为 `{"id": "...", "text": "..."}`，由 `--calibration_text_field text` 指定字段；id 必须唯一，并与 held-out evaluation id 无交集；自动切分时使用稳定 id hash，使输入重排不改变 split；
2. 只使用 calibration negatives 选择阈值 `tau`，判定规则固定为 `score >= tau`；
3. 候选阈值集合为每个有限 unique score 本身及其 `math.nextafter(score, +inf)`；逐个计算包含完整 tie group 的 empirical FPR，并选择满足 FPR `<= 0.01` 的最小有限阈值。若只有高于最大分数才满足，则使用 `math.nextafter(max_score, +inf)`，不得在 JSON 中写 Infinity；
4. 冻结 `tau` 后，在 held-out positives/negatives 上报告 TPR 和实际 FPR，不得使用 held-out positives 调参；
5. ROC-AUC 只在 held-out 数据上计算；报告正负样本数、split hash、`tau`、calibration FPR、held-out FPR/TPR，以及按 sample id bootstrap 的 95% CI；
6. bootstrap 次数、percentile/BCa 方法和 seed 必须预先固定并写入 evaluation manifest，默认可使用 2,000 次 percentile bootstrap。

正式 1% FPR 建议 calibration negatives 和 held-out negatives 各至少 1,000 条；数量不足时只能输出明确标记的 diagnostic 指标，不能宣称稳定的 1% FPR 结论。

### 14.4 质量与失败指标

检测率不能脱离文本质量报告。每次正式实验至少同时输出：

- `insertion_success`：`list1` 非空、最终输出非空，且每个 group 的最佳 presence 均达到 `min_group_presence`；
- `max_attempt_exhausted`：至少一个 group 达到总尝试上限仍未满足阈值；
- `empty_output`：该样本所有生成尝试都只返回空白/空串；
- 分开报告 `embedding_input_truncated` 和 `generation_output_truncated`，后者指达到 `max_new_tokens` 且未生成 EOS；
- requested-word presence、list overlap；
- text1/text2 的绝对与相对长度变化；
- 独立本地语义评估器得到的相似度；若只能复用 Nomic，必须标记为 Nomic proxy；
- 存在任务参考答案时，报告 text1/text2 的任务指标及 delta；
- 失败样本也必须保留在分母中，并给出明确的失败分类。

所有比例以正式 eligible id 集合为分母。聚合前拒绝 NaN/Inf；`k=0`、`k>N` 或完全生成失败按前述 fallback/score 规则进入指标，不得运行后静默 drop。

删除 paraphrase attack 后，本阶段只能声明 clean-condition baseline。若后续声称鲁棒性，必须由统一的本地 attack harness 对所有方法施加同一攻击。

## 15. 公共工具

新增：

```text
postmark/common.py
```

包含：

- `set_global_seed()`；
- 规范 JSON 和 SHA-256 计算；
- stable sample id、per-sample seed 和配置 hash；
- JSONL 校验、按 id 索引、原子追加和截断尾行恢复；
- stable word count；
- spaCy token/lemma normalization；
- calibration threshold、ROC/TPR、bootstrap CI 和质量指标辅助函数；
- dataclass 配置；
- 明确异常类型。

不要在模块 import 时：

- 自动下载 spaCy 模型；
- 加载大型模型；
- 读取不存在的资源。

spaCy 只能从已安装的本地包或用户提供的本地路径加载。缺少模型时给出明确的离线部署提示，不要静默联网下载。

## 16. 推荐运行流程

### 16.1 离线预部署与安装

以下资源必须预先放到本地：LLM、Nomic model、对应 tokenizer、spaCy 英文模型、独立 anchor corpus，以及 compat detector 所需的 Paragram 原始向量。联网下载或制作 wheelhouse 属于部署准备，不属于正式运行流程。

```bash
pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

正式命令全部传本地绝对路径，并保持 `--local_files_only` 开启。README 可以另列联网 staging 命令，但必须与离线运行命令分开。

### 16.2 构建 compat 候选词表

```bash
python -m postmark.build_candidate_words \
  --implementation_profile compat \
  --legacy_pickle_path valid_wtmk_words_in_wiki_base-only-f1000.pkl \
  --output_path resources/candidate_words.json
```

portable 模式可从 `wikitext_freq.json` 或独立用户语料构词，但输出 manifest 必须标记 `profile=portable`。

### 16.3 构建 Nomic anchor table

```bash
python -m postmark.build_nomic_anchor_pool \
  --implementation_profile compat \
  --selection_mode official_two_stage \
  --input_path /path/to/redpajama_english.jsonl \
  --text_field text \
  --corpus_revision <fixed-revision> \
  --corpus_manifest_sha256 <sha256> \
  --candidate_words_path resources/candidate_words.json \
  --embedder_path /path/to/nomic-embed-text-v1 \
  --tokenizer_path /path/to/bert-base-uncased-or-verified-tokenizer \
  --output_path resources/postmark_nomic_table.pt \
  --chunk_words 250 \
  --num_anchor_chunks 100000 \
  --batch_size 32 \
  --seed 42 \
  --max_length 512 \
  --pooling mean \
  --normalization l2 \
  --task_prefix "" \
  --chunking_algorithm sentbound_v1 \
  --mapping_algorithm_version 1 \
  --local_files_only
```

### 16.4 构建本地 Paragram detector 资源

```bash
python -m postmark.build_paragram_table \
  --vectors_path /path/to/local/paragram_vectors.txt \
  --vocab_path paragram_xxl_words.json \
  --output_path resources/paragram_table.pt
```

如果没有该资源，跳过本步并使用 portable detector；结果名称和 manifest 必须明确标记不可与论文口径直接比较。

### 16.5 对冻结文本嵌入水印

```bash
python -m postmark.watermark \
  --implementation_profile compat \
  --selection_mode official_two_stage \
  --input_path data/frozen_text1.jsonl \
  --output_path runs/postmark/watermarked.jsonl \
  --text_field text \
  --id_field id \
  --inserter_path /path/to/llama-3.1-8b-instruct \
  --inserter_tokenizer_path /path/to/llama-3.1-8b-instruct \
  --embedder_path /path/to/nomic-embed-text-v1 \
  --embedder_tokenizer_path /path/to/bert-base-uncased-or-verified-tokenizer \
  --table_path resources/postmark_nomic_table.pt \
  --prompt_path prompts/insert.txt \
  --ratio 0.12 \
  --iterate v2 \
  --group_size 10 \
  --min_group_presence 0.5 \
  --max_insert_attempts 1 \
  --max_new_tokens 600 \
  --torch_dtype bfloat16 \
  --local_files_only
```

### 16.6 只有 prompt 时生成并嵌入

prompt 模式可以保留，但用于正式对比时应先单独生成并冻结 `text1`，再让所有 baseline 共享该文件。base LLM 采样必须使用由 sample id 派生的 seed。

### 16.7 compat 成对检测与校准

```bash
python -m postmark.detect \
  --implementation_profile compat \
  --detector_profile compat \
  --selection_mode official_two_stage \
  --input_path runs/postmark/watermarked.jsonl \
  --calibration_path data/calibration.jsonl \
  --calibration_text_field text \
  --output_path runs/postmark/scored.jsonl \
  --paired \
  --negative_field text1 \
  --positive_field text2 \
  --embedder_path /path/to/nomic-embed-text-v1 \
  --embedder_tokenizer_path /path/to/bert-base-uncased-or-verified-tokenizer \
  --table_path resources/postmark_nomic_table.pt \
  --presence_mode paragram_fuzzy \
  --paragram_table_path resources/paragram_table.pt \
  --similarity_threshold 0.7 \
  --ratio 0.12 \
  --local_files_only
```

### 16.8 portable 检测

```bash
python -m postmark.detect \
  --implementation_profile compat \
  --detector_profile portable \
  --selection_mode official_two_stage \
  --input_path runs/postmark/watermarked.jsonl \
  --calibration_path data/calibration.jsonl \
  --calibration_text_field text \
  --output_path runs/postmark/scored-exact-lemma.jsonl \
  --paired \
  --negative_field text1 \
  --positive_field text2 \
  --embedder_path /path/to/nomic-embed-text-v1 \
  --embedder_tokenizer_path /path/to/bert-base-uncased-or-verified-tokenizer \
  --table_path resources/postmark_nomic_table.pt \
  --presence_mode exact_lemma \
  --ratio 0.12 \
  --local_files_only
```

该结果必须显示 `config_consistent=true`、`eligible_for_aggregate=true`、`paper_method_compatible=false`、`exact_paper_reproduction=false`，并与 compat detector 结果分表报告。这里 portable 的只是 detector；selector/table/tokenizer 仍保持 compat 配置。

## 17. 测试要求

### 17.1 资源与配置测试

- `test_resource_roundtrip.py` 验证 candidate words、anchor embeddings 和 candidate word embeddings 保存/加载后逐项一致；
- 两种 embedding 行数/维度对齐、全部有限且近似单位范数；
- 同一 seed 两次构建映射一致，不同 seed 映射不同；
- candidate/corpus/selected-chunks/table hash 可重算且一致；
- 篡改任一 metadata、tensor、模型/tokenizer fingerprint 或 selection config 时默认硬失败；
- 路径不同但 fingerprint 相同可通过；mismatch override 只产生 `config_consistent=false`、`eligible_for_aggregate=false` 的诊断输出；
- content hash 绑定 tensor 名称、dtype、shape、byte order 和字节内容。

### 17.2 `test_word_selection.py`

- 同一文本多次选词一致；
- compat 使用 `int(len(text.split()) * ratio)`，portable 使用各自声明的 round/clamp；
- 空文本和 `k=0` 行为明确，compat 不静默强制为 1；
- `k > len(candidate_words)` 不调用 top-k、不 clamp，并按预定义失败/score 规则输出；
- fixture 验证 anchor top-`3k` 后使用 candidate word embeddings 重排到 `k`；
- 构造样例证明二阶段结果可不同于 anchor-only；
- tie 按候选索引稳定处理；
- 不产生重复词。

### 17.3 `test_presence.py`

- Paragram exact token、大小写、标点、cosine 恰好为 `0.7`、低于阈值和 OOV；
- exact-lemma 的词形变化；
- Nomic fuzzy threshold 和缓存；
- 空 expected list 返回固定 score `0.0` 和失败标签，而不是除零、NaN 或运行后丢弃；
- compat/portable presence 结果带正确标签。

### 17.4 pipeline、resume 与 retry

- `test_jsonl_pipeline.py` 使用 mock inserter验证字段、manifest、flush 和 overwrite；
- `test_resume_by_id.py` 覆盖输入重排、追加/中间插入、重复 id、同 id 改文本、改 ratio/model/prompt、配置 hash 冲突、终态 completed/failed、非终态和尾行截断；验证只备份/截断最后一个坏行且中间坏行硬失败；
- 只跳过 id、输入 hash、selection hash、run hash 全匹配的终态记录，每个 id 最终恰好输出一次；
- mock sampling base LLM 验证完整运行与中断、重排后 resume 的 `text1/text2` 逐项一致；
- `test_retry_policy.py` 验证 `max_insert_attempts` 为每组总次数、greedy 不重复相同 prompt、后续只处理缺失词、较差候选不覆盖最佳、无提升早停，且每次诊断包含 prompt hash/missing words/presence/selected/stop reason；
- 若支持 sampling retry，每个 attempt seed 可复现且互不相同。

### 17.5 离线测试

- `test_offline_mode.py` 断言所有 HF loader 都收到 `local_files_only=True`；
- 禁用 socket 后 import、资源加载和 mock pipeline 不触网；
- 未缓存的远程 ID、缺失 tokenizer/spaCy/model 立即给出可操作的离线错误；
- 网络隔离 smoke test 在两个 offline 环境变量开启时，从 tiny corpus/vectors 构建 candidate/Nomic/Paragram 测试资源，再完成 watermark/detect，整个流程无网络访问。

### 17.6 指标和质量测试

- `test_metrics.py` 使用构造分数验证阈值方向、完整 tie group、有限 sentinel、calibration schema/id 隔离、输入重排稳定、冻结阈值和 CI seed 可复现；修改 held-out 分数不得改变 `tau`，calibration positives 不参与阈值；
- 样本不足时拒绝输出未标注的正式 1% FPR 指标；
- `test_quality_metrics.py` 覆盖空文本、零长度分母、k 边界/fallback、失败/两类截断分类、requested presence/list overlap、语义相似度、task metric delta、NaN/Inf 拒绝、覆盖数守恒、长度统计、聚合分位数和评估器 fingerprint。

## 18. 验收标准

1. 仓库中不存在 `import openai` 或 `import together`。
2. 主流程不读取 `openai_key.txt` 或 `together_key.txt`。
3. `requirements.txt` 不包含 `openai`、`together`。
4. `watermark.py --help` 不存在 GPT、OpenAI、Together 或 paraphraser 参数。
5. 不依赖：
   - `filtered_data_100k_unique_250w_sentbound_openai_embs.pkl`
   - `filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl`
   - `paragram_xxl.pkl`
6. 能生成同时包含 `anchor_embeddings` 和 `candidate_word_embeddings` 的 `postmark_nomic_table.pt` 及 manifest。
7. `compat` selector 对冻结 fixture 完成 anchor top-`3k` 加 candidate-word rerank top-`k`，结果逐词一致；anchor-only 仅作为消融。
8. 兼容词表由仓库现有候选词 pickle 保序转换，并保存 source/words hash；portable 词表有明确标签。
9. 能从用户预置的本地 Paragram 原始向量生成兼容检测资源，不依赖缺失的 `paragram_xxl.pkl`。
10. `paragram_fuzzy` 以 exact token 加 cosine `>= 0.7` 作为 compat 默认；exact-lemma/Nomic fuzzy 输出强制标记 portable。
11. 能对冻结的通用 JSONL 文本嵌入水印，并使用本地 HuggingFace 模型作为 inserter。
12. detector 根据候选文本重算 expected words，不读取保存的 `list1/list2`，但严格校验 keyed table 和 selection contract。
13. selector、detector 或 evaluation config、模型/tokenizer fingerprint、pooling、prefix 或资源 hash 不匹配时默认硬失败；override 输出标记 `config_consistent=false`、`eligible_for_aggregate=false`，不得进入正式汇总。
14. 任意资源篡改均被检测并拒绝。
15. resume 基于 id、输入 hash、配置 hash 和明确终态；只自动修复最后一个截断行，中间坏行硬失败，输入重排或中间插入样本后仍保证每个 id 恰好输出一次。
16. greedy 模式不会对相同 prompt 重复调用；retry 保留最佳候选并记录完整诊断。
17. 网络隔离且 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 时，能从 tiny 本地 fixtures 构建测试资源并完成 watermark/detect/test，且无 DNS/HTTP 尝试。
18. paired 模式同时输出 held-out ROC-AUC、calibration threshold、held-out TPR/FPR、95% CI、样本数、split hash 和 `evaluation_config_sha256`；小样本指标明确标记 diagnostic。
19. 正式报告覆盖全部 eligible id；`k=0`、`k>N` 和生成失败使用预定义 fallback/score，不得静默丢弃，并同时报告失败率、两类截断率、长度变化和语义/任务质量指标。
20. 所有 baseline 使用同一冻结 `text1`；anchor/dev/calibration/test 无数据泄漏。
21. 最小单元测试、compat fixture 和 2-5 条真实样本 smoke test 通过。
22. README 提供离线预部署、compat/portable 完整示例、已知限制和不可比性声明。


## 19. 实现顺序

### Phase 1：清理依赖

- 删除 OpenAI/Together import、类和 CLI；
- 删除 paraphraser；
- 清理 requirements；
- 确保模块可 import。

### Phase 2：资源生成

- compat candidate words 保序转换与 portable 构建；
- Nomic encoder；
- anchor corpus 固定 manifest、分块和稳定采样；
- anchor embeddings 与 candidate word embeddings 同时构建；
- 可选本地 Paragram detector 资源；
- table、模型/tokenizer fingerprint 和配置 manifest 保存；
- roundtrip 测试。

### Phase 3：选词器

- `NomicPostMarkEmbedder`；
- `official_two_stage` 默认选择器；
- `anchor_only`/`direct_word` 消融；
- compat fixture、稳定 tie-break 和可复现性测试。

### Phase 4：本地 inserter

- `LocalHFLLM`；
- iterative insertion；
- 已有文本模式；
- prompt/base LLM 模式和 per-sample seed；
- greedy retry、最佳候选保留和失败诊断；
- 基于 id/hash 的断点续跑。

### Phase 5：detector 与评测

- compat `paragram_fuzzy` keyed blind detector；
- portable exact-lemma 与 Nomic fuzzy；
- selection contract 硬校验；
- calibration/held-out paired metrics、bootstrap CI；
- 失败率、长度和质量指标。

### Phase 6：文档和验收

- 更新 README；
- 示例数据；
- 测试；
- 网络隔离下的 2-5 条真实样本 smoke test；
- compat/portable 不可比性和 clean-condition 声明；
- 全仓库搜索在线 API 和缺失 pickle 引用。

## 20. README 中必须声明的实现差异

英文：

> This repository contains PostMark-Local, a fully local reimplementation of the PostMark pipeline. Its compatibility profile preserves the official two-stage selector: random-anchor top-3k preselection followed by candidate-word embedding reranking to k. Because the original precomputed anchor embeddings are unavailable, this implementation rebuilds a deterministic Nomic table from a versioned local corpus. When locally provisioned Paragram vectors are available, the compatibility detector uses exact-token plus Paragram fuzzy matching at cosine threshold 0.7; exact-lemma and Nomic-fuzzy detectors are portable variants and are reported separately. Local inserters, regenerated anchors, and portable detectors can change both text quality and detection performance, so results must be labeled with the inserter, selector, table, and detector configuration and must not be treated as an exact reproduction of the paper's reported numbers.

中文：

> 本实现命名为 PostMark-Local，是 PostMark 流程的完全本地重实现。compat 模式保留官方二阶段 selector：先按随机 anchor 取 top-3k，再使用候选词自身 embedding 重排到 k。由于官方预计算 anchor embedding 不可用，本实现从带版本和哈希的本地语料重建固定 Nomic table。用户预先提供本地 Paragram 原始向量时，compat detector 使用 exact token 加 cosine threshold 0.7 的 Paragram fuzzy matching；exact-lemma 和 Nomic fuzzy 只属于分开报告的 portable 变体。本地 inserter、重建 anchor 和 portable detector 都可能改变质量与检测性能，因此结果必须标明 inserter、selector、table 和 detector 配置，不能视为论文原始数值的严格复现。

README 还必须准确说明：所谓 blind detection 是“不需要原文和保存的 list，但需要相同 keyed table”；删除 paraphrase attack 后仅评估 clean condition；论文开放设置使用的 inserter 与本地 8B 示例并不相同。

## 21. 额外实现注意事项

1. anchor table 构建一次后固定使用，不能每次随机重建。
2. 不要直接用 dev、calibration 或最终测试集构建 anchor pool。
3. detector 必须根据候选文本自身重新选词。
4. 不要用插入阶段 requested words 计算正式 detection score。
5. 保留 `list1/list2` 分析语义选择稳定性。
6. 构建 table、watermark、detector 三处共享同一 Nomic tokenizer、pooling、normalization、max length 和 task prefix。
7. Python、NumPy、PyTorch 都固定 seed；base LLM 采样另外使用由 sample id 派生的 seed，保证断点续跑一致。
8. 先用 LLM batch size 1 跑通。
9. 默认每组插入 10 个词；8B 模型能力不足时可降到 5。
10. 模型输出为空或失败时必须记录 sample id 和 attempt。
11. 不得在 import 时加载大型模型。
12. baseline 首先保留官方 `prompts/insert.txt`。
13. `list_overlap` 建议使用 Jaccard：
    ```python
    len(set(list1) & set(list2)) / max(1, len(set(list1) | set(list2)))
    ```
14. compat 默认使用 `int(len(text.split()) * ratio)`；portable 的 round/clamp 必须显式配置。
15. greedy decoding 不得对相同 prompt 重复调用；重试必须改变 prompt 或显式改为 sampling。
16. 所有正式结果按 `PostMark-Local-Nomic-<inserter>-<selector>-<detector>` 或等价完整命名记录。
17. table 是 keyed detection 的关键资源。公开 table 便于可复现，但不代表完成了针对密钥泄露或自适应攻击的安全评估。

## 22. Codex 最终交付内容

Codex 完成后应提供：

1. 修改文件列表；
2. 新增文件列表；
3. 删除的在线依赖列表；
4. compat 候选词、Nomic anchor table 和可选 Paragram table 构建命令；
5. watermark 命令；
6. compat detector、portable detector 和 calibration/held-out 评测命令；
7. 测试命令；
8. 网络隔离下的一次最小 smoke-test 输出；
9. 资源 manifest、selection/run config hash 和示例输出；
10. 已知限制；
11. 与官方原实现的差异说明；
12. 公平评测报告，包括失败率、截断率、长度变化、质量指标和置信区间。
