# PostMark 本地开源版改造方案

> 目标：在官方仓库 `lilakk/PostMark` 的基础上，改造成一个完全离线、只依赖本地 HuggingFace 模型的 PostMark baseline。  
> 本方案不追求复现论文表格中的原始数值，但应尽量保留 PostMark 的核心机制：  
> **文本语义编码 → 根据固定词表/语义锚点选出水印词 → 本地 LLM 改写并插入水印词 → 盲检测水印词出现率。**

## 1. 改造目标

必须实现以下要求：

1. 删除所有 OpenAI、Together.ai 和其他在线 API 相关代码、依赖、参数和密钥文件。
2. base LLM 和 inserter 均通过本地 HuggingFace 模型加载，例如本地的 Llama-3.1-8B-Instruct/Chat。
3. embedder 使用本地 `nomic-ai/nomic-embed-text-v1` 或用户指定的本地 Nomic 模型路径。
4. 不使用作者缺失的 `filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl`，而是在本地自行构建等价的 Nomic anchor pool。
5. 删除 paraphraser、`text3`、`list3` 和 paraphrase attack 相关逻辑。
6. 不依赖缺失的 `paragram_xxl.pkl`，自行实现 detector。
7. 支持用户自己的通用 JSONL 输入，而不是写死 OpenGen、LFQA 和 FActScore。
8. 保证 watermark embedding 和 detection 使用同一份固定资源文件，结果可复现。
9. 保留原始 PostMark 的 iterative insertion 主线，默认使用 `v2`：每次插入一小组水印词。
10. 提供最小测试、命令示例和验收标准。

## 2. 非目标

本次改造暂不实现：

- GPT-4、GPT-4o、GPT-3.5-Turbo 或任何在线模型；
- Together.ai；
- paraphrase attack；
- 论文中的人工评估和 GPT judge；
- 对论文所有表格数值的严格复现；
- 多语言统一支持；
- 大规模并行生成；
- 复杂攻击鲁棒性评估。

第一阶段应优先保证：

```text
构建资源 → 嵌入水印 → 盲检测 → 输出分数
```

能够稳定跑通。


## 3. 需要保留的 PostMark 核心机制

原始 PostMark 的关键不是 Nomic 模型本身，而是以下两部分共同工作：

1. **文本编码器**：将待处理文本编码为一个语义向量。
2. **固定的候选水印词—语义锚点映射表**：每个候选水印词随机绑定到一个预先计算好的文本片段 embedding。给定输入文本后，根据输入文本 embedding 与所有 anchor embeddings 的 cosine similarity，选出最相近的若干 anchor，再取出这些 anchor 对应的候选词。

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
    "version": 1,
    "embedder_path": "...",
    "candidate_words": list[str],
    "anchor_embeddings": torch.Tensor,   # [num_words, embedding_dim]
    "seed": 42,
    "normalization": "l2",
    "source": {
        "corpus_path": "...",
        "text_field": "text",
        "chunk_words": 250,
        "num_anchor_chunks": 100000
    }
}
```

`candidate_words[i]` 必须始终对应 `anchor_embeddings[i]`。

## 4. 建议的最终文件结构

```text
PostMark/
├── README.md
├── requirements.txt
├── prompts/
│   └── insert.txt
├── postmark/
│   ├── __init__.py
│   ├── hf_llm.py
│   ├── nomic_embedder.py
│   ├── resources.py
│   ├── watermark.py
│   ├── detect.py
│   ├── build_candidate_words.py
│   ├── build_nomic_anchor_pool.py
│   └── common.py
├── resources/
│   ├── candidate_words.json
│   └── postmark_nomic_table.pt
├── tests/
│   ├── test_resource_roundtrip.py
│   ├── test_word_selection.py
│   ├── test_presence.py
│   └── test_jsonl_pipeline.py
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
scipy
scikit-learn
spacy
nltk
tqdm
tiktoken
sentencepiece
protobuf
```

不要在 requirements 中固定安装 CUDA runtime、cuDNN 等大型 NVIDIA wheel。让 PyTorch/CUDA 由用户环境管理。

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
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        use_chat_template: bool = True,
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

- HuggingFace model ID；
- 本地模型目录；
- `device_map="auto"`；
- `float16`、`bfloat16`、`float32`；
- `trust_remote_code`；
- tokenizer 没有 pad token 时回退到 EOS token。

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

## 8. 构建候选水印词表

新增：

```text
postmark/build_candidate_words.py
```

默认优先使用官方仓库已有的 `wikitext_freq.json` 生成候选词，以尽量接近原实现。

### 8.1 默认过滤规则

- lowercase；
- 只包含英文字母；
- 长度至少 3；
- 频次不低于 `1000`；
- 词性为 noun、verb、adjective 或 adverb；
- 排除 proper noun；
- 去重并稳定排序。

建议使用 spaCy 的 lemma/POS，而不是逐词调用旧版 NLTK tagger。

输出：

```text
resources/candidate_words.json
```

格式：

```json
{
  "version": 1,
  "source": "wikitext_freq.json",
  "min_frequency": 1000,
  "pos": ["NOUN", "VERB", "ADJ", "ADV"],
  "words": ["ability", "able"]
}
```

### 8.2 备用语料模式

如果 `wikitext_freq.json` 不存在，允许从用户 JSONL 语料统计：

```bash
python -m postmark.build_candidate_words   --input_path data/corpus.jsonl   --text_field text   --output_path resources/candidate_words.json   --min_frequency 10   --max_words 20000
```

默认不要直接从最终测试集构建候选词表。优先使用独立语料、训练集或公开语料，避免 evaluation leakage。


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
--embedder_path
--output_path
--chunk_words
--num_anchor_chunks
--batch_size
--seed
--max_length
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

按空格词数切成约 `chunk_words` 长度的片段。尽量在句子边界结束，但第一版允许简单分块。

过滤：

- 太短片段；
- 空文本；
- 重复片段；
- 非英文占比过高的片段。

### 9.3 Nomic 编码

使用用户指定的本地路径：

```python
AutoTokenizer.from_pretrained(embedder_path, trust_remote_code=True)
AutoModel.from_pretrained(embedder_path, trust_remote_code=True)
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
- 资源记录模型路径和维度；
- batch 推理；
- 使用 `torch.inference_mode()`；
- 保存前移到 CPU。

### 9.4 固定词—anchor 映射

使用固定随机种子：

```python
rng = random.Random(seed)
```

步骤：

1. 对 anchor embeddings 索引做稳定随机采样；
2. 取前 `len(candidate_words)` 个；
3. 对选出的 anchor 顺序做固定 shuffle；
4. 与候选词建立一一对应；
5. 保存最终映射。

不要在运行时的 `NomicPostMarkEmbedder.__init__()` 中再次随机采样或 shuffle。

最终资源：

```text
resources/postmark_nomic_table.pt
```

必须供 watermark 和 detector 共同加载。

### 9.5 可选 direct-word 模式

可以额外实现：

```text
--table_mode direct_word
```

直接编码每个候选词。但默认必须使用：

```text
--table_mode random_anchor
```

因为 random anchor 更接近原作者实现：候选词随机绑定到文本片段语义向量，而不是绑定到候选词自身的词义 embedding。


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
        ratio: float = 0.12,
        max_length: int = 512,
        device: str | None = None,
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

默认：

```python
k = max(1, round(num_words_in_text * ratio))
```

增加：

```text
--min_watermark_words
--max_watermark_words
```

建议默认：

```text
min_watermark_words = 1
max_watermark_words = 64
```

### 10.2 选词逻辑

```python
text_embedding = encode_texts([text])           # [1, d]
scores = anchor_embeddings @ text_embedding.T  # 已归一化
indices = torch.topk(scores.squeeze(-1), k).indices
words = [candidate_words[i] for i in indices]
```

不要沿用原代码中疑似无效的二次筛选逻辑。

### 10.3 资源一致性检查

加载资源时验证：

- `len(candidate_words) == anchor_embeddings.shape[0]`；
- embedding dimension 与当前 Nomic 模型一致；
- normalization 为 `l2`；
- 无 NaN/Inf；
- 模型路径不同时给出明确警告；
- 表文件不存在时给出资源构建命令提示。


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
        max_insert_attempts: int = 3,
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
5. 检查该组词的 exact/lemma presence；
6. 若低于阈值，最多重试 `max_insert_attempts`；
7. 下一组基于上一次改写结果继续；
8. 得到 `text2`；
9. 对 `text2` 重新独立计算 `list2`。

输出：

```json
{
  "id": "0",
  "text1": "...",
  "list1": ["..."],
  "text2": "...",
  "list2": ["..."],
  "diagnostics": {
    "list_overlap": 0.73,
    "requested_word_presence": 0.81,
    "num_groups": 3,
    "num_attempts": 4
  }
}
```

`requested_word_presence` 是 `list1` 在 `text2` 中的出现率，只用于调试嵌入阶段。

### 11.3 断点续跑

- 输出已存在时读取已有记录数；
- 跳过已处理样本；
- append 继续；
- 每条写入后 flush；
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
--inserter_path
--embedder_path
--table_path
--prompt_path
--ratio
--min_watermark_words
--max_watermark_words
--iterate
--group_size
--min_group_presence
--max_insert_attempts
--max_new_tokens
--torch_dtype
--device_map
--use_chat_template
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

### 13.1 Presence 模式 A：`exact_lemma`

默认：

1. spaCy tokenize；
2. lowercase；
3. lemmatize；
4. 若期望词本身或 lemma 出现在文本 token/lemma 集合中，则认为 present。

```python
score = present_count / len(expected_words)
```

### 13.2 Presence 模式 B：`nomic_fuzzy`

可选：

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
- 默认仍使用 `exact_lemma`。

### 13.3 通用候选文本输入

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
  "watermark_score": 0.43
}
```

### 13.4 成对评估模式

读取 `text1/text2`，分别重算：

- `score1`：原始文本；
- `score2`：水印文本。

计算：

- ROC-AUC；
- TPR@1% FPR；
- 平均 `score1`；
- 平均 `score2`；
- 平均分数差。

```bash
python -m postmark.detect   --input_path runs/postmark/watermarked.jsonl   --paired   --negative_field text1   --positive_field text2   --output_path runs/postmark/scored.jsonl   ...
```

### 13.5 正式检测不得使用保存的 list

`list1/list2` 只用于调试、分析稳定性和验证重算结果。正式 detector 必须重新调用：

```python
selector.select_words(text)
```


## 14. 公共工具

新增：

```text
postmark/common.py
```

包含：

- `set_global_seed()`；
- JSONL 读取/追加；
- stable word count；
- spaCy token/lemma normalization；
- TPR 辅助函数；
- dataclass 配置；
- 明确异常类型。

不要在模块 import 时：

- 自动下载 NLTK 数据；
- 自动下载 spaCy 模型；
- 加载大型模型；
- 读取不存在的资源。

缺少 spaCy 模型时给出明确安装提示，不要静默联网下载。

## 15. 推荐运行流程

### 15.1 安装

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

模型必须提前下载到本地。

### 15.2 构建候选词表

```bash
python -m postmark.build_candidate_words   --frequency_path wikitext_freq.json   --output_path resources/candidate_words.json   --min_frequency 1000
```

### 15.3 构建 Nomic anchor table

```bash
python -m postmark.build_nomic_anchor_pool   --input_path data/anchor_corpus.jsonl   --text_field text   --candidate_words_path resources/candidate_words.json   --embedder_path /path/to/nomic-embed-text-v1   --output_path resources/postmark_nomic_table.pt   --chunk_words 250   --num_anchor_chunks 100000   --batch_size 32   --seed 42
```

### 15.4 已有文本时嵌入水印

```bash
python -m postmark.watermark   --input_path data/input.jsonl   --output_path runs/postmark/watermarked.jsonl   --text_field text   --inserter_path /path/to/llama-3.1-8b-instruct   --embedder_path /path/to/nomic-embed-text-v1   --table_path resources/postmark_nomic_table.pt   --prompt_path prompts/insert.txt   --ratio 0.12   --iterate v2   --group_size 10   --min_group_presence 0.5   --max_insert_attempts 3   --max_new_tokens 600   --torch_dtype bfloat16
```

### 15.5 只有 prompt 时生成并嵌入

```bash
python -m postmark.watermark   --input_path data/prompts.jsonl   --output_path runs/postmark/watermarked.jsonl   --prompt_field prompt   --base_llm_path /path/to/llama-3.1-8b-instruct   --inserter_path /path/to/llama-3.1-8b-instruct   --embedder_path /path/to/nomic-embed-text-v1   --table_path resources/postmark_nomic_table.pt   --prompt_path prompts/insert.txt   --ratio 0.12   --iterate v2   --max_new_tokens 600   --torch_dtype bfloat16
```

### 15.6 成对检测

```bash
python -m postmark.detect   --input_path runs/postmark/watermarked.jsonl   --output_path runs/postmark/scored.jsonl   --paired   --negative_field text1   --positive_field text2   --embedder_path /path/to/nomic-embed-text-v1   --table_path resources/postmark_nomic_table.pt   --presence_mode exact_lemma
```


## 16. 测试要求

### 16.1 `test_resource_roundtrip.py`

验证：

- table 保存后可重新加载；
- candidate words 数量与 embedding 行数一致；
- 同一 seed 两次构建映射一致；
- 不同 seed 映射不同；
- embedding 全部有限且近似单位范数。

### 16.2 `test_word_selection.py`

- 同一文本多次选词一致；
- `top_k` 数量正确；
- ratio 边界正确；
- 空文本行为明确；
- 不产生重复词。

### 16.3 `test_presence.py`

- exact token；
- 大小写；
- lemma 变化；
- 标点；
- 空 expected list；
- fuzzy 阈值行为。

### 16.4 `test_jsonl_pipeline.py`

使用 mock inserter：

- 输入两条 JSONL；
- 输出两条；
- 字段完整；
- 断点续跑不重复；
- `--overwrite` 正常。

## 17. 验收标准

1. 仓库中不存在 `import openai` 或 `import together`。
2. 主流程不读取 `openai_key.txt` 或 `together_key.txt`。
3. `requirements.txt` 不包含 `openai`、`together`。
4. `watermark.py --help` 不存在 GPT、OpenAI、Together 或 paraphraser 参数。
5. 不依赖：
   - `filtered_data_100k_unique_250w_sentbound_openai_embs.pkl`
   - `filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl`
   - `paragram_xxl.pkl`
6. 能生成 `postmark_nomic_table.pt`。
7. 能对用户 JSONL 已有文本嵌入水印。
8. 能使用本地 HuggingFace Llama-3.1-8B 类模型作为 inserter。
9. 能用相同 Nomic table 对 `text1/text2` 盲检测。
10. detector 不依赖保存的 `list1/list2`。
11. exact-lemma detector 输出 `watermark_score`。
12. paired 模式输出 ROC-AUC 和 TPR@1% FPR。
13. 最小单元测试通过。
14. README 提供完整离线示例。
15. 运行时不自动联网下载。


## 18. 实现顺序

### Phase 1：清理依赖

- 删除 OpenAI/Together import、类和 CLI；
- 删除 paraphraser；
- 清理 requirements；
- 确保模块可 import。

### Phase 2：资源生成

- candidate words；
- Nomic encoder；
- anchor corpus 分块；
- table 保存；
- roundtrip 测试。

### Phase 3：选词器

- `NomicPostMarkEmbedder`；
- 可复现性；
- selection 测试。

### Phase 4：本地 inserter

- `LocalHFLLM`；
- iterative insertion；
- 已有文本模式；
- prompt/base LLM 模式。

### Phase 5：detector

- exact-lemma blind detector；
- paired metrics；
- 可选 Nomic fuzzy。

### Phase 6：文档和验收

- 更新 README；
- 示例数据；
- 测试；
- 2–5 条真实样本 smoke test；
- 全仓库搜索在线 API 和缺失 pickle 引用。

## 19. README 中必须声明的实现差异

英文：

> This repository contains a fully local reimplementation of the PostMark baseline. It preserves the core semantic word-selection and LLM-based insertion pipeline, but does not use the unavailable precomputed embedding files from the original release. Instead, it builds a deterministic Nomic anchor table locally from a user-provided corpus. The detector is also rebuilt locally using exact lemma matching, with optional Nomic-based fuzzy matching. Therefore, this implementation is intended as a reproducible baseline rather than an exact reproduction of the paper's reported numbers.

中文：

> 本实现是 PostMark 的完全本地开源重实现。它保留语义水印词选择和基于 LLM 的插入主流程，但不使用官方仓库中已失效下载链接对应的预计算 embedding 文件，而是从用户提供的独立语料中自行构建固定的 Nomic anchor table。检测器也使用本地 exact-lemma 或可选 Nomic fuzzy matching 重建。因此，该实现用于可复现 baseline，而非严格复现论文原始数值。

## 20. 额外实现注意事项

1. anchor table 构建一次后固定使用，不能每次随机重建。
2. 不要直接用最终测试集构建 anchor pool。
3. detector 必须根据候选文本自身重新选词。
4. 不要用插入阶段 requested words 计算正式 detection score。
5. 保留 `list1/list2` 分析语义选择稳定性。
6. 构建 table、watermark、detector 三处共享同一 Nomic pooling。
7. Python、NumPy、PyTorch 都固定 seed，资源记录 seed。
8. 先用 LLM batch size 1 跑通。
9. 默认每组插入 10 个词；8B 模型能力不足时可降到 5。
10. 模型输出为空或失败时必须记录 sample id 和 attempt。
11. 不得在 import 时加载大型模型。
12. baseline 首先保留官方 `prompts/insert.txt`。
13. `list_overlap` 建议使用 Jaccard：
    ```python
    len(set(list1) & set(list2)) / max(1, len(set(list1) | set(list2)))
    ```

## 21. Codex 最终交付内容

Codex 完成后应提供：

1. 修改文件列表；
2. 新增文件列表；
3. 删除的在线依赖列表；
4. Nomic anchor table 构建命令；
5. watermark 命令；
6. detector 命令；
7. 测试命令；
8. 一次最小 smoke-test 输出；
9. 已知限制；
10. 与官方原实现的差异说明。
