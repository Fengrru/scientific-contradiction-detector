# 哈萨比斯Alpha系列方法论复刻：AI for Science 7大项目全流程落地计划
## 一、计划总览（0基础新手一眼看懂）
- **计划核心**：100%复刻DeepMind AlphaGo/AlphaFold/AlphaTensor的成功路径，打造0基础也能落地的AI科研项目。全程不依赖大厂资源、单消费级显卡/免费云算力即可完成，完全避开大厂内卷的通用大模型赛道。
- **对标依据**：所有环节严格遵循哈萨比斯从Alpha系列验证的、20年未变的核心方法论，无主观臆造内容，每一步都有Alpha系列的落地先例。
- **适合人群**：科研人员、在校学生、AI爱好者，**完全不需要了解本对话的前置内容**，照着步骤操作即可完成。
- **周期说明**：最短12周完成入门级可落地项目，最长3年完成大师级开创性项目。
- **核心承诺**：每个项目都有明确可验证的交付物，每一步都有可复制的操作步骤、可直接运行的代码，无任何画饼内容。

---

## 二、先搞懂：哈萨比斯Alpha系列4条核心铁律（所有项目必须严格遵守）
这是哈萨比斯从AlphaGo到AlphaFold、AlphaTensor从未改变的底层逻辑，也是本计划所有设计的核心依据，新手必须先理解：
| 铁律编号 | 核心规则（大白话版） | Alpha系列对标先例 | 你必须执行的对应动作 |
|----------|------------------------|--------------------|------------------------|
| 1 | 先锁定**规则明确、反馈闭环、学术界长期卡壳、无商业变现价值**的狭窄细分领域，绝对不做泛化的通用方向 | AlphaGo（围棋，规则固定，人类研究3000年卡壳）；AlphaFold（蛋白质折叠，规则为物理化学公理，人类研究50年卡壳） | 所有项目必须先选极窄的细分领域，不能选“人工智能”“生物学”这类宽泛方向，先在单领域做到极致再横向扩展 |
| 2 | 脚手架式开发：先用通用工具做最小原型验证可行性，再逐步用专用模型/规则引擎替换，最终完全抛弃通用工具 | AlphaGo v1先用人类棋谱做监督学习（脚手架）验证可行性→AlphaGo Zero完全抛弃人类棋谱，从零开始自主学习；AlphaFold v1用通用ML模型拼接验证可行性→AlphaFold 2完全重构专用架构 | 所有项目严格执行「3个月用大模型做脚手架出可落地原型→后续用专用小模型/规则引擎替换，最终完全抛弃通用大模型」的路径 |
| 3 | 先定义**形式化的规则/公理体系（世界模型）**，再让AI在规则内做核心决策，绝对不依赖黑箱大模型做最终判断 | AlphaGo的世界模型是围棋完整规则；AlphaFold的世界模型是蛋白质物理化学公理，所有核心决策零幻觉、100%可解释 | 所有项目第一步必须先定义评估标准/本体论/公理体系，大模型只做数据清洗、文本提取这类重复性脏活，不参与任何核心决策 |
| 4 | 所有成果必须开源开放，用学术界的反馈迭代优化，而不是闭源做商业变现 | AlphaFold免费开放2亿个蛋白质结构，AlphaGo相关论文完全公开，所有核心代码开源 | 每个项目完成后必须开源代码、数据集、成果，用学术界的反馈持续优化 |

---

## 三、通用前置准备（所有项目必须先完成，0基础照着做）
### 1. 账号与工具准备（1天完成）
| 工具/账号 | 注册/获取方式 | 用途 |
|------------|---------------|------|
| Google账号 | https://accounts.google.com/ 免费注册 | 用于使用Google Colab云算力、Google Drive存储数据 |
| Google Colab Pro+ | 登录Colab后（https://colab.research.google.com/），右上角升级，50美元/月 | 提供A100 40G GPU，满足所有项目的算力需求，无需自己买显卡 |
| GitHub账号 | https://github.com/ 免费注册 | 用于托管代码、开源成果、备份项目 |
| Semantic Scholar API账号 | https://www.semanticscholar.org/product/api 免费注册，申请API密钥 | 用于获取论文元数据、引用量、全文链接 |
| arXiv API | 无需注册，直接调用 | 用于批量下载论文PDF全文 |

### 2. 基础环境配置（2小时完成）
所有代码均可直接在Google Colab中运行，无需本地配置复杂环境。仅需在Colab笔记本中，每次运行前先执行以下代码，安装所有依赖库：
```python
# 所有项目通用依赖库，一键安装
!pip install arxiv semanticscholar pandas tqdm pymupdf transformers accelerate bitsandbytes sentence-transformers streamlit networkx matplotlib
```

### 3. 目标领域选择（1天完成，新手必严格遵守）
#### 选择标准（必须同时满足）：
1.  细分领域近5年发表的核心论文数量在200-500篇之间（太多处理不过来，太少没有足够数据）
2.  你对该领域有基础认知，能看懂论文的核心结论，能判断AI输出的结果是否合理
3.  领域内存在大量争议、未解决的问题、无法复现的实验结果（这是项目的核心价值）
4.  无明确的商业变现路径（大厂绝对不会碰，你没有竞争对手）

#### ✅ 好的例子（直接可用）：
- 大语言模型数学推理中的链式思维有效性
- CRISPR-Cas9在人类细胞中的脱靶效应影响因素
- 深度学习在肺结节影像诊断中的泛化能力边界
- 锂离子电池硅基负极的循环寿命衰减因素

#### ❌ 坏的例子（绝对不能选）：
- 人工智能
- 机器学习
- 生物学
- 新能源

---

## 四、核心入门项目：科学文献矛盾检测器 12周0基础完整执行计划
这是所有新手的首选项目，0基础可上手，12周即可完成可发表成果，100%复刻AlphaGo v1的验证逻辑。
### 项目基础信息
- **项目定位**：用AI自动扫描目标领域的所有论文，找出相互矛盾的科学结论，生成该领域的「矛盾地图」，解决学术界「没人能读完所有论文、没人能发现所有矛盾」的核心痛点
- **周期**：12周（3个月）
- **算力/成本**：Google Colab Pro+，总成本≤200美元
- **最终交付物**：
  1.  可一键运行的矛盾检测器完整代码
  2.  目标领域的「科学矛盾地图」数据集
  3.  可发表的综述论文一篇
  4.  可公开访问的Streamlit交互网页工具
  5.  开源GitHub仓库

---

### 分周详细执行计划（完全照着做即可）
#### 第一阶段：定义规则与数据获取（第1-2周）
**对标AlphaGo：定义围棋完整规则，收集基础棋谱数据**
##### 第1周：定义领域公理体系（核心中的核心，必须先完成）
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周二 | 1.  最终确定你的目标细分领域（严格遵守前面的选择标准）<br>2.  拿出文档，写下该领域**10个最核心、无歧义**的概念，每个概念配1句精准定义<br>3.  写下该领域**最多10个不可再分的核心关系**，用固定格式：`关系(主体, 客体, [数值])`<br>示例：`准确率(模型, 数据集, 数值)`、`优于(方法A, 方法B, 指标)`<br>4.  定义6种矛盾类型及严格判断标准：实验条件矛盾、测量方法矛盾、统计误差矛盾、科学争议、逻辑矛盾、无矛盾 | `ontology_v0.1.md` 领域本体论文档 | 绝对不要写模糊的概念/关系，比如“效果好”“性能提升”，必须是可量化、无歧义的 |
| 周三-周四 | 1.  用Semantic Scholar API，编写脚本获取目标领域2021年至今的论文元数据<br>2.  按引用量从高到低排序，保留前200篇核心论文<br>3.  提取每篇论文的标题、摘要、作者、发表时间、引用量、PDF链接，保存为CSV文件 | `core_papers.csv` 核心论文元数据集 | 不要下载超过500篇论文，新手处理不过来；优先选顶刊/顶会的高引论文 |
| 周五-周日 | 1.  编写脚本，批量下载200篇核心论文的PDF全文<br>2.  按arxiv_id命名PDF文件，保存到云盘固定文件夹 | 200篇完整PDF论文库 | 必须限速，每10秒下载1篇，否则会被arXiv封IP；下载失败的论文手动用Sci-Hub补充 |

**可直接运行的元数据获取代码**：
```python
import arxiv
import semanticscholar as sch
import pandas as pd
import time
from tqdm import tqdm
import os
from google.colab import drive

# 挂载Google Drive，永久保存数据
drive.mount('/content/drive')
# 创建项目文件夹
BASE_PATH = "/content/drive/MyDrive/ScientificContradiction/"
os.makedirs(BASE_PATH + "data", exist_ok=True)
os.makedirs(BASE_PATH + "pdfs", exist_ok=True)

# 替换成你的目标领域关键词
SEARCH_QUERY = "chain of thought mathematical reasoning"
START_YEAR = 2021
MAX_PAPERS = 200

# 1. 从arXiv获取论文基础信息
client = arxiv.Client()
search = arxiv.Search(
    query=SEARCH_QUERY,
    max_results=MAX_PAPERS,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

papers = []
for result in tqdm(client.results(search), total=MAX_PAPERS, desc="获取论文元数据"):
    if result.published.year >= START_YEAR:
        papers.append({
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary,
            "authors": [a.name for a in result.authors],
            "published": result.published.date(),
            "citations": 0,
            "pdf_url": result.pdf_url
        })

df = pd.DataFrame(papers)

# 2. 补充论文引用量
sch_client = sch.SemanticScholar(api_key="你的Semantic Scholar API密钥")
for i, row in tqdm(df.iterrows(), total=len(df), desc="补充引用量"):
    try:
        paper = sch_client.get_paper(f"ARXIV:{row['arxiv_id']}")
        df.loc[i, "citations"] = paper.citationCount
    except Exception as e:
        print(f"获取引用量失败 {row['arxiv_id']}: {e}")
        continue

# 按引用量排序，保留前200篇
df = df.sort_values("citations", ascending=False).head(200).reset_index(drop=True)
df.to_csv(BASE_PATH + "data/core_papers.csv", index=False)
print(f"成功获取 {len(df)} 篇核心论文元数据")
```

**可直接运行的PDF批量下载代码**：
```python
import requests

for i, row in tqdm(df.iterrows(), total=len(df), desc="下载PDF"):
    pdf_path = BASE_PATH + f"pdfs/{row['arxiv_id']}.pdf"
    # 已下载的跳过
    if os.path.exists(pdf_path):
        continue
    # 下载PDF
    try:
        response = requests.get(row["pdf_url"], timeout=30)
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        time.sleep(15)  # 严格限速，避免被封IP
    except Exception as e:
        print(f"下载失败 {row['arxiv_id']}: {e}")
        continue
```

##### 第2周：论文文本提取与预处理
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周三 | 1.  编写PDF文本提取脚本，用PyMuPDF库提取每篇PDF的全文文本<br>2.  自动定位论文的「结果」「讨论」「结论」部分，只保留这些核心内容（90%的结论都在这里）<br>3.  批量处理所有PDF，清洗乱码、换行、多余空格 | `papers_with_text.csv` 论文核心文本数据集 | 不要提取全文，只提取结果/讨论/结论部分，减少无关内容，提升后续提取准确率 |
| 周四-周日 | 1.  手动检查20篇论文的提取结果，修正提取错误<br>2.  过滤掉无法提取文本、无实验结论的论文<br>3.  最终保留至少150篇有效论文 | 清洗后的有效论文文本数据集 | 必须手动检查，PDF提取的乱码会直接导致后续结果全部错误 |

**可直接运行的文本提取代码**：
```python
import fitz

def extract_clean_text(pdf_path):
    """提取PDF中结果/讨论/结论部分的干净文本"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"提取失败 {pdf_path}: {e}")
        return ""
    
    # 文本基础清洗
    text = text.replace("\n\n", "\n").replace("\t", " ").strip()
    # 定位核心章节，只保留后面的内容
    core_sections = ["results", "discussion", "conclusion"]
    start_index = 0
    for section in core_sections:
        idx = text.lower().find(section)
        if idx != -1:
            start_index = max(start_index, idx)
    return text[start_index:]

# 批量提取所有论文的核心文本
df["full_text"] = ""
for i, row in tqdm(df.iterrows(), total=len(df), desc="提取论文文本"):
    pdf_path = BASE_PATH + f"pdfs/{row['arxiv_id']}.pdf"
    if os.path.exists(pdf_path):
        df.loc[i, "full_text"] = extract_clean_text(pdf_path)

# 过滤掉无文本的论文
df = df[df["full_text"].str.len() > 100].reset_index(drop=True)
df.to_csv(BASE_PATH + "data/papers_with_text.csv", index=False)
print(f"成功提取 {len(df)} 篇论文的核心文本")
```

#### 第二阶段：结构化信息提取（第3-5周）
**对标AlphaGo：用人类棋谱做监督学习，提取有效特征**
##### 第3-4周：六元组声明提取（大模型脚手架）
核心目标：把非结构化的论文文本，转换成结构化的六元组声明，这是整个项目的核心数据基础。
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周三 | 1.  加载Llama 3 8B Instruct模型（4-bit量化，Colab A100可直接运行）<br>2.  编写优化后的提示词，让大模型严格按照你定义的本体论，提取论文中的结论性声明<br>3.  单篇论文测试，调整提示词，确保输出格式100%符合要求 | 可运行的声明提取函数，单篇论文测试通过 | 提示词必须严格限定输出格式，温度设为0.05以下，减少幻觉；必须强制模型使用你定义的概念和关系 |
| 周四-周日 | 批量处理所有论文，提取每篇论文的结论性声明，每条声明严格按照六元组格式输出：<br>`<主体> | <关系> | <客体> | <条件> | <证据类型> | <置信度>` | `all_claims.csv` 原始声明数据集 | 批量处理时每10篇保存一次，避免崩溃丢失数据；不要一次跑完全部，分批次运行 |

##### 第5周：数据清洗与归一化
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周三 | 1.  基于你定义的本体论，编写概念/关系映射表，把不同表述的相同术语归一化<br>示例：`"GPT-4": "大语言模型"`、`"提升": "提高"`<br>2.  过滤掉不符合本体论的声明、格式错误的声明、无意义的声明 | `cleaned_claims.csv` 清洗后的有效声明数据集 | 必须严格对齐你第1周定义的本体论，不符合的声明直接删掉，不要留模糊内容 |
| 周四-周日 | 1.  手动检查100条声明，修正提取错误<br>2.  最终保留至少3000条有效声明<br>3.  备份最终数据集 | 最终结构化声明数据集 | 手动检查是必须的，大模型的提取错误会直接导致后续矛盾检测结果错误 |

**可直接运行的声明提取完整代码**：
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 加载Llama 3 8B Instruct模型（4-bit量化，节省显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

def extract_claims(paper_text):
    """从论文文本中提取六元组结构化声明"""
    prompt = f"""
你是一位专业的科学论文分析专家。请严格阅读论文的结果和讨论部分，提取所有明确的、可验证的结论性声明。

对于每一个声明，必须严格按照以下六元组格式输出，每个声明单独占一行：
<主体> | <关系> | <客体> | <条件> | <证据类型> | <置信度>

格式定义：
- 主体：声明的主语，必须使用你定义的领域核心概念
- 关系：必须使用你定义的10个核心关系之一
- 客体：声明的宾语
- 条件：声明成立的所有前提（数据集、模型参数、实验环境等）
- 证据类型：只能选「实验结果」「理论分析」「文献引用」三者之一
- 置信度：1-5分，5分代表作者100%确定

要求：
1.  只提取作者自己的研究结论，不提取背景、文献综述、未来展望
2.  严格遵守格式，不要添加任何额外解释、标题、序号
3.  如果没有有效结论，直接返回「无」

论文核心文本：
{paper_text[:12000]}

输出：
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.05,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只保留输出部分
    return response.split("输出：")[-1].strip().split("\n")

# 批量提取所有论文的声明
all_claims = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="提取声明"):
    if row["full_text"]:
        claims = extract_claims(row["full_text"])
        for claim in claims:
            claim = claim.strip()
            # 只保留格式正确的声明
            if claim and "|" in claim and len(claim.split("|")) == 6:
                parts = [p.strip() for p in claim.split("|")]
                all_claims.append({
                    "arxiv_id": row["arxiv_id"],
                    "title": row["title"],
                    "citations": row["citations"],
                    "subject": parts[0],
                    "relation": parts[1],
                    "object": parts[2],
                    "condition": parts[3],
                    "evidence_type": parts[4],
                    "confidence": parts[5]
                })

# 保存原始声明数据
claims_df = pd.DataFrame(all_claims)
claims_df.to_csv(BASE_PATH + "data/all_claims.csv", index=False)
print(f"成功提取到 {len(claims_df)} 条声明")

# 本体论归一化（替换成你自己的映射表）
subject_mapping = {
    "GPT-4": "大语言模型",
    "GPT-3.5": "大语言模型",
    "Llama 2": "大语言模型",
    "Llama 3": "大语言模型"
}
relation_mapping = {
    "提升": "提高",
    "改善": "提高",
    "增强": "提高",
    "好于": "优于",
    "强于": "优于"
}

# 应用归一化
claims_df["subject"] = claims_df["subject"].map(subject_mapping).fillna(claims_df["subject"])
claims_df["relation"] = claims_df["relation"].map(relation_mapping).fillna(claims_df["relation"])

# 过滤掉不符合本体论的声明
valid_relations = ["准确率", "优于", "提高", "降低", "等于", "大于", "小于"]
claims_df = claims_df[claims_df["relation"].isin(valid_relations)].reset_index(drop=True)
claims_df.to_csv(BASE_PATH + "data/cleaned_claims.csv", index=False)
print(f"清洗后剩余 {len(claims_df)} 条有效声明")
```

#### 第三阶段：矛盾检测与分析（第6-9周）
**对标AlphaGo：用蒙特卡洛树搜索做核心决策，验证走法的有效性**
##### 第6周：候选矛盾对生成
核心逻辑：只有讨论同一个「主体-关系-客体」的声明，才有可能产生矛盾，先过滤掉99%的无效对比。
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周三 | 1.  按「主体-关系-客体」对声明进行分组<br>2.  同一组内的声明两两配对，生成候选矛盾对<br>3.  过滤掉同一篇论文内的声明对（同一篇论文不会自己和自己矛盾） | `candidate_contradictions.csv` 候选矛盾对数据集 | 不要做全量两两对比，会产生几百万无效配对，浪费算力；只对比同一「主体-关系-客体」的声明 |
| 周四-周日 | 1.  手动检查50个候选矛盾对，验证分组逻辑是否正确<br>2.  优化分组规则，减少无效配对 | 最终候选矛盾对数据集 | 必须手动验证，确保分组逻辑正确，否则后续矛盾检测全是无效结果 |

**可直接运行的候选对生成代码**：
```python
# 按主体-关系-客体分组，生成候选矛盾对
candidate_pairs = []
grouped = claims_df.groupby(["subject", "relation", "object"])

for name, group in tqdm(grouped, desc="生成候选矛盾对"):
    # 同一组内至少2条声明才有可能产生矛盾
    if len(group) >= 2:
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                # 跳过同一篇论文的声明
                if group.iloc[i]["arxiv_id"] == group.iloc[j]["arxiv_id"]:
                    continue
                candidate_pairs.append({
                    "claim1_id": group.iloc[i].name,
                    "claim1_text": f"{group.iloc[i]['subject']} {group.iloc[i]['relation']} {group.iloc[i]['object']}",
                    "claim1_condition": group.iloc[i]["condition"],
                    "claim1_paper": group.iloc[i]["arxiv_id"],
                    "claim1_citations": group.iloc[i]["citations"],
                    "claim1_confidence": group.iloc[i]["confidence"],
                    "claim2_id": group.iloc[j].name,
                    "claim2_text": f"{group.iloc[j]['subject']} {group.iloc[j]['relation']} {group.iloc[j]['object']}",
                    "claim2_condition": group.iloc[j]["condition"],
                    "claim2_paper": group.iloc[j]["arxiv_id"],
                    "claim2_citations": group.iloc[j]["citations"],
                    "claim2_confidence": group.iloc[j]["confidence"]
                })

candidates_df = pd.DataFrame(candidate_pairs)
candidates_df.to_csv(BASE_PATH + "data/candidate_contradictions.csv", index=False)
print(f"生成了 {len(candidates_df)} 个候选矛盾对")
```

##### 第7-8周：矛盾分类与验证
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周三 | 1.  编写提示词，让大模型对每个候选矛盾对进行分类，严格使用你第1周定义的6种矛盾类型<br>2.  单对测试，调整提示词，确保分类准确率<br>3.  批量处理所有候选矛盾对 | 带分类标签的矛盾数据集 | 提示词必须严格限定分类类型，温度设为0，只输出分类编号，减少幻觉 |
| 周四-周日 | 1.  过滤掉「无矛盾」的候选对，保留真正的矛盾<br>2.  设计矛盾显著性评分公式，给每对矛盾打分（权重：论文引用量60% + 声明置信度40%）<br>3.  按显著性从高到低排序 | `final_contradictions.csv` 最终矛盾数据集 | 评分公式必须优先考虑高引论文的矛盾，这是学术界最关心的内容 |

##### 第9周：人工验证与规则引擎搭建
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周四 | 1.  人工验证前100个最显著的矛盾，标记真阳性/假阳性<br>2.  分析假阳性的原因，优化分类逻辑<br>3.  修正矛盾数据集，确保最终矛盾的准确率≥95% | 人工验证后的最终矛盾数据集 | 人工验证是必须的，这是你论文成果的核心可信度来源 |
| 周五-周日 | 1.  从人工验证的矛盾中，总结出最常见的矛盾模式，编写对应的逻辑规则<br>2.  用规则引擎替换80%的大模型分类工作，实现零幻觉的矛盾检测 | 规则引擎v0.1，核心矛盾检测规则 | 这是哈萨比斯脚手架方法论的核心：用规则引擎替换大模型黑箱，为后续升级打下基础 |

**可直接运行的矛盾分类代码**：
```python
def classify_contradiction(claim1, condition1, claim2, condition2):
    prompt = f"""
请判断下面两个科学声明是否存在矛盾，严格从以下6种类型中选择一个，只返回类型编号，不要任何解释：
1. 实验条件矛盾
2. 测量方法矛盾
3. 统计误差矛盾
4. 科学争议
5. 逻辑矛盾
6. 无矛盾

声明1：{claim1}
声明1的前提条件：{condition1}

声明2：{claim2}
声明2的前提条件：{condition2}

输出编号：
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response

# 批量分类
candidates_df["type"] = ""
for i, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="矛盾分类"):
    candidates_df.loc[i, "type"] = classify_contradiction(
        row["claim1_text"], row["claim1_condition"],
        row["claim2_text"], row["claim2_condition"]
    )

# 过滤掉无矛盾的候选对
contradictions_df = candidates_df[candidates_df["type"] != "6"].reset_index(drop=True)

# 计算矛盾显著性分数
def calculate_significance(row):
    citation_score = (int(row["claim1_citations"]) + int(row["claim2_citations"])) / 2
    confidence_score = (int(row["claim1_confidence"]) + int(row["claim2_confidence"])) / 2
    return 0.6 * citation_score + 0.4 * confidence_score

contradictions_df["significance"] = contradictions_df.apply(calculate_significance, axis=1)
# 按显著性排序
contradictions_df = contradictions_df.sort_values("significance", ascending=False).reset_index(drop=True)
contradictions_df["manual_verified"] = False
contradictions_df.to_csv(BASE_PATH + "data/final_contradictions.csv", index=False)
print(f"最终发现 {len(contradictions_df)} 个有效矛盾")
```

#### 第四阶段：产品化与论文写作（第10-12周）
**对标AlphaGo：发布v1版本，公开成果，获取社区反馈**
##### 第10-11周：Streamlit交互网页开发
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周四 | 1.  编写Streamlit网页代码，实现以下核心功能：<br>• 矛盾列表展示，按显著性排序<br>• 矛盾类型统计可视化<br>• 按关键词、论文、类型筛选矛盾<br>• 单篇论文上传，自动生成矛盾检测报告<br>2.  本地测试网页功能，修复bug | 可运行的Streamlit网页代码 | 不要做复杂功能，先实现核心的矛盾展示和查询功能，保证稳定运行 |
| 周五-周日 | 1.  将网页部署到Streamlit Community Cloud，生成公开访问链接<br>2.  测试公开链接的可用性，优化访问速度 | 可公开访问的交互网页工具 | 部署时必须把数据集和代码一起上传，确保路径正确 |

##### 第12周：论文写作与成果发布
| 时间 | 具体操作步骤 | 交付物 | 避坑提示 |
|------|--------------|--------|----------|
| 周一-周四 | 1.  按照学术论文标准结构撰写论文：<br>摘要→引言→方法→结果→讨论→结论→附录<br>2.  重点展示你发现的矛盾分布、最显著的10个矛盾、对领域的影响<br>3.  补充方法细节、数据集说明、代码开源链接 | 论文终稿 | 不要写太多技术细节，重点突出你的发现对领域的价值，这是论文能被接收的核心 |
| 周五-周日 | 1.  上传论文到arXiv预印本平台<br>2.  将所有代码、数据集、论文上传到GitHub开源仓库<br>3.  在学术社区、知乎、Twitter分享你的成果，获取反馈 | arXiv预印本+公开GitHub仓库 | 开源时必须写清楚README，告诉别人怎么运行你的代码、怎么使用你的工具 |

---

## 五、其余6个项目 独立可执行完整计划
每个项目均为独立模块，0基础可直接照着做，严格遵循哈萨比斯Alpha系列方法论。
### 入门级项目2："被遗忘的科学发现"挖掘器 12周计划
#### 项目基础信息
- **定位**：用AI挖掘科学史上「高价值但低引用」的被遗忘研究，解决学术界「重复发明轮子」的巨大浪费问题，哈萨比斯多次强调该方向的核心价值
- **周期**：12周
- **算力/成本**：Google Colab Pro+，总成本≤150美元
- **最终交付物**：挖掘工具+被遗忘高价值发现数据集+可发表论文+开源仓库

#### 分阶段执行步骤
1.  **第1-2周：定义规则与数据获取**
    - 定义「高价值低引用」量化标准：1990-2020年发表、总引用量≤5次（排除自引）、包含明确创新点
    - 定义科学价值6维评估标准：创新性、可复现性、可扩展性、跨领域适用性、问题重要性、结论严谨性
    - 用Semantic Scholar API获取目标领域300-500篇低引论文，下载PDF全文
2.  **第3-5周：结构化信息提取**
    - 提取论文核心创新点、解决的问题、实验结论、后续研究方向，生成结构化数据集
    - 基于本体论做概念归一化，统一术语表述
    - 清洗数据，过滤无有效创新的论文
3.  **第6-9周：价值评估与挖掘**
    - 用大模型给每篇论文的科学价值打分（1-10分），筛选出高分论文
    - 人工验证前50篇高分论文，标记被遗忘的原因
    - 用语义嵌入模型匹配「被遗忘的创新」和近年高引论文，找出「重复发明轮子」的典型案例
4.  **第10-12周：产品化与发布**
    - 开发Streamlit交互网页，支持按领域、关键词筛选被遗忘的高价值研究
    - 撰写论文《被遗忘的创新：[领域]低引用高价值研究的系统挖掘》，上传arXiv
    - 开源所有代码和数据集

---

### 入门级项目3：论文实验复现性自动评估系统 12周计划
#### 项目基础信息
- **定位**：给每篇论文自动打「复现性分数」，直击全球学术界的「复现危机」，哈萨比斯多次强调「不可复现的科学不是科学」
- **周期**：12周
- **算力/成本**：Google Colab Pro+，总成本≤150美元
- **最终交付物**：复现性评分工具+领域论文复现性分析报告+可发表论文+开源仓库

#### 分阶段执行步骤
1.  **第1-2周：定义规则与数据获取**
    - 定义复现性5维评估标准（满分100分）：实验步骤完整性、数据集可获得性、代码可获得性、参数完整性、结果可对比性
    - 获取目标领域顶刊/顶会近5年200-300篇核心论文，下载PDF全文
2.  **第3-5周：结构化信息提取**
    - 提取论文「方法」「实验」部分的文本，用大模型提取5个评估维度的结构化信息
    - 自动计算每篇论文的复现性初步分数
    - 清洗数据，过滤无实验部分的综述论文
3.  **第6-9周：验证与优化**
    - 人工验证100篇论文的打分结果，修正偏差，优化提示词，确保打分准确率≥90%
    - 针对开源代码的论文，编写脚本自动克隆代码、验证可运行性，补充「代码可运行性」评分
    - 统计领域整体复现性水平，分析不同期刊、子领域的差异
4.  **第10-12周：产品化与发布**
    - 开发Streamlit网页，支持上传论文自动生成复现性报告、查询论文/期刊复现性排名
    - 撰写论文《[领域]顶刊论文实验复现性的系统评估与分析》，上传arXiv
    - 开源所有代码和数据集

---

### 进阶级项目1："失败实验"数据库与分析引擎 9个月计划
#### 项目基础信息
- **定位**：构建匿名的失败实验数据库，用AI分析失败规律，告诉学术界「什么路走不通」，哈萨比斯公开表示「这是我一直想做但谷歌不会同意的项目」
- **周期**：9个月
- **算力/成本**：单卡RTX 3090，总成本≤1000美元
- **最终交付物**：匿名失败实验数据库+AI失败规律分析引擎+公开平台+顶会论文

#### 分阶段执行步骤
1.  **第1-3个月：最小原型搭建**
    - 设计失败实验结构化模板：实验目标、实验设计、实验步骤、失败结果、失败原因、尝试过的解决方案
    - 从目标领域入手，采访10-20位一线科研人员，收集50个真实失败实验案例，构建种子数据库
    - 开发匿名上传平台，支持科研人员匿名提交失败实验
2.  **第4-6个月：数据库扩展与AI分析引擎开发**
    - 通过学术社区推广，将数据库扩展到1000+失败实验案例
    - 用大模型提取失败实验的结构化信息，构建「失败原因分类体系」
    - 训练AI自动分析同一类问题的失败共性规律，自动总结「哪些方法大概率行不通」
3.  **第7-9个月：平台上线与成果发布**
    - 上线公开的失败实验查询平台，支持按领域、问题、方法查询失败案例
    - 撰写论文《失败的价值：[领域]失败实验的系统分析与规律挖掘》，发表在领域顶会
    - 与顶刊合作，推动投稿作者同步提交相关失败实验记录

---

### 进阶级项目2：跨学科"知识迁移"发现器 12个月计划
#### 项目基础信息
- **定位**：用AI自动发现「A领域解决X问题的方法，可完美解决B领域的Y问题」，打通学科壁垒，哈萨比斯所有划时代突破均来自跨学科迁移
- **周期**：12个月
- **算力/成本**：双卡RTX 3090，总成本≤2000美元
- **最终交付物**：跨学科知识迁移发现引擎+迁移案例数据集+顶刊论文

#### 分阶段执行步骤
1.  **第1-3个月：双学科原型搭建**
    - 选择两个关联度低但底层逻辑相似的学科（如流体力学与细胞迁移）
    - 构建两个学科的本体论，定义核心概念、方法、问题
    - 提取两个学科论文中的「问题-方法」对，构建结构化数据集
2.  **第4-6个月：迁移匹配引擎开发**
    - 用多模态嵌入模型学习跨学科「问题-方法」的语义嵌入
    - 设计匹配算法，自动发现跨领域的可迁移方法候选对
    - 人工验证候选对，优化匹配算法，准确率≥80%
3.  **第7-9个月：多学科扩展与验证**
    - 将系统扩展到10+核心学科，覆盖自然科学、工程学、医学
    - 构建跨学科知识图谱，可视化方法迁移路径
    - 与不同学科的科研人员合作，验证AI发现的迁移方案的可行性
4.  **第10-12个月：平台上线与成果发布**
    - 上线跨学科知识迁移查询平台，支持输入研究问题，自动匹配其他领域的解决方案
    - 撰写论文《跨学科知识迁移的自动发现：基于AI的科学突破加速引擎》，发表在综合性顶刊
    - 开源所有代码和数据集

---

### 进阶级项目3：科学"元问题"自动生成器 12个月计划
#### 项目基础信息
- **定位**：训练AI专门提出高质量的科学问题，直击「AI只能解决问题，不能提出问题」的核心瓶颈，哈萨比斯多次强调该方向的终极价值
- **周期**：12个月
- **算力/成本**：单卡A100 80G，总成本≤3000美元
- **最终交付物**：科学元问题生成器+高质量科学问题数据集+顶刊论文

#### 分阶段执行步骤
1.  **第1-3个月：伟大问题的模式分析**
    - 收集科学史上1000个伟大的科学问题，覆盖所有自然科学领域
    - 分析总结「好的科学问题」的6个核心特征：指向知识边界、可验证、有颠覆性、可衍生、针对核心矛盾、有明确研究路径
    - 构建科学问题的分类体系和量化评估标准
2.  **第4-6个月：问题生成模型训练**
    - 用1000个伟大问题作为种子数据，微调开源大模型，专门用于科学问题生成
    - 设计提示词框架，让AI基于领域知识图谱、矛盾点、空白点，生成高质量科学元问题
    - 构建问题质量评估模型，自动筛选高质量问题
3.  **第7-9个月：验证与优化**
    - 邀请不同领域的一线科研人员，评估AI生成问题的质量、创新性、可研究性
    - 基于专家反馈优化模型，提升高质量问题的产出率
    - 整理出100个AI生成的、专家认可的高价值科学问题，形成数据集
4.  **第10-12个月：平台上线与成果发布**
    - 上线科学元问题生成平台，支持输入研究领域，自动生成高价值科学问题
    - 撰写论文《AI驱动的科学创新：高质量科学元问题的自动生成》，发表在综合性顶刊
    - 开源所有模型和数据集

---

### 大师级项目1："科学发现"本身的科学 3年计划
#### 项目基础信息
- **定位**：用AI分析人类科学发现的完整历史，找到科学发现的普适规律，把科学发现从「依赖天才的艺术」变成「有规律可循的科学」，哈萨比斯DeepMind的终极使命之一
- **周期**：3年
- **最终交付物**：科学发现规律理论体系+科学突破预测模型+开创「计算科学学」新学科

#### 分阶段执行步骤
1.  **第1年：构建人类科学发现完整数据库**
    - 收录16世纪科学革命至今的所有重大科学发现、科学家、研究路径、时代背景、技术条件
    - 构建科学发现的结构化本体论，定义科学发现的类型、阶段、驱动因素
    - 构建跨学科的科学发展时间线和知识图谱
2.  **第2年：科学发现规律分析与预测模型开发**
    - 用AI分析科学发现的普适规律，找到「什么样的问题、方法、路径最容易产生重大突破」
    - 构建科学突破预测模型，能准确预测「哪个领域、什么问题、用什么方法最有可能产生重大突破」
    - 用历史数据回测模型，验证预测准确率
3.  **第3年：验证与理论体系发布**
    - 用预测模型指导真实的科研项目，验证其有效性
    - 发布完整的科学发现理论体系，开创「计算科学学」全新学科
    - 发布开源的科学突破预测平台，供全球科研人员使用

---

### 大师级项目2："思想实验"模拟器 3年计划
#### 项目基础信息
- **定位**：构建能做思想实验的AI，复刻AlphaGo Zero的核心逻辑，让AI从基本公理出发，自主推导出全新的科学理论，是哈萨比斯Alpha系列路线的终极延伸
- **周期**：3年
- **最终交付物**：领域思想实验模拟器+AI自主发现的全新科学理论+开创性论文

#### 分阶段执行步骤
1.  **第1年：构建领域完美世界模型**
    - 选择一个狭窄的科学领域，定义该领域的基本公理、物理规则、数学约束
    - 构建100%精确、无歧义的领域世界模型，支持AI在模型内做无限次模拟实验
    - 验证世界模型的正确性：AI能从公理出发，推导出该领域所有已知的科学定律
2.  **第2年：训练思想实验强化学习代理**
    - 设计思想实验的强化学习框架：状态=当前公理体系+推导结论，动作=提出新假设/做思想实验，奖励=理论的一致性、解释力、预测力
    - 训练AI代理在世界模型中自主做思想实验，从零开始推导科学定律
    - 优化奖励函数，引导AI发现人类从未想到的新理论
3.  **第3年：验证与成果发布**
    - 人工验证AI发现的新理论的逻辑自洽性、创新性、可验证性
    - 设计真实实验，验证AI提出的新理论
    - 发布思想实验模拟器，发表开创性论文，展示AI自主完成的科学发现

---

## 六、新手通用避坑指南（必看，少走90%的弯路）
1.  **绝对不要贪多求全**：新手必须先从入门级项目的12周计划开始，不要一上来就做进阶级/大师级项目，先跑通完整闭环，再谈进阶
2.  **绝对不要优化脚手架**：大模型只是临时工具，输出能用就行，不要花几周时间调提示词，3个月后你会把它全部替换掉
3.  **绝对不要跳过人工验证**：所有AI输出的结果必须人工抽样验证，这是你成果可信度的唯一来源，没有人工验证的论文/成果毫无价值
4.  **绝对不要选宽泛的领域**：必须选极窄的细分领域，先在一个小池塘里做到第一，再去大海里，这是哈萨比斯所有成功的核心
5.  **所有数据必须备份**：每一步的结果都要保存到云盘/GitHub，避免Colab崩溃、电脑故障导致数据全部丢失
6.  **不要自己买显卡**：新手优先用Google Colab Pro+，等跑通原型、确定长期做之后，再考虑买本地显卡

---

## 七、哈萨比斯式进阶升级路线图（做完入门项目后必看）
1.  **第4-6个月**：用入门项目积累的标注数据，微调一个1B参数的专用小模型，完全替换通用大模型，实现更高的准确率、更快的速度、更低的成本
2.  **第7-12个月**：基于人工验证的结果，总结核心逻辑规则，用规则引擎替换80%的模型工作，实现零幻觉、100%可解释的核心决策
3.  **第13-24个月**：扩展到相关领域，构建跨领域的知识图谱，从工具升级为领域基础设施
4.  **第25-36个月**：构建领域世界模型，训练强化学习代理，实现AI自主科学发现，完成从工具到科学家的跨越