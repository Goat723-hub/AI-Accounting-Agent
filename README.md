# AI Accounting Agent

用 LangGraph 搭的工资表处理 Agent。

它要回答的不是「怎么用 LLM 算工资」，而是一个更具体的问题：

> **当流程里有一步必须靠 LLM 判断、但这一步又不能出错时，怎么把它安全地放进生产流程？**

---

## 核心设计原则

> ### LLM 有提议权，人有批准权，机器只有执行权。

这句话在这个项目里不是口号，是**结构上的事实**，有三处代码支撑：

| 角色 | 谁 | 结构上如何保证 |
|---|---|---|
| **提议** | LLM | 只在一个节点里被调用，出口只有一个「候选列表」 |
| **批准** | 人 | `human_review` 是 LLM 节点的**无条件出边**，图里不存在绕过它的路 |
| **执行** | 纯 Python | `calculate` / `validate` 全程零 LLM 调用 |

设计动机：**工资算错是要赔钱的。** 认列名（「季度激励」算不算奖金）确实需要语义理解，这是 LLM 真正擅长而规则表做不到的；但「认出来之后算不算数」是**责任问题**，不是能力问题 —— 一旦映射定下来，计算和校验必须**同输入同输出**，不能被模型的随机性影响。

---

## 流程

```
                          START
                            │
                            ▼
        parse ──────────────┴── 文件处理员 · 纯 openpyxl，零 LLM
                            │
                            ▼
        extract ────────────┴── 信息提取员 · 列名 → 标准字段（白名单）
                            │
                            ▼
        memory_lookup ──────┴── 规则记忆员 · 只查历史规则，不改任何数据
                            │
                            ▼
        ◆ route_after_memory_lookup ◆  纯函数，只读 State，不做任何副作用
                            │
     ┌────────────────┬─────┴──────────┬────────────────┐
     ▼                ▼                ▼                ▼
 有完全不认识     有 pending/       有 approved       全认得
 的列              rejected 记录     规则可用
     │                │                │                │
     ▼                │                │                │
 llm_rule_analysis     │                │                │
 ★ 全流程唯一调用 LLM   │                │                │
     │                │                │                │
     └────────┬───────┘                │                │
              ▼                        │                │
        human_review ◄─────────────────┘                │
        ★ 人工闸门 · LLM 的提议必过这一关                 │
              │                                          │
       ┌──────┴──────┐                                   │
    approve        reject                                │
       │              │                                  │
       ▼              ▼                                  │
 apply_memory_rules  exception ──→ END                   │
       │                                                 │
       └─────────────────────┬───────────────────────────┘
                             ▼
                        calculate ── 纯 Python，零 LLM
                             │
                             ▼
                        validate ─── 与表内实发对账，定位差额
                             │
                            END
```

**三个关键位置**（读代码时按这个顺序找）：

1. **LLM 在哪调用** —— [llm_node.py](llm_node.py) 的 `analyze_unknown_fields()` 里那一句 `client.invoke(...)`。全项目只有这一处。
2. **结果在哪进 State** —— 同一个文件的 `llm_rule_analysis_node()` 返回值 `{"llm_analysis": ..., "problems": ...}`，由 LangGraph 合并进 State。
3. **谁决定下一步** —— 三个 Router 纯函数：`route_after_parse` / `route_after_memory_lookup` / `route_after_human_review`。**LLM 节点不决定路由**，它到 `human_review` 是一条无条件边。

---

## 快速开始

```bash
pip install -r requirements.txt

cp .env.example .env          # 然后把 DEEPSEEK_API_KEY 换成你自己的
python payroll_agent.py       # 跑 5 个场景的完整演示
```

`.env` 里填 DeepSeek 的 key（[申请地址](https://platform.deepseek.com)）。**不填也能跑** —— 流程不会崩，只是遇到没见过的列名时提不出候选，直接转人工。

---

## 演示的 5 个场景

| 场景 | 输入 | 看点 |
|---|---|---|
| 一 | 正常工资表 | 走完全程到 `validate` |
| 二 | 文件不存在 | 在 `parse` 就拐进 `exception`，带原因 |
| 三 | 脏数据（`¥8,000`、空值、文本） | `problems` 一路带到 State 末尾，不抛异常 |
| 四 | 表里出现没见过的列 | **LLM 提议 → 人工批准 → 规则入库 → 第二次跑 0 次 LLM 调用** |
| 五 | 人工否决 | 直接停下，**不算半张表** |

场景四是整个设计最值得看的一段 —— 它演示了「规则学习闭环」的经济价值：

```
4a. 人工不表态   → 调 1 次 LLM，候选存为 pending，这个人算不出来
4b. 人工批准     → 调 0 次 LLM，规则入库，算出工资
4c. 再跑一遍     → 调 0 次 LLM，直接命中规则
```

**同一个字段被问第二次就是浪费钱。** pending 状态就是为这件事存在的。

---

## 三个设计决策

### 1. 提示词不是安全边界

`llm_node.py` 的 `_parse_candidates()` 是一道**闸门**，逐条丢弃不合格的模型输出：

- 编了一个不存在的标准字段 → 丢
- 提了一个没问它的列 → 丢
- 键名写错 / 缺字段 / `confidence` 不是 0~1 的数（含 `true`，因为 `bool` 是 `int` 子类）→ 丢
- 返回空内容 / 不是合法 JSON / 带 markdown 代码块 → 分别处理

**为什么不能只靠 prompt 里写「必须是这 7 个之一」**：那是一段建议，不是一道约束。模型完全可能返回一个不存在的字段名，而那个值会一路存进 `memory.json`，之后每张表都按它算 —— 而且**没人知道是什么时候错的**。

关键点：**丢弃是可追踪的**。每一条都写进 `problems`，所以「LLM 提了但被挡掉」这件事看得见，不是静默消失。

### 2. 规则库有三个状态，不是一个

`memory.py` 里的规则只有三种状态：

| 状态 | 谁能产生 | 机器能用吗 |
|---|---|---|
| `approved` | **只有人能批准** | ✅ 能 |
| `pending` | LLM 提议后自动记录 | ❌ 不能 |
| `rejected` | 只有人能否决 | ❌ 不能 |

`pending` 是这套设计的关键：它让 LLM 的提议**被记住但不生效**。

- 不记的话，同一个字段每跑一次就烧一次 LLM 的钱
- 记了之后，人工复核时看到的是**同一份候选**，不会因为模型每次输出不同而反复改判断

而 `query_rule()` 默认只返回 `approved` —— 所以哪怕 `memory.json` 里躺着 100 条 pending，计算结果也不会被影响一条。

### 3. 异常不炸图，但也不能被吞掉

三个层次：

- `parser.py` **永远不抛异常**，脏数据记进 `problems`，`ok=False` 走失败出口
- LLM 调用失败**不抛异常**，包成一条 problem 返回。网络抖一下不该让整张工资表算不出来
- 走到 `exception` 节点时**必须有原因** —— `halt_reason` 是强制的不变量。算不出来和算出来但没人成功，在日志里是两种不同的状态

---

## 项目结构

```
AI-Accounting-Agent/
├── payroll_agent.py     Agent 主流程：State / Router / Graph / 日志 / demo
├── llm_node.py          唯一调用 LLM 的节点（提议权）
├── parser.py            文件处理：xlsx → 结构化数据（零 LLM）
├── extraction.py        信息提取：列名 → 标准字段（白名单）
├── calculation.py       计算：社保 / 个税 / 实发（纯 Python）
├── validation.py        校验：算出来的 vs 表里写的
├── memory.py            规则库：approved / pending / rejected 三态
├── requirements.txt
├── .env.example         配置模板（可提交）
├── .gitignore           .env* 被排除在外
└── data/
    └── payroll_demo.xlsx
```

**分层原则**：`parser` / `extraction` / `calculation` / `validation` 是**独立可测的纯逻辑**，互相不依赖，也不依赖 Agent。Agent 只负责「决定下一步让谁干」。

---

## 技术栈

- **LangGraph** —— StateGraph / 节点 / 条件边
- **LangChain OpenAI** —— 用 OpenAI 兼容协议调 DeepSeek
- **openpyxl** —— 读 xlsx
- **python-dotenv** —— 从 `.env` 读 key

---

## 安全

- API Key **从环境变量读，不硬编码**。代码里只有 `os.getenv("DEEPSEEK_API_KEY")`，没有值
- `.gitignore` 排除 `.env*`（含 `.env.local` 等变体），但保留 `.env.example` 模板
- 换 key **不需要改任何代码**，只改 `.env` 一行

---

## 已知限制

诚实说明，这些是**有意没做**的：

- **人不表态的字段不会中断流程**（[payroll_agent.py:1005](payroll_agent.py#L1005)）。目前只有**明确否决**才 halt。生产环境里「不表态」也应该阻止出结果 —— 这是个业务权衡，不是遗漏
- **没接真实财务软件**。凭证生成是 RPA 接口桩
- **只支持 xlsx**。PDF / 扫描件需要另加一层 LLM 提取
- **单表处理**，没有批量 / 并发
