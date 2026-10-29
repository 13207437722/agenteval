# AGENTS.md

> 本文件为 Codex 在本仓库开发时的指导规范和长期遵循规则。
> 所有开发活动必须严格按照以下原则执行，不得违背。

## 语言约定

**中文为主，英文为辅**：除了少量特殊的不能替换的或者重要的英文专业词，所有回复都用中文。

## 一、项目概览与定位

### 1.1 项目性质
**AgentEval** 是一个基于 Agent Trace 的自动化评测系统，用于 LLM Agent 质量评估，**这是一个真实的求职项目**。

**核心目标**：
1. 将简历中已写的全部技术功能真正实现并运行验证
2. 掌握每个模块的设计原理、数据流、代码实现、测试和实验过程
3. 能够独立向技术面试官解释每个技术细节

**关键约束**：
- 简历是最终验收范围，不能增减功能
- 淘宝文章是方法参考，不代表拥有其源码
- DeepEval 是成熟开源框架，优先复用
- 所有量化指标必须真实运行验证，不能造假

### 1.2 参考资料职责划分

| 资料 | 职责 | 使用方式 |
|------|------|---------|
| 简历（references/简历.pdf） | **项目需求和验收范围** | 每条主张都要实现并有证据 |
| 淘宝文章（references/agent第二个项目.pdf） | **工程方法与设计参考** | 学习方法论，不复制源码 |
| DeepEval 文档 | **评测框架能力** | 复用成熟能力，聚焦自研 |

### 1.3 系统定位
**AgentEval 不是**：
- ❌ 开发另一个能聊天、购物或执行复杂任务的 Agent
- ❌ 复制淘宝的海量数仓、分布式推理资源池
- ❌ 学术研究或通用 LLM 评测框架

**AgentEval 是**：
- ✅ 读取 Agent 多轮对话及工具执行轨迹
- ✅ 清洗异构、带噪声的日志
- ✅ 使用分层 Rubric 与 LLM-as-a-Judge 评价执行和回复质量
- ✅ 批量生成评测结果、与人工标签对照、归因错误、回归验证
- ✅ 个人工程项目（200 条评测集），不是生产系统

## 二、开发流程与核心原则

### 2.1 最高优先级

**快速完成真实可运行的 AgentEval，让用户掌握完整执行逻辑，最终发布 GitHub。**

### 2.2 核心开发原则

1. **功能不缩水**
   - 简历中的全部技术主张都必须实现
   - 淘宝文章为方法参考，必须真正掌握核心技术点

2. **最小工程结构**
   - 使用职责明确的 Python 模块、DeepEval、Pydantic、pytest、JSONL
   - 不建立不必要的分层架构和过度抽象

3. **文档精简**
   - 只维护 `README.md`、`AGENTS.md`、`LEARN.md`
   - 实验结果保存到 `results/`，无需大量独立规划文档
   - 删除或归档过时的规划文档

4. **直接开发，连续推进**
   - 需求已明确时，不反复规划或等待逐阶段批准
   - 先实现，再运行，再修复
   - 按依赖顺序连续完成：Trace 处理 → Rubric → Judge → 归因 → 回归
   - 完成可独立验证的工作后再汇报

5. **真实运行**
   - 必须实际执行 pytest、CLI 和实验
   - 真实 LLM API 未配置时，完成离线部分并明确列出待验证项
   - **绝不能将 Mock 当成真实 Judge 实验**

6. **指标真实**
   - 200 条 Trace 和五类异常必须真实构建
   - 88% 一致率与 25% 误判率降低仅是待验证的简历主张
   - **禁止硬编码指标、修改标签、用 Mock 冒充真实结果**

7. **快速学习**
   - 在 `LEARN.md` 中用一个具体电商 Agent Case 贯穿讲解
   - 覆盖：Trace 输入 → 清洗 → 证据重建 → Rubric → Judge → 聚合 → 归因 → 回归
   - 每步对应实际函数和数据变化

8. **严格区分来源**
   - 淘宝：方法论参考（六层漏斗、跨轮隔离、幻觉压制）
   - DeepEval：成熟框架复用（GEval、pytest、批量评测）
   - 自研：Trace 预处理、短路逻辑、四层归因、回归测试

### 2.3 Git 提交规范

- ✅ 可以在本地 commit
- ❌ **最终验收前不要执行 git push**（由用户控制）
- ❌ **不提交 references/、.env、密钥或个人资料**

### 2.4 禁止事项

**实验与指标**：
- ❌ 硬编码指标（如 `consistency_rate = 0.88`）
- ❌ 修改 Ground Truth 凑结果
- ❌ 用 Mock 冒充真实模型调用
- ✅ 实际结果与简历数字不同时，如实报告真实值和差异原因

**过度设计**：
- ❌ 前端 Dashboard（除非用户要求）
- ❌ 复杂微服务架构
- ❌ Redis、Kubernetes 等重量级组件
- ✅ 保持简单、聚焦核心功能

### 2.5 完成标准

用户能够：
1. 在新环境安装依赖
2. 运行测试（pytest）
3. 执行批量评测
4. 查看真实结果
5. 通过 `README.md` 与 `LEARN.md` 理解整个系统

## 三、核心技术点与关键亮点

### 3.1 跨轮证据隔离（核心亮点，必须掌握）

**问题**（淘宝文章 3.5 节第 5 点）：
- 最初把所有轮次的 tool_calls 都传给 Judge
- 模型会用前一轮的工具返回来评判当前轮的 Agent 回复
- 导致跨轮信息互相干扰串味，通过率明显偏低

**解决方案**：
- 只传递当前轮的 `tool_calls` 和 `tool_results`
- 历史轮次只保留对话文本（user query + agent reply）
- 不传递前序轮次的工具调用细节

**实现**：`agenteval/preprocessing/isolator.py`

**效果**：通过率显著提升（淘宝文章实测）

**这是简历面试的核心亮点**，必须能清楚解释！

### 3.2 CNT 六层漏斗 Rubric（完全采用淘宝方法论）

```
CTX（上下文门控） → OOSR（越界拒答） → OOC（能力边界诚实）
→ TCC（工具执行链路） → CNT（内容与文本） → RED（安全红线，并联）
```

**短路机制**：
- 串行层：任一层判 0 或 N/A → 后续层输出 N/A
- RED 并联层：始终独立检查，不受短路影响

**核心价值**：
- 证据不足时不强行评测，降低误判
- 分层定位问题（失败在 TCC 层工具调用，还是 CNT 层回复质量）

**实现**：`agenteval/evaluator/orchestrator.py` 的 `CNTEvaluator` 类

### 3.3 幻觉压制技术（Prompt 工程）

**淘宝文章 3.5 节第 2 点，四项技术**：

1. **事实锚定**："仅基于输入信息判断，不引入外部知识或假设"
2. **证据引用**："先摘录相关原文，再做判断"
3. **Few-shot 边界案例**：提供 2-3 个边界 case 的标注示例
4. **禁止推断**："信息不足时输出'剔除'而非猜测性打分"

**实现**：在每层 GEval metric 的 `criteria` 和 `evaluation_steps` 中体现

### 3.4 四层错误归因（简历核心功能）

1. **数据层归因**：输入数据本身有问题（字段缺失、噪声未清洗）
2. **Prompt 层归因**：Judge 对规则理解有偏差（缺少示例、规则模糊）
3. **Rubric 层归因**：评测标准本身与技术预期不一致
4. **Agent 层归因**：排除前三层后，确认为 Agent 真实质量问题

**实现**：`agenteval/attribution/` 四个模块

**价值**：对每个失败 case 精准定位根因，而不是笼统说"评测失败"

### 3.5 四层数据清洗 Pipeline

**淘宝文章 3.1 节，完全采用**：

1. **第一层**：修复数据格式不一致（异构格式适配、字段归一化）
2. **第二层**：语义对齐（时间锚点补充、用户语义映射）
3. **第三层**：屏蔽干扰信号（tool_calls 噪声过滤）
4. **第四层**：过滤规则精化（按轮次隔离）

**实现**：`agenteval/preprocessing/` 四个模块

## 四、DeepEval 复用边界

### 4.1 直接复用（约 40%）
- ✅ GEval (LLM-as-a-Judge)
- ✅ ConversationalTestCase（多轮对话）
- ✅ pytest 集成
- ✅ 批量评测（evals_iterator）
- ✅ Pydantic + JSON Schema
- ✅ 自定义模型（Codex, GPT-4）
- ✅ 结果记录

### 4.2 完全自研（约 60%）
- ⚠️ Trace 预处理（异构格式适配、噪声过滤、跨轮隔离、时间锚点）
- ⚠️ CNT 六层短路逻辑（DeepEval 无层级依赖）
- ⚠️ 四层错误归因
- ⚠️ Bad Case 回放与回归测试
- ⚠️ Judge-Human 一致性检验
- ⚠️ Baseline vs Improved 对比实验

**复用原则**：不重复造轮子，但业务特定功能必须自研

## 五、开发命令与工作流

### 5.1 环境设置

```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装 DeepEval
pip install -U deepeval

# 安装项目依赖
pip install -r requirements.txt
```

### 5.2 运行评测

```bash
# 批量评测
python -m agenteval.cli.evaluate --input data/agent_traces_v1.0.jsonl

# 单条评测
python -m agenteval.cli.evaluate --case-id <id>

# 失败案例回放
python -m agenteval.cli.replay --case-id <id>

# 运行实验对比
python -m agenteval.cli.experiment --baseline v1 --improved v2
```

### 5.3 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_isolator.py

# 运行集成测试
pytest tests/integration/
```

## 六、简历主张与技术证据映射

**简历中的每项主张都必须有对应的技术证据**：

### 6.1 200 条 Agent Trace
- **证据**：`data/agent_traces_v1.0.jsonl` 文件
- **验收**：`wc -l data/agent_traces_v1.0.jsonl` → 200+

### 6.2 5 类典型异常
- **证据**：`docs/anomaly_types.md` 定义文档
- **验收**：每类异常至少 10+ 样本

### 6.3 人工标注一致率 88%
- **证据**：`results/consistency_report.md`
- **计算**：`consistent_count / total_count`
- **验收**：真实运行后如实汇报（可能不到 88%）

### 6.4 误判率相对降低 25%
- **证据**：`docs/experiments.md` 实验报告
- **计算**：`(Baseline误判率 - Improved误判率) / Baseline误判率`
- **验收**：真实运行后如实汇报（可能不到 25%）

## 七、面试追问准备（必须掌握）

### 7.1 淘宝文章与你的项目关系？
- "我参考了淘宝的**方法论**（六层漏斗、跨轮隔离、幻觉压制等）"
- "但没有其源码，也没有复制其分布式架构"
- "我基于 **DeepEval**，自己实现了 Trace 预处理、CNT 六层判定、批量评测流程"
- "规模不同（200 条 vs 日均百万条），但方法论一致"

### 7.2 DeepEval 做了什么，你自己做了什么？
- "**DeepEval 提供**：GEval、pytest 集成、批量评测基础设施（40%）"
- "**我自己实现**：Trace 预处理、跨轮隔离、六层短路、四层归因（60%）"
- "DeepEval 解决通用 LLM 评测，我解决 Agent Trace 评测的业务特定问题"

### 7.3 跨轮证据隔离具体怎么实现的？
- "问题：把所有轮次 tool_calls 都传给 Judge，模型会用前几轮数据评判当前轮"
- "方案：只传当前轮的 tool_calls，历史对话只保留文本"
- "实现：`isolate_turn_evidence()` 函数提取 `turns[turn_index]` 的工具调用"
- "效果：通过率显著提升"

### 7.4 如果实际一致率只有 75%，怎么办？
- "如实报告真实一致率是 75%"
- "分析原因：Rubric 定义不够明确、Prompt 幻觉压制不够、人工标注主观差异"
- "改进方向：增加 Few-shot、细化 Rubric、扩大标注样本量"
- "88% 是淘宝多轮迭代后的收敛水平，我的初版可能还有优化空间"

## 八、文档体系（精简版）

### 8.1 核心文档（必须维护）
- `README.md`：项目说明、快速开始、安装、使用示例
- `LEARN.md`：用一个电商 Agent Case 贯穿讲解整个系统
- `AGENTS.md`：本文件，开发规范和长期遵循规则

### 8.2 实验结果（保存到 results/）
- `results/` 目录：评测结果、一致性报告、实验对比
- 无需创建大量独立规划文档

### 8.3 代码文档
- 每个模块的 docstring 和类型注解
- 关键函数的设计理由和使用示例

### 8.4 历史文档（可归档到 docs/archive/）
- `docs/requirements.md`、`docs/architecture.md` 等前期规划文档
- 保留作为参考，但不作为主要文档维护

## 九、关键文件与目录（精简版）

```
AgentEval/
├── agenteval/              # 主代码目录（职责明确的 Python 模块）
│   ├── preprocessing.py     # Trace 预处理 + 跨轮证据隔离（自研）
│   ├── rubrics.py           # CNT 六层 Rubric 定义（自研）
│   ├── evaluator.py         # 短路逻辑 + DeepEval 集成（自研 + DeepEval）
│   ├── attribution.py       # 四层归因分析（自研）
│   ├── regression.py        # Bad Case 回放与回归（自研）
│   ├── models.py            # Pydantic 数据模型
│   └── cli.py               # 命令行工具
├── tests/                  # pytest 测试
├── data/                   # 数据集（200 条 Trace + 人工标注）
├── results/                # 评测结果、实验报告
├── references/             # 参考资料（不提交）
├── README.md               # 项目说明
├── LEARN.md                # 电商 Agent Case 贯穿讲解
├── AGENTS.md               # 本文件
└── requirements.txt        # 依赖
```

**关键差异**：
- 不创建过度细分的子包（preprocessing/、attribution/ 等）
- 使用单个 Python 文件 + 清晰的类/函数组织
- 保持简单，聚焦核心功能

---

**最后更新时间**：2026-10-24
**版本**：v2.0 - 快速开发版
**状态**：开始实现
