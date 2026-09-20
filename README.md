# 饮用水混配优化 API（Drinking Water Blending API）

基于 **Python + FastAPI + SciPy + SQLite** 的饮用水混配优化服务。用户提交目标体积、
钙 / 镁 / 碱度区间以及 2～6 种水源的浓度、可用量与单位成本，服务求解满足全部区间
与供给上限的**最低成本配比**；每个水源还可设置可选的 **`batch_size`（批次体积，m³）**，
设置后该水源的用量必须是 `batch_size` 的**非负整数倍**且不超过 `max_volume`，未设置时
保持连续变量。每个水源还可设置可选的 **`minimum_run_volume`（投运门槛，m³）**，设置后
该水源的用量**只能为 0 或不低于该门槛**（由二元启用变量 `y_i ∈ {0,1}` 在同一 MIP 中
联立）；请求可设置可选的 **`minimum_active_sources`（最少投运水源数 K）**，要求至少 K 个
**设置了门槛的水源**达到门槛（`Σy_i ≥ K`）。连续、批次与投运约束在**同一个混合整数规划
（MIP）**中求解，不会先求连续解再四舍五入，也**不会先求解再剔除小用量**。成本并列时取
**按水源 ID 排序后用量向量字典序最小**的方案；问题不可行时返回**各指标在同一混合整数
可行集上的可达范围**。支持以**幂等键**保存方案并按 ID 查询。对于已保存且状态为
`optimal` 的方案，还可通过**供给缩减预案接口**在不改动原方案的前提下（完整保留批次与
投运约束），按"先最小化总调整量、再最小化成本、并列取用量向量字典序最小"重新配水。

## 快速开始（单条命令：安装 + 建库 + 启动）

```bash
docker compose up --build
```

该命令完成依赖安装、SQLite 建库（服务启动时自动执行 `CREATE TABLE IF NOT EXISTS`）
并启动服务。

- 服务地址：**http://localhost:8000**
- 交互式文档（Swagger UI）：**http://localhost:8000/docs**
- 健康检查：**http://localhost:8000/health**

所有接口均挂载在 `http://localhost:8000/api/v1/...` 下（容器内监听 `0.0.0.0:8000`，
Compose 将宿主机 **8000** 端口映射到容器 **8000** 端口）。

不使用 Docker 时，本地等价的一条命令（Python ≥ 3.11）：

```bash
pip install -r requirements.txt && python -c "from app.db import init_db; init_db()" && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 配置

均通过环境变量配置，默认值如下：

| 变量      | 默认值              | 说明                                   |
| --------- | ------------------- | -------------------------------------- |
| `DB_PATH` | `./data/blends.db`（容器内 `/data/blends.db`） | SQLite 数据库文件路径 |
| `HOST`    | `0.0.0.0`           | 监听地址（`python -m app.main` 时生效） |
| `PORT`    | `8000`              | 监听端口                               |

Compose 默认将宿主机 `8000` 映射到容器 `8000`，并通过命名卷 `blends-data` 持久化数据库。

## 单位约定

- 体积：m³（`target_volume`、`max_volume`、响应中的 `volume`、可选的 `batch_size`、
  可选的 `minimum_run_volume`）
- 浓度：mg/L（钙、镁；碱度按 mg/L CaCO₃ 计）
- 成本：任意货币单位 / m³

## 投运约束（`minimum_run_volume` 与 `minimum_active_sources`，可选）

每个水源可携带可选字段 `minimum_run_volume`（有限正数，单位 m³）。设置后为该水源引入
二元**启用变量** `y_i ∈ {0,1}`，在**同一个 MIP** 中施加 big-M 联立约束：

```
x_i ≤ U_i · y_i        # 未启用（y_i=0）时只能取 0
x_i ≥ m_i · y_i        # 一旦启用（y_i=1），用量不得低于门槛 m_i
```

其中 `U_i` 是该水源在供给上限（预案时为缩减后的新上限）下、折算到批次格点后的最大可取
体积。因此该水源用量只能是 **0** 或 **≥ `minimum_run_volume`**，介于 `(0, m_i)` 之间的
用量一律不可行——这是 MIP 可行集本身的性质，**不存在"先求解再剔除小用量"的后处理**。
对批次水源，启用后的最低用量按 `⌈m_i / batch_size⌉ · batch_size` 计（门槛向上对齐到批次
格点）；若 `minimum_run_volume > max_volume`，该水源只能停用。

请求级字段 `minimum_active_sources`（非负整数，可省略）给出最少投运水源数 K：

```
Σ_{i ∈ 门槛水源} y_i ≥ K
```

**只有设置了 `minimum_run_volume` 的水源参与计数**；连续 / 仅批次水源即使被大量使用也不
计入。K 大于当前供给上限下能够达到门槛的水源数（如某水源新上限被预案削到门槛以下）时，
问题不可行，并在 `achievable_ranges.active_sources` 中返回 `{requested: K, available: n}`
（缩减上限下可达门槛的水源数 n）。设为 `0` 与省略等价。

- 投运门槛、批次约束、连续用量三者**联合建模、一次求解**：二元变量与整数批次数变量、
  连续体积变量同处一个 MIP（HiGHS），最低成本目标与"成本并列取按 ID 字典序最小"规则
  保持不变（字典序细化只固定用量变量，二元启用变量在每一级 MIP 中自由取值）。
- 不可行响应中的可达范围在**同一约束集**上计算：供给上限、目标体积、批次、投运门槛与
  `Σy_i ≥ K` 全部保留（对每个指标分别求解含全部二元 / 整数变量的 min/max MIP）；门槛
  造成的目标体积空隙同样返回不超过目标体积的最大可达体积。
- 保存请求（`request_json`）、幂等指纹（SHA-256 规范化 JSON）与供给缩减预案都会保留
  `minimum_run_volume` 与 `minimum_active_sources`：改动任一门槛或 K 即构成不同请求，
  重放同一幂等键返回 `409`；预案时门槛从保存的原始请求恢复，缩减后的新上限若低于门槛
  则该水源只能停用（其最低批次按新上限重新折算）。

## 批次约束（`batch_size`，可选）

每个水源可携带可选字段 `batch_size`（有限正数，单位 m³）：

- **设置后**：引入非负整数批次数变量 `z_i ∈ ℤ≥0`，该水源实际用量
  `x_i = batch_size · z_i`，且 `x_i ≤ max_volume`；因此非整数倍的用量（包括
  `max_volume` 本身不是整数倍时超出的零头）都不会被采用，批次数上限按
  `⌊max_volume / batch_size⌋` 计；若 `batch_size > max_volume`，该水源只能取 0。
- **未设置 / 为 `null`**：`z_i` 保持连续，`x_i = z_i`，行为与原 LP 完全一致。
- 连续水源与批次水源在**同一个混合整数模型**中联合求解（SciPy `linprog` 的
  HiGHS 后端，`integrality=` 声明整数变量），**禁止先求连续解再四舍五入**——
  连续最优解取整后可能违反指标区间或目标体积等式。
- 目标体积本身无法由批次拼出时（即使总供给名义上充足），同样返回
  `status="infeasible"`，并在 `achievable_ranges` 中给出不超过目标体积的最大
  可拼出体积及各指标在**同一混合整数可行集**上的可达范围。

## API

### `POST /api/v1/blends/solve` — 只求解，不落库

```bash
curl -X POST http://localhost:8000/api/v1/blends/solve \
  -H 'Content-Type: application/json' \
  -d '{
    "target_volume": 1000,
    "calcium":    {"min": 40, "max": 60},
    "magnesium":  {"min": 5,  "max": 20},
    "alkalinity": {"min": 100, "max": 200},
    "sources": [
      {"id": "W1", "max_volume": 600, "cost": 1.0, "calcium": 100, "magnesium": 30, "alkalinity": 250},
      {"id": "W2", "max_volume": 800, "cost": 2.0, "calcium": 20,  "magnesium": 5,  "alkalinity": 80}
    ]
  }'
```

最优响应（`200`）：

```json
{
  "status": "optimal",
  "plan_id": null,
  "total_volume": 1000.0,
  "total_cost": 1500.0,
  "blend": [
    {"source_id": "W1", "volume": 500.0, "cost": 500.0},
    {"source_id": "W2", "volume": 500.0, "cost": 1000.0}
  ],
  "indicators": {"calcium": 60.0, "magnesium": 17.5, "alkalinity": 165.0}
}
```

需要整批供水时，在对应水源上加 `batch_size`（可与连续水源混用）：

```bash
curl -X POST http://localhost:8000/api/v1/blends/solve \
  -H 'Content-Type: application/json' \
  -d '{
    "target_volume": 1000,
    "calcium": {"min": 40, "max": 80},
    "sources": [
      {"id": "A", "max_volume": 1000, "cost": 2.0, "calcium": 100, "magnesium": 10, "alkalinity": 100},
      {"id": "B", "max_volume": 1000, "cost": 1.0, "calcium": 20,  "magnesium": 10, "alkalinity": 100,
       "batch_size": 300}
    ]
  }'
```

上例的连续松弛最优解是 A=500、B=500（钙 60，成本 1500），但 B=500 不是 300 的
整数倍；MIP 直接在批次数空间求解，最优解为 **A=400、B=600（2 个批次）**，
钙 52，成本 1400：

```json
{
  "status": "optimal",
  "plan_id": null,
  "total_volume": 1000.0,
  "total_cost": 1400.0,
  "blend": [
    {"source_id": "A", "volume": 400.0, "cost": 800.0},
    {"source_id": "B", "volume": 600.0, "cost": 600.0}
  ],
  "indicators": {"calcium": 52.0, "magnesium": 10.0, "alkalinity": 100.0}
}
```

离散约束导致无解时返回 `200`、`"status": "infeasible"`。例如两个水源均按 300 m³
批次供水、各上限 600，目标体积 1000 无法由 300 的整数倍拼出（最大可拼 900）：

```json
{
  "status": "infeasible",
  "message": "the target volume 1000 cannot be assembled under the batch-size constraints; the largest attainable volume not above it is 900",
  "achievable_ranges": {
    "volume":  {"requested": 1000.0, "max": 900.0},
    "calcium": {"min": 50.0, "max": 50.0}
  }
}
```

注意可达范围也是用**同一个混合整数模型**（对每个指标分别做一次 MIP 最小化 / 最大化）
算出的离散投影，而不是连续松弛的范围。

投运门槛与最少投运水源数通过 `minimum_run_volume` 与 `minimum_active_sources` 表达。
下例中 A（钙 100，成本 1）设置 300 m³ 投运门槛，目标 500 m³、钙区间 [15, 67]：连续最优
本应是 A=250、B=250（钙 60，成本 750），但 A=250 落在禁区 `(0, 300)` 内；启用 A 则
A ≥ 300（钙 ≥ 68）又违反区间，MIP 只能让 A 完全停用、由 B 供足 500 m³（钙 20，成本 1000）：

```bash
curl -X POST http://localhost:8000/api/v1/blends/solve \
  -H 'Content-Type: application/json' \
  -d '{
    "target_volume": 500,
    "calcium": {"min": 15, "max": 67},
    "sources": [
      {"id": "A", "max_volume": 500, "cost": 1.0, "calcium": 100, "magnesium": 10,
       "alkalinity": 100, "minimum_run_volume": 300},
      {"id": "B", "max_volume": 500, "cost": 2.0, "calcium": 20, "magnesium": 10,
       "alkalinity": 100}
    ]
  }'
```

```json
{
  "status": "optimal",
  "plan_id": null,
  "total_volume": 500.0,
  "total_cost": 1000.0,
  "blend": [
    {"source_id": "A", "volume": 0.0, "cost": 0.0},
    {"source_id": "B", "volume": 500.0, "cost": 1000.0}
  ],
  "indicators": {"calcium": 20.0, "magnesium": 10.0, "alkalinity": 100.0}
}
```

`minimum_active_sources` 要求至少指定数量的门槛水源投运。下例两个水源门槛均为 300、
目标 700、K=2，单水源方案（A=700，成本 700）被计数约束排除；两个水源必须同时启用，
最优为 A=400、B=300（成本 1000）：

```bash
curl -X POST http://localhost:8000/api/v1/blends/solve \
  -H 'Content-Type: application/json' \
  -d '{
    "target_volume": 700,
    "minimum_active_sources": 2,
    "sources": [
      {"id": "A", "max_volume": 700, "cost": 1.0, "calcium": 50, "magnesium": 10,
       "alkalinity": 100, "minimum_run_volume": 300},
      {"id": "B", "max_volume": 700, "cost": 2.0, "calcium": 50, "magnesium": 10,
       "alkalinity": 100, "minimum_run_volume": 300}
    ]
  }'
```

计数约束无法满足时（如只有 1 个门槛水源却要求 K=2）返回 `"status": "infeasible"`，并给出
`achievable_ranges.active_sources`；目标体积在门槛造成的"用量空隙"中无法拼出（例如各水源
用量只能为 0 或 [300, 400]，目标 500 落在单源 ≤400、双源 ≥600 之间）时，同样在
`achievable_ranges.volume` 给出不超过目标的最大可达体积，各指标可达范围也在保留全部
投运 / 批次约束的同一 MIP 上计算。

### `POST /api/v1/blends` — 求解并以幂等键保存

幂等键可放在请求体 `idempotency_key` 字段或 `Idempotency-Key` 请求头中（二者至少其一，
body 优先）：

```bash
curl -X POST http://localhost:8000/api/v1/blends \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: order-2026-0001' \
  -d '<同上请求体>'
```

- 首次创建返回 `201`，响应体含 `plan_id` 与 `created_at`；
- 以**相同键 + 相同请求**重发：返回 `200` 与已保存的方案，并带响应头
  `X-Idempotent-Replay: true`（不会重复求解或新建记录）；
- 以**相同键 + 不同请求**重发：返回 `409 Conflict`（仅增删或改动
  `batch_size`、`minimum_run_volume`，或改动 `minimum_active_sources`
  也算不同请求——这些设置全部参与指纹）；
- 缺少幂等键：返回 `422`。

### `GET /api/v1/blends/{plan_id}` — 按 ID 查询已保存方案

```bash
curl http://localhost:8000/api/v1/blends/1
```

不存在时返回 `404`。

### `POST /api/v1/blends/{plan_id}/contingency` — 供给缩减预案

针对**已保存且状态为 `optimal`** 的方案，给定部分水源的**新供给上限**，重新求解
一份缩减预案。请求体是"已有水源 ID → 新供给上限"的映射（至少一项，
`0 ≤ 新上限 ≤ 原上限`）；**未列出的水源沿用原上限**。服务读取保存时的原始请求
与配比，保持目标体积、指标区间、水质（各水源浓度）和单位成本不变，按以下优先级
优化：

1. 先最小化各水源用量变化绝对值之和 `Σ|x_i − x_i⁰|`（**总调整量**）；
2. 在总调整量最小的配比中，再最小化**总成本**；
3. 仍并列时，按水源 ID 升序的用量向量取**字典序最小**。

水源的 `batch_size` 从保存的原始请求中恢复并**全程保留**：批次水源的新用量仍须为
批次的非负整数倍；新上限按 `⌊新上限 / batch_size⌋` 折算为批次数上限。新上限为非
整除值时，其零头不可用。同样，各水源的 `minimum_run_volume` 与请求的
`minimum_active_sources` 也从保存请求中恢复并**全程保留**：新上限低于门槛时该水源只能
停用（若同时带批次，最低批次数按 `⌈门槛 / batch_size⌉` 并以新上限折算）；缩减导致可达
门槛的水源数不足 K 时预案不可行，返回体含 `achievable_ranges.active_sources`。

预案**不会被保存，也不会修改原方案**；接口可对同一方案反复调用。

```bash
# 假设方案 1 的配比为 W1=500、W2=500（W1 原上限 600、W2 原上限 800），
# 将 W1 的供给上限缩减到 400：
curl -X POST http://localhost:8000/api/v1/blends/1/contingency \
  -H 'Content-Type: application/json' \
  -d '{"W1": 400}'
```

可行时返回 `200`，含新配比、各水源用量差、总调整量及新旧成本差：

```json
{
  "status": "optimal",
  "plan_id": 1,
  "total_volume": 1000.0,
  "total_cost": 1600.0,
  "cost_delta": 100.0,
  "total_adjustment": 200.0,
  "blend": [
    {"source_id": "W1", "original_volume": 500.0, "new_volume": 400.0,
     "volume_delta": -100.0, "new_cap": 400.0, "cost": 400.0},
    {"source_id": "W2", "original_volume": 500.0, "new_volume": 600.0,
     "volume_delta": 100.0, "new_cap": 800.0, "cost": 1200.0}
  ],
  "indicators": {"calcium": 52.0, "magnesium": 15.0, "alkalinity": 148.0}
}
```

- 新上限不构成约束时（例如等于或高于当前用量），原配比不变，
  `total_adjustment` 与 `cost_delta` 均为 `0.0`；
- 新上限为 `0` 合法，表示完全停用该水源。

新上限下无可行配比时返回 `200` 与不可行原因（不保存任何内容，也不改原方案）：

```json
{
  "status": "infeasible",
  "plan_id": 1,
  "message": "total available volume 900 is below the target 1000; ...",
  "achievable_ranges": {
    "volume": {"requested": 1000.0, "max": 900.0},
    "calcium": {"min": 20.0, "max": 20.0, "...": "..."}
  }
}
```

错误状态码：

| 情况                                       | 状态码 |
| ------------------------------------------ | ------ |
| 方案 ID 不存在                             | `404`  |
| 原方案状态不是 `optimal`（如 `infeasible`）| `422`  |
| 请求体为空映射、含未知水源 ID              | `422`  |
| 新上限为负 / 非有限值，或**超过原上限**    | `422`  |
| 请求体不是 JSON 对象                       | `422`  |
| `batch_size` / `minimum_run_volume` 为 0、负数、NaN/±Infinity、布尔值或非数字 | `422` |
| `minimum_active_sources` 为负数、小数、布尔值或非数字 | `422` |

### 不可行响应

问题不可行时（求解与保存接口均）返回 `200` 及结构化说明，例如钙下限 70 超过可达上限：

```json
{
  "status": "infeasible",
  "message": "calcium: requested minimum 70 exceeds the achievable maximum 68",
  "achievable_ranges": {
    "calcium":    {"min": 36.0, "max": 68.0, "requested_min": 70.0, "requested_max": 90.0},
    "magnesium":  {"min": 10.0, "max": 20.0, "requested_min": 5.0,  "requested_max": 20.0},
    "alkalinity": {"min": 114.0, "max": 182.0, "requested_min": 100.0, "requested_max": 200.0}
  }
}
```

可达范围指：在供给上限、目标体积、**批次约束与投运约束**下（若目标体积无法拼出或
总供给不足，则改用可实现的最大体积），各指标在混配水中可实现的浓度区间（对每个指标
分别在同一混合整数模型上求解 min/max 两个 MIP 得到的投影范围，二元启用变量与计数约束
一并保留）。总可用水量不足目标体积、目标体积不在批次格点上，或门槛 / 计数约束使正体积
配比不存在时，额外返回 `achievable_ranges.volume.max`；`minimum_active_sources` 无法满足
时额外返回 `achievable_ranges.active_sources = {requested, available}`。不可行方案同样会
被保存，可按 ID 查询。

## 规则与语义

- **优化模型**：混合整数线性规划（SciPy `linprog`，HiGHS 求解器，整数变量通过
  `integrality=` 声明）。变量为各水源的取水单位 `z_i ≥ 0` 与门槛水源的二元启用变量
  `y_i ∈ {0,1}`：未设置 `batch_size` 时 `z_i` 连续、用量 `x_i = z_i`；设置批次 `b_i`
  时 `z_i` 为非负整数、用量 `x_i = b_i·z_i`。设置门槛 `m_i` 的水源满足
  `x_i ≤ U_i·y_i`、`x_i ≥ m̂_i·y_i`（批次水源 `m̂_i = ⌈m_i/b_i⌉·b_i`，否则 `m̂_i = m_i`）；
  设置 K=`minimum_active_sources` 时 `Σ_{门槛} y_i ≥ K`。目标 `min Σ cost_i·x_i`；
  约束 `Σx_i = 目标体积`、
  `lo_j·V ≤ Σ conc_ij·x_i ≤ hi_j·V`（钙/镁/碱度）、
  `0 ≤ z_i ≤ ⌊max_volume_i / b_i⌋`（连续水源为 `x_i ≤ max_volume_i`）。连续、批次与
  投运变量联合求解，**不会先求连续解再取整，也不会先求解再剔除小用量**。
- **成本并列**：在所有最优解中，按水源 ID 升序排列的用量向量取**字典序最小**者
  （依次最小化 z₁, z₂, …；通过一系列 MIP 精确实现，缩放系数非负故与用量向量字典序
  一致；二元启用变量不参与字典序目标，只在每一级 MIP 中受同一组结构约束支配）。
  水源 ID 为字符串，按字典序排序。
- **区间缺省**：`calcium` / `magnesium` / `alkalinity` 任一区间可省略，表示该指标不设限。
- **校验**：水源数量 2～6、ID 唯一、`min ≤ max`、数值非负（目标体积为正）；
  `batch_size` 与 `minimum_run_volume` 可省略或为 `null`，否则必须是**有限正数**
  （拒绝 0、负数、NaN、±Infinity 与布尔值；数字字符串照常转换）；
  `minimum_active_sources` 可省略或为 `null`，否则必须是**非负整数**（拒绝负数、小数、
  布尔值与非数字）；不满足返回 `422`。
- **幂等**：以幂等键唯一索引存储；请求体（不含键本身，含各水源的 `batch_size`、
  `minimum_run_volume` 与请求级 `minimum_active_sources`）做 SHA-256 指纹比对，防止键被
  不同请求复用。
- **不可行分析**：仍返回 `status=infeasible`；可达范围用**同一个混合整数模型**计算
  （先判定目标体积等式在批次格点与投运 / 计数约束下是否可行，不可行时改求不超过目标
  体积的最大可达体积，再逐指标求 MIP 投影；K 无法满足时直接给出可达门槛水源数），因此
  离散与投运约束造成的不可行与连续情形同样可解释。
- **供给缩减预案**：对已保存的 `optimal` 方案，用 L1 线性化（在真实体积空间引入偏差
  变量 `d_i ≥ |x_i − x_i⁰| = |b_i·z_i − x_i⁰|`，最小化 `Σd_i`）先求最小总调整量；
  再将其作为等式约束固定，依次求解成本最小化与字典序并列（共三级 MIP，数值不稳时以
  紧不等式回退）。批次约束、整数性、非整除上限折算、投运门槛与 `Σy_i ≥ K` 计数全程
  保留（缩减上限使门槛不可达的水源只能停用，计入 `available`）。预案仅根据保存的原始
  请求与配比即时计算，**不落库、不改写原方案**；新上限下不可行时同样在混合整数可行集
  上返回各指标可达范围与原因。未知方案 `404`，非 `optimal` 原方案或非法上限 `422`。

## 测试

```bash
pip install -r requirements-dev.txt && pytest
```

共 112 个用例，覆盖求解器（连续最优性、字典序并列、混合整数批次、非整除上限、
`batch_size=1` 的整数性、批次格点不可行、离散可达范围、投运门槛的 0 / ≥门槛语义、
门槛与批次格点联立、`minimum_active_sources` 计数、门槛抬高可达区间、门槛造成的体积
空隙、预案的最小调整 / 成本次级目标 / 字典序并列 / 批次保留 / 投运保留 / 不可行）与
API（幂等回放 / 冲突、`batch_size` / `minimum_run_volume` / `minimum_active_sources`
参与指纹、按 ID 查询、预案可行 / 不可行 / 校验错误、有限正数与非负整数校验、参数校验
等）。

## 项目结构

```
app/
  main.py     # FastAPI 路由与生命周期（启动时自动建库）
  solver.py   # SciPy 混合整数规划：最低成本 + 字典序并列 + 批次/投运二元变量 + 供给缩减预案 + 不可行可达范围
  models.py   # Pydantic 请求/响应模型与校验（batch_size/minimum_run_volume 有限正数、minimum_active_sources 非负整数校验）
  db.py       # SQLite 持久化（幂等键唯一约束、原子写入）
  config.py   # 环境变量配置（DB_PATH / HOST / PORT）
tests/        # pytest 单元与端到端测试
Dockerfile    # python:3.12-slim 镜像，非 root 运行，含健康检查
docker-compose.yml
```
