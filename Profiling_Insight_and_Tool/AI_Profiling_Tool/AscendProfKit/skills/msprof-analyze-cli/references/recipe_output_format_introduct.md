# recipe结果和cluster_analysis.db交付件表结构说明

> [!NOTE]
>
> - msprof-analyze配置--mode参数时可分析并输出cluster_analysis.db交付件，本节介绍该交付件的表结构和字段说明。
>
> - 部分分析能力不会生成cluster_analysis.db。

## cluster_step_trace_time.csv

数据解析模式为communication_matrix、communication_time或all时均生成。

A列： Step数，是采集性能数据时设置的，一般来说集群性能数据采集一个step足够，如果采集多个step，需要先筛选一下。

B列： Type，主要分两种，rank和stage，和后面的index强相关，可以理解为一个是单卡rank，一个是rank group（pp 并行的stage），如果type为stage，则后面D-K列信息为rank group下的最大值。

C列：Index，与type相关，表示卡号。

D列：Computing， 此列统计计算时间。

E列：Communication(Not Overlapped)，此列统计未被掩盖的通信耗时。

F列：Overlapped，统计计算与通信重叠的耗时。

G列：Communication，通信时间的全部耗时。

H列：Free，空闲时间，指device侧既不在通信也不在计算的耗时，可能在做sdma拷贝或者空闲等待。

I列：Stage时间，I、J、K列属于pp并行时有效的数值，stage时间代表除receive算子时间外的时间。

J列：Bubble时间，指receive时间的总和。

K列：Communication（Not Overlapped and Exclude Receive）指剔除receive算子外的并且不被掩盖的通信时间。

L列：Preparing，指迭代开始到首个计算或通信算子运行的时间。

M列：DP Index，指集群数据按照并行策略切分后所属DP组的索引， 如果没有采集则不显示。

N列：PP Index，指集群数据按照并行策略切分后所属PP组的索引，如果没有采集则不显示。

O列：TP Index，指集群数据按照并行策略切分后所属TP组的索引，如果没有采集则不显示。

**Tips**：先筛选B列type为stage， 看stage间是否有问题，再筛选B列type为rank，看rank是否有问题，根据以下几点排查。

* 根据Computing的时间差异判断是否有慢卡，或者有负载不均衡的现象。

* 根据Free统计是否有host bound或者分布不均现象。

* 根据Communication（Not Overlapped and Exclude Receive）时间判断是否通信耗时占比过大。

* 根据Bubble时间的占比和理论计算公式判断bubble设置是否合理，stage间是否有不均衡现象。

以上时间理论上都应该保持相对持平，即最大值与最小值的差值不超过5%，否则就可能出现慢卡。

## cluster_communication_matrix.json

数据解析模式为communication_matrix或all时生成。

直接打开json（vscode或json查看器），搜索"Total"，会有多个搜索结果，一般来说链路带宽信息的结构：

```bash
{src_rank}-{dst_rank}: {
    "Transport Type": "LOCAL",
    "Transit Time(ms)": 0.02462,
    "Transit Size(MB)": 16.777216,
    "Bandwidth(GB/s)": 681.4466
}
```

**Tips**：可以根据rank互联的带宽以及链路类型，判断是否有慢链路的问题。

- LOCAL是片内拷贝，速度最高。
- HCCS或PCIE是节点内片间拷贝，速度居中。
- RDMA是节点间拷贝，速度最低。

## cluster_communication.json

数据解析模式为communication_time或all时生成。
主要为通信耗时数据。

## compute_op_sum

设置-m compute_op_sum时，会生成以下表。

### ComputeOpAllRankStats

说明：

基于db格式的集群性能数据，针对全部rank的数据，以OpType和TaskType分组，对计算算子的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| OpType   | TEXT    | 计算算子类型                           |
| TaskType | TEXT    | 算子执行的加速器类型                    |
| Count    | INTEGER | 以OpType和TaskType分组进行统计的算子数量 |
| MeanNs   | REAL    | 耗时的平均值                           |
| StdNs    | REAL    | 耗时的标准差                           |
| MinNs    | REAL    | 耗时的最小值                           |
| Q1Ns     | REAL    | 耗时的25%分位数                        |
| MedianNs | REAL    | 耗时的50%分位数                        |
| Q3Ns     | REAL    | 耗时的75%分位数                        |
| MaxNs    | REAL    | 耗时的最大值                           |
| SumNs    | REAL    | 耗时的总和                             |

### ComputeOpPerRankStatsByOpType

说明：

基于db格式的集群性能数据，针对每个rank的数据，以OpType和TaskType分组，对计算算子的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| OpType   | TEXT    | 计算算子类型                           |
| TaskType | TEXT    | 算子执行的加速器类型                    |
| Count    | INTEGER | 以OpType和TaskType分组进行统计的算子数量 |
| MeanNs   | REAL    | 耗时的平均值                           |
| StdNs    | REAL    | 耗时的标准差                           |
| MinNs    | REAL    | 耗时的最小值                           |
| Q1Ns     | REAL    | 耗时的25%分位数                        |
| MedianNs | REAL    | 耗时的50%分位数                        |
| Q3Ns     | REAL    | 耗时的75%分位数                        |
| MaxNs    | REAL    | 耗时的最大值                           |
| SumNs    | REAL    | 耗时的总和                             |
| Rank     | INTEGER | rank_id                               |

### ComputeOpPerRankStatsByOpName

说明：

配置--exclude_op_name参数时不会生成该表；
基于db格式的集群性能数据，针对每个rank的数据，以OpName、OpType、TaskType和InputShapes分组，对计算算子的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| OpName      | TEXT    | 计算算子名字                           |
| OpType      | TEXT    | 计算算子类型                           |
| TaskType    | TEXT    | 算子执行的加速器类型                    |
| InputShapes | TEXT    | 算子的输入维度                         |
| Count       | INTEGER | 这个分组的算子数量                     |
| MeanNs      | REAL    | 耗时的平均值                           |
| StdNs       | REAL    | 耗时的标准差                           |
| MinNs       | REAL    | 耗时的最小值                           |
| Q1Ns        | REAL    | 耗时的25%分位数                        |
| MedianNs    | REAL    | 耗时的50%分位数                        |
| Q3Ns        | REAL    | 耗时的75%分位数                        |
| MaxNs       | REAL    | 耗时的最大值                           |
| SumNs       | REAL    | 耗时的总和                             |
| Rank        | INTEGER | rank_id                               |

## cann_api_sum

设置-m cann_api_sum时，会生成以下表。

### CannApiSum

说明：

基于db格式的集群性能数据，针对全部rank的数据，对每一种API（名字不同）的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| name           | TEXT    | API名字                           |
| timeRatio      | REAL    | API的耗时占所有API总耗时的百分比 |
| totalTimeNs    | INTEGER | API的总耗时                    |
| totalCount     | INTEGER | API的数量                      |
| averageNs      | REAL    | 耗时的平均值                       |
| Q1Ns           | REAL    | 耗时的25%分位数                    |
| medNs          | REAL    | 耗时的50%分位数                    |
| Q3Ns           | REAL    | 耗时的75%分位数                    |
| minNs          | REAL    | 耗时的最小值                       |
| maxNs          | REAL    | 耗时的最大值                       | 
| stdev          | REAL    | 耗时的标准差                       |
| minRank        | TEXT    | minNs对应的rank的集合              |
| maxRank        | TEXT    | maxNs对应的rank的集合              |

### CannApiSumRank

说明：

基于db格式的集群性能数据，针对每个rank的数据，对每一种API（名字不同）的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| name          | TEXT    | API名字            |
| durationRatio | REAL    | API的耗时占卡内所有API总耗时的百分比 |
| totalTimeNs   | INTEGER | API的总耗时        |
| totalCount    | INTEGER | API的数量          |
| averageNs     | REAL    | 耗时的平均值       |
| minNs         | REAL    | 耗时的最小值       |
| Q1Ns          | REAL    | 耗时的25%分位数    |
| medNs         | REAL    | 耗时的50%分位数    |
| Q3Ns          | REAL    | 耗时的75%分位数    |
| maxNs         | REAL    | 耗时的最大值       |
| stdev         | REAL    | 耗时的标准差       |
| rank          | INTEGER | rank_id           |

## hccl_sum

设置-m hccl_sum时，会生成以下表。

### HcclAllRankStats

说明：

基于db格式的集群性能数据，针对全部rank的数据，对每一种通信算子类型（例如hcom_broadcast_）的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| OpType   | TEXT    | 通信算子类型    |
| Count    | INTEGER | 数量           |
| MeanNs   | REAL    | 耗时的平均值    |
| StdNs    | REAL    | 耗时的标准差    |
| MinNs    | REAL    | 耗时的最小值    |
| Q1Ns     | REAL    | 耗时的25%分位数 |
| MedianNs | REAL    | 耗时的50%分位数 |
| Q3Ns     | REAL    | 耗时的75%分位数 |
| MaxNs    | REAL    | 耗时的最大值    | 
| SumNs    | REAL    | 耗时的总和      |

### HcclPerRankStats

说明：

基于db格式的集群性能数据，针对每个rank的数据，对每一种通信算子类型（例如hcom_broadcast_）的耗时进行统计分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| OpType   | TEXT    | 通信算子类型    |
| Count    | INTEGER | 数量           |
| MeanNs   | REAL    | 耗时的平均值    |
| StdNs    | REAL    | 耗时的标准差    |
| MinNs    | REAL    | 耗时的最小值    |
| Q1Ns     | REAL    | 耗时的25%分位数 |
| MedianNs | REAL    | 耗时的50%分位数 |
| Q3Ns     | REAL    | 耗时的75%分位数 |
| MaxNs    | REAL    | 耗时的最大值    | 
| SumNs    | REAL    | 耗时的总和      |
| Rank     | INTEGER | rank_id        |

### HcclGroupNameMap

说明：

通信域内包含的rank。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| GroupName | TEXT | 通信域，例如：{ip_address}%enp67s0f5_60000_0_1708156014257149 |
| GroupId   | TEXT | 通信域的hash值的后三位 |
| Ranks     | TEXT | 该通信域的所有rank |

### HcclTopOpStats

说明：

基于db格式的集群性能数据，对所有rank的通信算子的耗时进行分析，展示耗时平均值排名TOP N（默认为 15）的通信算子的数据。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| OpName   | TEXT    | 通信算子名，例如hcom_allReduce__606_0_1 |
| Count    | INTEGER | 数量           |
| MeanNs   | REAL    | 耗时的平均值    |
| StdNs    | REAL    | 耗时的标准差    |
| MinNs    | REAL    | 耗时的最小值    |
| Q1Ns     | REAL    | 耗时的25%分位数 |
| MedianNs | REAL    | 耗时的50%分位数 |
| Q3Ns     | REAL    | 耗时的75%分位数 |
| MaxNs    | REAL    | 耗时的最大值    | 
| SumNs    | REAL    | 耗时的总和      |
| MinRank  | INTEGER | 该通信算子耗时最小的rank |
| MaxRank  | INTEGER | 该通信算子耗时最大的rank |

## slow_rank

设置-m slow_rank时，会生成以下表。

### SlowRank

说明：

基于db格式的集群性能数据，进行慢卡分析。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| rankId          | INTEGER | 慢卡    |
| slowAffectCount | INTEGER | 该rank影响了多少次通信 |

### SlowOpStats

说明：

基于db格式的集群性能数据，慢卡瓶颈位置对应的通信算子统计信息。

格式：

| 字段名       | 类型 | 含义             |
|-----------| ---- |----------------|
| SlowRank  | TEXT    | 慢卡rankId       |
| OpName    | TEXT    | 通信算子名          |
| GroupName | TEXT    | 通信域名称          |
| Timestamp | TEXT    | 通信算子时间戳        |
| Count     | INTEGER | 数量             |
| MeanNs    | REAL    | 耗时的平均值         |
| StdNs     | REAL    | 耗时的标准差         |
| MinNs     | REAL    | 耗时的最小值         |
| Q1Ns      | REAL    | 耗时的25%分位数      |
| MedianNs  | REAL    | 耗时的50%分位数      |
| Q3Ns      | REAL    | 耗时的75%分位数      |
| MaxNs     | REAL    | 耗时的最大值         | 
| SumNs     | REAL    | 耗时的总和          |
| MinRank   | INTEGER | 该通信算子耗时最小的rank |
| MaxRank   | INTEGER | 该通信算子耗时最大的rank |

## freq_analysis

说明：

基于db格式的集群性能数据，分析aicore frequency，提供npu降频一键检测能力。频率分为三种情况：

* 正常情况下，应当稳定在1800MHz；
* 当npu空闲时间较长时，设备会自动降频，会掉到800MHz；
* 当npu因为各种原因，出现降频现象时，除了1800MHz，800MHz，还会出现其他异常频率。

设置-m freq_analysis时，如果发生降频，会生成以下表。

### FreeFrequencyRanks

说明：

对应第二种情况：当npu空闲时间较长时，设备会自动降频，会掉到800MHz。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| rankId          | INTEGER | global rank |
| aicoreFrequency | TEXT    | [800, 1800] |

### AbnormalFrequencyRanks

说明：

对应第三种情况：当npu因为各种原因，出现降频现象时，除了1800MHz，800MHz，还会出现其他异常频率。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| rankId          | INTEGER | global rank |
| aicoreFrequency | TEXT    | 异常频率列表；例如：[800, 1150, 1450, 1800] |

## ep_load_balance

说明：

集群训练场景下，MOE负载不均指的是，在分布式环境下，不同的专家模型处理的任务量不均衡，导致某些专家过载（处理过多任务），而其他专家闲置。这种负载不均会降低系统的整体效率，甚至可能导致性能瓶颈。

设置-m ep_load_balance时，会生成以下表。

### EPTokensSummary

说明：

基于db格式的集群性能数据，分析GroupedMatmul算子的shape信息。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| rank               | INTEGER | global rank |
| epRanks            | TEXT    | 同一个ep(Expert Parallelism)的rank集合，例如0,1 |
| inputShapesSummary | INTEGER | 该rank的GroupedMatmul算子的inputshapes的第一个维度的总和 |

### TopEPTokensInfo

说明：

负载不均的ep。

格式：

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| epRanks    | TEXT    | 负载不均的ep(Expert Parallelism)的rank集合，例如0,1 |
| tokensDiff | INTEGER | 同一个ep内最大值与最小值之间的差值 |

## cluster_time_summary

设置-m cluster_time_summary时，会生成以下表。

说明：和cluster_step_trace_time.csv相似，之后考虑替代它。

### ClusterTimeSummary

说明：

基于db格式的集群性能数据，针对全部rank的数据，对集群的一些耗时进行统计分析，可以用来定位性能问题。

格式：
**下表的时间单位都是us**

| 字段名 | 类型 | 含义 |
| ------ | ---- | ---- |
| rank           | INTEGER | global rank |
| step           | INTEGER | 迭代id |
| stepTime       | REAL    | 整个迭代耗时 |
| computation    | REAL    | 计算时间的全部耗时 |
| communicationNotOverlapComputation       | REAL | 未被计算掩盖的通信耗时 |
| communicationOverlapComputation          | REAL | 计算与通信重叠的耗时 |
| communication  | REAL    | 通信时间的全部耗时 |
| free           | REAL    | 空闲时间，指device侧既不在通信也不在计算、并且不包含异步拷贝的总耗时 |
| communicationWaitStageTime               | REAL | 通信等待总耗时 |
| communicationTransmitStageTime           | REAL | 通信传输总耗时 |
| memory         | REAL    | 异步拷贝的总耗时 |
| memoryNotOverlapComputationCommunication | REAL | 不被计算和通信掩盖的异步拷贝的总耗时 | 
| taskLaunchDelayAvgTime                   | REAL | 下发耗时，指所有task从host侧api的开始时间到device侧task的开始时间的平均耗时 |
