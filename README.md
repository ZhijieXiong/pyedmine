[![](./asset/img/pypi_icon.png)](https://pypi.org/project/edmine/)

[文档] | [相关论文列表] | [数据集信息] | [模型榜单]

[文档]: https://zhijiexiong.github.io/sub-page/pyedmine/document/site/index.html
[数据集信息]: https://zhijiexiong.github.io/sub-page/pyedmine/datasetInfo.html
[相关论文列表]: https://zhijiexiong.github.io/sub-page/pyedmine/paperCollection.html
[模型榜单]: https://zhijiexiong.github.io/sub-page/pyedmine/rankingList.html

PyEdmine是一个面向研究者的，易于开发与复现的**教育数据挖掘**代码库

PyEdmine提出了一个统一的实验流程，用于进行知识追踪、认知诊断、习题推荐和学习路径推荐的实验

PyEdmine约定了一个统一、易用的数据文件格式用于数据集处理，并已支持 14 个 benchmark dataset

PyEdmine设计了一套代码框架用于训练和评估模型，并且基于该代码框架已经实现了26个知识追踪模型、7个认知诊断模型、3个习题推荐模型、4个学习路径推荐模型


<p align="center">
  <img src="asset/img/ExperimentalFlowChart.jpg" alt="PeEdmine 实验流程图" width="600">
  <br>
  <b>图片</b>: PyEdmine 实验流程图
</p>

各任务的具体实验设置请查阅[模型榜单](https://zhijiexiong.github.io/sub-page/pyedmine/rankingList.html)上的说明


- [安装](#安装)
  - [从pip安装](#从pip安装)
  - [从源文件安装（推荐）](#从源文件安装推荐)
  - [主要依赖](#主要依赖)
- [快速开始](#快速开始)
  - [概览](#概览)
  - [目录配置](#目录配置)
  - [数据预处理](#数据预处理)
  - [数据集划分](#数据集划分)
  - [模型训练](#模型训练)
  - [模型评估](#模型评估)
    - [知识追踪](#知识追踪)
    - [认知诊断](#认知诊断)
    - [习题推荐](#习题推荐)
    - [学习路径推荐](#学习路径推荐)
  - [模型自动调参](#模型自动调参)
  - [绘制学生知识状态变化图](#绘制学生知识状态变化图)
- [PyEdmine 重要发布](#pyedmine-重要发布)
- [数据集扩展](#数据集扩展)
- [参考代码库](#参考代码库)
- [贡献](#贡献)
- [免责声明](#免责声明)


## 安装

### 从pip安装

```bash
pip install edmine
```

### 从源文件安装（推荐）
```bash
git clone git@github.com:ZhijieXiong/pyedmine.git && cd pyedmine
pip install -e .
```

### 主要依赖
必须依赖：pandas、numpy、sklearn、torch

非必需依赖：dgl 是部分认知诊断模型所需的；hyperopt 用于自动化参数调优；wandb 用于记录实验数据；tqdm 用于模型评估阶段。

## 快速开始
### 概览
请从 GitHub 下载 PyEdmine 的源代码，然后使用 `examples` 目录中提供的脚本完成数据预处理、数据集划分、模型训练与模型评估。PyEdmine 框架的基本流程如下，请按顺序执行：

1、目录配置：通过 `settings.json` 文件配置数据与模型的存放路径，然后运行`set_up.py`以生成必要的目录；

2、数据预处理：下载原始数据集并放置到指定位置，然后使用 `examples` 中提供的脚本进行数据预处理，以获得统一格式的数据文件。数据集信息可在 [这里](https://zhijiexiong.github.io/sub-page/pyedmine/datasetInfo.html) 查看；

3、数据集划分：对执行了统一格式处理后的数据，基于特定实验设置进行数据集划分。PyEdmine 提供了五种实验设置：两种知识追踪任务的设置（分别借鉴 [PYKT](https://dl.acm.org/doi/abs/10.5555/3600270.3601617) 与 [SFKT](https://dl.acm.org/doi/10.1145/3583780.3614988)）、一种认知诊断任务的设置（借鉴 [NCD](https://ojs.aaai.org/index.php/AAAI/article/view/6080)）、一种离线习题推荐任务和一种离线学习路径推荐任务的设置；

4、模型训练：`examples` 中提供了每个模型的训练启动脚本，更多信息可参考 [这里](https://zhijiexiong.github.io/sub-page/pyedmine/document/site/index.html)；

5、模型评估：`examples` 中也提供了每个模型的评估脚本，并根据不同任务实现了不同维度与粒度的评估指标，包括冷启动评估、无偏评估等；

6、其它特性：（1）PyEdmine 针对部分模型实现了基于贝叶斯优化的自动参数调整方法；（2）PyEdmine 可通过参数设置启用 wandb 功能；（3）绘制学生知识状态变化图。

每一步的具体操作说明，请参阅下文。

### 目录配置
在`examples`目录下创建`settings.json`文件，在该文件中配置数据目录和模型目录，格式如下
```json
{
  "FILE_MANAGER_ROOT": "/path/to/save/data",
  "MODELS_DIR": "/path/to/save/model"
}
```
然后运行脚本
```bash
python examples/set_up.py
```
则会自动生成（内置处理代码的）数据集的原始文件存放目录和经过统一处理的文件的存放目录 ，其中各数据集的原始存放目录（位于`/path/to/save/data/dataset_raw`）如下
```
.
├── SLP
│   ├── family.csv
│   ├── psycho.csv
│   ├── school.csv
│   ├── student.csv
│   ├── term-bio.csv
│   ├── term-chi.csv
│   ├── term-eng.csv
│   ├── term-geo.csv
│   ├── term-his.csv
│   ├── term-mat.csv
│   ├── term-phy.csv
│   ├── unit-bio.csv
│   ├── unit-chi.csv
│   ├── unit-eng.csv
│   ├── unit-geo.csv
│   ├── unit-his.csv
│   ├── unit-mat.csv
│   └── unit-phy.csv
├── assist2009
│   └── skill_builder_data.csv
├── assist2009-full
│   └── assistments_2009_2010.csv
├── assist2012
│   └── 2012-2013-data-with-predictions-4-final.csv
├── assist2015
│   └── 2015_100_skill_builders_main_problems.csv
├── assist2017
│   └── anonymized_full_release_competition_dataset.csv
├── edi2020
│   ├── images
│   ├── metadata
│   │   ├── answer_metadata_task_1_2.csv
│   │   ├── answer_metadata_task_3_4.csv
│   │   ├── question_metadata_task_1_2.csv
│   │   ├── question_metadata_task_3_4.csv
│   │   ├── student_metadata_task_1_2.csv
│   │   ├── student_metadata_task_3_4.csv
│   │   └── subject_metadata.csv
│   ├── test_data
│   │   ├── quality_response_remapped_private.csv
│   │   ├── quality_response_remapped_public.csv
│   │   ├── test_private_answers_task_1.csv
│   │   ├── test_private_answers_task_2.csv
│   │   ├── test_private_task_4.csv
│   │   ├── test_private_task_4_more_splits.csv
│   │   ├── test_public_answers_task_1.csv
│   │   ├── test_public_answers_task_2.csv
│   │   └── test_public_task_4_more_splits.csv
│   └── train_data
│       ├── train_task_1_2.csv
│       └── train_task_3_4.csv
├── junyi2015
│   ├── junyi_Exercise_table.csv
│   ├── junyi_ProblemLog_original.csv
│   ├── relationship_annotation_testing.csv
│   └── relationship_annotation_training.csv
├── moocradar
│   ├── problem.json
│   ├── student-problem-coarse.json
│   ├── student-problem-fine.json
│   └── student-problem-middle.json
├── poj
│   └── poj_log.csv
├── slepemapy-anatomy
│   └── answers.csv
├── statics2011
│   └── AllData_student_step_2011F.csv
└── xes3g5m
    ├── kc_level
    │   ├── test.csv
    │   └── train_valid_sequences.csv
    ├── metadata
    │   ├── kc_routes_map.json
    │   └── questions.json
    └── question_level
        ├── test_quelevel.csv
        └── train_valid_sequences_quelevel.csv
```

### 数据预处理
你可以选择使用我们的数据集预处理脚本
```bash
python data_preprocess/kt_data.py
```
该脚本会生成数据集经过统一格式处理后的文件（位于`/path/to/save/data/dataset/dataset_preprocessed`）

注意：`Ednet-kt1`数据集由于原始数据文件数量太多，需要首先使用脚本`examples/data_preprocess/generate_ednet_raw.py`对用户的数据按照5000为单位进行聚合，并且因为该数据集过于庞大，所以预处理默认是只使用交互序列最长5000名用户的数据

或者你可以直接下载已处理好的[数据集文件](https://drive.google.com/drive/folders/1f5hw6PSKWDanVhVVqU1qS-_RxNYNdl9v?usp=sharing)

### 数据集划分
你可以选择使用我们提供的数据集划分脚本，划分好的数据集文件将存放在`/path/to/save/data/dataset/settings/[setting_name]`下
```bash
python examples/knowledge_tracing/prepare_dataset/pykt_setting.py  # 知识追踪
python examples/cognitive_diagnosis/prepare_dataset/ncd_setting.py  # 认知诊断
python examples/exercise_recommendation/preprare_dataset/offline_setting.py  # 习题推荐

```
你也可以直接下载划分后的数据集文件（[pykt_setting](https://www.alipan.com/s/Lek2EDxPfUJ),[sfkt_setting](https://www.alipan.com/s/NfUiLwfoAsK), [ncd_setting](https://drive.google.com/drive/folders/1TDap7nmwPQ7EP4FUpyv6hvo8UkDBeh0R?usp=sharing), [ER_offline_setting](https://www.alipan.com/s/BJQHQn3waA6), [CD4ER_offline_setting](https://drive.google.com/drive/folders/13HHuyOQq31hCP9V8rNUF70ppWvlivxHS?usp=sharing)），然后将其存放在`/path/to/save/data/dataset/settings`目录下

或者你也可以参照我们提供的数据集划分脚本来设计自己的实验处理流程

### 模型训练
对于无需生成包含额外信息的模型，直接运行训练代码即可，如
```bash
python examples/knowledge_tracing/train/dkt.py  # 使用默认参数训练DKT模型
python examples/cognitive_diagnosis/train/ncd.py  # 使用默认参数训练NCD模型
```
对于需要预先生成额外信息的模型，例如DIMKT需要预先计算难度信息、HyperCD需要预先构造知识点超图信息，则需要先运行模型对应的额外信息生成脚本，如
```bash
python examples/knowledge_tracing/dimkt/get_difficulty.py  # 生成DIMKT需要的难度信息
python examples/cognitive_diagnosis/hyper_cd/construct_hyper_graph.py  # 生成HyperCD需要的图信息
```

学习路径推荐任务需要知识追踪模型作为环境模拟器，因此需要先训练好一个知识追踪模型，PyEdmine目前实现了基于qDKT和LPKT4LPR的环境模拟器

基于Epoch的训练器，训练时会得到类似如下的输出
```bash
2025-06-19 10:59:21 start loading and processing dataset
2025-06-19 10:59:38 start training
2025-06-19 10:59:44 epoch 1   , valid performances are main metric: 0.76521  , AUC: 0.76521  , ACC: 0.84833  , MAE: 0.23686  , RMSE: 0.34025  , train loss is predict loss: 0.406844    , current best epoch is 1
2025-06-19 11:00:11 epoch 2   , valid performances are main metric: 0.77796  , AUC: 0.77796  , ACC: 0.85032  , MAE: 0.23244  , RMSE: 0.33654  , train loss is predict loss: 0.376817    , current best epoch is 2
2025-06-19 11:00:40 epoch 3   , valid performances are main metric: 0.78149  , AUC: 0.78149  , ACC: 0.85163  , MAE: 0.22629  , RMSE: 0.33514  , train loss is predict loss: 0.371912    , current best epoch is 3
2025-06-19 11:01:08 epoch 4   , valid performances are main metric: 0.78366  , AUC: 0.78366  , ACC: 0.85256  , MAE: 0.22437  , RMSE: 0.33424  , train loss is predict loss: 0.369758    , current best epoch is 4
2025-06-19 11:01:37 epoch 5   , valid performances are main metric: 0.78437  , AUC: 0.78437  , ACC: 0.85268  , MAE: 0.21839  , RMSE: 0.33416  , train loss is predict loss: 0.368626    , current best epoch is 4

...

2025-06-19 11:06:12 epoch 37  , valid performances are main metric: 0.78987  , AUC: 0.78987  , ACC: 0.85457  , MAE: 0.2147   , RMSE: 0.33187  , train loss is predict loss: 0.362751    , current best epoch is 21
2025-06-19 11:06:17 epoch 38  , valid performances are main metric: 0.7907   , AUC: 0.7907   , ACC: 0.85463  , MAE: 0.21792  , RMSE: 0.3316   , train loss is predict loss: 0.362828    , current best epoch is 21
2025-06-19 11:06:23 epoch 39  , valid performances are main metric: 0.78943  , AUC: 0.78943  , ACC: 0.85388  , MAE: 0.22209  , RMSE: 0.33233  , train loss is predict loss: 0.362957    , current best epoch is 21
2025-06-19 11:06:29 epoch 40  , valid performances are main metric: 0.79026  , AUC: 0.79026  , ACC: 0.85434  , MAE: 0.21326  , RMSE: 0.33218  , train loss is predict loss: 0.362876    , current best epoch is 21
2025-06-19 11:06:35 epoch 41  , valid performances are main metric: 0.79023  , AUC: 0.79023  , ACC: 0.8546   , MAE: 0.22441  , RMSE: 0.33173  , train loss is predict loss: 0.362758    , current best epoch is 21
best valid epoch: 21  , train performances in best epoch by valid are main metric: 0.79207  , AUC: 0.79207  , ACC: 0.85297  , MAE: 0.22056  , RMSE: 0.33278  , main_metric: 0.79207  , 
valid performances in best epoch by valid are main metric: 0.7898   , AUC: 0.7898   , ACC: 0.85434  , MAE: 0.21901  , RMSE: 0.33197  , main_metric: 0.7898   , 
```
基于Step的训练器，训练时会得到类似如下的输出
```bash
2025-08-01 19:16:44 start loading and processing dataset
2025-08-01 19:17:08 start training
2025-08-01 19:17:28 step 100      : train loss is concept state loss: 0.867828    , concept action loss: -1.51221    , question state loss: 0.338242    , question action loss: -1.80328    , 
2025-08-01 19:17:50 step 200      : train loss is concept state loss: 0.818596    , concept action loss: -1.46902    , question state loss: 0.308322    , question action loss: -1.78957    , 
2025-08-01 19:18:14 step 300      : train loss is concept state loss: 0.823793    , concept action loss: -1.48536    , question state loss: 0.309225    , question action loss: -2.42813    , 
2025-08-01 19:18:35 step 400      : train loss is concept state loss: 0.701109    , concept action loss: -1.35137    , question state loss: 0.235002    , question action loss: -3.50641    , 
2025-08-01 19:18:58 step 500      : train loss is concept state loss: 0.738613    , concept action loss: -1.4045     , question state loss: 0.258047    , question action loss: -3.9537     , 
2025-08-01 19:32:58 step 500      , valid performance are
main metric: -0.008321982840624414
step5, AP: -0.033304, APR: -0.0063331, RP: -0.033304, RPR: -0.0063331, NRP: -0.061017, NRPR: -0.011474, 
step10, AP: -0.046469, APR: -0.0041848, RP: -0.046469, RPR: -0.0041848, NRP: -0.08442 , NRPR: -0.0075014, 
step20, AP: -0.067674, APR: -0.0033046, RP: -0.067674, RPR: -0.0033046, NRP: -0.12283 , NRPR: -0.0059905, 

...

2025-08-01 21:53:24 step 5100     : train loss is concept state loss: 0.212986    , concept action loss: -0.765349   , question state loss: 0.0368868   , question action loss: -1.31573    , 
2025-08-01 21:53:46 step 5200     : train loss is concept state loss: 0.199054    , concept action loss: -0.732374   , question state loss: 0.0336832   , question action loss: -1.18531    , 
2025-08-01 21:54:08 step 5300     : train loss is concept state loss: 0.208855    , concept action loss: -0.761285   , question state loss: 0.0397747   , question action loss: -1.28841    , 
2025-08-01 21:54:28 step 5400     : train loss is concept state loss: 0.178077    , concept action loss: -0.706407   , question state loss: 0.0257976   , question action loss: -1.04622    , 
2025-08-01 21:54:49 step 5500     : train loss is concept state loss: 0.191855    , concept action loss: -0.728117   , question state loss: 0.0346215   , question action loss: -1.20783    , 
2025-08-01 22:08:29 step 5500     , valid performance are
main metric: -0.012395949990968961
step5, AP: -0.05172 , APR: -0.0097354, RP: -0.05172 , RPR: -0.0097354, NRP: -0.097195, NRPR: -0.018089, 
step10, AP: -0.06721 , APR: -0.0062151, RP: -0.06721 , RPR: -0.0062151, NRP: -0.12677 , NRPR: -0.011617, 
step20, AP: -0.083493, APR: -0.0039493, RP: -0.083493, RPR: -0.0039493, NRP: -0.15846 , NRPR: -0.0074827, 

best valid step: 500      
valid performance by best valid epoch is {"5": {"AP": -0.03330377663327104, "APR": -0.00633306784213987, "RP": -0.03330377663327104, "RPR": -0.00633306784213987, "NRP": -0.061017051242375234, "NRPR": -0.011473998633936285}, "10": {"AP": -0.04646934891696771, "APR": -0.004184800090990535, "RP": -0.04646934891696771, "RPR": -0.004184800090990535, "NRP": -0.08442007529036973, "NRPR": -0.007501426604322975}, "20": {"AP": -0.06767395292607752, "APR": -0.0033046125319343496, "RP": -0.06767395292607752, "RPR": -0.0033046125319343496, "NRP": -0.12283069568440122, "NRPR": -0.005990523283613981}}
```
如果训练模型时*use_wandb*参数为True，则可以在[wandb](https://wandb.ai/)上查看模型的损失变化和指标变化

### 模型评估
如果训练模型时*save_model*参数为True，则会将模型参数文件保存至`/path/to/save/model`目录下，那么可以使用测试集对模型进行评估，如
```bash
python examples/knowledge_tracing/evaluate/sequential_dlkt.py --model_dir_name [model_dir_name] --dataset_name [dataset_name] --test_file_name [test_file_name]
```
其中知识追踪和认知诊断模型除了常规的指标评估外，还可以进行一些细粒度的指标评估，例如冷启动评估，知识追踪的多步预测等，这些评估都可以通过设置对应的参数开启。

以下是不同指标的含义，

#### 知识追踪
- overall 从序列的第2个交互开始预测
- core 论文[Do We Fully Understand Students’ Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing](https://arxiv.org/abs/2308.07779)提出的指标
- user warm start, seqStart25 从序列的第25个交互开始预测
- double warm start, seqStart5QueNum5 从序列的第5个交互开始预测，并且只预测训练中出现次数大于等于5的习题
- user cold start, seqEnd5 只预测序列的前5个交互
- question cold start, queNum5 只预测训练集中出现次数小于等于5的习题
- multi step 论文[pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models](https://dl.acm.org/doi/abs/10.5555/3600270.3601617)中提到的两种多步预测
- first trans 只预测每个学生交互序列中第一次接触到的知识点
#### 认知诊断
- overall 预测全部测试集
- user cold start, userNum5 只预测训练集中出现次数小于等于5的学生
- question cold start, questionNum5 只预测训练集中出现次数小于等于5的习题
#### 习题推荐
- KG4EX_ACC 论文[KG4Ex: An Explainable Knowledge Graph-Based Approach for Exercise Recommendation](https://dl.acm.org/doi/10.1145/3583780.3614943)中提出的指标，本榜单公布的结果基于DKT计算
- KG4EX_NOV 同KG4EX_ACC
- OFFLINE_ACC 将学生未来练习的习题作为标签，计算准确率
- OFFLINE_NDCG 将学生未来练习的习题作为标签，计算NDCG
- PERSONALIZATION_INDEX 计算给不同学生推荐习题的差异度，作为个性化的指标
#### 学习路径推荐
$m_{start}$和$m_{end}$分别是目标知识点的初始分数和最终分数，$m_{full}$是知识点的满分，$l$是路径长度
- AP = $m_{end} - m_{start}$
- APR = $\frac{m_{end} - m_{start}}{l}$
- RP = $\frac{AP}{m_{full}}$
- RPR = $\frac{RP}{l}$
- NRP = $\frac{AP}{m_{full} - m_{start}}$
- NRPR = $\frac{NRP}{l}$

你也可以下载已经[训练好的模型](https://drive.google.com/drive/folders/1KxLgcVDoZwswopCRQEVnBKn4K4gs3lRf?usp=sharing)在我们提供的实验设置上进行模型评估

### 模型自动调参
PyEdmine还支持基于贝叶斯网络的自动调参功能，如
```bash
python examples/cognitive_diagnosis/train/ncd_search_params.py
```
该脚本基于代码中的*parameters_space*变量设置搜参空间

### 绘制学生知识状态变化图
PyEdmine支持使用热力图展示学生知识状态变化过程，对应代码在

```bash
python examples/roster/kt_plot.py
```

效果如下图所示

<img src="asset/img/trace_related_cs_change.png" alt="trace_related_cs_change" width="600">
<img src="asset/img/trace_selected_cs_change.png" alt="trace_selected_cs_change" width="600">
<img src="asset/img/trace_single_concept_change.png" alt="trace_single_concept_change" width="600">

## PyEdmine 重要发布
| Releases | Date      |
|----------|-----------|
| v0.1.0   | 3/26/2025 |
| v0.1.1   | 3/31/2025 |
| v0.2.0   | 4/9/2025 |
| v0.2.1   | 1/8/2025 |
| v0.2.2   | 3/8/2025 |

- `v0.1.0` 初始发布版本
- `v0.1.1` 修复了一些bug，增加了5个知识追踪模型，即ATDKT、CLKT、DTransformer、GRKT、HDLPKT
- `v0.2.0` beta版本，但是GRKT模型训练会报错（NaN），尚未解决
- `v0.2.1` 修复了一些bug，集成了学习路径推荐任务
- `v0.2.2` 修复了学习路径推荐的一些bug

## 数据集扩展
[edi2020-task-34-question.json](./edi2020-task34-question.json)是在 **EDi2020 Task 3&4** 提供的数学题目图像数据基础上，进行的非正式扩展版本。原始数据集中仅包含题目图像，未提供对应的文本信息。为增强其在知识追踪与文本建模任务中的适用性，我补充提取了题目的文本内容，并参考了 [Kaggle Eedi: Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) 的数据格式进行组织，以便于后续使用。

文本提取流程相对简化，主要包括：

使用 OCR 工具识别图像中的文字；

对于 OCR 无法有效识别的题目，使用多模态大模型生成文本描述；

结合人工进行了简单核对与修正。

尽管整体文本信息具有较高准确性，但仍可能存在个别提取错误。这是一个**非官方的扩展版本**，欢迎社区参考与使用，但建议在具体研究中结合自身需求进行验证与清洗。

## 参考代码库

- [PYKT](https://github.com/pykt-team/pykt-toolkit)
- [EduKTM](https://github.com/bigdata-ustc/EduKTM)
- [EduCDM](https://github.com/bigdata-ustc/EduCDM)
- [RecBole](https://github.com/RUCAIBox/RecBole)
- [More_Simple_Reinforcement_Learning](https://github.com/lansinuote/More_Simple_Reinforcement_Learning)
- [其它论文代码仓库](https://zhijiexiong.github.io/sub-page/pyedmine/paperCollection.html)

## 贡献

如果您遇到错误或有任何建议，请通过 [Issue](https://github.com/ZhijieXiong/pyedmine/issuesWe) 进行反馈

我们欢迎任何形式的贡献，包括推荐论文将其添加到[论文列表](https://zhijiexiong.github.io/sub-page/pyedmine/paperCollection.html)中、修复 bug、添加新特性、或提供已训练的模型权重。

如果您希望推荐论文，请在[Discussion](https://github.com/ZhijieXiong/pyedmine/discussions/7)中进行推荐。

如果您希望贡献代码，且没有合并冲突，可以直接提交 Pull Request；若存在潜在冲突或重大更改，请先通过 issue 描述问题，再提交 Pull Request。

如果您希望提供已训练模型的权重，请发送邮件至 18800118477@163.com，并附上模型权重和训练脚本，或包含这些内容的可访问链接。

若您提供的是 PyEdmine 尚未实现的模型，请先通过 Pull Request 贡献模型实现代码，再通过邮件联系。

## 免责声明
PyEdmine 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的
