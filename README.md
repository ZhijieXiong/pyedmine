# PyEdmine

[![](./asset/img/pypi_icon.png)](https://pypi.org/project/edmine/)

[文档] | [数据集信息] | [教育数据挖掘论文列表] | [模型榜单] | [English]

[文档]: https://zhijiexiong.github.io/sub-page/pyedmine/document/site/index.html
[数据集信息]: https://zhijiexiong.github.io/sub-page/pyedmine/datasetInfo.html
[教育数据挖掘论文列表]: https://zhijiexiong.github.io/sub-page/pyedmine/paperCollection.html
[模型榜单]: https://zhijiexiong.github.io/sub-page/pyedmine/rankingList.html
[English]: README_EN.md

PyEdmine 是一个面向研究者的，易于开发与复现的**教育领域数据挖掘**代码库。
目前已实现了20+个知识追踪模型、7个认知诊断模型、3个习题推荐模型。
我们约定了一个统一、易用的数据文件格式，并已支持 14 个 benchmark dataset。
此外，我们设计了一个统一的实验设置，该设置下，知识追踪模型和认知诊断模型可以在习题推荐任务上进行评估。


<p align="center">
  <img src="asset/img/ExperimentalFlowChart.jpg" alt="PeEdmine 实验流程图" width="600">
  <br>
  <b>图片</b>: PyEdmine 实验流程图
</p>


## 安装

### 从pip安装

```bash
pip install edmine
```

### 从源文件安装
```bash
git clone git@github.com:ZhijieXiong/pyedmine.git && cd pyedmine
pip install -e .
```

## 快速开始
如果你从GitHub下载了PyEdmine的源码，你可以使用`examples`里提供的脚本进行数据预处理、数据集划分、模型训练和模型评估：

### 配置数据和模型的存放目录
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
该脚本会生成数据集经过统一格式处理后的文件（位于`/path/to/save/data/dataset_preprocessed`）

注意：`Ednet-kt1`数据集由于原始数据文件数量太多，需要首先使用脚本`examples/data_preprocess/generate_ednet_raw.py`对用户的数据按照5000为单位进行聚合，并且因为该数据集过于庞大，所以预处理默认是只使用随机抽选的5000名用户的数据

或者你可以直接下载已处理好的[数据集文件](https://www.alipan.com/s/WGGnC3uqgq6)

### 数据集划分
你可以选择使用我们提供的数据集划分脚本，划分好的数据集文件将存放在`/path/to/save/data/settings/setting_name`下
```bash
python examples/knowledge_tracing/prepare_dataset/pykt_setting.py  # 知识追踪
python examples/cognitive_diagnosis/prepare_dataset/ncd_setting.py  # 认知诊断
python examples/exercise_recommendation/preprare_dataset/offline_setting.py  # 习题推荐
```
你也可以直接下载划分后的数据集文件（[KT](https://www.alipan.com/s/Lek2EDxPfUJ), [CD](https://www.alipan.com/s/ZVkqDhtdbpV), [ER](https://www.alipan.com/s/BJQHQn3waA6), [CD4ER](https://www.alipan.com/s/YCojzoGtYPu)），然后将其存放在对应的目录下

或者你也可以参照我们提供的数据集划分脚本来设计自己的实验处理流程

### 训练模型
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
训练时会得到类似如下的输出
```bash
2025-03-06 02:12:35 epoch 1   , valid performances are main metric: 0.7186   , AUC: 0.7186   , ACC: 0.64765  , MAE: 0.41924  , RMSE: 0.46919  , train loss is predict loss: 0.588902    , current best epoch is 1
2025-03-06 02:12:37 epoch 2   , valid performances are main metric: 0.72457  , AUC: 0.72457  , ACC: 0.63797  , MAE: 0.42329  , RMSE: 0.47456  , train loss is predict loss: 0.556672    , current best epoch is 2
2025-03-06 02:12:39 epoch 3   , valid performances are main metric: 0.72014  , AUC: 0.72014  , ACC: 0.63143  , MAE: 0.43218  , RMSE: 0.47536  , train loss is predict loss: 0.551513    , current best epoch is 2
2025-03-06 02:12:40 epoch 4   , valid performances are main metric: 0.71843  , AUC: 0.71843  , ACC: 0.65182  , MAE: 0.41843  , RMSE: 0.46837  , train loss is predict loss: 0.548907    , current best epoch is 2
2025-03-06 02:12:42 epoch 5   , valid performances are main metric: 0.72453  , AUC: 0.72453  , ACC: 0.65276  , MAE: 0.41841  , RMSE: 0.46684  , train loss is predict loss: 0.547639    , current best epoch is 2
...
2025-03-06 02:13:44 epoch 31  , valid performances are main metric: 0.72589  , AUC: 0.72589  , ACC: 0.65867  , MAE: 0.40794  , RMSE: 0.46316  , train loss is predict loss: 0.532516    , current best epoch is 16
2025-03-06 02:13:47 epoch 32  , valid performances are main metric: 0.72573  , AUC: 0.72573  , ACC: 0.65426  , MAE: 0.41602  , RMSE: 0.46415  , train loss is predict loss: 0.532863    , current best epoch is 16
2025-03-06 02:13:49 epoch 33  , valid performances are main metric: 0.72509  , AUC: 0.72509  , ACC: 0.6179   , MAE: 0.43133  , RMSE: 0.48417  , train loss is predict loss: 0.532187    , current best epoch is 16
2025-03-06 02:13:52 epoch 34  , valid performances are main metric: 0.72809  , AUC: 0.72809  , ACC: 0.63938  , MAE: 0.41994  , RMSE: 0.47377  , train loss is predict loss: 0.533765    , current best epoch is 16
2025-03-06 02:13:54 epoch 35  , valid performances are main metric: 0.72523  , AUC: 0.72523  , ACC: 0.63852  , MAE: 0.42142  , RMSE: 0.47327  , train loss is predict loss: 0.531101    , current best epoch is 16
2025-03-06 02:13:57 epoch 36  , valid performances are main metric: 0.72838  , AUC: 0.72838  , ACC: 0.61986  , MAE: 0.43105  , RMSE: 0.48364  , train loss is predict loss: 0.532342    , current best epoch is 16
best valid epoch: 16  , train performances in best epoch by valid are main metric: 0.74893  , AUC: 0.74893  , ACC: 0.72948  , MAE: 0.34608  , RMSE: 0.42706  , main_metric: 0.74893  , 
valid performances in best epoch by valid are main metric: 0.72902  , AUC: 0.72902  , ACC: 0.59389  , MAE: 0.43936  , RMSE: 0.49301  , main_metric: 0.72902  , 
```
如果训练模型时*use_wandb*参数为True，则可以在[wandb](https://wandb.ai/)上查看模型的损失变化和指标变化

### 评估模型
如果训练模型时*save_model*参数，则会将模型参数文件保存至`/path/to/save/model`目录下，那么可以使用测试集对模型进行评估，如
```bash
python examples/knowledge_tracing/evaluate/sequential_dlkt.py --model_dir_name [model_dir_name] --dataset_name [dataset_name] --test_file_name [test_file_name]
```
其中知识追踪和认知诊断模型除了常规的指标评估外，还可以进行一些细粒度的指标评估，例如冷启动评估，知识追踪的多步预测等，这些评估都可以通过设置对应的参数开启

你也可以下载已经[训练好的模型](https://zhijiexiong.github.io/sub-page/pyedmine/document/site/index.html)在我们提供的实验设置上进行模型评估

### 自动调参
PyEdmine还支持基于贝叶斯网络的自动调参功能，如
```bash
python examples/cognitive_diagnosis/train/ncd_search_params.py
```
该脚本基于代码中的*parameters_space*变量设置搜参空间

## PyEdmine 重要发布
| Releases | Date      |
|----------|-----------|
| v0.1.0   | 3/26/2025 |
| v0.1.1   | 3/31/2025 |

- `v0.1.0` 初始发布版本
- `v0.1.1` 修复了一些bug，增加了5个知识追踪模型，即ATDKT、CLKT、DTransformer、GRKT、HDLPKT
- 
## 参考代码库

- [PYKT](https://github.com/pykt-team/pykt-toolkit)
- [EduDATA](https://github.com/bigdata-ustc/EduData)
- [EduKTM](https://github.com/bigdata-ustc/EduKTM)
- [EduCDM](https://github.com/bigdata-ustc/EduCDM)
- [RecBole](https://github.com/RUCAIBox/RecBole)

## 贡献

如果您遇到错误或有任何建议，请通过 [Issue](https://github.com/ZhijieXiong/pyedmine/issuesWe) 进行反馈

我们欢迎关于修复错误、添加新特性的任何贡献。

如果想贡献代码，请先在issue中提出问题，然后再提PR。

## 免责声明
PyEdmine 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的。
