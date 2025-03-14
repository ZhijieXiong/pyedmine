import dspy


class SelectUserHistoryExercise(dspy.Signature):
    """你需要根据学生的历史练习记录，推断ta是否能做对指定习题，首先请根据待预测的习题，从学生的历史练习中选择你认为对预测有帮助的记录，其中学生的历史练习记录将会以`interaction_id1: practice time; related concepts\ninteraction_id2: practice time; related concepts\n...`的格式展示，你只需要按照`qid1,qid2, ...`的格式回复哪些习题练习记录是你想要查看的即可"""
    question = dspy.InputField(desc="待预测的习题")
    history_interactions = dspy.InputField(desc="学生历史练习记录的信息，每一行的格式为`interaction_id1: practice time; related concepts`，其中practice time表示这个练习记录是发生在多久之前，单位为天，related concepts表示这次练习习题关联的知识点")
    selected_history_interactions = dspy.OutputField(desc="你认为对于预测学生是否能做对习题有帮助的习题练习记录，格式为`interaction_id1,interaction_id1, ...`，例如`1,3`表示查看interaction_id为1和2的学生历史练习记录")


class PredictUserAnswerCorrectness(dspy.Signature):
    """你需要根据学生的历史练习记录，推断ta是否能做对指定习题"""
    question = dspy.InputField(desc="待预测的习题")
    history_exercised_questions = dspy.InputField(desc="学生历史练习记录的信息，每一行的格式为`practice time: answer result; question text; related concepts`，其中answer result为1表示学生做对这道习题，为0表示做错，practice time表示这个练习记录是发生在多久之前，单位为天，question text是习题的文本信息，related concepts表示这次练习习题关联的知识点，例如`3: 0; question text; related concepts`表示学生3天前做错了这道习题")
    predict_result = dspy.OutputField(desc="基于学生的历史练习记录，你认为学生是否能做对习题，只需要回答`Y`或者`N`，Y表示认为能做对，N表示会做错")
    predict_explanation = dspy.OutputField(desc="你对于预测结果的解释")