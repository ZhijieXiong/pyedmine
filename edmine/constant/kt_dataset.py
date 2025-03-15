from copy import deepcopy


def datasets_useful_cols(datasets_merged=None):
    result = {
        "assist2009": ["order_id", "user_id", "problem_id", "correct", "skill_id", "school_id", "skill_name",
                       'attempt_count', 'hint_count', "ms_first_response", "overlap_time"],
        "assist2009-full": ["order_id", "user_id", "problem_id", "correct", "list_skill_ids", "list_skills",
                            "attempt_count", "ms_first_response_time", "answer_type", "student_class_id",
                            "school_id"],
        "assist2012": ["problem_id", "user_id", "end_time", "correct", "skill_id", "overlap_time", "school_id", "skill",
                       'attempt_count', 'hint_count', "ms_first_response"],
        "assist2017": ["studentId", "MiddleSchoolId", "problemId", "skill", "timeTaken", "startTime", "correct",
                       "hintCount", "attemptCount"],
        "slepemapy-anatomy": ["user", "item_asked", "item_answered", "context_name", "type", "time", "response_time",
                              "ip_country"],
        "statics2011": ["Anon Student Id", "Problem Hierarchy", "Problem Name", "Step Name", "First Attempt",
                        "First Transaction Time", "Hints"],
        "junyi2015": ["user_id", "exercise", "correct", "time_done", "time_taken", "time_taken_attempts", "count_hints",
                      "count_attempts"]
    }
    algebra2005 = ["Anon Student Id", "Problem Name", "Step Name", "First Transaction Time", "Correct First Attempt",
                   "Hints", "Step Duration (sec)"]
    result["algebra2005"] = deepcopy(algebra2005)
    result["algebra2005"].append("KC(Default)")

    result["algebra2006"] = deepcopy(algebra2005)
    result["algebra2006"].append("KC(Default)")

    result["algebra2008"] = deepcopy(algebra2005)
    result["algebra2008"].append("KC(SubSkills)")

    result["bridge2algebra2006"] = deepcopy(algebra2005)
    result["bridge2algebra2006"].append("KC(SubSkills)")

    result["bridge2algebra2008"] = deepcopy(algebra2005)
    result["bridge2algebra2008"].append("KC(SubSkills)")

    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)


def datasets_renamed(datasets_merged=None):
    result = {
        "assist2009": {
            "problem_id": "question_id",
            "correct": "correctness",
            "skill_id": "concept_id",
            "skill_name": "concept_name",
            "attempt_count": "num_attempt",
            "hint_count": "num_hint",
            "ms_first_response": "use_time_first_attempt",
            "overlap_time": "use_time"
        },
        "assist2009-full": {
            "problem_id": "question_id",
            "correct": "correctness",
            "list_skill_ids": "concept_id",
            "ms_first_response_time": "use_time_first_attempt",
            "list_skills": "concept_name",
            "student_class_id": "class_id",
            "attempt_count": "num_attempt",
            "answer_type": "question_type"
        },
        "assist2012": {
            "problem_id": "question_id",
            "correct": "correctness",
            "skill_id": "concept_id",
            "end_time": "timestamp",
            "overlap_time": "use_time",
            "skill": "concept_name",
            "attempt_count": "num_attempt",
            "hint_count": "num_hint",
            "ms_first_response": "use_time_first_attempt",
        },
        "assist2015": {
            "sequence_id": "question_id",
            "correct": "correctness"
        },
        "assist2017": {
            "problemId": "question_id",
            "correct": "correctness",
            "skill": "concept_id",
            "studentId": "user_id",
            "MiddleSchoolId": "school_id",
            "timeTaken": "use_time",
            "startTime": "timestamp",
            "attemptCount": "num_attempt",
            "hintCount": "num_hint",
        },
        "SLP": {
            "student_id": "user_id",
            "concept": "concept_id",
            "time_access": "timestamp"
        },
        "statics2011": {
            "Anon Student Id": "user_id",
            "Problem Hierarchy": "concept_id",
            "First Transaction Time": "timestamp",
            "First Attempt": "correctness",
            "Hints": "num_hint"
        },
        "ednet-kt1": {
            "tags": "concept_id",
            "elapsed_time": "use_time",
            "correct": "correctness"
        },
        "slepemapy-anatomy": {
            "user": "user_id",
            "time": "timestamp",
            "response_time": "use_time",
            "type": "question_type",
            "ip_country": "country_id",
            "context_name": "concept_id"
        },
        "poj": {
            "User": "user_id",
            "Submit Time": "timestamp",
            "Problem": "question_id",
            "Result": "correctness"
        },
        "junyi2015": {
            "exercise": "question_name",
            "time_done": "timestamp",
            "time_taken": "use_time",
            "time_taken_attempts": "use_time_first_attempt",
            "count_hints": "num_hint",
            "count_attempts": "num_attempt",
            "correct": "correctness"
        }
    }
    algebra2005 = {
        "Anon Student Id": "user_id",
        "Correct First Attempt": "correctness",
        "First Transaction Time": "timestamp",
        "Step Duration (sec)": "use_time",
        "Hints": "num_hint"
    }
    result["algebra2005"] = deepcopy(algebra2005)
    result["algebra2005"]["KC(Default)"] = "concept_id"

    result["algebra2006"] = deepcopy(algebra2005)
    result["algebra2006"]["KC(Default)"] = "concept_id"

    result["algebra2008"] = deepcopy(algebra2005)
    result["algebra2008"]["KC(SubSkills)"] = "concept_id"

    result["bridge2algebra2006"] = deepcopy(algebra2005)
    result["bridge2algebra2006"]["KC(SubSkills)"] = "concept_id"

    result["bridge2algebra2008"] = deepcopy(algebra2005)
    result["bridge2algebra2008"]["KC(SubSkills)"] = "concept_id"

    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)


def datasets_seq_keys(datasets_merged=None):
    result = {
        "assist2009": ["question_seq", "correctness_seq", "use_time_seq", "use_time_first_seq",
                       "num_hint_seq", "num_attempt_seq"],
        "assist2009-full": ["question_seq", "correctness_seq", "use_time_first_seq", "num_attempt_seq"],
        "assist2012": ["question_seq", "correctness_seq", "time_seq", "use_time_seq", "use_time_first_seq",
                       "num_hint_seq", "num_attempt_seq"],
        "assist2015": ["question_seq", "correctness_seq", "answer_score_seq"],
        "assist2017": ["question_seq", "correctness_seq", "time_seq", "use_time_seq", "num_hint_seq",
                       "num_attempt_seq"],
        "edi2020-task1": ["question_seq", "correctness_seq", "time_seq", "age_seq"],
        "edi2020-task34": ["question_seq", "correctness_seq", "time_seq", "age_seq"],
        "SLP": ["question_seq", "correctness_seq", "time_seq", "mode_seq", "answer_score_seq"],
        "slepemapy-anatomy": ["question_seq", "correctness_seq", "time_seq", "use_time_seq"],
        "statics2011": ["question_seq", "correctness_seq", "time_seq", "num_hint_seq"],
        "ednet-kt1": ["question_seq", "correctness_seq", "time_seq", "use_time_seq"],
        "algebra2005": ["question_seq", "correctness_seq", "time_seq", "use_time_seq", "num_hint_seq"],
        "junyi2015": ["question_seq", "correctness_seq", "time_seq", "use_time_seq", "use_time_first_seq",
                      "num_hint_seq", "num_attempt_seq"],
        "poj": ["question_seq", "correctness_seq", "error_type_seq", "time_seq"],
    }
    result["algebra2006"] = result["algebra2005"]
    result["algebra2008"] = result["algebra2005"]
    result["bridge2algebra2006"] = result["algebra2005"]
    result["bridge2algebra2008"] = result["algebra2005"]
    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)
