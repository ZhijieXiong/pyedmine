import numpy as np
from longling import as_list


def as_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    else:
        return np.asarray(as_list(obj))


def promotion_report(initial_score, final_score, path_length, weights=None):
    """

    Parameters
    ----------
    initial_score: list or array
    final_score: list or array
    path_length: list or array

    Returns
    -------
    report: dict
    
        AP:
            absolute promotion =  final_score - initial_score
        APR:
            absolute promotion rate =  \frac{absolute promotion}{path_length}
        RP:
            relative promotion =  \frac{final_score - initial_score}{full_score}
        RPR:
            relative promotion rate = \frac{relative promotion}{path_length}
        NRP:
            normalized relative promotion = \frac{final_score - initial_score}{full_score - initial_score}
        NRPR:
            normalized relative promotion rate = \frac{normalized relative promotion}{path_length}
    """
    ret = {}

    initial_score = as_array(initial_score)
    final_score = as_array(final_score)

    absp = final_score - initial_score

    if weights is not None:
        absp *= as_array(weights)

    ret["AP"] = absp

    absp_rate = absp / as_array(path_length)
    absp_rate[absp_rate == np.inf] = 0
    ret["APR"] = absp_rate

    full_score = as_array([1] * len(initial_score))

    relp = absp / full_score
    ret["RP"] = relp

    relp_rate = absp / (full_score * path_length)
    relp_rate[relp_rate == np.inf] = 0
    ret["RPR"] = relp_rate

    ret["NRP"] = absp / (full_score - initial_score)

    norm_relp_rate = absp / ((full_score - initial_score) * path_length)
    norm_relp_rate[norm_relp_rate == np.inf] = 0
    ret["NRPR"] = norm_relp_rate

    return {k: np.average(v) for k, v in ret.items()}
