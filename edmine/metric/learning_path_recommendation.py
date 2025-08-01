import numpy as np


def promotion_report(initial_scores, final_scores, path_lengths, weights=None):
    """

    Parameters
    ----------
    initial_scores: list or array
    final_scores: list or array
    path_lengths: list or array

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
    if len(initial_scores) == 0 or len(final_scores) == 0 or len(path_lengths) == 0:
        return {
            "AP": -1,
            "APR": -1,
            "RP": -1,
            "RPR": -1,
            "NRP": -1,
            "NRPR": -1
        }
    ret = {}

    initial_scores = np.asarray(initial_scores)
    final_scores = np.asarray(final_scores)

    absp = final_scores - initial_scores

    if weights is not None:
        absp *= np.asarray(weights)

    ret["AP"] = absp

    absp_rate = absp / np.asarray(path_lengths)
    absp_rate[absp_rate == np.inf] = 0
    ret["APR"] = absp_rate

    full_score = np.asarray([1] * len(initial_scores))

    relp = absp / full_score
    ret["RP"] = relp

    relp_rate = absp / (full_score * path_lengths)
    relp_rate[relp_rate == np.inf] = 0
    ret["RPR"] = relp_rate

    ret["NRP"] = absp / (full_score - initial_scores)

    norm_relp_rate = absp / ((full_score - initial_scores) * path_lengths)
    norm_relp_rate[norm_relp_rate == np.inf] = 0
    ret["NRPR"] = norm_relp_rate

    return {k: np.average(v) for k, v in ret.items()}
