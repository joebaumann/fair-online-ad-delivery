import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from statistics import mean
from utils import *
from pathlib import Path
from scipy.stats import powerlaw
import logging
from datetime import datetime
from scenario_config import get_scenario_parameters, get_simulation_parameters


class PowerLawDistribution:
    def __init__(self, k, n, loc=0, scale=1, seed=123123):
        logging.info(f"""new PowerLawDistribution instance:
            - k: {k}
            - n: {n}""")
        self.k = k
        self.n = n

        rng = np.random.default_rng(seed=seed)

        # draw samples from powerlaw distribution
        samples = powerlaw.rvs(k, loc=loc, scale=scale, size=n, random_state=rng)
        self.s = pd.Series(samples, name="probabilities")

        self.thresholds, self.y_pred_dict = self.generate_lower_bound_thresholds()
        #self.thresholds, self.y_pred_dict = self.generate_upper_and_lower_bound_thresholds()

    
    def plot_hist(self, mycolor="blue", nr_of_bins=1000, mylabel="", log=False):
        # display hist
        count, bins, ignored = plt.hist(self.s, bins=nr_of_bins, density=True, align='mid', color=mycolor, alpha=0.5, label=mylabel, log=log)
        plt.axis('tight')
        #plt.show()


    def generate_lower_bound_thresholds(self, nr_of_thresholds=100, unconstrained_optimal_threshold=None):
        results, bin_edges = pd.qcut(self.s, q=nr_of_thresholds, retbins=True, duplicates="drop", labels=False)
        thresholds = list(bin_edges)
        if len(thresholds) < nr_of_thresholds:
            logging.info(f"nr_of_thresholds is set to {nr_of_thresholds} but sample size is just {len(self.s)}. This resulted in duplicate thresholds. Non-unique thresholds were droped, which resulted in a total of {len(thresholds)} thresholds.")

        if min(self.s) < 0.1 and min(self.s) >= 0:
            thresholds.insert(0,0.0)
        if max(self.s) > 0.9 and max(self.s) <= 1:
            thresholds.append(1.0)
        
        if unconstrained_optimal_threshold:
            if unconstrained_optimal_threshold > max(thresholds) or unconstrained_optimal_threshold < min(thresholds):
                logging.info(f"Error: unconstrained_optimal_threshold must be between the minimum and the maximum threshold.")
                return
            
            # insert threshold of optimal unconstrained classifier
            if unconstrained_optimal_threshold not in thresholds and float(unconstrained_optimal_threshold) not in thresholds:
                for i,t in enumerate(thresholds):
                    if t>unconstrained_optimal_threshold:
                        thresholds.insert(i, float(unconstrained_optimal_threshold))
                        break
                if unconstrained_optimal_threshold not in thresholds and float(unconstrained_optimal_threshold) not in thresholds:
                    thresholds.insert(len(thresholds)-1, float(unconstrained_optimal_threshold))
        
        y_pred_dict = {t:self.s >= t for t in thresholds}
        
        return thresholds, y_pred_dict


    def generate_upper_and_lower_bound_thresholds(self, nr_of_thresholds=100, unconstrained_optimal_threshold=None):
        results, bin_edges = pd.qcut(self.s, q=nr_of_thresholds, retbins=True, duplicates="drop", labels=False)
        thresholds = list(bin_edges)
        if len(thresholds) < nr_of_thresholds:
            logging.info(f"nr_of_thresholds is set to {nr_of_thresholds} but sample size is just {len(self.s)}. This resulted in duplicate thresholds. Non-unique thresholds were droped, which resulted in a total of {len(thresholds)} thresholds.")

        if min(self.s) < 0.1 and min(self.s) >= 0:
            thresholds.insert(0,0.0)
        if max(self.s) > 0.9 and max(self.s) <= 1:
            thresholds.append(1.0)
        
        threshold_tuples = []
        
        minimum, maximum = min(thresholds), max(thresholds)

        for t in thresholds:
            new_tuple = (minimum, t)
            if minimum != t and new_tuple not in threshold_tuples:
                threshold_tuples.append(new_tuple)
        for t in thresholds:
            new_tuple = (t, maximum)
            if maximum != t and new_tuple not in threshold_tuples:
                threshold_tuples.append(new_tuple)
        
        if unconstrained_optimal_threshold:
            if unconstrained_optimal_threshold > maximum or unconstrained_optimal_threshold < minimum:
                logging.info(f"Error: unconstrained_optimal_threshold must be between the minimum and the maximum threshold.")
                return

            # insert threshold of optimal unconstrained classifier
            if (unconstrained_optimal_threshold,maximum) not in threshold_tuples and (float(unconstrained_optimal_threshold),maximum) not in threshold_tuples:
                # check lower-bount thresholds 
                for i,t in enumerate(threshold_tuples):
                    if t[0]>unconstrained_optimal_threshold:
                        threshold_tuples.insert(i, (float(unconstrained_optimal_threshold),maximum))
                        break
                if (unconstrained_optimal_threshold,maximum) not in threshold_tuples and (float(unconstrained_optimal_threshold),maximum) not in threshold_tuples:
                    threshold_tuples.append( (float(unconstrained_optimal_threshold),maximum))
            
            if (minimum,unconstrained_optimal_threshold) not in threshold_tuples and (minimum,float(unconstrained_optimal_threshold)) not in threshold_tuples:
                # check upper-bount thresholds 
                for i,t in enumerate(threshold_tuples):
                    if unconstrained_optimal_threshold>t[1]:
                        threshold_tuples.insert(i-1, (minimum, float(unconstrained_optimal_threshold)))
                        break
                if (minimum,unconstrained_optimal_threshold) not in threshold_tuples and (minimum,float(unconstrained_optimal_threshold)) not in threshold_tuples:
                    threshold_tuples.insert(0, (minimum, float(unconstrained_optimal_threshold)))
        
        y_pred_dict = {t: self.s.between(t[0], t[1]) for t in threshold_tuples}

        return threshold_tuples, y_pred_dict


    def generate_thresholds_old(self, threshold_nr=1000):
        thresholds = []
        for i in np.linspace(1/threshold_nr, 1, threshold_nr).tolist():
            thresholds.append((0, i))
        for i in np.linspace(1/threshold_nr, 1-1/threshold_nr, threshold_nr-1).tolist():
            thresholds.append((i, 1))
        return thresholds
    
    def apply_decision_rule_old(self, threshold): # based on existing thresholds
        # lower-bound threshold
        lower_bound_thresholds = [i for (i,j) in self.thresholds if j==1]
        closest_existing_threshold = min(lower_bound_thresholds,key=lambda x:abs(x-threshold))
        decisions = self.y_pred_dict[(closest_existing_threshold,1)]
        probabilities = self.s
        return decisions, probabilities
    
    def apply_decision_rule(self, threshold):
        # lower-bound threshold
        decisions = self.s >= threshold
        probabilities = self.s
        return decisions, probabilities

def optimal_unconstrained_decision_rule(alpha, beta):
    # alpha: individual utility for Y=1 & D=1
    # alpha: individual utility for D=0
    optimal_lower_bound_treshold = beta/alpha
    return optimal_lower_bound_treshold

def all_possible_selections_under_limited_ressources(limited_ressources, probabilities_a, probabilities_b, metrics_prob, alpha, beta_group_a, beta_group_b):

    probabilities_a_sorted = probabilities_a.sort_values(ascending=True)
    probabilities_b_sorted = probabilities_b.sort_values(ascending=True)

    decisions_a = probabilities_a_sorted.between(probabilities_a_sorted.iloc[-limited_ressources], 1)
    decisions_b = probabilities_b_sorted.between(probabilities_b_sorted.iloc[-limited_ressources], 1)
    
    selected_a_sorted = probabilities_a_sorted[decisions_a]
    selected_b_sorted = probabilities_b_sorted[decisions_b]
    
    all_possible_selections = {}
    for index_a, a in selected_a_sorted.items():

        decisions_a[index_a] = False # remove lowest individual from group a
        
        if sum(decisions_a)+sum(decisions_b) <= limited_ressources and sum(decisions_a)>0 and sum(decisions_b)>0:

            # key takes the form: (<nr of selected top candidates group a>, <nr of selected top candidates group a>)
            all_possible_selections[ (sum(decisions_a), sum(decisions_b)) ] = {
                "decisions_a": decisions_a.copy(deep=True),
                "decisions_b": decisions_b.copy(deep=True),
                "threshold_a": a,
                "threshold_a": selected_b_sorted.min(),
                "utility": get_utility(probabilities_a, alpha, beta_group_a, decisions_a) + get_utility(probabilities_b, alpha, beta_group_b, decisions_b),
                "results": calculate_metrics(metrics_prob, decisions_a, probabilities_a, decisions_b, probabilities_b)
                }

        for index_b, b in selected_b_sorted.items():
            
            decisions_b[index_b] = False # remove lowest individual from group b

            if sum(decisions_a)+sum(decisions_b) <= limited_ressources and sum(decisions_a)>0 and sum(decisions_b)>0:
            
                all_possible_selections[ (sum(decisions_a), sum(decisions_b)) ] = {
                    "decisions_a": decisions_a.copy(deep=True),
                    "decisions_b": decisions_b.copy(deep=True),
                    "threshold_a": a,
                    "threshold_a": b,
                    "utility": get_utility(probabilities_a, alpha, beta_group_a, decisions_a) + get_utility(probabilities_b, alpha, beta_group_b, decisions_b),
                    "results": calculate_metrics(metrics_prob, decisions_a, probabilities_a, decisions_b, probabilities_b)
                    }
        
        # reset decisions b
        decisions_b = probabilities_b_sorted.between(probabilities_b_sorted.iloc[-limited_ressources], 1)

    return all_possible_selections


def get_utility(probabilities, alpha, beta, decisions):
        return beta * sum(decisions == 0) + sum(alpha * probabilities[decisions == 1])

def get_utilities(probabilities, alpha, beta):
        return alpha * probabilities - beta


def optimal_unconstrained_decision_rule_under_limited_ressources(limited_ressources, probabilities_a, probabilities_b, alpha, beta_group_a, beta_group_b):

    # combine all values of both groups and sort by expected utility
    lowest_selected_expected_utility = pd.concat([get_utilities(probabilities_a, alpha, beta_group_a), get_utilities(probabilities_b, alpha, beta_group_b)], ignore_index=True).sort_values(ascending=True).iloc[-limited_ressources]

    lowest_selected_value_a = (lowest_selected_expected_utility + beta_group_a) / alpha
    lowest_selected_value_b = (lowest_selected_expected_utility + beta_group_b) / alpha

    return lowest_selected_value_a, lowest_selected_value_b

def optimal_fair_decision_rule(metrics_prob, fairness_requirement, group_a, group_b, probabilities_a, probabilities_b, alpha, beta_group_a, beta_group_b, max_fairness_diff=None, limited_ressources=None, fixed_impressions=None, fairness_margin=0.1):

    fairness_function = metrics_prob[fairness_requirement]

    rates_a = [fairness_function(group_a.y_pred_dict[t], probabilities_a) for t in group_a.thresholds]
    rates_b = [fairness_function(group_b.y_pred_dict[t], probabilities_b) for t in group_b.thresholds]
    nr_of_selections_a = [sum(group_a.y_pred_dict[t]) for t in group_a.thresholds]
    nr_of_selections_b = [sum(group_b.y_pred_dict[t]) for t in group_b.thresholds]

    # TODO: check if min(nr_of_selections_ a plus b is bigger than limit --> if yes, error --> more thresholds or more resources required
    
    utilities = []
    fairness_scores = []
    thresholds_a = []
    thresholds_b = []
    for rate_a, nr_selected_a, threshold_a in zip(rates_a, nr_of_selections_a, group_a.thresholds):

        rate_a = np.float64(rate_a)
        decisions_a, _ = group_a.apply_decision_rule(threshold_a)
        threshold_b = None

        if limited_ressources:
            # if the number of selected individuals in group a already exceeds the limited resources, then this threshold is not possible
            if nr_selected_a > limited_ressources:
                #utilities.append(-np.inf)
                #fairness_scores.append(None)
                #thresholds_a.append(None)
                #thresholds_b.append(None)
                continue
            elif nr_selected_a+min(nr_of_selections_b) > limited_ressources:
                # if the number of selected individuals from group a plus the minimum number of selected individuals from group b is bigger than the limited resources, then this threshold is not possible
                #utilities.append(-np.inf)
                #fairness_scores.append(None)
                #thresholds_a.append(None)
                #thresholds_b.append(None)
                continue
            else:
                # there are possible solutions with this threshold for group a. check which threshold_b combinations are possible
                mask_array = np.array([nr_selected_a+i for i in nr_of_selections_b])
                if fixed_impressions:
                    # if fixed budget, then we need to select the threshold that is closest to the limited resources
                    subset_idx = np.argmin(np.array(np.abs(mask_array - limited_ressources)))
                    threshold_b = group_b.thresholds[np.arange(mask_array.shape[0])[subset_idx]]
                    #print(f"fixed budget of {limited_ressources} --> final number of selected individuals total: {nr_selected_a + nr_of_selections_b[np.arange(mask_array.shape[0])[subset_idx]]} // a: {nr_selected_a} // b: {nr_of_selections_b[np.arange(mask_array.shape[0])[subset_idx]]}")
                else:
                    mask = (mask_array <= limited_ressources)
        else:
            mask = np.array([True for _ in nr_of_selections_b])

        if threshold_b is not None:
            # the budget is fixed, so there is only one possible threshold_b
            pass
        elif max_fairness_diff:
            # get max utility threshold amongst all those that are within the fairness boundaries
            fairness_scores = np.abs(rates_b - rate_a)
            #mask_array = np.array([nr_selected_a+i for i in fairness_scores])
            # combine possibilities within the budget limit and the fairness boundaries
            mask = np.logical_and(mask, (fairness_scores <= max_fairness_diff))
            
            # check if there is any possible solution
            if sum(mask) == 0:
                # there is no solution left, discard this threshold
                #utilities.append(-np.inf)
                #fairness_scores.append(None)
                #thresholds_a.append(None)
                #thresholds_b.append(None)
                continue
            
            # loop through all possible threshold combinations and choose the one with the highest utility
            utilities_of_applicable_solutions = []
            for i, rate_b in enumerate(rates_b):
                if mask[i]:
                    # this rate combination is fair enough, let's calculate the utility
                    temp_decisions_b, probabilities_b = group_b.apply_decision_rule(group_b.thresholds[i])
                    temp_utility = get_utility(probabilities_a, alpha, beta_group_a, decisions_a) + get_utility(probabilities_b, alpha, beta_group_b, temp_decisions_b)
                    utilities_of_applicable_solutions.append(temp_utility)
                else:
                    # this is not within the fairness boundaries
                    utilities_of_applicable_solutions.append(-np.inf)
            
            optimal_fair_solution_index = np.argmax(np.array(utilities_of_applicable_solutions))
            threshold_b = group_b.thresholds[optimal_fair_solution_index]

        else:    
            # get the fairest threshold for group b among all those that are possible (i.e., that are within the limited resources)
            subset_idx = np.argmin(np.array(np.abs(rates_b - rate_a))[mask])
            threshold_b = group_b.thresholds[np.arange(len(nr_of_selections_b))[mask][subset_idx]]

        decisions_b, _ = group_b.apply_decision_rule(threshold_b)

        utility = get_utility(probabilities_a, alpha, beta_group_a, decisions_a) + get_utility(probabilities_b, alpha, beta_group_b, decisions_b)
        fairness_score_diff = abs(rates_a[group_a.thresholds.index(threshold_a)]-rates_b[group_b.thresholds.index(threshold_b)])

        fairness_scores.append(fairness_score_diff)
        utilities.append(utility)
        thresholds_a.append(threshold_a)
        thresholds_b.append(threshold_b)
    
    logging.info(f"there are {len(fairness_scores)} possible threshold combinations, that satisfy the requirements")
    # get all indices of the fairest solutions
    fairness_scores = np.array(fairness_scores)
    utilities = np.array(utilities)
    solutions_within_fairness_margin = np.where(fairness_scores <= fairness_margin)[0]
    logging.info(f"{len(solutions_within_fairness_margin)} of those {len(fairness_scores)} threshold combinations result in an acceptable fairness score, i.e., one that corresponds to a smaller between group difference than the specified fairness-margin of {fairness_margin}")
    if len(solutions_within_fairness_margin) == 0:
        logging.info(f"there are no solutions within the fairness margin of {fairness_margin}, therefore we select the fairest solution")
        solutions_within_fairness_margin = np.where(fairness_scores == fairness_scores.min())[0]
    # among all solutions within the fairness margin, get the one with the highest utility
    index_of_best_among_fairest_solutions = solutions_within_fairness_margin[np.argmax(utilities[solutions_within_fairness_margin])]
    highest_utility = utilities[index_of_best_among_fairest_solutions]
    ideal_threshold_a = thresholds_a[index_of_best_among_fairest_solutions]
    ideal_threshold_b = thresholds_b[index_of_best_among_fairest_solutions]
    return highest_utility, ideal_threshold_a, ideal_threshold_b


def ppv_prob(decisions, probabilities):
    return 1 if (len(probabilities) <= 0 or len(probabilities[decisions==1]) <= 0) else mean(probabilities[decisions==1])

def forate_prob(decisions, probabilities):
    return 0 if (len(probabilities) <= 0 or len(probabilities[decisions==0]) <= 0) else mean(probabilities[decisions==0])

def tpr_prob(decisions, probabilities):
    return 1 if len(probabilities) <= 0 else ( sum(probabilities[decisions==1]) / sum(probabilities)  )

def fpr_prob(decisions, probabilities):
    return 0 if len(probabilities) <= 0 else ( sum(1 - probabilities[decisions==1]) / sum(1 - probabilities)  )

def acceptance_rate_prob(decisions, probabilities):
    return sum(decisions == 1) / len(decisions)

def nr_of_selected_individuals(decisions, probabilities):
    return sum(decisions == 1)

metrics_prob = {
    "ppv": ppv_prob,
    "for": forate_prob,
    "tpr": tpr_prob,
    "fpr": fpr_prob,
    "acceptance_rate": acceptance_rate_prob,
    "nr_selected": nr_of_selected_individuals
}

def calculate_metrics(metrics, decisions_a, probabilities_a, decisions_b, probabilities_b):
    metric_values = {}
    for metric_name, fairness_function in metrics.items():
        fairness_value_a = fairness_function(decisions_a, probabilities_a)
        fairness_value_b = fairness_function(decisions_b, probabilities_b)
        metric_values[metric_name] = [fairness_value_a, fairness_value_b]

    return metric_values

def calculate_metrics_for_one_group(metrics, decisions, probabilities, group):
    metric_values = {"group":group}
    for metric_name, fairness_function in metrics.items():
        fairness_value = fairness_function(decisions, probabilities)
        metric_values[metric_name] = fairness_value

    return metric_values

def get_scenario(scenario, sensitivity_value=None, seed=None):
    alpha, beta_group_a, beta_group_b, k_group_a, k_group_b, n_group_a, n_group_b = get_simulation_parameters(scenario, sensitivity_value)

    logging.info(f"""SIMULATION PARAMETERS
        - alpha: {alpha}
        - beta_group_a: {beta_group_a}
        - beta_group_b: {beta_group_b}
        - k_group_a: {k_group_a}
        - k_group_b: {k_group_b}
        - n_group_a: {n_group_a}
        - n_group_b: {n_group_b}
    """)

    group_a = PowerLawDistribution(k=k_group_a, n=n_group_a, seed=seed)
    group_b = PowerLawDistribution(k=k_group_b, n=n_group_b, seed=seed+10000)

    return group_a, group_b, alpha, beta_group_a, beta_group_b

def run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources, fairness_requirement, fixed_impressions=None, max_fairness_diff=None, fairness_margin=0.1):
    
    group_a_threshold = optimal_unconstrained_decision_rule(alpha=alpha, beta=beta_group_a)
    group_b_threshold = optimal_unconstrained_decision_rule(alpha=alpha, beta=beta_group_b)
    
    logging.info(f"Individuals of group a are shown the ad if their probability of clicking on it is > {group_a_threshold} this corresponds to {sum(group_a.s>group_a_threshold)/(group_a.n)*100}% of group a, i.e., {sum(group_a.s>group_a_threshold)} out of {group_a.n} users")
    logging.info(f"Individuals of group b are shown the ad if their probability of clicking on it is > {group_b_threshold} this corresponds to {sum(group_b.s>group_b_threshold)/(group_b.n)*100}% of group b, i.e., {sum(group_b.s>group_b_threshold)} out of {group_b.n} users")

    decisions_a, probabilities_a = group_a.apply_decision_rule(group_a_threshold)
    decisions_b, probabilities_b = group_b.apply_decision_rule(group_b_threshold)

    if fairness_requirement:
        highest_utility, group_a_threshold, group_b_threshold = optimal_fair_decision_rule(metrics_prob, fairness_requirement, group_a, group_b, probabilities_a, probabilities_b, alpha, beta_group_a, beta_group_b, max_fairness_diff=max_fairness_diff, limited_ressources=limited_ressources, fixed_impressions=fixed_impressions, fairness_margin=fairness_margin)

        decisions_a, probabilities_a = group_a.apply_decision_rule(group_a_threshold)
        decisions_b, probabilities_b = group_b.apply_decision_rule(group_b_threshold)

    elif limited_ressources:
        # fairness is not required, we just want to get the optimal decision rule not exceeding the limited ressources
        if limited_ressources and sum(decisions_a) + sum(decisions_b) <= limited_ressources:
            logging.info(f"The optimal solution does not exceed the available ressources.")
        else:
            logging.info(f"The optimal solution exceeds the available ressources!")
            lowest_selected_value_a, lowest_selected_value_b = optimal_unconstrained_decision_rule_under_limited_ressources(limited_ressources, probabilities_a, probabilities_b, alpha, beta_group_a, beta_group_b)

            group_a_threshold = lowest_selected_value_a
            group_b_threshold = lowest_selected_value_b

            decisions_a, probabilities_a = group_a.apply_decision_rule(group_a_threshold)
            decisions_b, probabilities_b = group_b.apply_decision_rule(group_b_threshold)
    

    results = calculate_metrics(metrics_prob, decisions_a, probabilities_a, decisions_b, probabilities_b)
    results["utility"] = get_utility(probabilities_a, alpha, beta_group_a, decisions_a) + get_utility(probabilities_b, alpha, beta_group_b, decisions_b)
    
    return results, group_a_threshold, group_b_threshold


def run_one_scenario(scenario, limited_ressources, fairness_requirement):

    group_a, group_b, alpha, beta_group_a, beta_group_b = get_scenario(scenario)

    metrics_solution, group_a_threshold, group_b_threshold = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources, fairness_requirement)

    logging.info(f"metrics_solution: {metrics_solution}")

    logging.info(f"total selected = {sum(metrics_solution['nr_selected'])}")

    logging.info(f"--- done end ---")


def run_fairness_utility_tradeoff_simulation_for_degrees_of_fairness(scenario, limited_ressources, fairness_requirement):

    group_a, group_b, alpha, beta_group_a, beta_group_b = get_scenario(scenario)

    metrics_solution_no_fairness, _, _ = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources, False, max_fairness_diff=None)
    logging.info(f"metrics_solution NO FAIRNESS: {metrics_solution_no_fairness}")

    metrics_solution_full_fairness, _, _ = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources, fairness_requirement, max_fairness_diff=None)
    logging.info(f"metrics_solution FULL FAIRNESS: {metrics_solution_full_fairness}")

    min_fairness = abs(metrics_solution_no_fairness[fairness_requirement][0] - metrics_solution_no_fairness[fairness_requirement][1])
    max_fairness = abs(metrics_solution_full_fairness[fairness_requirement][0] - metrics_solution_full_fairness[fairness_requirement][1])

    nr_of_solutions = 5
    all_solutions = {
        (0, min_fairness): metrics_solution_no_fairness
    }

    for i, max_fairness_diff in enumerate(np.linspace(min_fairness, max_fairness, num=nr_of_solutions, endpoint=False)):
        if i == 0:
            continue
        logging.info(f"")
        logging.info(f"    --- calculate solution for fairness value {i} of {nr_of_solutions} ---")
        results, _, _ = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources, fairness_requirement, max_fairness_diff=max_fairness_diff)
        all_solutions[(i, max_fairness_diff)] = results
        
    all_solutions[(nr_of_solutions, max_fairness)] = metrics_solution_full_fairness

    logging.info(f"--- done calculating all solutions ---")


    df = pd.DataFrame()

    for k,v in all_solutions.items():

        sol = {
            "max_fairness_diff_index": k[0],
            "ppv_a": v["ppv"][0],
            "ppv_b": v["ppv"][1],
            "ppv": v["ppv"][0] - v["ppv"][1],
            "for_a": v["for"][0],
            "for_b": v["for"][1],
            "for": v["for"][0] - v["for"][1],
            "tpr_a": v["tpr"][0],
            "tpr_b": v["tpr"][1],
            "tpr": v["tpr"][0] - v["tpr"][1],
            "fpr_a": v["fpr"][0],
            "fpr_b": v["fpr"][1],
            "fpr": v["fpr"][0] - v["fpr"][1],
            "acceptance_rate_a": v["acceptance_rate"][0],
            "acceptance_rate_b": v["acceptance_rate"][1],
            "acceptance_rate": v["acceptance_rate"][0] - v["acceptance_rate"][1],
            "utility": v["utility"]
            }

        df = pd.concat([df, pd.DataFrame([sol])], ignore_index=True)



    sns.set_context("paper")
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

    plt.clf()
    palette = iter(COLOR_PALETTE)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="ppv", alpha=0.6, label="ppv", color=next(palette))
    sns.lineplot(data=df, x="max_fairness_diff_index", y="for", alpha=0.6, label="for", color=next(palette))
    sns.lineplot(data=df, x="max_fairness_diff_index", y="tpr", alpha=0.6, label="tpr", color=next(palette))
    sns.lineplot(data=df, x="max_fairness_diff_index", y="fpr", alpha=0.6, label="fpr", color=next(palette))
    sns.lineplot(data=df, x="max_fairness_diff_index", y="acceptance_rate", alpha=0.6, label="acceptance_rate", color=next(palette))
    plt.ylabel("parity metrics difference (group a - group b)")
    plt.xlabel("degree of fairness (0: full utility / 1: max fairness)")
    #plt.legend()
    ax2 = plt.twinx()
    sns.lineplot(data=df, x="max_fairness_diff_index", y="utility", color='black',alpha=0.6, label="utility", ax=ax2)
    plt.tight_layout()
    plt.savefig('degrees_of_fairness/scenario_' + str(scenario) + '_' + fairness_requirement + '_diff.pdf')

    plt.clf()
    palette = iter(COLOR_PALETTE)
    col= next(palette)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="ppv_a", marker='o', alpha=0.6, label="ppv_a", color=col)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="ppv_b", marker='v', alpha=0.6, label="ppv_b", color=col)
    col= next(palette)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="for_a", marker='o', alpha=0.6, label="for_a", color=col)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="for_b", marker='v', alpha=0.6, label="for_b", color=col)
    col= next(palette)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="tpr_a", marker='o', alpha=0.6, label="tpr_a", color=col)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="tpr_b", marker='v', alpha=0.6, label="tpr_b", color=col)
    col= next(palette)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="fpr_a", marker='o', alpha=0.6, label="fpr_a", color=col)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="fpr_b", marker='v', alpha=0.6, label="fpr_b", color=col)
    col= next(palette)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="acceptance_rate_a", marker='o', alpha=0.6, label="acceptance_rate_a", color=col)
    sns.lineplot(data=df, x="max_fairness_diff_index", y="acceptance_rate_b", marker='v', alpha=0.6, label="acceptance_rate_b", color=col)
    plt.ylabel("parity metrics difference (group a - group b)")
    plt.xlabel("degree of fairness (0: full utility / 1: max fairness)")
    #plt.legend()
    ax2 = plt.twinx()
    sns.lineplot(data=df, x="max_fairness_diff_index", y="utility", color='black',alpha=0.6, label="utility", ax=ax2)
    plt.tight_layout()
    plt.savefig('degrees_of_fairness/scenario_' + str(scenario) + '_' + fairness_requirement + '.pdf')

@timer
def run_fairness_utility_tradeoff_simulation_for_different_criteria(path, scenario, limited_ressources, fixed_impressions, fairness_requirements, fairness_margin, sensitivity_parameter, sensitivity_vals, seed):
    logging.info(f"""
    RUNNING FAIRNESS UTILITY TRADEOFF SIMULATION:
        - path: {path}
        - scenario: {scenario}
        - limited_ressources: {limited_ressources}
        - fairness_requirements: {fairness_requirements}
        - sensitivity_parameter: {sensitivity_parameter}
        - sensitivity_vals: {sensitivity_vals}
    """)
    df_columns = metrics_prob.keys()
    def unpack_results(results, df_columns):
        results_list = [["utility", results["utility"]]]
        for k in df_columns:
            results_men = results[k][0]
            results_women = results[k][1]
            results_list.append([f"{k}_m", results_men])
            results_list.append([f"{k}_w", results_women])
            results_list.append([f"{k}_diff", results_men - results_women])
            results_list.append([f"{k}_diff_abs", abs(results_men - results_women)])
            # limit results to small non-zero values to avoid division by zero errors calculating ratios
            if results_men == 0:
                results_men = 0.00001
            if results_women == 0:
                results_women = 0.00001
            results_list.append([f"{k}_ratio", min(results_men/results_women, results_women/results_men)])
        return results_list

    all_solutions = []
    avg_probabilities = []
    sim_counter = 0
    for sensitivity_value in sensitivity_vals:
        
        sim_counter += 1
        logging.info(f"\n\n-------\nRUNNING SIMULATION {sim_counter} OF {len(sensitivity_vals)}\n  --> calculating solution for {sensitivity_parameter}={sensitivity_value}\n-------\n")

        group_a, group_b, alpha, beta_group_a, beta_group_b = get_scenario(scenario, sensitivity_value, seed)

        # write the groups' probabilities to disk
        probabilities_filename = Path(path, "probabilities.csv")
        # if file does not exist write header
        if not probabilities_filename.is_file():
            pd.DataFrame({'probabilities':group_a.s,'group':'m', 'k':group_a.k}).to_csv(probabilities_filename, mode='a',index=False)
        else: # else it exists so append without writing the header
            pd.DataFrame({'probabilities':group_a.s,'group':'m', 'k':group_a.k}).to_csv(probabilities_filename, mode='a',index=False, header=False)
        pd.DataFrame({'probabilities':group_b.s,'group':'w', 'k':group_b.k}).to_csv(probabilities_filename, mode='a',index=False, header=False)
                
        avg_p_a, avg_p_b = group_a.s.mean()*100, group_b.s.mean()*100
        avg_probabilities.append([sensitivity_value, avg_p_a, "m"])
        avg_probabilities.append([sensitivity_value, avg_p_b, "w"])
        logging.info(f"average probabilities (in %) for groups a and b are {avg_p_a} and {avg_p_b}")

        """
        # hist plots moved to separate module
        group_a.plot_hist(nr_of_bins=2000, mylabel="group a")
        group_b.plot_hist(mycolor="orange", nr_of_bins=2000, mylabel="group b")
        plt.xlim(0,0.02)
        plt.ylim(0,100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}/distributions_{sensitivity_parameter}_{sensitivity_value}.pdf",bbox_inches='tight')
        #plt.savefig(f"{path}/distributions_{sensitivity_parameter.replace("k","")}_{sensitivity_value}.pdf",bbox_inches='tight')
        plt.clf()
        
        # make same plot on log scale
        group_a.plot_hist(nr_of_bins=20, mylabel="group a", log=True)
        group_b.plot_hist(mycolor="orange", nr_of_bins=20, mylabel="group b", log=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}/distributions_{sensitivity_parameter}_{sensitivity_value}_log.pdf",bbox_inches='tight')
        plt.clf()
        
        # make same plot on log scale ONLY FOR GROUP A
        group_a.plot_hist(nr_of_bins=20, mylabel="group a", log=True)
        plt.xlabel(r'$p$')
        plt.ylabel(r'frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}/distribution_log_group_A.pdf",bbox_inches='tight')
        plt.clf()
        """
        if limited_ressources == -1:
            # the fixed budget should correspond to the optimal budget for the unconstrained case
            limited_ressources_unconstrained = None
        else:
            limited_ressources_unconstrained = limited_ressources
        metrics_solution_no_fairness, group_a_threshold, group_b_threshold = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources_unconstrained, False, fixed_impressions=fixed_impressions, max_fairness_diff=None)
        logging.info(f"metrics_solution NO FAIRNESS: {metrics_solution_no_fairness}")
        results = unpack_results(metrics_solution_no_fairness, df_columns)
        for metric, value in results:
            all_solutions.append(["none", sensitivity_value, group_a_threshold, group_b_threshold, metric, value])

        if limited_ressources == -1:
            # the fixed budget should correspond to the optimal budget for the unconstrained case
            limited_ressources_constrained = sum(metrics_solution_no_fairness["nr_selected"])
        else:
            limited_ressources_constrained = limited_ressources
            
        for fairness_requirement in fairness_requirements:

            metrics_solution_full_fairness, group_a_threshold, group_b_threshold = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources_constrained, fairness_requirement, fixed_impressions=fixed_impressions, max_fairness_diff=None, fairness_margin=fairness_margin)
            logging.info(f" -- {fairness_requirement} --\nmetrics_solution FULL FAIRNESS: {metrics_solution_full_fairness}")
            logging.info(f"   --> thresholds for a/b: {group_a_threshold}/{group_b_threshold}")

            min_fairness = abs(metrics_solution_no_fairness[fairness_requirement][0] - metrics_solution_no_fairness[fairness_requirement][1])
            max_fairness = abs(metrics_solution_full_fairness[fairness_requirement][0] - metrics_solution_full_fairness[fairness_requirement][1])
            logging.info(f"for the criterion {fairness_requirement}, min and max fairness are {min_fairness} and {max_fairness}")
            results = unpack_results(metrics_solution_full_fairness, df_columns)
            for metric, value in results:
                all_solutions.append([fairness_requirement, sensitivity_value, group_a_threshold, group_b_threshold, metric, value])

    df_avg_p = pd.DataFrame(avg_probabilities, columns=["sensitivity_value", "average click probability (in %)", "group"])
    df = pd.DataFrame(all_solutions, columns=["criterion", "sensitivity_value", "group_a_threshold", "group_b_threshold", "metric", "value"])
    
    df_avg_p.to_csv(Path(path, "simulation_results_df_avg_p.csv"), index=False)
    df.to_csv(Path(path, "simulation_results.csv"), index=False)
    
    logging.info(f"--- done calculating all solutions and writing solutions to disk ---")


def run_limited_resources_simulation(scenario):
    
    fairness_requirement = False

    group_a, group_b, alpha, beta_group_a, beta_group_b = get_scenario(scenario)

    if scenario == 1:
        limited_ressources = 85000
    elif scenario == 2:
        limited_ressources = 200000
    elif scenario == 3:
        limited_ressources = 10000
    elif scenario == 4:
        limited_ressources = 32000
    elif scenario == 5:
        limited_ressources = 18000
    
    limits = list(range(int(limited_ressources/50),limited_ressources,int(limited_ressources/50)))
    limits.append(limited_ressources)

    df_a = pd.DataFrame()
    df_b = pd.DataFrame()

    for limit in limits:
        logging.info(f"")
        logging.info(f"running limit {limit} for scenario {scenario}")
        logging.info(f"")
        metrics_solution, _, _ = run_scenario(group_a, group_b, alpha, beta_group_a, beta_group_b, limit, fairness_requirement)
        #logging.info(f"metrics_solution: {metrics_solution}")

        solution_group_a = {k:v[0] for k,v in metrics_solution.items() if k != "utility"}
        solution_group_a["limit"] = limit
        solution_group_a["group"] = "group_a"
        solution_group_b = {k:v[1] for k,v in metrics_solution.items() if k != "utility"}
        solution_group_b["limit"] = limit
        solution_group_b["group"] = "group_b"
        
        df_a = pd.concat([df_a, pd.DataFrame([solution_group_a])], ignore_index=True)
        df_b = pd.concat([df_b, pd.DataFrame([solution_group_b])], ignore_index=True)
    
    sns.set_theme()

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10,10), sharex=True)
    fig.suptitle('Scenario ' + str(scenario) + ': results under limited resources without fairness constraints')
    fig.tight_layout()

    ax1.plot(df_a.limit, df_a.ppv - df_b.ppv, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a - group_b")
    ax1.set_title("ppv diff")
    ax1.legend()

    ax2.plot(df_a.limit, df_a["for"] - df_b["for"], color="blue", alpha=0.5, marker="o", markersize=3, label="group_a - group_b")
    ax2.set_title("for diff")

    ax3.plot(df_a.limit, df_a.tpr - df_b.tpr, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a - group_b")
    ax3.set_title("tpr diff")

    ax4.plot(df_a.limit, df_a.fpr - df_b.fpr, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a - group_b")
    ax4.set_title("fpr diff")

    ax5.plot(df_a.limit, df_a.acceptance_rate - df_b.acceptance_rate, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a - group_b")
    ax5.set_title("acceptance_rate diff")
    ax5.set_xlabel("max nr of selected individuals")

    ax6.plot(df_a.limit, df_a.nr_selected - df_b.nr_selected, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a - group_b")
    ax6.set_title("nr_selected diff")
    ax6.set_xlabel("max nr of selected individuals")

    plt.subplots_adjust(left=0.1, bottom=0.1, wspace=0.3, hspace=0.1)

    plt.savefig('figures/limited_resources_scenario_' + str(scenario) + '_diff.pdf')


    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10,10), sharex=True)
    fig.suptitle('Scenario ' + str(scenario) + ': results under limited resources without fairness constraints')
    fig.tight_layout()

    ax1.plot(df_a.limit, df_a.ppv, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a")
    ax1.plot(df_b.limit, df_b.ppv, color="red", alpha=0.5, marker="o", markersize=3, label="group_b")
    ax1.set_title("ppv")
    ax1.legend()

    ax2.plot(df_a.limit, df_a["for"], color="blue", alpha=0.5, marker="o", markersize=3, label="group_a")
    ax2.plot(df_b.limit, df_b["for"], color="red", alpha=0.5, marker="o", markersize=3, label="group_b")
    ax2.set_title("for")

    ax3.plot(df_a.limit, df_a.tpr, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a")
    ax3.plot(df_b.limit, df_b.tpr, color="red", alpha=0.5, marker="o", markersize=3, label="group_b")
    ax3.set_title("tpr")

    ax4.plot(df_a.limit, df_a.fpr, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a")
    ax4.plot(df_b.limit, df_b.fpr, color="red", alpha=0.5, marker="o", markersize=3, label="group_b")
    ax4.set_title("fpr")

    ax5.plot(df_a.limit, df_a.acceptance_rate, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a")
    ax5.plot(df_b.limit, df_b.acceptance_rate, color="red", alpha=0.5, marker="o", markersize=3, label="group_b")
    ax5.set_title("acceptance_rate")
    ax5.set_xlabel("max nr of selected individuals")

    ax6.plot(df_a.limit, df_a.nr_selected, color="blue", alpha=0.5, marker="o", markersize=3, label="group_a")
    ax6.plot(df_b.limit, df_b.nr_selected, color="red", alpha=0.5, marker="o", markersize=3, label="group_b")
    ax6.set_title("nr_selected")
    ax6.set_xlabel("max nr of selected individuals")

    plt.subplots_adjust(left=0.1, bottom=0.1, wspace=0.3, hspace=0.1)

    plt.savefig('figures/limited_resources_scenario_' + str(scenario) + '.pdf')


def plot_probability_distributions(path, scenario=4, sensitivity_vals=np.linspace(0.05, 0.005, num=10)):
    
    group_a, _, _, _, _ = get_scenario(scenario=scenario)
    plt.hist(group_a.s, bins=20, density=True, align='mid', color="blue", alpha=0.5, label=r"men", log=True, histtype='step', linestyle='dashed')
    palette = iter(COLOR_PALETTE)
    """
    for sensitivity_value in sensitivity_vals:
        color=next(palette)
        print(f"\n-------\ncalculating solution for {sensitivity_parameter}={sensitivity_value}\n-------\n")
        _, group_b, _, _, _ = get_scenario(scenario, sensitivity_value)
        
        plt.hist(group_b.s, bins=20, density=True, align='mid', color=color, alpha=0.5, label=rf"women ($k={sensitivity_value}$)", log=True, histtype='step')
    """

    # TODO: fix cause step somehow doesnt work for several histograms, maybe because of different bin edges...

    plt.xlim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(path, "probability_distributions_log.pdf"), bbox_inches='tight')
    plt.clf()


def directory_and_logfile_setup(path, scenario, seed):
    simulation_start_date = datetime.now().strftime("%Y %m %d %H:%M:%S")
    sanitized_date = simulation_start_date.replace(' ', '_').replace(':', '_')
    
    path = Path(path, f"scenario_{scenario}", f"simulation_seed_{seed}_{sanitized_date}")
    # create directory if it does not exist yet
    path.mkdir(parents=True, exist_ok=True)

    #logging.basicConfig(filename=f'{path}/logs-{simulation_start_date}.log', encoding='utf-8', format='%(levelname)s:%(message)s', level=logging.DEBUG) #only works from python 3.9
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(Path(path, 'logs.log'), 'w', 'utf-8')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    root_logger.addHandler(handler)
    logging.captureWarnings(True)

    # ignore matplotlib output because it's overwhelming
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith('matplotlib'):
            logger.disabled = True

    logging.info(f'new simulation started at: {simulation_start_date} (format: %Y %m %d %H:%M:%S)')
    logging.info(f'SEED: {seed}')

    return path
    


if __name__ == "__main__":

    # run simulation: python run_simulation.py -p results -s A --seed 123
    # get help with: python run_simulation.py -h

    parser = argparse.ArgumentParser(description='Simulating impressions for an online ad platform.', argument_default=argparse.SUPPRESS)

    parser.add_argument('-p', '--path', type=str, required=True, help='The name of the directory where the results should be stored.')
    parser.add_argument('-s', '--scenario', type=str, required=True, help='The name of the scenario to simulate.')
    parser.add_argument('--seed', type=int, required=False, help='The seed used for the random number generation..')

    args = parser.parse_args()
    args = vars(args)

    path = args.pop('path', None)
    scenario = args.pop('scenario', None)
    seed = args.pop('seed', None)

    print(f"  running scenario {scenario} with seed {seed}")

    # provide limited ressources as number of impressions to be achieved maximally
    # -1 means limited ressources is derived dynamically from the number of impressions in the unconstrained case
    # if fixed_impressions is True, the full budget must be spent
    limited_ressources, fixed_impressions = -1, True
    #limited_ressources, fixed_impressions = False, False
    #fairness_requirement = False
    #fairness_requirement = "ppv"
    # TODO: move to run script

    fairness_margin = 0.01

    #run_one_scenario(scenario=1, limited_ressources=limited_ressources, fairness_requirement=fairness_requirement)

    #run_limited_resources_simulation(scenario=1)
    #run_limited_resources_simulation(scenario=2)
    #run_limited_resources_simulation(scenario=3)
    #run_limited_resources_simulation(scenario=4)
    #run_limited_resources_simulation(scenario=5)

    #run_fairness_utility_tradeoff_simulation_for_degrees_of_fairness(scenario=5, limited_ressources=None, fairness_requirement="fpr")

    # load paramters for specified scenario

    sensitivity_parameter, sensitivity_vals, fairness_requirements = get_scenario_parameters(scenario)

    path = directory_and_logfile_setup(path, scenario, seed)
    
    run_fairness_utility_tradeoff_simulation_for_different_criteria(path=path, scenario=scenario, limited_ressources=limited_ressources, fixed_impressions=fixed_impressions, fairness_requirements=fairness_requirements, fairness_margin=fairness_margin, sensitivity_parameter=sensitivity_parameter, sensitivity_vals=sensitivity_vals, seed=seed)

    # plot proability distributions for all scenarios
    #plot_probability_distributions(path=path, scenario=3, sensitivity_vals=np.linspace(0.05, 0.005, num=10))