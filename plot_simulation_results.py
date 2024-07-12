import math
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
from scenario_config import get_scenario_parameters


def get_colors():
    color_indices = [0, 1, 2, 3, 4, 5, 7, 10, 13, 40]
    colors = iter([COLOR_PALETTE[index] for index in color_indices])
    return colors


def get_mitigation_label(criterion):
    if criterion in ["none", "Unconstrained"]:
        return "Unconstrained"
    if criterion == "acceptance_rate":
        return "Mitigation: statistical parity"
    elif criterion == "tpr":
        return "Mitigation: equality of opportunity"
    else:
        return f"Mitigation: {criterion.upper()} parity"


def get_mitigation_label2(criterion):
    if criterion == "none":
        return "Unconstrained"
    else:
        return "Mitigation: Any fairness constraint (Statistical, TPR, FPR,\nPPV, or FOR parity) has the same effect"


def get_fairness_metric_label(l):
    # check if l contains "_diff" or "_ratio"
    if "_diff" in l:
        l = l.replace("_diff", " difference")
    elif "_ratio" in l:
        l = l.replace("_ratio", " ratio")
    elif "_m" in l:
        l = l.replace("_m", " m")
    elif "_w" in l:
        l = l.replace("_w", " w")

    if "acceptance_rate" in l:
        return l.replace("acceptance_rate", "Acceptance rate")
    else:
        l = l.split(" ")
        return l[0].upper() + " " + l[1]


def get_fairness_constraint_label(criterion):
    if criterion == "none":
        return "Unconstrained"
    if criterion == "acceptance_rate":
        return "Statistical parity"
    elif criterion == "tpr":
        return "Equality of opportunity"
    elif criterion == "fpr":
        return "FPR parity"
    elif criterion == "ppv":
        return "Predictive parity"
    elif criterion == "for":
        return "FOR parity"
    else:
        return criterion


@timer
def plot_results(path, scenario, sensitivity_parameter, sensitivity_vals, generate_all_plots=False):

    df_avg_p = pd.DataFrame()
    df = pd.DataFrame()
    scores = pd.DataFrame()

    path = Path(path, f"scenario_{scenario}")

    # loop through all simulations of the specified scenario and aggregate the results
    for simulation in path.iterdir():
        if "simulation_seed_" in str(simulation):
            # read results from files of one simulation and add to dataframes
            df_avg_p = pd.concat([df_avg_p, pd.read_csv(Path(simulation, "simulation_results_df_avg_p.csv")).assign(
                simulation_name=simulation)], ignore_index=True)
            df = pd.concat([df, pd.read_csv(Path(simulation, "simulation_results.csv")).assign(
                simulation_name=simulation)], ignore_index=True)
            scores = pd.concat([scores, pd.read_csv(Path(simulation, "probabilities.csv")).assign(
                simulation_name=simulation)], ignore_index=True)

    plotting_start_date = datetime.now().strftime("%Y %m %d %H:%M:%S")
    sanitized_date = plotting_start_date.replace(' ', '_').replace(':', '_')

    path = Path(path, f"plots_{sanitized_date}")
    # create directory if it does not exist yet
    path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(
        Path(path, 'logs-plotting.log'), 'w', 'utf-8')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    root_logger.addHandler(handler)
    logging.captureWarnings(True)

    # ignore matplotlib output because it's overwhelming
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith('matplotlib'):
            logger.disabled = True

    logging.info(
        f'plotting started at: {plotting_start_date} (format: %Y %m %d %H:%M:%S)')

    # plot average click probabilities
    plt.figsize = (3, 3)
    set_plot_style()
    # plt.title(f'average click probabilities by group (scenario {scenario})')
    sns.lineplot(data=df_avg_p, x="sensitivity_value",
                 y="average click probability (in %)", hue="group", sort=True, errorbar=None)
    plt.xlabel(sensitivity_parameter)
    plt.xticks(sensitivity_vals)
    if "C" in scenario:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.ylabel(r"Average click probability in %")
    plt.legend(frameon=False)
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(
        Path(path, f"average_click_probabilities_{scenario}.pdf"), bbox_inches='tight')
    plt.clf()

    df.sort_values(by=['criterion', 'metric'])
    fairness_criteria = df["criterion"].unique()

    # plot fairness utility tradeoff for different fairness criteria
    set_plot_style()
    # plt.title(f'fairness-utility-tradeoff (scenario {scenario})')
    df_utility = df[df["metric"] == "utility"]
    df_utility["utility"] = df.apply(lambda x: x["value"]/df_utility.loc[((df_utility["sensitivity_value"] ==
                                     x["sensitivity_value"]) & (df_utility["simulation_name"] == x["simulation_name"])), "value"].max()*100, axis=1)
    sns.lineplot(data=df_utility, x="sensitivity_value", y="utility",
                 hue="criterion", style="criterion", markers=True, sort=True, alpha=0.5)
    plt.ylabel(r"utility in % of unconstrained utility")
    plt.xlabel(sensitivity_parameter)
    # plt.xticks(sensitivity_vals)
    if "C" in scenario:
        plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(Path(
        path, f"fairness_utility_tradeoff_relative_{scenario}.pdf"), bbox_inches='tight')
    plt.clf()

    if generate_all_plots:

        k_values_all = sorted(scores['k'].unique())
        if len(k_values_all) > 3:
            k_values_to_plot = []
            k_values_to_plot.append(k_values_all[0])
            k_values_to_plot.append(k_values_all[-1])
            k_values_to_plot.append(k_values_all[len(k_values_all) // 2])
            k_values_to_plot.remove(
                list(scores.loc[scores['group'] == 'm', ['k']].iloc[0])[0])
        else:
            k_values_to_plot = k_values_all

        scores_to_plot = scores[(scores['group'] == 'm') | (
            scores['k'].isin(k_values_to_plot))]
        # plot histogram of probabilities
        scores_to_plot["Group"] = scores_to_plot.apply(
            lambda x: fr"{x['group']} ($k_{x['group']}={x['k']}$)", axis=1)

        # make same plot on log scale
        set_plot_style()
        sns.histplot(scores_to_plot, x="probabilities", hue="Group",
                     element="step", stat="density", common_norm=True, bins=10, fill=False)
        plt.yscale('log')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(
            Path(path, f"probability_distributions_log_{scenario}.pdf"), bbox_inches='tight')
        plt.clf()

        # for each fairness criteria, plot the tradeoff with other types of fairness
        metrics_detailed = [x for x in df.metric.unique() if ("_m" in x or "_w" in x) and not any(
            forbidden in x for forbidden in ["_abs", "nr_selected"])]

        fairness_metric_types_and_labels = [
            ("_diff", "group differences"), ("_ratio", "min group ratios")]

        for metric_type, metric_label in fairness_metric_types_and_labels:
            # plot results for 2 types of fairness metrics: difference and minimum ratio
            metrics = [x for x in df.metric.unique() if (metric_type in x and not any(
                forbidden in x for forbidden in ["_abs", "nr_selected"]))]
            palette = get_colors()
            color_palette_fairness = {
                k: next(palette) for k in sorted(metrics)}
            color_palette_fairness_detailed, markers = {}, {}
            for m in sorted(metrics_detailed):
                color_palette_fairness_detailed[m] = color_palette_fairness[m.strip(
                    "_m").strip("_w")+metric_type]
                if m[-1] == 'm':
                    markers[m] = 'v'
                elif m[-1] == 'w':
                    markers[m] = 'o'

            nr_of_scenarios = len(fairness_criteria)
            set_plot_style()
            ncols = 2
            nrows = math.ceil(nr_of_scenarios/ncols)
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(10, 9))
            # delete subfigures if too many
            [fig.delaxes(axes[nrows-1][ncols-1-i])
             for i in range(ncols * nrows-nr_of_scenarios)]

            for criterion, ax in zip(fairness_criteria, axes.ravel()):
                # plot difference in metrics between both groups
                ax.title.set_text(f"{get_mitigation_label(criterion)}")
                # select only difference metrics
                sns.lineplot(ax=ax, data=df[df["metric"].isin(metrics) & (df["criterion"] == criterion)], x="sensitivity_value",
                             y="value", hue="metric", style="metric", markers=True, sort=True, alpha=0.5, palette=color_palette_fairness)
                ax.set(ylabel=metric_label)
                ax.get_legend().remove()
                ax.set(xlabel=sensitivity_parameter)
                ax.set(xticks=sensitivity_vals)
                ax.set_xticklabels([str(round(s, 3))
                                   for s in sensitivity_vals], rotation=90)

            # fig.supylabel(r"parity metric differences (group m $-$ group w)")
            if "C" in scenario:
                ax.invert_xaxis()
            elif 'D' in scenario:
                # only plot the legend for Scenario D
                handles, labels = ax.get_legend_handles_labels()
                nr_of_legen_columns = len(handles)
                labels = [get_fairness_metric_label(l) for l in labels]
                plt.figlegend(handles=handles, labels=labels, title="Fairness metric", title_fontproperties={
                              'weight': 'bold'}, loc='lower center', ncol=nr_of_legen_columns, bbox_to_anchor=(0.5, -0.04), labelspacing=0.)
            plt.tight_layout()
            plt.savefig(
                Path(path, f"fairness{metric_type}_{scenario}.pdf"), bbox_inches='tight')
            plt.clf()

        set_plot_style()
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(10, 9))
        # delete subfigures if too many
        [fig.delaxes(axes[nrows-1][ncols-1-i])
         for i in range(ncols * nrows-nr_of_scenarios)]

        # plot detailed metrics for both groups
        for criterion, ax in zip(fairness_criteria, axes.ravel()):
            ax.title.set_text(f"{get_mitigation_label(criterion)}")
            # select only group-specific metrics
            sns.lineplot(ax=ax, data=df[df["metric"].isin(metrics_detailed) & (df["criterion"] == criterion)], x="sensitivity_value",
                         y="value", hue="metric", style="metric", markers=markers, sort=True, alpha=0.5, palette=color_palette_fairness_detailed)
            ax.set(ylabel=r"group values")
            ax.get_legend().remove()
            ax.set(xlabel=sensitivity_parameter)
            ax.set(xticks=sensitivity_vals)
            ax.set_xticklabels([str(round(s, 3))
                               for s in sensitivity_vals], rotation=90)

        # fig.supylabel(r"group-specific metrics")
        if "C" in scenario:
            ax.invert_xaxis()
        elif 'D' in scenario:
            # only plot the legend for Scenario D
            handles, labels = ax.get_legend_handles_labels()
            nr_of_legen_columns = len(handles)
            labels = [get_fairness_metric_label(l) for l in labels]
            plt.figlegend(handles=handles, labels=labels, title="Group-specific metric", title_fontproperties={
                          'weight': 'bold'}, loc='lower center', ncol=nr_of_legen_columns, bbox_to_anchor=(0.5, -0.04), labelspacing=0.)
        plt.tight_layout()
        plt.savefig(
            Path(path, f"fairness_{scenario}_detailed.pdf"), bbox_inches='tight')
        plt.clf()

        # for the paper, redo the same plots again for scenarios A and B but with some slight changes
        if ("A" in scenario) or ("B" in scenario):
            if "A" in scenario:
                # for scenario A, only plot the unconstrained case, because even with mitigations, the results remain unchanged
                fairness_criteria = fairness_criteria[:1]
                figsize = (5, 3.25)
            if "B" in scenario:
                # for scenario A, only plot the unconstrained case and the one under statistical parity, because the results remain unchanged for any other fairness constraint
                fairness_criteria = fairness_criteria[:2]
                figsize = (10, 3.25)

            for metric_type, metric_label in fairness_metric_types_and_labels:
                # plot results for 2 types of fairness metrics: difference and minimum ratio
                metrics = [x for x in df.metric.unique() if (metric_type in x and not any(
                    forbidden in x for forbidden in ["_abs", "nr_selected"]))]
                palette = get_colors()
                color_palette_fairness = {
                    k: next(palette) for k in sorted(metrics)}
                color_palette_fairness_detailed, markers = {}, {}
                for m in sorted(metrics_detailed):
                    color_palette_fairness_detailed[m] = color_palette_fairness[m.strip(
                        "_m").strip("_w")+metric_type]
                    if m[-1] == 'm':
                        markers[m] = 'v'
                    elif m[-1] == 'w':
                        markers[m] = 'o'

                nr_of_scenarios = len(fairness_criteria)
                set_plot_style()
                ncols = min(2, nr_of_scenarios)
                nrows = math.ceil(nr_of_scenarios/ncols)
                fig, axes = plt.subplots(
                    nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
                # delete subfigures if too many
                [fig.delaxes(axes[nrows-1][ncols-1-i])
                 for i in range(ncols * nrows-nr_of_scenarios)]

                if "B" in scenario:
                    axes = axes.ravel()
                else:
                    axes = [axes]

                for criterion, ax in zip(fairness_criteria, axes):
                    # plot difference in metrics between both groups
                    ax.title.set_text(f"{get_mitigation_label2(criterion)}")
                    # select only difference metrics
                    sns.lineplot(ax=ax, data=df[df["metric"].isin(metrics) & (df["criterion"] == criterion)], x="sensitivity_value",
                                 y="value", hue="metric", style="metric", markers=True, sort=True, alpha=0.5, palette=color_palette_fairness)
                    ax.set(ylabel=metric_label)
                    ax.get_legend().remove()
                    ax.set(xlabel=sensitivity_parameter)
                    ax.set(xticks=sensitivity_vals)
                    ax.set_xticklabels([str(round(s, 3))
                                       for s in sensitivity_vals], rotation=90)
                    # ax.set_ylim([-0.35, 0.4])

                # fig.supylabel(r"parity metric differences (group m $-$ group w)")
                # handles, labels = ax.get_legend_handles_labels()
                # nr_of_legen_columns = len(handles)
                # plt.figlegend(handles=handles, loc='lower center', ncol=nr_of_legen_columns, bbox_to_anchor=(0.5, -0.04), labelspacing=0.)
                plt.tight_layout()
                plt.savefig(
                    Path(path, f"fairness{metric_type}_{scenario}_paper.pdf"), bbox_inches='tight')
                plt.clf()

            set_plot_style()
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize)
            # delete subfigures if too many
            [fig.delaxes(axes[nrows-1][ncols-1-i])
             for i in range(ncols * nrows-nr_of_scenarios)]

            if "B" in scenario:
                axes = axes.ravel()
            else:
                axes = [axes]

            # plot detailed metrics for both groups
            for criterion, ax in zip(fairness_criteria, axes):
                ax.title.set_text(f"{get_mitigation_label2(criterion)}")
                # select only group-specific metrics
                sns.lineplot(ax=ax, data=df[df["metric"].isin(metrics_detailed) & (df["criterion"] == criterion)], x="sensitivity_value",
                             y="value", hue="metric", style="metric", markers=markers, sort=True, alpha=0.5, palette=color_palette_fairness_detailed)
                ax.set(ylabel=r"group values")
                ax.get_legend().remove()
                ax.set(xlabel=sensitivity_parameter)
                ax.set(xticks=sensitivity_vals)
                ax.set_xticklabels([str(round(s, 3))
                                   for s in sensitivity_vals], rotation=90)

            # fig.supylabel(r"group-specific metrics")
            # handles, labels = ax.get_legend_handles_labels()
            # nr_of_legen_columns = len(handles)
            # plt.figlegend(handles=handles, loc='lower center', ncol=nr_of_legen_columns, bbox_to_anchor=(0.5, -0.03), labelspacing=0.)
            plt.tight_layout()
            plt.savefig(
                Path(path, f"fairness_{scenario}_detailed_paper.pdf"), bbox_inches='tight')
            plt.clf()

        # palette = iter(COLOR_PALETTE[7:])
        # color_palette_fairness_detailed, markers = {"m": next(pal), "w": }, {"m": 'v', "w": 'o'}
        markers = {"m": 'v', "w": 'o'}
        for criterion in ["acceptance_rate", "tpr"]:
            df_leveling_down = df[(df["criterion"] == criterion) & (
                df["metric"].isin([f"{criterion}_m", f"{criterion}_w"]))]
            df_leveling_down_reference = df[(df["criterion"] == 'none') & (
                df["metric"].isin([f"{criterion}_m", f"{criterion}_w"]))]

            if df_leveling_down.shape[0] == 0:
                # continue if this fairness criterion has not been simulated
                continue

            df_leveling_down[f"group"] = df_leveling_down.apply(
                lambda x: x["metric"][-1:], axis=1)
            df_leveling_down[f"{criterion}_leveling_down"] = df_leveling_down.apply(lambda x: x["value"]-df_leveling_down_reference.loc[(
                df['simulation_name'] == x['simulation_name']) & (df['sensitivity_value'] == x['sensitivity_value']) & (df['metric'] == x['metric']), 'value'].values[0], axis=1)

            # plot leveling down
            set_plot_style()
            # plot the utility results
            plt.title(f'Scenario {scenario}')
            # select only difference metrics
            sns.lineplot(data=df_leveling_down, x="sensitivity_value",
                         y=f"{criterion}_leveling_down", hue="group", style="group", markers=markers, sort=True, alpha=0.5)
            plt.axhline(0, color='black', linestyle='dashed', alpha=0.5)
            plt.ylabel(r"$V(d^{\ast}_c) - V(d^{\ast})$")
            plt.xlabel(sensitivity_parameter)
            plt.xticks(sensitivity_vals)
            # rotate the xtick labels
            plt.xticks(rotation=90)
            if "C" in scenario:
                plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(
                Path(path, f"leveling_down_{criterion}_{scenario}.pdf"), bbox_inches='tight')
            plt.clf()

    logging.info(f"\n--- PLOTTING DONE FOR SCENARIO {scenario} ---\n")


@timer
def plot_all_fairness_accuracy_tradeoffs(path, scenarios):

    plotting_start_date = datetime.now().strftime("%Y %m %d %H:%M:%S")
    sanitized_date = plotting_start_date.replace(' ', '_').replace(':', '_')

    path = Path(path)

    # set up figure
    scenario_dirs = [dir for dir in path.iterdir() if "scenario_" in str(
        dir) and str(dir).rsplit('_', 1)[-1] in scenarios]
    nr_of_scenarios = len(scenario_dirs)

    set_plot_style()
    ncols = 4
    nrows = math.ceil(nr_of_scenarios/ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=False, sharey=True, figsize=(11, 3))
    # delete subfigures if too many
    [fig.delaxes(axes[nrows-1][ncols-1-i])
     for i in range(ncols * nrows-nr_of_scenarios)]

    # make code palette for the different metrics using the same colors as for the criteria
    colors = list(get_colors())
    palette = {
        'none': colors[9],
        'acceptance_rate': colors[0],
        'tpr': colors[4],
        'ppv': colors[3],
        'for': colors[1],
        'fpr': colors[2],
    }

    # for scenario_dir in path.iterdir():
    for (path, scenario), ax in zip(sorted([(dir, str(dir).rsplit('_', 1)[-1]) for dir in scenario_dirs], key=lambda tup: tup[1]), axes.ravel()):
        # loop through all scenarios
        print(f"plotting scenario {scenario}")
        sensitivity_parameter, sensitivity_vals, _ = get_scenario_parameters(
            scenario)

        df_avg_p = pd.DataFrame()
        df = pd.DataFrame()
        scores = pd.DataFrame()

        # loop through all simulations of the specified scenario and aggregate the results
        for simulation in path.iterdir():
            if "simulation_seed_" in str(simulation):
                # read results from files of one simulation and add to dataframes
                df_avg_p = pd.concat([df_avg_p, pd.read_csv(Path(simulation, "simulation_results_df_avg_p.csv")).assign(
                    simulation_name=simulation)], ignore_index=True)
                df = pd.concat([df, pd.read_csv(Path(simulation, "simulation_results.csv")).assign(
                    simulation_name=simulation)], ignore_index=True)
                scores = pd.concat([scores, pd.read_csv(Path(simulation, "probabilities.csv")).assign(
                    simulation_name=simulation)], ignore_index=True)

        # plt.title(f'fairness-utility-tradeoff (scenario {scenario})')
        df_utility = df[df["metric"] == "utility"]
        df_utility["utility"] = df.apply(lambda x: x["value"]/df_utility.loc[((df_utility["sensitivity_value"] ==
                                         x["sensitivity_value"]) & (df_utility["simulation_name"] == x["simulation_name"])), "value"].max()*100, axis=1)
        sns.lineplot(ax=ax, data=df_utility, x="sensitivity_value", y="utility", hue="criterion", style="criterion",
                     markers=True, sort=True, alpha=0.5, palette=palette, legend=True if 'A' in scenario else False)
        # ax.title.set_text(f"Scenario {scenario}")
        # ax.set(ylabel=r"utility in % of max. utility")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, ls=':')

        ax.set(ylabel=r"")
        # ax.get_legend().remove()
        ax.set(xlabel=sensitivity_parameter)
        # ax.set(xticks=sensitivity_vals)
        # ax.set_xticklabels([str(round(s, 3)) for s in sensitivity_vals], rotation = 90)

        if scenario == 'C':
            ax.set_xticks([0.01, .02, .03, .04, .05])
            ax.set_xticklabels(['.01', '.02', '.03', '.04', '.05'])
            ax.invert_xaxis()

        elif scenario == 'C2':
            ax.set_xticks([0.01, .1, .2, .3, .4])
            ax.set_xticklabels(['.01', '.1', '.2', '.3', '.4'])
            ax.invert_xaxis()

        else:
            ax.set_xticks([0, .2, .4, .6, .8, 1])
            ax.set_xticklabels(['0', '.2', '.4', '.6', '.8', '1'])

        s = f"Scenario {scenario}"
        ax.annotate(text=s, xy=(0.2, 0.05), xycoords='axes fraction')

        if 'A' in scenario:

            handles, labels = ax.get_legend_handles_labels()
            nr_of_legen_columns = 1
            labels = [get_fairness_constraint_label(l) for l in labels]

            ax.legend(handles=handles, labels=[l.replace('of ', 'of\n') for l in labels],
                      ncol=1,
                      bbox_to_anchor=(4.3, 1),
                      ncols=1, mode="expand", borderaxespad=0.,
                      frameon=False, labelspacing=2)

    axes[0].set_ylabel("utility in % of\nunconstrained utility")
    # fig.supylabel(r"utility in % of unconstrained utility")

    # #handles = [Line2D([0], [0], marker=markers[label], label=label, color=color) for label, color in pal.items()]
    # handles, labels = ax.get_legend_handles_labels()
    # nr_of_legen_columns = len(handles)
    # labels = [get_fairness_constraint_label(l) for l in labels]
    # plt.figlegend(handles=handles, labels=labels, title="Fairness constraint", title_fontproperties={'weight':'bold'}, loc='lower center', ncol=nr_of_legen_columns, bbox_to_anchor=(0.5, -0.06), labelspacing=0.)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0, right=0.8)
    # set siez similar to plt.set_size_inches(11, 3)
    plt.gcf().set_size_inches(12, 3)
    # create directory if it does not exist yet
    plots_directory = f"plots_{sanitized_date}"
    fig_path = Path(path.parent, plots_directory)
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(
        fig_path, f"fairness_utility_tradeoff_relative_scenarios_{'_'.join(scenarios)}3.pdf"), bbox_inches='tight')
    plt.clf()

    return plots_directory


@timer
def plot_leveling_down_effects(path, scenarios, plots_directory):

    path = Path(path)

    # set up figure
    scenario_dirs = [dir for dir in path.iterdir() if "scenario_" in str(
        dir) and str(dir).rsplit('_', 1)[-1] in scenarios]
    nr_of_scenarios = len(scenario_dirs)

    set_plot_style()
    ncols = 2
    nrows = math.ceil(nr_of_scenarios/ncols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(9, 3))
    # delete subfigures if too many
    [fig.delaxes(axes[nrows-1][ncols-1-i])
     for i in range(ncols * nrows-nr_of_scenarios)]

    # make code palette for the different metrics using the same colors as for the criteria
    colors = list(get_colors())
    palette = {
        'none': colors[9],
        'acceptance_rate': colors[0],
        'tpr': colors[4],
        'ppv': colors[3],
        'for': colors[1],
        'fpr': colors[2],
    }

    # for scenario_dir in path.iterdir():
    for (path, scenario), ax in zip(sorted([(dir, str(dir).rsplit('_', 1)[-1]) for dir in scenario_dirs], key=lambda tup: tup[1]), axes.ravel()):
        # loop through all scenarios
        sensitivity_parameter, sensitivity_vals, _ = get_scenario_parameters(
            scenario)

        df_avg_p = pd.DataFrame()
        df = pd.DataFrame()
        scores = pd.DataFrame()

        # loop through all simulations of the specified scenario and aggregate the results
        for simulation in path.iterdir():
            if "simulation_seed_" in str(simulation):
                # read results from files of one simulation and add to dataframes
                df_avg_p = pd.concat([df_avg_p, pd.read_csv(Path(simulation, "simulation_results_df_avg_p.csv")).assign(
                    simulation_name=simulation)], ignore_index=True)
                df = pd.concat([df, pd.read_csv(Path(simulation, "simulation_results.csv")).assign(
                    simulation_name=simulation)], ignore_index=True)
                scores = pd.concat([scores, pd.read_csv(Path(simulation, "probabilities.csv")).assign(
                    simulation_name=simulation)], ignore_index=True)

        # palette = iter(COLOR_PALETTE[7:])
        # color_palette_fairness_detailed, markers = {"m": next(pal), "w": }, {"m": 'v', "w": 'o'}
        markers = {"m": 'v', "w": 'o'}
        criterion = "acceptance_rate"

        df_leveling_down = df[(df["criterion"] == criterion) & (
            df["metric"].isin([f"{criterion}_m", f"{criterion}_w"]))]
        df_leveling_down_reference = df[(df["criterion"] == 'none') & (
            df["metric"].isin([f"{criterion}_m", f"{criterion}_w"]))]

        df_leveling_down[f"group"] = df_leveling_down.apply(
            lambda x: x["metric"][-1:], axis=1)
        df_leveling_down[f"{criterion}_leveling_down"] = df_leveling_down.apply(lambda x: x["value"]-df_leveling_down_reference.loc[(
            df['simulation_name'] == x['simulation_name']) & (df['sensitivity_value'] == x['sensitivity_value']) & (df['metric'] == x['metric']), 'value'].values[0], axis=1)

        # plot user utility to investigate leveling down effects
        ax.title.set_text(f"Scenario {scenario}")
        # select only leveling down metric in terms of acceptance rate
        sns.lineplot(ax=ax, data=df_leveling_down, x="sensitivity_value",
                     y=f"{criterion}_leveling_down", hue="group", style="group", markers=markers, sort=True, alpha=0.5)
        ax.axhline(0, color='black', linestyle='dashed', alpha=0.5)
        ax.set(ylabel=r"")
        ax.set(xlabel=sensitivity_parameter)
        ax.set(xticks=sensitivity_vals)
        ax.set_xticklabels([str(round(s, 3))
                           for s in sensitivity_vals], rotation=90)
        if scenario == 'D' or 'fixedImpressions' in str(path):
            ax.get_legend().remove()

    fig.supylabel(r"$V(d^{\ast}_c) - V(d^{\ast})$")
    set_plot_style()
    plt.tight_layout()

    fig_path = Path(path.parent, plots_directory)
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(fig_path, f"leveling_down.pdf"), bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":

    # generate plots for a specific scenario: python plot_simulation_results.py -p results -s A
    # generate plot with all fairness-accuracy-tradeoff of a list of scenarios: python plot_simulation_results.py -p results -sc A B C D
    # get help with: python plot_simulation_results.py -h

    parser = argparse.ArgumentParser(
        description='Simulating impressions for an online ad platform.', argument_default=argparse.SUPPRESS)

    parser.add_argument('-p', '--path', type=str, required=False,
                        help='The name of the directory where the results should be stored.')
    parser.add_argument('-s', '--scenario', type=str, required=False,
                        help='The name of the simulated scenario for which the plots should be generated.')
    parser.add_argument('-sc', '--scenarios', nargs='+', default=[], required=False,
                        help='The name of the simulated scenario that should be included in a combined fairness-accuracy-tradeoff plot.')

    args = parser.parse_args()
    args = vars(args)

    path = args.pop('path', None)
    scenario = args.pop('scenario', None)
    scenarios = args.pop('scenarios', [])
    path = args.pop('path', "results")
    scenarios = args.pop('scenario', ["A2", "B2", "C2", "D2"])

    if scenario:
        # generate plots for a single scenario
        print(f"  plotting results for scenario {scenario}")
        # load paramters for specified scenario
        sensitivity_parameter, sensitivity_vals, fairness_requirements = get_scenario_parameters(
            scenario)
        # plot_results(path, scenario, sensitivity_parameter, sensitivity_vals)
        # plotting all plots is slow
        plot_results(path, scenario, sensitivity_parameter,
                     sensitivity_vals, generate_all_plots=True)

    if scenarios != []:
        # generate a combined fairness-accuracy-tradeoff plot for a list of scenarios
        print(
            f"  generate a combined fairness-accuracy-tradeoff plot for scenarios {scenarios}")
        plots_directory = plot_all_fairness_accuracy_tradeoffs(path, scenarios)
        print(f"  generate the leveling down plots for scenarios B and D")
        if ('B' in scenarios or 'B2' in scenarios) and ('D' in scenarios or 'D2' in scenarios):
            plot_leveling_down_effects(path, ['B', 'D'], plots_directory)
