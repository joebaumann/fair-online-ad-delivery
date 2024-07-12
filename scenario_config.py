import numpy as np
import pandas as pd


def get_scenario_parameters(scenario):
    """
    Scenarios A,B,C,D are presented in the paper's experiment section.
    Scenarios A2,B2,C2,D2 are presented in the paper's appendix to show the larger effect sizes for situations with generally higher likelihoods of clicking on the high-stake ad (i.e., larger values for k_m).
    """

    fairness_requirements = ["acceptance_rate", "ppv", "for", "tpr", "fpr"]
    # fairness_requirements=["acceptance_rate", "ppv"]

    if scenario == "A":
        sensitivity_parameter, sensitivity_vals = r"$\alpha$", np.linspace(
            0.03, 1, num=10)
    elif scenario == "B":
        sensitivity_parameter, sensitivity_vals = r"$\beta_w$", np.linspace(
            0.03, 1, num=10)
    elif scenario == "C":
        sensitivity_parameter, sensitivity_vals = r"$k_w$", np.linspace(
            0.05, 0.005, num=10)
    elif scenario == "D":
        # just as scenario 3 but with distribution difference
        sensitivity_parameter, sensitivity_vals = r"$\beta_w$", np.linspace(
            0.03, 1, num=10)

    elif scenario == "A2":
        sensitivity_parameter, sensitivity_vals = r"$\alpha$", np.linspace(
            0.03, 1, num=10)
    elif scenario == "B2":
        sensitivity_parameter, sensitivity_vals = r"$\beta_w$", np.linspace(
            0.03, 1, num=10)
    elif scenario == "C2":
        sensitivity_parameter, sensitivity_vals = r"$k_w$", np.linspace(
            0.4, 0.01, num=10)
    elif scenario == "D2":
        # just as scenario 3 but with distribution difference
        sensitivity_parameter, sensitivity_vals = r"$\beta_w$", np.linspace(
            0.03, 1, num=10)

    return sensitivity_parameter, sensitivity_vals, fairness_requirements


def get_simulation_parameters(scenario, sensitivity_value=None):
    """
    Scenarios A,B,C,D are presented in the paper's experiment section.
    Scenarios A2,B2,C2,D2 are presented in the paper's appendix to show the larger effect sizes for situations with generally higher likelihoods of clicking on the high-stake ad (i.e., larger values for k_m).
    """

    n_m = 1000
    n_w = 1000
    k_m = 0.05
    k_w = 0.05
    beta_m = 0.03
    beta_w = 0.03
    alpha = 0.2

    if scenario == "A":
        alpha = sensitivity_value

    elif scenario == "B":
        beta_w = sensitivity_value

    elif scenario == "C":
        k_w = sensitivity_value
        alpha = 1

    elif scenario == "D":
        k_w = 0.01
        beta_w = sensitivity_value

    elif scenario == "A2":
        k_m = 0.4
        k_w = 0.4
        alpha = sensitivity_value

    elif scenario == "B2":
        k_m = 0.4
        k_w = 0.4
        beta_w = sensitivity_value

    elif scenario == "C2":
        k_m = 0.4
        alpha = 1
        k_w = sensitivity_value

    elif scenario == "D2":
        k_m = 0.4
        k_w = 0.01
        beta_w = sensitivity_value

    return alpha, beta_m, beta_w, k_m, k_w, n_m, n_w


def get_latex_table(scenarios):
    data = []
    for scenario in scenarios:
        sensitivity_parameter, sensitivity_vals, _ = get_scenario_parameters(
            scenario)
        alpha, beta_m, beta_w, k_m, k_w, n_m, n_w = get_simulation_parameters(
            scenario, sensitivity_vals[0])
        if sensitivity_parameter == r"$\alpha$":
            alpha = str(sensitivity_vals)
        elif sensitivity_parameter == r"$\beta_w$":
            beta_w = str(sensitivity_vals)
        elif sensitivity_parameter == r"$k_w$":
            k_w = str(sensitivity_vals)

        row = {
            "Scenario": scenario,
            r"$\alpha$": alpha,
            r"$k_m$": k_m,
            r"$beta_m$": beta_m,
            r"$k_w$": k_w,
            r"$\beta_w$": beta_w,
            r"$n_m$": n_m,
            r"$n_w$": n_w,
        }
        data.append(row)

    df = pd.DataFrame(data)
    df = df[[r"$\alpha$", r"$k_m$", r"$beta_m$", r"$k_w$", r"$\beta_w$"]]
    print(df)
    print(df.to_latex(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    # generate df with all parameters of all scenarios and output as latex table
    # scenarios = ["A", "B", "C", "D"]
    scenarios = ["A2", "B2", "C2", "D2"]
    get_latex_table(scenarios)
