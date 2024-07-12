# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output, dash_table
from numpy import histogram
import plotly.express as px
import pandas as pd

from run_simulation import *

group_a = None
group_b = None

app = Dash(__name__)

app = Dash(__name__)

colors = {
    'background': '#191919',
    'text': '#7FDBFF'
}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})


def plot_histograms(group_a, group_b, group_a_threshold, group_b_threshold, nr_of_bins=2000):
    df_a = group_a.s.to_frame()
    df_a["group"] = "a"
    df_b = group_b.s.to_frame()
    df_b["group"] = "b"
    df = pd.concat([df_a, df_b])
    fig = px.histogram(df, x="probabilities", nbins=nr_of_bins, color="group",
                       opacity=0.3, barmode="overlay", width=1100, height=600)
    if colors is not None:
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text']
        )
    fig.add_vline(x=group_a_threshold, line_dash='dash', line_color='yellow')
    fig.add_vline(x=group_b_threshold, line_dash='dash', line_color='green')
    return fig


def get_optimal_unconstrained_decision_rules_OLD(alpha, beta_group_a, beta_group_b):

    print("Log: calculating solutions")
    group_a_threshold = optimal_unconstrained_decision_rule(
        alpha=alpha, beta=beta_group_a)
    print("Individuals of group a are shown the ad if their probability of clicking on it is >", group_a_threshold)
    group_b_threshold = optimal_unconstrained_decision_rule(
        alpha=alpha, beta=beta_group_b)
    print("Individuals of group b are shown the ad if their probability of clicking on it is >", group_b_threshold)

    return group_a_threshold, group_b_threshold


def calculate_results(group_a, group_b, group_a_threshold, group_b_threshold):

    decisions_a, probabilities_a = group_a.apply_decision_rule(
        group_a_threshold)
    decisions_b, probabilities_b = group_b.apply_decision_rule(
        group_b_threshold)

    # metrics_solution = calculate_metrics(metrics_prob, decisions_a, probabilities_a, decisions_b, probabilities_b)

    # print("metrics_solution:", metrics_solution)

    result_group_a = calculate_metrics_for_one_group(
        metrics_prob, decisions_a, probabilities_a, group="group a")
    result_group_a["decision_rule"] = group_a_threshold
    result_group_a["decision_rule_color"] = "yellow"
    result_group_a["BR"] = mean(group_a.s)
    result_group_b = calculate_metrics_for_one_group(
        metrics_prob, decisions_b, probabilities_b, group="group b")
    result_group_b["decision_rule"] = group_b_threshold
    result_group_b["BR"] = mean(group_b.s)
    result_group_b["decision_rule_color"] = "green"

    diffs = {
        "group": "DIFFS:",
        "ppv": result_group_a["ppv"] - result_group_b["ppv"],
        "for": result_group_a["for"] - result_group_b["for"],
        "tpr": result_group_a["tpr"] - result_group_b["tpr"],
        "fpr": result_group_a["fpr"] - result_group_b["fpr"],
        "acceptance_rate": result_group_a["acceptance_rate"] - result_group_b["acceptance_rate"],
        "nr_selected": result_group_a["nr_selected"] - result_group_b["nr_selected"],
        "decision_rule": result_group_a["decision_rule"] - result_group_b["decision_rule"],
        "BR": result_group_a["BR"] - result_group_b["BR"],
        "decision_rule_color": "-"
    }

    metrics_solutions = pd.DataFrame(
        [result_group_a, result_group_b, diffs]).round(decimals=5)

    table_results = metrics_solutions.to_dict('records')
    table_columns = [{'id': c, 'name': c} for c in metrics_solutions.columns]

    return table_results, table_columns


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='ONLINE ADS & GROUP FAIRNESS: SIMULATION',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.H2(children='Probability distribution group a', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Div(children=[

        html.Label(" mu_group_a "),
        dcc.Input(id="mu_group_a", type="number", value=0,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" sigma_group_a "),
        dcc.Input(id="sigma_group_a", type="number", value=1,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" n_group_a "),
        dcc.Input(id="n_group_a", type="number", value=100000,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" weight_group_a "),
        dcc.Input(id="weight_group_a", type="number", value=0.01,
                  debounce=True, style={'marginRight': '0px'}),

    ], style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.H2(children='Probability distribution group b', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.Div(children=[

        html.Label(" mu_group_b "),
        dcc.Input(id="mu_group_b", type="number", value=0,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" sigma_group_b "),
        dcc.Input(id="sigma_group_b", type="number", value=1,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" n_group_b "),
        dcc.Input(id="n_group_b", type="number", value=100000,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" weight_group_b "),
        dcc.Input(id="weight_group_b", type="number", value=0.01,
                  debounce=True, style={'marginRight': '0px'}),

    ], style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.H2(children='Utility functions', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.Div(children=[

        html.Label(" alpha "),
        dcc.Input(id="alpha", type="number", value=1,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" beta_group_a "),
        dcc.Input(id="beta_group_a", type="number", value=0.01,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" beta_group_b "),
        dcc.Input(id="beta_group_b", type="number", value=0.01,
                  debounce=True, style={'marginRight': '0px'}),

    ], style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.H2(children='Constraints', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    html.Div(children=[

        html.Label(" limited_ressources "),
        dcc.Input(id="limited_ressources", type="number", value=0,
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" fairness_requirement "),
        dcc.Input(id="fairness_requirement", type="text", value="",
                  debounce=True, style={'marginRight': '0px'}),
        html.Label(" max_fairness_diff "),
        dcc.Input(id="max_fairness_diff", type="number", value=0,
                  debounce=True, style={'marginRight': '0px'}),

    ], style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Br(),

    html.Div(id="summary_group_a", style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Div(id="summary_group_b", style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Div(children=[

        dcc.Graph(
            id='probability_histograms',
            style={'display': 'inline-block'}
        ),

        dash_table.DataTable(
            id="solutions_table",
            data=[],
            columns=[],

            style_header={
                'backgroundColor': colors['background'],
                'color': colors['text']
            },
            style_data={
                'backgroundColor': colors['background'],
                'color': colors['text'],
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_cell={
                'font-family': 'bold',
                'font-size': '26px',
                'text-align': 'center'
            },
            fill_width=False
        ),

        html.Span(
            id="utility",
            style={
                # 'textAlign': 'center',
                'color': colors['text']
            }
        )

    ],
        style={
        'textAlign': 'center',
        'color': colors['text']
    }
    ),



])


@app.callback(
    Output(component_id="summary_group_a", component_property='children'),
    Output(component_id="summary_group_b", component_property='children'),
    Output('probability_histograms', 'figure'),
    Output("solutions_table", "data"),
    Output('solutions_table', 'columns'),
    Output('utility', 'children'),
    Input("mu_group_a", "value"),
    Input("sigma_group_a", "value"),
    Input("n_group_a", "value"),
    Input("weight_group_a", "value"),
    Input("mu_group_b", "value"),
    Input("sigma_group_b", "value"),
    Input("n_group_b", "value"),
    Input("weight_group_b", "value"),
    Input("alpha", "value"),
    Input("beta_group_a", "value"),
    Input("beta_group_b", "value"),
    Input("limited_ressources", "value"),
    Input("fairness_requirement", "value"),
    Input("max_fairness_diff", "value")
)
def print_summary_of_inputs(mu_group_a, sigma_group_a, n_group_a, weight_group_a, mu_group_b, sigma_group_b, n_group_b, weight_group_b, alpha, beta_group_a, beta_group_b, limited_ressources, fairness_requirement, max_fairness_diff):
    global group_a
    global group_b
    print("")
    print("############################################     RUN NEW SIMULATION     ############################################")

    if group_a is None or [mu_group_a, sigma_group_a, n_group_a, weight_group_a] != [group_a.mu, group_a.sigma, group_a.n, group_a.weight]:
        print("Log: GROUP A")
        group_a = LogNormalDistribution(
            mu=mu_group_a, sigma=sigma_group_a, n=n_group_a, weight=weight_group_a)

    if group_b is None or [mu_group_b, sigma_group_b, n_group_b, weight_group_b] != [group_b.mu, group_b.sigma, group_b.n, group_b.weight]:
        print("Log: GROUP B")
        # group_b = LogNormalDistribution(mu=0, sigma=1, n=10000, weight= 0.005)
        group_b = LogNormalDistribution(
            mu=mu_group_b, sigma=sigma_group_b, n=n_group_b, weight=weight_group_b)

    summary_group_a = f"Summary of inputs [a]: mu_group_a={mu_group_a} / sigma_group_a={sigma_group_a} / n_group_a={n_group_a} / weight_group_a={weight_group_a} / alpha={alpha} / beta_group_a={beta_group_a} / beta_group_b={beta_group_b}"
    summary_group_b = f"Summary of inputs [b]: mu_group_b={mu_group_b} / sigma_group_b={sigma_group_b} / n_group_b={n_group_b} / weight_group_b={weight_group_b} / alpha={alpha} / beta_group_b={beta_group_b} / beta_group_b={beta_group_b}"

    if limited_ressources == 0:
        limited_ressources = None
    if fairness_requirement == "":
        fairness_requirement = False
    if max_fairness_diff == 0:
        max_fairness_diff = None

    print("limited_ressources:", limited_ressources, "fairness_requirement:",
          fairness_requirement, "max_fairness_diff:", max_fairness_diff)

    print("Log: calculating solutions")
    results, group_a_threshold, group_b_threshold = run_case(
        group_a, group_b, alpha, beta_group_a, beta_group_b, limited_ressources, fairness_requirement, max_fairness_diff=max_fairness_diff)

    utility = "Total utility = " + str(results["utility"])

    print("Log: plotting histogram & table")
    group_histogram = plot_histograms(
        group_a, group_b, group_a_threshold, group_b_threshold)

    table_results, table_columns = calculate_results(
        group_a, group_b, group_a_threshold, group_b_threshold)

    print("Log: plotting done")
    print("")

    return summary_group_a, summary_group_b, group_histogram, table_results, table_columns, utility


if __name__ == '__main__':
    app.run_server(debug=True)
