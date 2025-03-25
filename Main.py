import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, no_update
import plotly.express as px
import pandas as pd
import plotly.io as pio
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import dash_ag_grid as dag
import datetime
from dash_bootstrap_templates import load_figure_template
import numpy as np
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.ar_model import AutoReg
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn import  linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import holidays

#INITIALIZING GLOBAL VARIABLES
active_index_power_analysis = 0
active_index_weather_analysis = 0
active_index_filter_methods = 0
active_index_wrapper_methods = 0
active_index_autoregression = 0
active_index_linear = 0
active_index_vector = 0
active_index_tree = 0
active_index_forest = 0
active_index_neural = 0
active_index_validation = 0
selected_buildings = []

#LOADING DATA
dataframe_cleaned_data = pd.read_csv('All_data_cleaned_South.csv')
dataframe_cleaned_data['Date + Time'] = pd.to_datetime(dataframe_cleaned_data['Date + Time'])
dataframe_cleaned_data = dataframe_cleaned_data.set_index('Date + Time', drop=True)
start_date = datetime.date(2017, 1, 1)
end_date = datetime.date(2018, 12, 31)
date_range = pd.date_range(start_date, end_date, freq='W')

def add_features(dataframe):
    dataframe_features = dataframe.copy()
    dataframe_features['Weekday'] = dataframe_features.index.day
    dataframe_features['Month'] = dataframe_features.index.month
    dataframe_features['Hour'] = dataframe_features.index.hour
    dataframe_features['Weekend'] = dataframe_features.index.weekday >= 5
    dataframe_features['Weekend'] = dataframe_features['Weekend'].astype(int)
    dataframe_features['Weekend_or_holiday'] = dataframe_features['Weekend'] | dataframe_features['Holiday']
    dataframe_features = dataframe_features.drop(columns=['Holiday', 'Weekend'])
    dataframe_features['Previous_hour'] = dataframe_features['Power [kW]'].shift(1)
    dataframe_features['Previous_day'] = dataframe_features['Power [kW]'].shift(24)
    dataframe_features['Previous_week'] = dataframe_features['Power [kW]'].shift(24 * 7)
    dataframe_features = dataframe_features.dropna()
    amp = 1 / 24
    time = dataframe_features['Hour'].values
    dataframe_features['Sin_hour'] = 10 * np.sin(2 * np.pi * amp * time - 8)
    dataframe_features['HDH'] = np.maximum(0, -dataframe_features['Temperature [°C]'] + 21)
    dataframe_features['logtemp'] = np.log(dataframe_features['Temperature [°C]'])
    return dataframe_features

dataframe_features = add_features(dataframe_cleaned_data)
all_features = dataframe_features.columns[1:]
selected_features = []
Z_all_features = dataframe_features.values
Y_all_features  = Z_all_features[:,0] #Target Variable
X_all_features = Z_all_features[:,1:] #Features

dataframe_selected_features = None
X_selected_features = None
Y_selected_features = None
performance_df = pd.DataFrame()
AR_results = pd.DataFrame()
LR_results = pd.DataFrame()
SVR_results = pd.DataFrame()
DT_results = pd.DataFrame()
RF_results = pd.DataFrame()
NN_results = pd.DataFrame()
trained_models = []
selected_model = str()
dataframe_test_data = pd.DataFrame()
coef_AR = None
model_LR = None
model_SVR = None
model_DT = None
model_RF = None
model_NN = None
scaler_SVR = None
scaler_NN = None

#INITIALIZING THE APP WITH CUSTOM THEME SUPERHERO AND IMPORTING BOOTSTRAP ICONS
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.SUPERHERO, dbc.icons.BOOTSTRAP,"https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css",
], suppress_callback_exceptions=True)
app.title = "IST - Electricity Forecasting Tool"
load_figure_template("superhero")

#LAYOUT OF THE WELCOME PAGE
welcome_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("⚡ IST - Electricity Forecasting Tool ⚡", className="text-center text-primary mb-4", style={"fontSize": "4rem"}), width=12)
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        html.P("Welcome to this electricity forecasting tool for the 4 biggest buildings of the IST Campus. Please select "
                               "the building for which you want to forecast the electricity consumption. Then, press the run dashboard"
                               " button to start the forecasting tool.", className="card-text text-center", style={"fontSize": "1.2rem"})
                    ),
                    color="secondary",
                    className="mb-3",
                    style={"width": "100%"}
                ),
                width=9
            )
        ], justify="center"),

        # Building Selection Buttons
        dbc.Row([
            dbc.Col(dbc.Button(
        [html.I(className="bi bi-buildings-fill me-2", id="icon-central", style={"font-size": "3rem"}), html.Br(), "Central Building"],
                id="btn-central", color="secondary", outline=True, className="w-100 d-flex flex-column align-items-center justify-content-center", style={"height": "150px", "border-radius": "10px"}), width=3),
        dbc.Col(dbc.Button(
        [html.I(className="bi bi-buildings-fill me-2", id="icon-civil", style={"font-size": "3rem"}), html.Br(), "Civil Building"],
                id="btn-civil", color="secondary", outline=True, className="w-100 d-flex flex-column align-items-center justify-content-center", style={"height": "150px", "border-radius": "10px"}), width=3),
        dbc.Col(dbc.Button(
        [html.I(className="bi bi-buildings-fill me-2", id="icon-north", style={"font-size": "3rem"}), html.Br(), "North Tower"],
                id="btn-north", color="secondary", outline=True, className="w-100 d-flex flex-column align-items-center justify-content-center", style={"height": "150px", "border-radius": "10px"}), width=3),
        dbc.Col(dbc.Button(
        [html.I(className="bi bi-buildings-fill me-2", id="icon-south", style={"font-size": "3rem"}), html.Br(), "South Tower"],
                id="btn-south", color="secondary", outline=True, className="w-100 d-flex flex-column align-items-center justify-content-center", style={"height": "150px", "border-radius": "10px"}), width=3)
        ], justify="center", className="mb-4"),

        # Run Dashboard Button (larger)
        dbc.Row([
            dbc.Col(
                dbc.Button("Run the tool 🚀", id="start-btn", color="success", style= {"font-size": "1.5rem"}),
                width="auto"
            )
        ], justify="center"),

        dbc.Alert("⚠️ Please select a building before proceeding! ⚠️",
                  id="alert-no-selection",
                  color="danger",
                  dismissable=True,
                  is_open=False,
                  style={"position": "fixed", "top": "10%", "left": "50%", "transform": "translate(-50%, -50%)", "zIndex": 1050, "font-size": "1.2rem"}
        ),
    ], className="content-box p-5"),
], style={
    "display": "flex",
    "align-items": "center",
    "justify-content": "center",
    "height": "100vh",
    "width": "100vw"
})


#CREATION OF THE NAVIGATION BAR FOR THE MAIN DASHBOARD PAGE
dashboard_layout = dbc.Navbar(
    dbc.Container([
        # Left-aligned title (clickable, redirects to home)
        dbc.NavbarBrand("IST - Energy Forecasting Tool", href="/", className='ms-3'),

        # Right-aligned menu items
        dbc.Nav([
            dbc.DropdownMenu(
                label="Raw Data",
                nav=True,
                className='mx-3',
                children=[
                    dbc.DropdownMenuItem("Power Data", href="/power-data", id='tab-raw-power-data'),
                    dbc.DropdownMenuItem("Weather Data", href="/weather-data", id='tab-raw-weather-data'),
                ]
            ),
            dbc.DropdownMenu(
                label="Data Analysis",
                nav=True,
                className='mx-3',
                children=[
                    dbc.DropdownMenuItem("Power Data", href="/analysis-power-data", id='tab-analyis-power-data'),
                    dbc.DropdownMenuItem("Weather Data", href="/analysis-weather-data", id='tab-analyis-weather-data'),
                ]
            ),
            # Other Navigation Links
            dbc.NavItem(dbc.NavLink("Feature Selection", href="/feature-selection", className='mx-3'), id="tab-features"),
            # Models Dropdown
            dbc.DropdownMenu(
                label="Train Models",
                nav=True,
                className='mx-3',
                children=[
                    dbc.DropdownMenuItem("Train Autoregression", href="/models/autoregression"),
                    dbc.DropdownMenuItem("Train Linear Regression", href="/models/linear-regression"),
                    dbc.DropdownMenuItem("Train Support Vector Regression", href="/models/svr"),
                    dbc.DropdownMenuItem("Train Decision Tree", href="/models/decision-tree"),
                    dbc.DropdownMenuItem("Train Random Forest", href="/models/random-forest"),
                    dbc.DropdownMenuItem("Train Neural Network", href="/models/neural-network"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Choose Best Model", href="/models/choose-model")
                ]
            ),
            dbc.NavItem(dbc.NavLink("Model Validation", href="/model-validation", className='mx-3')),

            # Vertical stripe separator
            html.Div(style={"borderLeft": "2px solid white", "height": "40px"},className='mx-3'),

            # Help Button
            dbc.NavItem(dbc.NavLink("Help", href="/help", className="mx-3")),
        ], className="ms-auto", navbar=True)
    ], fluid=True),
    color="primary", dark=True, className="mb-4"
)

#CREATING THE PAGE FOR THE RAW POWER DATA
def make_rangeslider(id):
    rangeslider = dbc.Row(dcc.RangeSlider(
            id = id,
            min = 0,
            max = len(date_range) - 1,
            step = 1,
            marks = None,
            value = [0, len(date_range) - 1],
            tooltip={"placement": "bottom", "template": "Week {value}"},
            className='mt-2 dbc'))
    return rangeslider

power_data_layout = dbc.Container([
    dbc.Card([dcc.Graph(
            id="power-graph",
            figure= px.line(
                dataframe_cleaned_data,  # DataFrame
                x= dataframe_cleaned_data.index,  # X-axis
                y="Power [kW]",  # Y-axis
                template="superhero"
            )), make_rangeslider('power-slider')]),
    ], fluid=True, className="p-4")

#CREATING THE PAGE FOR THE RAW WEATHER DATA
weather_data_layout = dbc.Container([
    dbc.Card([dbc.Tabs([

        dbc.Tab(dcc.Graph(
            id="temp-graph",
            figure= px.line(
                dataframe_cleaned_data,
                x= dataframe_cleaned_data.index,
                y="Temperature [°C]",
                template="superhero"
            )), label="Temperature"),
        dbc.Tab(dcc.Graph(
            id="humidity_graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Relative Humidity [%]",
                template="superhero"
            )), label="Relative Humidity"),
        dbc.Tab(dcc.Graph(
            id="wind-graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Windspeed [m/s]",
                template="superhero"
            )), label="Windspeed"),
        dbc.Tab(dcc.Graph(
            id="gusts-graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Wind Gusts [m/s]",
                template="superhero"
            )), label="Wind Gusts"),
        dbc.Tab(dcc.Graph(
            id="pressure-graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Air Pressure [mbar]",
                template="superhero"
            )), label="Air Pressure"),
        dbc.Tab(dcc.Graph(
            id="radiation-graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Solar Radiation [W/m²]",
                template="superhero"
            )), label="Solar Radiation"),
        dbc.Tab(dcc.Graph(
            id="rain-graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Rain [mm/h]",
                template="superhero"
            )), label="Rain"),
        dbc.Tab(dcc.Graph(
            id="acc-rain-graph",
            figure=px.line(
                dataframe_cleaned_data,
                x=dataframe_cleaned_data.index,
                y="Accumulated rain on day [mm]",
                template="superhero"
            )), label="Accumulated Rain/day"),
    ]), make_rangeslider('weather-data-slider')]),

    ], fluid=True, className="p-4")

#CREATING THE PAGE FOR THE EXPLORATORY DATA ANALYSIS OF THE POWER DATA

def make_layout_analysis(id_left_button, id_right_button,id_content):
    layout = dbc.Spinner(dbc.Row([
            dbc.Col(
                dbc.Button(
                    html.I(className="bi bi-arrow-left"),  # Font Awesome left arrow icon
                    id=id_left_button,
                    color="primary",
                    style={"width": "50px", "height": "50px"}  # Style for better button appearance
                ),
                width="auto",  # Button will take only the space it needs
                className="d-flex align-items-center mx-2"
            ),
            dbc.Col(
                html.Div(id=id_content, className="mt-1", style={"width": "100%"}),
                width=True,
            ),
            dbc.Col(
                dbc.Button(
                    html.I(className="bi bi-arrow-right"),  # Font Awesome right arrow icon
                    id=id_right_button,
                    color="primary",
                    style={"width": "50px", "height": "50px"}  # Style for better button appearance
                ),
                width="auto",  # Button will take only the space it needs
                className="d-flex align-items-center mx-2"
            ),
        ],
        align="center",  # Align the row's content vertically to the center
        className="mt-2"
    ), color ='primary', delay_show=100)
    return layout

def make_table_analysis(column):
    table_header = [html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))]
    row1 = html.Tr([html.Td("Mean"), html.Td(f"{dataframe_cleaned_data[column].mean():.2f}")])
    row2 = html.Tr([html.Td("Median"), html.Td(f"{dataframe_cleaned_data[column].median():.2f}")])
    row3 = html.Tr([html.Td("Standard Deviation"), html.Td(f"{dataframe_cleaned_data[column].std():.2f}")])
    row4 = html.Tr([html.Td("Minimum"), html.Td(f"{dataframe_cleaned_data[column].min():.2f}")])
    row5 = html.Tr([html.Td("Maximum"), html.Td(f"{dataframe_cleaned_data[column].max():.2f}")])

    table_body = [html.Tbody([row1, row2, row3, row4, row5])]

    content = dbc.Spinner(html.Div(children=[
        html.H4("Statistical Values", className='text-center mt-2'),
        dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
    ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

def make_boxplot_analysis(column):
    content = dbc.Spinner(html.Div(children=[
        html.H4("Boxplot", className='text-center mt-2'),
        dcc.Graph(figure=px.box(dataframe_cleaned_data,
                                y=column,
                                template='superhero'))
    ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

def make_scatter_analysis(column):
    Q1 = dataframe_cleaned_data[column].quantile(0.25)
    Q3 = dataframe_cleaned_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataframe_with_outliers = dataframe_cleaned_data.copy()
    dataframe_with_outliers["Outlier"] = dataframe_with_outliers[column].apply(
        lambda x: "Outlier" if x < lower_bound or x > upper_bound else "Normal")

    fig = px.scatter(
        dataframe_with_outliers,
        x=dataframe_with_outliers.index,
        y=column,
        color="Outlier",  # Color points based on outlier status
        color_discrete_map={"Outlier": "red", "Normal": "darkorange"},  # Set colors
        template="superhero")

    content = dbc.Spinner(html.Div(children=[
        html.H4("Scatter plot with outliers in red", className='text-center mt-2'),
        dcc.Graph(figure=fig)
    ], className="mt-1", style={"width": "100%"}), color = 'primary', delay_show=100)
    return content

def make_histogram_analysis(column):
    fig = px.histogram(dataframe_cleaned_data, x=column)
    content = dbc.Spinner(html.Div(children=[
        html.H4("Histogram", className='text-center mt-2'),
        dcc.Graph(figure=fig)
    ]), color='primary', delay_show=100)
    return content

def make_aggregate_layout(graph_id):
    content = dbc.Spinner(html.Div([
        dbc.Row([
            dbc.Col(html.H4("Averages per hour/day/month", className="mt-2 ps-10"), width="auto"),
            dbc.Col(
                dbc.DropdownMenu(
                    label="Select Aggregation",
                    children=[
                        dbc.DropdownMenuItem("Hourly", id="hourly", n_clicks=0),
                        dbc.DropdownMenuItem("Daily", id="daily", n_clicks=0),
                        dbc.DropdownMenuItem("Monthly", id="monthly", n_clicks=0),
                    ],
                    id="agg-dropdown", direction="down", className="mb-3 mt-3"
                ),
                width="auto"
            ),
        ], className="justify-content-center align-items-center"),
        dcc.Graph(id=graph_id)
    ]), color = 'primary', delay_show=100)
    return content

def make_aggregate_graph(column,button_id):
    df_agg = dataframe_cleaned_data.copy()
    if button_id == "hourly":
        df_agg = df_agg.groupby(df_agg.index.hour).mean()
        df_agg.index = [str(i) for i in range(24)]
        df_agg.index.name = "Hour"
    elif button_id == "daily":
        df_agg = df_agg.groupby(df_agg.index.weekday).mean()
        df_agg.index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_agg.index.name = "Day"
    elif button_id == "monthly":
        df_agg = df_agg.groupby(df_agg.index.month).mean()
        df_agg.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df_agg.index.name = "Month"
    fig = px.bar(df_agg, x=df_agg.index, y=column)
    return fig


data_analysis_power_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-power-button', 'right-power-button', 'content-display')))

#CREATING THE PAGE FOR THE EXPLORATORY DATA ANALYSIS OF THE WEATHER DATA

data_analysis_weather_layout = dbc.Container([
    dbc.Card(dbc.Tabs([
        dbc.Tab(make_layout_analysis('left-temp-button', 'right-temp-button', 'content-temp'), label='Temperature'),
        dbc.Tab(make_layout_analysis('left-humidity-button', 'right-humidity-button', 'content-humidity'), label='Relative Humditity'),
        dbc.Tab(make_layout_analysis('left-wind-button', 'right-wind-button', 'content-wind'), label='Windspeed'),
        dbc.Tab(make_layout_analysis('left-gusts-button', 'right-gusts-button', 'content-gusts'), label= 'Wind Gusts'),
        dbc.Tab(make_layout_analysis('left-pressure-button', 'right-pressure-button', 'content-pressure'), label='Air Pressure'),
        dbc.Tab(make_layout_analysis('left-radiation-button', 'right-radiation-button', 'content-radiation'), label='Solar Radiation'),
        dbc.Tab(make_layout_analysis('left-rain-button', 'right-rain-button', 'content-rain'), label='Rain'),
        dbc.Tab(make_layout_analysis('left-acc-rain-button', 'right-acc-rain-button', 'content-acc-rain'), label ='Accumulated Rain/day')
    ], id='tabs-weather-analysis')),
    ], fluid=True, className="p-4")

#CREATING THE PAGE FOR THE FEATURE SELECTION

def show_filter_method(scoring_metric):
    global all_features
    features = SelectKBest(k=5, score_func=scoring_metric)
    fit = features.fit(X_all_features, Y_all_features)
    scores = fit.scores_
    fig = px.bar(x=all_features, y=scores, labels={'x': 'Features', 'y': 'Score'})
    content = dbc.Spinner(html.Div(children=[
        html.H4(f"Filter method with {scoring_metric.__name__}", className='text-center mt-2'),
        dcc.Graph(figure=fig)
    ]), color='primary', delay_show=100)
    return content

def show_wrapper_method(number_of_features):
    global all_features
    model = DecisionTreeRegressor()
    rfe = RFE(model, n_features_to_select= number_of_features)
    fit = rfe.fit(X_all_features, Y_all_features)
    ranking = pd.DataFrame({'Feature': all_features, 'Ranking': fit.ranking_})
    ranking["Adjusted Ranking"] = ranking["Ranking"].max() - ranking["Ranking"] + 1  # Invert ranking
    ranking = ranking.sort_values(by="Adjusted Ranking", ascending=False)
    fig = px.bar(ranking, x='Feature', y="Adjusted Ranking", labels={'x': 'Features', 'y': 'Ranking'})
    content = dbc.Spinner(html.Div(children=[
        html.H4(f"Wrapper method with {str(number_of_features)} features chosen", className='text-center mt-2'),
        dcc.Graph(figure=fig)
    ]), color='primary', delay_show=100)
    return content

def show_graph_embedded():
    global all_features
    model = RandomForestRegressor()
    model.fit(X_all_features, Y_all_features)
    feature_importance = model.feature_importances_
    scores = pd.DataFrame({"Feature": all_features, "Score": feature_importance})
    fig = px.bar(scores, x='Feature', y="Score", labels={'x': 'Features', 'y': 'Score'})
    content = dbc.Spinner(html.Div(children=[
        html.H4('Embedded method with Random Forest Model', className='text-center mt-2'),
        dcc.Graph(figure=fig)
    ]), color='primary', delay_show=100)
    return content


feature_selection_layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Select the features for the model:"),
                dbc.CardBody([
                    dbc.Checklist(
                        id="features-checklist",
                        options=[{"label": f, "value": f} for f in all_features],
                        value=selected_features,
                        switch=True
                    ),
                ])
            ], id = 'card-features', className="h-100"),  # h-100 ensures both cards are the same height
            width=3  # 1/4 of the width
        ),

        # Right Card (Feature Selection Methods)
        dbc.Col(
            dbc.Card(
                dbc.Tabs(
                    id="feature-selection-tabs",
                    active_tab="filter",
                    children=[
                        dbc.Tab(make_layout_analysis('left-filter-button', 'right-filter-button', 'filter-content'),
                                label="Filter Methods"),
                        dbc.Tab(make_layout_analysis('left-wrapper-button', 'right-wrapper-button', 'wrapper-content'),
                                label="Wrapper Methods"),
                        dbc.Tab(dbc.Spinner(html.Div(id='embedded-content',className="mt-1", style={
                                            "width": "100%",
                                            "height": "500px"
                            }), color='primary', delay_show=100), label="Embedded Methods")
                    ],
                ), className="h-100"),
            width=9  # 3/4 of the width
        )
    ], className="mt-4"),
    dcc.Store(id='selected-features-display')
], fluid=True)

#CREATING THE PAGE FOR THE AUTOREGRESSION MODEL
def calculate_model_performance(model_name, test, predictions):
    global performance_df
    MAE = abs(metrics.mean_absolute_error(test, predictions))
    MBE = abs(np.mean(test - predictions))
    MSE = abs(metrics.mean_squared_error(test, predictions))
    RMSE = abs(np.sqrt(MSE))
    cvRMSE = abs(RMSE / np.mean(test))
    NMBE = abs(MBE / np.mean(test))
    performance_df[model_name] = [MAE, MBE, MSE, RMSE, cvRMSE, NMBE]

def autoregression_model(X,Y):
    global coef_AR
    split_point = len(Y) - 1000
    train, test = Y[0:split_point], Y[split_point:]
    window = 1  # model only looks at the previous value
    model = AutoReg(train, lags=window)
    model_fit = model.fit()

    coef_AR = model_fit.params
    history = train[len(train) - window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef_AR[0]
        for d in range(window):
            yhat += coef_AR[d + 1] * lag[window - d - 1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
    return test, predictions

autoregression_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-autoregression-button', 'right-autoregression-button', 'content-autoregression')))

#CREATING THE PAGE FOR THE LINEAR REGRESSION MODEL
def linear_regression_model(X, Y):
    global model_LR
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model_LR = linear_model.LinearRegression()
    model_LR.fit(X_train, y_train)
    y_pred_LR = model_LR.predict(X_test)
    return y_test, y_pred_LR

linear_regression_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-linear-button', 'right-linear-button', 'content-linear')))

#CREATING THE PAGE FOR THE SUPPORT VECTOR REGRESSION MODEL
def support_vector_regression(X,Y):
    global model_SVR, scaler_SVR
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    ss_X = StandardScaler()
    scaler_SVR = StandardScaler()
    X_train_ss = ss_X.fit_transform(X_train)
    y_train_ss = scaler_SVR.fit_transform(y_train.reshape(-1, 1))
    model_SVR = SVR(kernel='rbf')
    model_SVR.fit(X_train_ss, y_train_ss)
    y_pred_SVR = model_SVR.predict(ss_X.fit_transform(X_test))
    y_pred_SVR2 = scaler_SVR.inverse_transform(y_pred_SVR.reshape(-1, 1))
    return y_test, y_pred_SVR2.flatten()

support_vector_regression_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-vector-button', 'right-vector-button', 'content-vector')))

#CREATING THE PAGE FOR THE DECISION TREE MODEL
def decision_tree(X,Y):
    global model_DT
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model_DT = DecisionTreeRegressor(min_samples_leaf=10)
    model_DT.fit(X_train, y_train)
    y_pred_DT = model_DT.predict(X_test)
    return y_test, y_pred_DT

decision_tree_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-tree-button', 'right-tree-button', 'content-tree')))

#CREATING THE PAGE OF THE RANDOM FOREST MODEL
def random_forest(X,Y):
    global model_RF
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    best_params = {'n_estimators': 300,
                   'max_depth': 20,
                   'min_samples_leaf': 4,
                   'min_samples_split': 10}
    model_RF = RandomForestRegressor(**best_params)
    model_RF.fit(X_train, y_train)
    y_pred_RF = model_RF.predict(X_test)
    return y_test, y_pred_RF

random_forest_layout  = dbc.Container(dbc.Card(
    make_layout_analysis('left-forest-button', 'right-forest-button', 'content-forest')))

#CREATING THE PAGE FOR THE NEURAL NETWORK MODEL
def neural_network(X,Y):
    global model_NN, scaler_NN
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    ss_X = StandardScaler()
    scaler_NN = StandardScaler()
    X_train_ss = ss_X.fit_transform(X_train)
    y_train_ss = scaler_NN.fit_transform(y_train.reshape(-1, 1))
    model_NN = MLPRegressor(hidden_layer_sizes=(30, 30, 30), max_iter=300)
    model_NN.fit(X_train_ss, y_train_ss)
    y_pred_NN = model_NN.predict(ss_X.transform(X_test))
    y_pred_NN2 = scaler_NN.inverse_transform(y_pred_NN.reshape(-1, 1))
    return y_test, y_pred_NN2.flatten()

neural_network_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-neural-button', 'right-neural-button', 'content-neural')))

#CREATING THE PAGE FOR CHOOSING THE BEST MODEL
best_model_layout = dbc.Spinner(dbc.Container([
    html.Div(id="selected-model-output", style={'display': 'none'}),
    dbc.Row([  # Row for the performance metrics card
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Select the Best Model:", className='text-center'),  # Card header with the text
                dbc.CardBody([
                    # Radio items for selecting the best model
                    dbc.RadioItems(
                        id="best-model-radio",
                        options=[{'label': model, 'value': model} for model in
                                 ['Autoregression', 'Linear Regression', 'Support Vector Regression', 'Decision Tree',
                                  'Random Forest', 'Neural Network']],
                        value=None,  # Default value is None, no model selected initially
                        inline=True,  # Display radio items inline (horizontal)
                        className='text-center'
                    ),
                    # Graphs for the performance metrics
                    dbc.Row([  # Row for the graphs
                        dbc.Col(dcc.Graph(id=f"MAE-chart"), width=6),  # First graph
                        dbc.Col(dcc.Graph(id=f"MBE-chart"), width=6),  # Second graph
                    ], className="mt-3"),  # Add top margin for spacing between radio items and graphs

                    dbc.Row([  # Second row of graphs
                        dbc.Col(dcc.Graph(id=f"MSE-chart"), width=6),  # Third graph
                        dbc.Col(dcc.Graph(id=f"RMSE-chart"), width=6),  # Fourth graph
                    ], className="mt-3"),

                    dbc.Row([  # Third row of graphs
                        dbc.Col(dcc.Graph(id=f"cvRMSE-chart"), width=6),  # Fifth graph
                        dbc.Col(dcc.Graph(id=f"NMBE-chart"), width=6),  # Sixth graph
                    ], className="mt-3"),
                ])
            ], className="mb-4"),  # Add margin bottom for spacing between cards
        ),
    ]),
], fluid=True), color='primary', delay_show=100)

#CREATING THE MODEL VALIDATION PAGE
model_validation_layout = dbc.Container(dbc.Card(
    make_layout_analysis('left-validation-button', 'right-validation-button', 'content-validation')))

#CREATING THE HELP PAGE
help_layout = html.Div([
    dbc.Container([
        html.P("Welcome to the IST Electricity Forecasting Tool! This helpful guide explains the functions of the different tabs.", className='text-center'),

        dbc.Accordion([
            dbc.AccordionItem([
                html.P("Here you can view the raw power data of the selected building and the raw weather data. The used data is from 2017"
                       "and 2018"),
            ], title="⚡ Raw Data"),

            dbc.AccordionItem([
                html.P("This tab allows you to analyze both power and weather data for selected buildings."),
                html.Ul([
                    html.Li([
                        html.B("Power Data: "),
                        "Gain insights into consumption patterns and trends for the selected building. "
                    ]),
                    html.Li([
                        html.B("Weather Data: "),
                        "Analyze seasonal trends and variations in different weather parameters, "
                        "helping to understand their impact on energy consumption."
                    ])
                ])
            ], title="📊 Data Analysis"),

            dbc.AccordionItem([
                html.P("In this tab, the features that will be used in the model can be selected."
                       "To help the user choose the final features, the available features are ranked based on the following three"
                       "methods:"),
                html.Ul([
                    html.Li([
                        html.B('Filter methods:'),
                        'Uses measures to score the different features (correlation, mutual information, T-test, F-test,...)'
                    ]),
                    html.Li([
                        html.B('Embedded methods:'),
                        'Methods that are based on the results of testing subsets of features in preliminary models. We use a DecisionTree model as the estimator model: '
                    ]),
                    html.Li([
                        html.B('Wrapper methods:'),
                        'Searches the space of features and model parameters simultaneously. Here we use a RandomForest model as the estimator model:'
                    ])
                ])
            ], title="🎛️ Feature Selection"),

            dbc.AccordionItem([
                html.P("Compare different forecasting models to predict the electricity consumption."),
                html.Ul([
                    html.Li("The different models first need to be trained, based on the features that were selected."),
                    html.Li("The user can then select the best model, which performance will be validated in the Model Validation Tab."),
                ])
            ], title="🤖 Models"),

            dbc.AccordionItem([
                html.P("Validate the chosen model with the 2019 data."),
            ], title="✅ Model Validation"),

        ], start_collapsed=True),
    ], className="p-4 content-box")
])

#CREATING THE GENERAL LAYOUT OF THE APP, WITH THE ID PAGE_CONTENT THE DIFFERENT PAGES CAN BE SHOWN
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
], fluid=True)

#CALLBACK FOR THE BUILDINGS BUTTONS ON THE WELCOME PAGE: callback is activated if one of the buttons is clicked

@app.callback(
    Input("btn-central", "n_clicks"),
    Input("btn-civil", "n_clicks"),
    Input("btn-north", "n_clicks"),
    Input("btn-south", "n_clicks"),
    prevent_initial_call=True
)

def toggle_building(central, civil, north, south):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update  # Prevents callback errors

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id in selected_buildings:
        selected_buildings.remove(button_id)  # Remove the building_id if it was already selected
    else:
        selected_buildings.clear()
        selected_buildings.append(button_id)

#CALLBACK THAT CHANGES THE COLOR OF THE BUTTONS

@app.callback(
    [Output("btn-central", "color"), Output("btn-civil", "color"),
     Output("btn-north", "color"), Output("btn-south", "color"),
     Output("icon-central", "className"), Output("icon-civil", "className"),
     Output("icon-north", "className"), Output("icon-south", "className")],
     [Input("btn-central", "n_clicks"), Input("btn-civil", "n_clicks"),
     Input("btn-north", "n_clicks"), Input("btn-south", "n_clicks")],
)

def update_button_and_icon_colors(central, civil, north, south):

    def get_color(button_id):
        return "primary" if button_id in selected_buildings else "secondary"

    def get_icon_class(button_id):
        return "bi bi-buildings-fill text-primary me-2" if button_id in selected_buildings else "bi bi-buildings-fill text-secondary me-2"

    return [
        get_color("btn-central"), get_color("btn-civil"),
        get_color("btn-north"), get_color("btn-south"),
        get_icon_class("btn-central"), get_icon_class("btn-civil"),
        get_icon_class("btn-north"), get_icon_class("btn-south"),
    ]

#CALLBACK THAT SWITCHES AN ALERT WHEN NO BUILDINGS ARE SELECTED OR MOVES TO THE DASHBOARD IF BUILDINGS ARE SELECTED + LOADS THE RIGHT DATA

@app.callback(
    [Output("alert-no-selection", "is_open"),
     Output("url", "pathname")],
     Input("start-btn", "n_clicks"),
    prevent_initial_call=True,
    allow_missing=True
)

def show_alert(button_clicked):
    global dataframe_cleaned_data, dataframe_features, Z_all_features, Y_all_features, X_all_features
    global AR_results, LR_results, SVR_results, DT_results, RF_results, NN_results, trained_models, performance_df
    global dataframe_test_data
    if button_clicked and selected_buildings == []:  # If no buildings are selected
        return True, dash.no_update  # Show alert, stay on welcome page
    elif button_clicked:
        if 'btn-north' in selected_buildings:
            dataframe_cleaned_data = pd.read_csv('All_data_cleaned_North.csv')
            dataframe_test_data = pd.read_csv('testData_2019_NorthTower.csv')
            dataframe_test_data.rename(columns={'Date': 'Date + Time', 'North Tower (kWh)': 'Power [kW]', 'temp_C': 'Temperature [°C]',
                                                'HR': 'Relative Humidity [%]',
                                                'windSpeed_m/s': 'Windspeed [m/s]', 'windGust_m/s': 'Wind Gusts [m/s]',
                                                'pres_mbar': 'Air Pressure [mbar]',
                                                'solarRad_W/m2': 'Solar Radiation [W/m²]', 'rain_mm/h': 'Rain [mm/h]',
                                                'rain_day': 'Accumulated rain on day [mm]'}, inplace=True)
            print('North loaded')
        elif 'btn-south' in selected_buildings:
            dataframe_cleaned_data = pd.read_csv('All_data_cleaned_South.csv')
            print('South loaded')
            dataframe_test_data = pd.read_csv('testData_2019_SouthTower.csv')
            dataframe_test_data.rename(
                columns={'Date': 'Date + Time', 'South Tower (kWh)': 'Power [kW]', 'temp_C': 'Temperature [°C]',
                         'HR': 'Relative Humidity [%]',
                         'windSpeed_m/s': 'Windspeed [m/s]', 'windGust_m/s': 'Wind Gusts [m/s]',
                         'pres_mbar': 'Air Pressure [mbar]',
                         'solarRad_W/m2': 'Solar Radiation [W/m²]', 'rain_mm/h': 'Rain [mm/h]',
                         'rain_day': 'Accumulated rain on day [mm]'}, inplace=True)
        elif 'btn-civil' in selected_buildings:
            dataframe_cleaned_data = pd.read_csv('All_data_cleaned_Civil.csv')
            print('Civil loaded')
            dataframe_test_data = pd.read_csv('testData_2019_Civil.csv')
            dataframe_test_data.rename(
                columns={'Date': 'Date + Time', 'Civil (kWh)': 'Power [kW]', 'temp_C': 'Temperature [°C]',
                         'HR': 'Relative Humidity [%]',
                         'windSpeed_m/s': 'Windspeed [m/s]', 'windGust_m/s': 'Wind Gusts [m/s]',
                         'pres_mbar': 'Air Pressure [mbar]',
                         'solarRad_W/m2': 'Solar Radiation [W/m²]', 'rain_mm/h': 'Rain [mm/h]',
                         'rain_day': 'Accumulated rain on day [mm]'}, inplace=True)
        elif 'btn-central' in selected_buildings:
            dataframe_cleaned_data = pd.read_csv('All_data_cleaned_Central.csv')
            print('Central loaded')
            dataframe_test_data = pd.read_csv('testData_2019_Central.csv')
            dataframe_test_data.rename(
                columns={'Date': 'Date + Time', 'Central (kWh)': 'Power [kW]', 'temp_C': 'Temperature [°C]',
                         'HR': 'Relative Humidity [%]',
                         'windSpeed_m/s': 'Windspeed [m/s]', 'windGust_m/s': 'Wind Gusts [m/s]',
                         'pres_mbar': 'Air Pressure [mbar]',
                         'solarRad_W/m2': 'Solar Radiation [W/m²]', 'rain_mm/h': 'Rain [mm/h]',
                         'rain_day': 'Accumulated rain on day [mm]'}, inplace=True)
        dataframe_cleaned_data['Date + Time'] = pd.to_datetime(dataframe_cleaned_data['Date + Time'])
        dataframe_cleaned_data = dataframe_cleaned_data.set_index('Date + Time', drop=True)
        dataframe_features = add_features(dataframe_cleaned_data)
        Z_all_features = dataframe_features.values
        Y_all_features = Z_all_features[:, 0]  # Target Variable
        X_all_features = Z_all_features[:, 1:]  # Features
        trained_models = []
        performance_df = pd.DataFrame()
        dataframe_test_data['Date + Time'] = pd.to_datetime(dataframe_test_data['Date + Time'])
        portugal_holidays = holidays.Portugal(years=[2019])
        dataframe_test_data["Holiday"] = dataframe_test_data['Date + Time'].apply(lambda x: 1 if x in portugal_holidays else 0)
        dataframe_test_data = dataframe_test_data.set_index('Date + Time', drop=True)
        dataframe_test_data = add_features(dataframe_test_data)
        return False, "/dashboard"  # Hide alert, navigate to dashboard
    return False, "/dashboard"  # Hide alert, navigate to dashboard

#CALLBACK TO SWITCH BETWEEN PAGES
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    global active_index_power_analysis
    active_index_power_analysis = 0
    if pathname == "/dashboard":
        return html.Div([dashboard_layout, help_layout])
    elif pathname == "/power-data":
        return html.Div([dashboard_layout, power_data_layout])
    elif pathname == "/weather-data":
        return html.Div([dashboard_layout, weather_data_layout])
    elif pathname == "/analysis-power-data":
        return html.Div([dashboard_layout, data_analysis_power_layout])
    elif pathname == "/analysis-weather-data":
        return html.Div([dashboard_layout, data_analysis_weather_layout])
    elif pathname == "/feature-selection":
        return html.Div([dashboard_layout, feature_selection_layout])
    elif pathname == "/models/autoregression":
        return html.Div([dashboard_layout, autoregression_layout])
    elif pathname == "/models/linear-regression":
        return html.Div([dashboard_layout, linear_regression_layout])
    elif pathname == "/models/svr":
        return html.Div([dashboard_layout, support_vector_regression_layout])
    elif pathname == "/models/decision-tree":
        return html.Div([dashboard_layout, decision_tree_layout])
    elif pathname == "/models/random-forest":
        return html.Div([dashboard_layout, random_forest_layout])
    elif pathname == "/models/neural-network":
        return html.Div([dashboard_layout, neural_network_layout])
    elif pathname == "/models/choose-model":
        return html.Div([dashboard_layout, best_model_layout])
    elif pathname == "/model-validation":
        return html.Div([dashboard_layout, model_validation_layout])
    elif pathname == "/help":
        return html.Div([dashboard_layout, help_layout])
    return welcome_layout

#WORKING OF THE RANGE SLIDER WITH THE POWER GRAPH

@app.callback(
    Output('power-graph', 'figure'),
    [Input('power-slider', 'value'),
     Input("tab-raw-power-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]
    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]
    fig = px.line(filtered_df, x=filtered_df.index, y='Power [kW]')
    return fig

#WORKING OF THE RANGE SLIDER WITH THE TEMPERATURE GRAPH

@app.callback(
    Output('temp-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
     Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Temperature [°C]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE HUMIDITY GRAPH

@app.callback(
    Output('humidity_graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Relative Humidity [%]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE WINDSPEED GRAPH

@app.callback(
    Output('wind-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Windspeed [m/s]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE WIND GUSTS GRAPH

@app.callback(
    Output('gusts-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Wind Gusts [m/s]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE AIR PRESSURE GRAPH

@app.callback(
    Output('pressure-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Air Pressure [mbar]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE RADIATION GRAPH

@app.callback(
    Output('radiation-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Solar Radiation [W/m²]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE RAIN GRAPH

@app.callback(
    Output('rain-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Rain [mm/h]',
    )
    return fig

#WORKING OF THE RANGE SLIDER WITH THE ACCUMULATED RAIN GRAPH

@app.callback(
    Output('acc-rain-graph', 'figure'),
    [Input('weather-data-slider', 'value'),
    Input("tab-raw-weather-data", "n_clicks")],
)
def update_graph(slider_range, n_clicks):
    start_idx, end_idx = slider_range
    start_date = date_range[start_idx]
    end_date = date_range[end_idx]

    filtered_df = dataframe_cleaned_data.loc[start_date:end_date]

    fig = px.line(
        filtered_df,
        x=filtered_df.index,
        y='Accumulated rain on day [mm]',
    )
    return fig

#WORKING OF THE SLIDES FOR THE POWER DATA ANALYSIS
@app.callback(
    Output("content-display", "children"),
    [Input("left-power-button", "n_clicks"),
     Input("right-power-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_power_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-power-button':
        active_index_power_analysis -= 1  # Go left
    if button_id == 'right-power-button':
        active_index_power_analysis += 1  # Go right

    if active_index_power_analysis == 5:
        active_index_power_analysis = 0
    if active_index_power_analysis == -1:
        active_index_power_analysis = 4

    # Define the content for each index
    if active_index_power_analysis == 0:
        content = make_table_analysis('Power [kW]')
    elif active_index_power_analysis == 1:
        content = make_boxplot_analysis('Power [kW]')
    elif active_index_power_analysis == 2:
        content = make_scatter_analysis('Power [kW]')
    elif active_index_power_analysis == 3:
        content = make_histogram_analysis('Power [kW]')
    elif active_index_power_analysis == 4:
        content = make_aggregate_layout('agg-power-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN POWER DATA ANALYSIS
@app.callback(
    Output("agg-power-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-power-button", 'n_clicks'),
     Input('right-power-button', "n_clicks")]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Power [kW]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE TEMP DATA ANALYSIS
@app.callback(
    Output("content-temp", "children"),
    [Input("left-temp-button", "n_clicks"),
     Input("right-temp-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-temp-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-temp-button':
        active_index_weather_analysis += 1  # Go right
    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    print(active_index_weather_analysis)
    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Temperature [°C]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Temperature [°C]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Temperature [°C]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Temperature [°C]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-temp-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN TEMP DATA ANALYSIS
@app.callback(
    Output("agg-temp-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-temp-button", 'n_clicks'),
     Input('right-temp-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Temperature [°C]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE HUMIDITY DATA ANALYSIS
@app.callback(
    Output("content-humidity", "children"),
    [Input("left-humidity-button", "n_clicks"),
     Input("right-humidity-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-humidity-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-humidity-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    print(active_index_weather_analysis)
    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Relative Humidity [%]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Relative Humidity [%]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Relative Humidity [%]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Relative Humidity [%]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-humidity-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN HUMIDITY DATA ANALYSIS
@app.callback(
    Output("agg-humidity-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-humidity-button", 'n_clicks'),
     Input('right-humidity-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Relative Humidity [%]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE WIND DATA ANALYSIS
@app.callback(
    Output("content-wind", "children"),
    [Input("left-wind-button", "n_clicks"),
     Input("right-wind-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-wind-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-wind-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Windspeed [m/s]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Windspeed [m/s]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Windspeed [m/s]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Windspeed [m/s]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-wind-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN WIND DATA ANALYSIS
@app.callback(
    Output("agg-wind-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-wind-button", 'n_clicks'),
     Input('right-wind-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Windspeed [m/s]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE GUSTS DATA ANALYSIS
@app.callback(
    Output("content-gusts", "children"),
    [Input("left-gusts-button", "n_clicks"),
     Input("right-gusts-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-gusts-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-gusts-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Wind Gusts [m/s]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Wind Gusts [m/s]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Wind Gusts [m/s]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Wind Gusts [m/s]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-gusts-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN GUSTS DATA ANALYSIS
@app.callback(
    Output("agg-gusts-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-gusts-button", 'n_clicks'),
     Input('right-gusts-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Wind Gusts [m/s]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE PRESSURE DATA ANALYSIS
@app.callback(
    Output("content-pressure", "children"),
    [Input("left-pressure-button", "n_clicks"),
     Input("right-pressure-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-pressure-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-pressure-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Air Pressure [mbar]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Air Pressure [mbar]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Air Pressure [mbar]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Air Pressure [mbar]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-pressure-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN PRESSURE DATA ANALYSIS
@app.callback(
    Output("agg-pressure-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-pressure-button", 'n_clicks'),
     Input('right-pressure-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Air Pressure [mbar]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE RADIATION DATA ANALYSIS
@app.callback(
    Output("content-radiation", "children"),
    [Input("left-radiation-button", "n_clicks"),
     Input("right-radiation-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-radiation-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-radiation-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Solar Radiation [W/m²]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Solar Radiation [W/m²]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Solar Radiation [W/m²]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Solar Radiation [W/m²]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-radiation-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN RADIATION DATA ANALYSIS
@app.callback(
    Output("agg-radiation-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-radiation-button", 'n_clicks'),
     Input('right-radiation-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Solar Radiation [W/m²]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE RAIN DATA ANALYSIS
@app.callback(
    Output("content-rain", "children"),
    [Input("left-rain-button", "n_clicks"),
     Input("right-rain-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-rain-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-rain-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Rain [mm/h]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Rain [mm/h]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Rain [mm/h]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Rain [mm/h]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-rain-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN RAIN DATA ANALYSIS
@app.callback(
    Output("agg-rain-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-rain-button", 'n_clicks'),
     Input('right-rain-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right,tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Rain [mm/h]', button_id)
    return fig

#WORKING OF THE SLIDES FOR THE ACC RAIN DATA ANALYSIS
@app.callback(
    Output("content-acc-rain", "children"),
    [Input("left-acc-rain-button", "n_clicks"),
     Input("right-acc-rain-button", "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_weather_analysis

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-acc-rain-button':
        active_index_weather_analysis -= 1  # Go left
    if button_id == 'right-acc-rain-button':
        active_index_weather_analysis += 1  # Go right

    if active_index_weather_analysis == 5:
        active_index_weather_analysis = 0
    if active_index_weather_analysis == -1:
        active_index_weather_analysis = 4

    # Define the content for each index
    if active_index_weather_analysis == 0:
        content = make_table_analysis('Accumulated rain on day [mm]')
    elif active_index_weather_analysis == 1:
        content = make_boxplot_analysis('Accumulated rain on day [mm]')
    elif active_index_weather_analysis == 2:
        content = make_scatter_analysis('Accumulated rain on day [mm]')
    elif active_index_weather_analysis == 3:
        content = make_histogram_analysis('Accumulated rain on day [mm]')
    elif active_index_weather_analysis == 4:
        content = make_aggregate_layout('agg-acc-rain-graph')
    return content

#WORKING OF THE DROPDOWN MENU FOR THE AGGREGATION GRAPH IN ACC RAIN DATA ANALYSIS
@app.callback(
    Output("agg-acc-rain-graph", "figure"),
    [Input("hourly", "n_clicks"),
    Input("daily", "n_clicks"),
    Input("monthly", "n_clicks"),
    Input("left-acc-rain-button", 'n_clicks'),
     Input('right-acc-rain-button', "n_clicks"),
     Input('tabs-weather-analysis', 'active_tab')]
)

def update_aggregate(n_hourly, n_daily, n_monthly, n_left, n_right, tab_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "hourly"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    fig = make_aggregate_graph('Accumulated rain on day [mm]', button_id)
    return fig

#SHOWING OF THE FEATURE SELECTION GRAPHS: FILTER METHOD

@app.callback(
    Output("filter-content", "children"),
    [Input("left-filter-button", "n_clicks"),
     Input("right-filter-button", "n_clicks"),
     Input('feature-selection-tabs', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_filter_methods

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-filter-button':
        active_index_filter_methods -= 1  # Go left
    if button_id == 'right-filter-button':
        active_index_filter_methods += 1  # Go right

    if active_index_filter_methods == 2:
        active_index_filter_methods = 0
    if active_index_filter_methods == -1:
        active_index_filter_methods = 1

    # Define the content for each index
    if active_index_filter_methods == 0:
        content = show_filter_method(f_regression)
    elif active_index_filter_methods == 1:
        content = show_filter_method(mutual_info_regression)
    return content

#SHOWING OF THE FEATURE SELECTION GRAPHS: WRAPPER METHOD

@app.callback(
    Output("wrapper-content", "children"),
    [Input("left-wrapper-button", "n_clicks"),
     Input("right-wrapper-button", "n_clicks"),
     Input('feature-selection-tabs', 'active_tab')]
)
def update_content(left_clicks, right_clicks, tab_click):
    global active_index_wrapper_methods

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-wrapper-button':
        active_index_wrapper_methods -= 1  # Go left
    if button_id == 'right-wrapper-button':
        active_index_wrapper_methods += 1  # Go right

    if active_index_wrapper_methods == 3:
        active_index_wrapper_methods = 0
    if active_index_wrapper_methods == -1:
        active_index_wrapper_methods = 2

    # Define the content for each index
    if active_index_wrapper_methods == 0:
        content = show_wrapper_method(1)
    elif active_index_wrapper_methods == 1:
        content = show_wrapper_method(2)
    elif active_index_wrapper_methods == 2:
        content = show_wrapper_method(3)
    return content

#UPDATING THE GRAPH OF THE EMBEDDED METHOD

@app.callback(
    Output("embedded-content", "children"),
     Input('feature-selection-tabs', 'active_tab')
)
def update_content(tab_click):
    content = show_graph_embedded()
    return content

#UPDATING THE SELECTED FEATURES AND MAKING THE NUMPY-ARRAYS TO TRAIN THE MODELS

@app.callback(
    Output('selected-features-display', 'children'),
    [Input('features-checklist', 'value')]
)
def update_selected_features(features_checklist):
    global selected_features, dataframe_selected_features, X_selected_features, Y_selected_features, trained_models, performance_df
    if features_checklist != selected_features and features_checklist != []:
        selected_features = features_checklist
        columns_to_keep = ['Power [kW]'] + selected_features
        dataframe_selected_features = dataframe_features.loc[:, dataframe_features.columns.intersection(columns_to_keep)]
        X_selected_features = dataframe_selected_features.values[:,1:]
        Y_selected_features = dataframe_selected_features.values[:,0]
        print(f'I changed selected features to {selected_features}')
        trained_models = []
        performance_df = pd.DataFrame()
    return html.Ul([html.Li(feature) for feature in selected_features])

#SWITHES ON THE CHECKBOXES THAT ARE IN SELECTED_FEATURES WHEN THE TAB IS OPENED

@app.callback(
    Output("card-features", "children"),
    Input("url", "pathname")
)

def update_tab_content(pathname):
    if pathname == "/feature-selection":
        return [dbc.CardHeader("Select the features for the model:"),
                dbc.CardBody([
                    dbc.Checklist(
                        id="features-checklist",
                        options=[{"label": f, "value": f} for f in all_features],
                        value= selected_features,
                        switch=True
                    ),
                ])]



#SLIDES FOR THE AUTOREGRESSION MODEL

@app.callback(
    Output("content-autoregression", "children"),
    [Input("left-autoregression-button", "n_clicks"),
     Input("right-autoregression-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_autoregression, AR_results, trained_models

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-autoregression-button':
        active_index_autoregression -= 1  # Go left
    if button_id == 'right-autoregression-button':
        active_index_autoregression += 1  # Go right

    if active_index_autoregression == 3:
        active_index_autoregression = 0
    if active_index_autoregression == -1:
        active_index_autoregression = 2

    if 'Autoregression' not in trained_models:
        test,predictions = autoregression_model(X_selected_features, Y_selected_features)
        AR_results = pd.DataFrame()
        AR_results['Test_AR'] = test
        AR_results['Pred_AR'] = predictions
        calculate_model_performance('Autoregression', test = test, predictions = predictions)
        trained_models.append('Autoregression')
        print('I trained the model')
    else:
        test = AR_results['Test_AR']
        predictions = AR_results['Pred_AR']
        print('I did not train the model')
    # Define the content for each index
    if active_index_autoregression == 0:
        table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
        table_body = [html.Tbody(html.Tr([html.Td(performance_df.at[i, 'Autoregression']) for i in range(len(performance_df))]))]

        content = dbc.Spinner(html.Div(children=[
            html.H4("Performance Metrics Autoregression", className='text-center mt-2'),
            dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_autoregression == 1:
        fig = px.scatter(x=test, y=predictions, labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'}, template='superhero')
        content = dbc.Spinner(html.Div(children=[
            html.H4("Scatter Plot Autoregression", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_autoregression == 2:
        fig = px.line(y=[test,predictions], template='superhero')
        fig.update_traces(name="Real Power Consumption")
        fig.update_traces(name="Predicted Power Consumption")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power Consumption",
            legend_title=''
        )
        content = dbc.Spinner(html.Div(children=[
            html.H4("Line Plot Autoregression", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

#SLIDES FOR THE LINEAR REGRESSION MODEL

@app.callback(
    Output("content-linear", "children"),
    [Input("left-linear-button", "n_clicks"),
     Input("right-linear-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_linear, LR_results

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-linear-button':
        active_index_linear -= 1  # Go left
    if button_id == 'right-linear-button':
        active_index_linear += 1  # Go right

    if active_index_linear == 3:
        active_index_linear = 0
    if active_index_linear == -1:
        active_index_linear = 2

    if 'Linear Regression' not in trained_models:
        test, predictions = linear_regression_model(X_selected_features, Y_selected_features)
        LR_results = pd.DataFrame()
        LR_results['Test_LR'] = test
        LR_results['Pred_LR'] = predictions
        calculate_model_performance('Linear Regression', test=test, predictions=predictions)
        trained_models.append('Linear Regression')
        print('I trained the model')
    else:
        test = LR_results['Test_LR']
        predictions = LR_results['Pred_LR']
        print('I did not train the model')
    # Define the content for each index
    if active_index_linear == 0:
        table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
        table_body = [html.Tbody(html.Tr([html.Td(performance_df.at[i, 'Linear Regression']) for i in range(len(performance_df))]))]

        content = dbc.Spinner(html.Div(children=[
            html.H4("Performance Metrics Linear Regression", className='text-center mt-2'),
            dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_linear == 1:
        fig = px.scatter(x=test, y=predictions, labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'}, template='superhero')
        content = dbc.Spinner(html.Div(children=[
            html.H4("Scatter Plot Linear Regression", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_linear == 2:
        fig = px.line(y=[test[1:200],predictions[1:200]], template='superhero')
        fig.update_traces(name="Real Power Consumption")
        fig.update_traces(name="Predicted Power Consumption")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power Consumption",
            legend_title=''
        )
        content = dbc.Spinner(html.Div(children=[
            html.H4("Line Plot Linear Regression", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

#SLIDES FOR THE SUPPORT VECTOR REGRESSION MODEL

@app.callback(
    Output("content-vector", "children"),
    [Input("left-vector-button", "n_clicks"),
     Input("right-vector-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_vector, SVR_results

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-vector-button':
        active_index_vector -= 1  # Go left
    if button_id == 'right-vector-button':
        active_index_vector += 1  # Go right

    if active_index_vector == 3:
        active_index_vector = 0
    if active_index_vector == -1:
        active_index_vector = 2

    if 'Support Vector Regression' not in trained_models:
        test, predictions = support_vector_regression(X_selected_features, Y_selected_features)
        SVR_results = pd.DataFrame()
        SVR_results['Test_SVR'] = test
        SVR_results['Pred_SVR'] = predictions
        calculate_model_performance('Support Vector Regression', test = test, predictions = predictions)
        trained_models.append('Support Vector Regression')
        print('I trained the model')
    else:
        test = SVR_results['Test_SVR']
        predictions = SVR_results['Pred_SVR']
        print('I did not train the model')

    # Define the content for each index
    if active_index_vector == 0:
        table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
        table_body = [html.Tbody(html.Tr([html.Td(performance_df.at[i, 'Support Vector Regression']) for i in range(len(performance_df))]))]

        content = dbc.Spinner(html.Div(children=[
            html.H4("Performance Metrics Support Vector Regression", className='text-center mt-2'),
            dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_vector == 1:
        fig = px.scatter(x=test, y=predictions, labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'}, template='superhero')
        content = dbc.Spinner(html.Div(children=[
            html.H4("Scatter Plot Support Vector Regression", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_vector == 2:
        fig = px.line(y=[test[1:200],predictions[1:200]], template='superhero')
        fig.update_traces(name="Real Power Consumption")
        fig.update_traces(name="Predicted Power Consumption")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power Consumption",
            legend_title=''
        )
        content = dbc.Spinner(html.Div(children=[
            html.H4("Line Plot Support Vector Regression", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

#SLIDES FOR THE DECISION TREE MODEL

@app.callback(
    Output("content-tree", "children"),
    [Input("left-tree-button", "n_clicks"),
     Input("right-tree-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_tree, DT_results

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-tree-button':
        active_index_tree -= 1  # Go left
    if button_id == 'right-tree-button':
        active_index_tree += 1  # Go right

    if active_index_tree == 3:
        active_index_tree = 0
    if active_index_tree == -1:
        active_index_tree = 2

    if 'Decision Tree' not in trained_models:
        test, predictions = decision_tree(X_selected_features, Y_selected_features)
        DT_results = pd.DataFrame()
        DT_results['Test_DT'] = test
        DT_results['Pred_DT'] = predictions
        calculate_model_performance('Decision Tree', test = test, predictions = predictions)
        trained_models.append('Decision Tree')
        print('I trained the model')
    else:
        test = DT_results['Test_DT']
        predictions = DT_results['Pred_DT']
        print('I did not train the model')

    # Define the content for each index
    if active_index_tree == 0:
        table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
        table_body = [html.Tbody(html.Tr([html.Td(performance_df.at[i, 'Decision Tree']) for i in range(len(performance_df))]))]

        content = dbc.Spinner(html.Div(children=[
            html.H4("Performance Metrics Decision Tree", className='text-center mt-2'),
            dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_tree == 1:
        fig = px.scatter(x=test, y=predictions, labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'}, template='superhero')
        content = dbc.Spinner(html.Div(children=[
            html.H4("Scatter Plot Decision Tree", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_tree == 2:
        fig = px.line(y=[test[1:200],predictions[1:200]], template='superhero')
        fig.update_traces(name="Real Power Consumption")
        fig.update_traces(name="Predicted Power Consumption")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power Consumption",
            legend_title=''
        )
        content = dbc.Spinner(html.Div(children=[
            html.H4("Line Plot Decision Tree", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

#SLIDES FOR THE RANDOM FOREST MODEL

@app.callback(
    Output("content-forest", "children"),
    [Input("left-forest-button", "n_clicks"),
     Input("right-forest-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_forest, RF_results

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-forest-button':
        active_index_forest -= 1  # Go left
    if button_id == 'right-forest-button':
        active_index_forest += 1  # Go right

    if active_index_forest == 3:
        active_index_forest = 0
    if active_index_forest == -1:
        active_index_forest = 2

    if 'Random Forest' not in trained_models:
        test, predictions = random_forest(X_selected_features, Y_selected_features)
        RF_results = pd.DataFrame()
        RF_results['Test_RF'] = test
        RF_results['Pred_RF'] = predictions
        calculate_model_performance('Random Forest', test = test, predictions = predictions)
        trained_models.append('Random Forest')
        print('I trained the model')
    else:
        test = RF_results['Test_RF']
        predictions = RF_results['Pred_RF']
        print('I did not train the model')

    # Define the content for each index
    if active_index_forest == 0:
        table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
        table_body = [html.Tbody(html.Tr([html.Td(performance_df.at[i, 'Random Forest']) for i in range(len(performance_df))]))]

        content = dbc.Spinner(html.Div(children=[
            html.H4("Performance Metrics Random Forest", className='text-center mt-2'),
            dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_forest == 1:
        fig = px.scatter(x=test, y=predictions, labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'}, template='superhero')
        content = dbc.Spinner(html.Div(children=[
            html.H4("Scatter Plot Random Forest", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_forest == 2:
        fig = px.line(y=[test[1:200],predictions[1:200]], template='superhero')
        fig.update_traces(name="Real Power Consumption")
        fig.update_traces(name="Predicted Power Consumption")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power Consumption",
            legend_title=''
        )
        content = dbc.Spinner(html.Div(children=[
            html.H4("Line Plot Random Forest", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

#SLIDES FOR THE NEURAL NETWORK MODEL

@app.callback(
    Output("content-neural", "children"),
    [Input("left-neural-button", "n_clicks"),
     Input("right-neural-button", "n_clicks")]
)
def update_content(left_clicks, right_clicks):
    global active_index_neural, NN_results

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'left-neural-button':
        active_index_neural -= 1  # Go left
    if button_id == 'right-neural-button':
        active_index_neural += 1  # Go right

    if active_index_neural == 3:
        active_index_neural = 0
    if active_index_neural == -1:
        active_index_neural = 2

    if 'Neural Network' not in trained_models:
        test, predictions = neural_network(X_selected_features, Y_selected_features)
        NN_results = pd.DataFrame()
        NN_results['Test_NN'] = test
        NN_results['Pred_NN'] = predictions
        calculate_model_performance('Neural Network', test = test, predictions = predictions)
        trained_models.append('Neural Network')
        print('I trained the model')
    else:
        test = NN_results['Test_NN']
        predictions = NN_results['Pred_NN']
        print('I did not train the model')

    # Define the content for each index
    if active_index_neural == 0:
        table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
        table_body = [html.Tbody(html.Tr([html.Td(performance_df.at[i, 'Neural Network']) for i in range(len(performance_df))]))]

        content = dbc.Spinner(html.Div(children=[
            html.H4("Performance Metrics Neural Network", className='text-center mt-2'),
            dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_neural == 1:
        fig = px.scatter(x=test, y=predictions, labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'}, template='superhero')
        content = dbc.Spinner(html.Div(children=[
            html.H4("Scatter Plot Neural Network", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    elif active_index_neural == 2:
        fig = px.line(y=[test[1:200],predictions[1:200]], template='superhero')
        fig.update_traces(name="Real Power Consumption")
        fig.update_traces(name="Predicted Power Consumption")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power Consumption",
            legend_title=''
        )
        content = dbc.Spinner(html.Div(children=[
            html.H4("Line Plot Neural Network", className='text-center mt-2'),
            dcc.Graph(figure=fig)
        ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
    return content

#CHOSE COMPARISON OF METRICS
@app.callback(
    [Output('MAE-chart', 'figure'),
     Output('MBE-chart', 'figure'),
     Output('MSE-chart', 'figure'),
     Output('RMSE-chart', 'figure'),
     Output('cvRMSE-chart', 'figure'),
     Output('NMBE-chart', 'figure'),
     Output('best-model-radio', 'options')],
    [Input('url', 'pathname')]  # Trigger when the tab value changes
)
def update_graphs_on_tab_change(pathname):
    if pathname == "/models/choose-model":
        mae_fig = px.bar(x = trained_models, y=[performance_df.iloc[0, :].values], labels={'x': 'Model', 'y': 'Value'}, title='MAE')
        mae_fig.update_layout(showlegend=False)
        mbe_fig = px.bar(x = trained_models, y=[performance_df.iloc[1, :].values], labels={'x': 'Model', 'y': 'Value'}, title='MBE')
        mbe_fig.update_layout(showlegend=False)
        mse_fig = px.bar(x = trained_models, y=[performance_df.iloc[2, :].values], labels={'x': 'Model', 'y': 'Value'}, title='MSE')
        mse_fig.update_layout(showlegend=False)
        rmse_fig = px.bar(x = trained_models, y=[performance_df.iloc[3, :].values], labels={'x': 'Model', 'y': 'Value'}, title='RMSE')
        rmse_fig.update_layout(showlegend=False)
        cvrmse_fig = px.bar(x = trained_models, y=[performance_df.iloc[4, :].values], labels={'x': 'Model', 'y': 'Value'},title='cvRMSE')
        cvrmse_fig.update_layout(showlegend=False)
        nmbe_fig = px.bar(x = trained_models, y=[performance_df.iloc[5, :].values], labels={'x': 'Model', 'y': 'Value'}, title='NMBE')
        nmbe_fig.update_layout(showlegend=False)
        options = [{'label': model, 'value': model} for model in trained_models]
        return mae_fig, mbe_fig, mse_fig, rmse_fig, cvrmse_fig, nmbe_fig, options
    return dash.no_update

#CHANGING THE SELECTED MODEL:

@app.callback(
    Output("selected-model-output", "children"),  # Where to display the selected model
    Input("best-model-radio", "value")  # Listen for changes in the radio button selection
)
def update_selected_model(selected_button):
    global selected_model
    if selected_button != '':
        selected_model = selected_button
        print(selected_model)
    return f"Selected Model: {selected_model}"


#WORKING OF THE SLIDES FOR THE MODEL VALIDATION
@app.callback(
    Output("content-validation", "children"),
    [Input("left-validation-button", "n_clicks"),
     Input("right-validation-button", "n_clicks"),]
)

def update_content(left_clicks, right_clicks):
    global selected_model, active_index_validation, scaler_SVR, scaler_NN

    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if selected_model == '':
        content = html.H4('No model selected')
        return content
    else:
        if button_id == 'left-validation-button':
            active_index_validation -= 1  # Go left
        if button_id == 'right-validation-button':
            active_index_validation += 1  # Go right

        if active_index_validation == 3:
            active_index_validation = 0
        if active_index_validation == -1:
            active_index_validation = 2

        if selected_model == 'Autoregression':
            test = dataframe_test_data['Power [kW]']
            history = dataframe_cleaned_data['Power [kW]'][len(dataframe_cleaned_data['Power [kW]']) - 1:]
            history = [history[i] for i in range(len(history))]
            predictions = list()
            for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length - 1, length)]
                yhat = coef_AR[0]
                for d in range(1):
                    yhat += coef_AR[d + 1] * lag[1 - d - 1]
                obs = test[t]
                predictions.append(yhat)
                history.append(obs)
        elif selected_model == 'Linear Regression':
            test = dataframe_test_data['Power [kW]']
            features = dataframe_test_data[selected_features].values
            predictions = model_LR.predict(features)
        elif selected_model == 'Support Vector Regression':
            test = dataframe_test_data['Power [kW]']
            features = dataframe_test_data[selected_features].values
            predictions_scaled = model_SVR.predict(features)
            predictions = scaler_SVR.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        elif selected_model == 'Decision Tree':
            test = dataframe_test_data['Power [kW]']
            features = dataframe_test_data[selected_features].values
            predictions = model_DT.predict(features)
        elif selected_model == 'Random Forest':
            test = dataframe_test_data['Power [kW]']
            features = dataframe_test_data[selected_features].values
            predictions = model_RF.predict(features)
        elif selected_model == 'Neural Network':
            test = dataframe_test_data['Power [kW]']
            features = dataframe_test_data[selected_features].values
            predictions_scaled = model_NN.predict(features)
            predictions = scaler_NN.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        # Define the content for each index
        if active_index_validation == 0:
            MAE = abs(metrics.mean_absolute_error(test, predictions))
            MBE = abs(np.mean(test - predictions))
            MSE = abs(metrics.mean_squared_error(test, predictions))
            RMSE = abs(np.sqrt(MSE))
            cvRMSE = abs(RMSE / np.mean(test))
            NMBE = abs(MBE / np.mean(test))
            table_header = [html.Thead(html.Tr([html.Th("MAE"), html.Th("MBE"), html.Th("MSE"), html.Th("RMSE"), html.Th("cvRMSE"), html.Th("NMBE")]))]
            table_body = [html.Tbody(html.Tr([html.Td(MAE), html.Td(MBE), html.Th(MSE), html.Th(RMSE), html.Th(cvRMSE), html.Th(NMBE)]))]
            content = dbc.Spinner(html.Div(children=[
                html.H4(f"Performance Metrics {selected_model}", className='text-center mt-2'),
                dbc.Table(table_header + table_body, bordered=True, className='mt-5', size='md')
            ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
        elif active_index_validation == 1:
            fig = px.scatter(x=test, y=predictions,
                             labels={'x': 'Real Power Consumption', 'y': 'Predicited Power Consumption'},
                             template='superhero')
            content = dbc.Spinner(html.Div(children=[
                html.H4(f"Scatter Plot {selected_model}", className='text-center mt-2'),
                dcc.Graph(figure=fig)
            ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)
        elif active_index_validation == 2:
            fig = px.line(x = dataframe_test_data.index, y=[test, predictions], template='superhero')
            fig.update_traces(name="Real Power Consumption")
            fig.update_traces(name="Predicted Power Consumption")
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Power Consumption",
                legend_title=''
            )
            content = dbc.Spinner(html.Div(children=[
                html.H4(f"Line Plot {selected_model}", className='text-center mt-2'),
                dcc.Graph(figure=fig)
            ], className="mt-1", style={"width": "100%"}), color='primary', delay_show=100)

        return content


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
