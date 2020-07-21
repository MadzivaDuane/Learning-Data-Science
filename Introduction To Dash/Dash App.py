#------------------------------------------------------------------------------------------------------------------
"""Building my first Dash App"""
#------------------------------------------------------------------------------------------------------------------

"""Goal:
Build a functional beginner dashboard"""

#------------------------------------------------------------------------------------------------------------------
"""Import packages"""
#------------------------------------------------------------------------------------------------------------------
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd 
import numpy as np 
import plotly.offline as pyo 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#initialize app
app = dash.Dash(__name__)

#------------------------------------------------------------------------------------------------------------------
"""Load and Clean Dataset """
#------------------------------------------------------------------------------------------------------------------
path = '/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Data/'
bees_data = pd.read_csv(path+'intro_bees.csv')
bees_data.head()

df = bees_data.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
df.head(20)
df.reset_index(inplace = True)
df.head(20)



#------------------------------------------------------------------------------------------------------------------
"""Dataframes and Functions"""
#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
"""App Layout"""
#------------------------------------------------------------------------------------------------------------------
#define app layout
app.layout = html.Div([
    html.H1("Web Application Dashboard with Dash", style = {
                                                            "text-align": "center", "font-size": "30px",
                                                            "background-color": "#198805", "font-family": "courier"
                                                            }),
    
    dcc.Dropdown(id = "select_year",
                options = [
                    {"label": "2015", "value": 2015},
                    {"label": "2016", "value": 2016},
                    {"label": "2017", "value": 2017},
                    {"label": "2018", "value": 2018}],
                multi = False,
                value = 2015,
                style = {
                        "width": "40%"
                }),

    html.Div(id = "output_container", children = []),
    html.Br(),

    dcc.Graph(id = "my_bee_map", figure = [])
])

#------------------------------------------------------------------------------------------------------------------
"""Call Backs"""
#------------------------------------------------------------------------------------------------------------------
@app.callback(
    [Output(component_id = "output_container", component_property = "children"), Output(component_id = "my_bee_map", component_property = "figure")],
    [Input(component_id = "select_year", component_property = "value")]
)

def update_figure(year_selected):
    print(year_selected)
    print(type(year_selected))

    container = "The year chosen is: {}".format(year_selected)

    dff = df.copy()
    dff = dff[dff["Year"] == year_selected]
    dff = dff[dff["Affected by"] == "Varroa_mites"]

    #use plotly express to create plot 
    fig = px.choropleth(
        data_frame = dff,
        locationmode = "USA-states",
        locations = "state_code",
        scope = "usa",
        color = "Pct of Colonies Impacted",
        hover_data = ["State", "Pct of Colonies Impacted"],
        color_continuous_scale = px.colors.sequential.YlOrRd,
        labels = {"Pct of Colonies Impacted": "% of Bee Colonies"},
        template = "plotly_dark")

    return container, fig 

#------------------------------------------------------------------------------------------------------------------
"""Run App"""
#------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1')

