from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

df = pd.read_csv('complete_data.csv')
features = df.columns[1:-1]


scaler = StandardScaler()

df = pd.read_csv('complete_data.csv')
df2 = df.fillna(0)
df2['Party'].replace(['I','D', 'R'],
                        [0, 1,2], inplace=True)
df2 = df2.iloc[:,2:-1]
scaled_df = df2.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)

kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_df)
cluster_id = pd.DataFrame(kmeans.predict(scaled_df))
cluster_id.columns=['cluster_id']

n_components = 4

pca = PCA(n_components=n_components)
components = pca.fit_transform(scaled_df)
total_var = pca.explained_variance_ratio_.sum() * 100
dfpcp = df2.copy()
dfpcp['Party'].replace([0, 1,2],
                        ['I','D', 'R'], inplace=True)
fig2 = px.parallel_coordinates(df2.iloc[:,1:10].join(cluster_id),template =  "plotly_dark" ,color="cluster_id")


loadings = pca.components_.T * np.sqrt(pca.explained_variance_)*10

fig3 = px.scatter(components, x=0, y=1, color=scaled_df.join(cluster_id)['cluster_id'], template = "plotly_dark", height=500, title='PCA Loadings',
        labels={
                     "0": "Principle Component 1",
                     "1": "Principle Component 2",
                     "color": "Cluster"
                 },)

for i, feature in enumerate(features):
    fig3.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig3.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
    if i == 28:
        break


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


opts = [{'label' : i, 'value' : i} for i in features[2:-1]]

app = Dash(__name__)
app.layout = html.Div(children=[
    html.Div(children=[
        dcc.Dropdown(id = 'opt', 
                        options = features[2:-1],
                        value = features[2],style = {'width': '300px', 'display': 'inline-block'}),
         html.H1('Multivariate Wellness Dashboard by Nicholas Barrett',style={'display': 'inline-block', 'color' : 'white' ,'marginLeft' : 100})
        
    ]),                
    html.Div(children=[
        
        dcc.Graph(id="choro", style={'display': 'inline-block'}, clickData= {'points': [{'location': 'NY'}]}),
        dcc.Graph(id="line", style={'display': 'inline-block'}),
        dcc.Graph(id="biplot", style={'display': 'inline-block'},figure = fig3)
    ]), 
    html.Div(children=[
        dcc.Graph(id="pcp",figure = fig2,style={'width': '120vh', 'height': '45vh','display': 'inline-block'}),
        dcc.Graph(id="PCA", style={'width': '70vh', 'height': '45vh','display': 'inline-block'}),
        
    ]),

   
                    ])


@app.callback(
    Output("choro", "figure"), 
    Input("opt", "value"),
    )
def display_choropleth(X):
    df = pd.read_csv('complete_data.csv')
    df2 = df.fillna(0)
    #df2 = df2.loc[df2['Year'] == Y]
    
    fig = px.choropleth(df2, 
              locations = 'State',
              color=X, 
              animation_frame="Year",
              color_continuous_scale="Inferno",
              locationmode='USA-states',
              scope="usa",
              
              title= X +' by State',
              height=500,
              template =  "plotly_dark" 
             )
    fig.update_layout(clickmode='event+select')
    return fig


@app.callback(
    Output("PCA", "figure"), 
    Input("opt", "value"),
    )
def display_PCA(X):

    labels = {str(i): f"PC {i+1}" for i in range(n_components)}
    labels['color'] = X
    fig = px.scatter_matrix(
        components,
        color = df2[X],
        dimensions=range(n_components),
        labels = labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
        template =  "plotly_dark" 
    )
    fig.update_traces(diagonal_visible=False),
    
    return fig

@app.callback(
    Output("line", "figure"), 
    Input("opt", "value"),
    Input("choro", "clickData"),
    )
def line_plot(X,clickData):
    state = clickData['points'][0]['location']
    df = pd.read_csv('complete_data.csv')
    df2 = df.fillna(0)
    df2['Party'].replace(['I','D', 'R'],
                        [0, 1,2], inplace=True)
    fig = px.line(df2.loc[df2["State"]==state], x="Year", y=X,template =  "plotly_dark", title= state + ' ' + X + ' over time', height=500)
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)

# %%
