# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %%
df1 = pd.read_csv('merged_data.csv',encoding='cp1252')
df2 = pd.read_csv('Minimum Wage Data.csv',encoding='cp1252')
# %%
df1.head
#%%
df2.head
#%%

# Get Crosswalk From Generic Website
cw_location = 'http://app02.clerk.org/menu/ccis/Help/CCIS%20Codes/'
cw_filename = 'state_codes.html'
states = pd.read_html(cw_location + cw_filename)[0]

# Create New Variable With State Abbreviations
state_code_map = dict(zip(states['Description'],states['Code']))
#%%
df2['State'] = df2['State'].map(state_code_map)
#%%
df2.head(5)
#%%
df1.rename(columns = {'state':'State','year':'Year'},inplace = True)
# %%
df_merged = pd.merge(df1,df2, on=['Year','State'],how = 'left')
df_merged.head()
#%%
df_merged.to_csv('merged_wage.csv')

#%%
df1 = pd.read_csv('merged_wage3.csv',encoding='cp1252')
df2 = pd.read_csv('states_party_strength_cleaned2.csv',encoding='cp1252')
#%%
df2['State'] = df2['State'].map(state_code_map)

#%%
df_merged = pd.merge(df1,df2, on=['Year','State'],how = 'left')
df_merged.head()
#%%
df_merged.to_csv('complete_data.csv')
# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%%
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

pca = PCA(n_components=4)
#fit PCA model to data
pca_fit = pca.fit(scaled_df)

#%%
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black')

#%%
import matplotlib.pyplot as plt
import numpy as np

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

#%%
y_kmeans = kmeans.predict(scaled_df)
# %%
import plotly.express as px
cluster_id = pd.DataFrame(kmeans.predict(scaled_df))
cluster_id.columns=['cluster_id']
#%%
fig2 = px.parallel_coordinates(df2.iloc[:,1:15].join(cluster_id), dimensions=["income", "State.Minimum.Wage.2020.Dollars", "Party", "pregnancyrate1517", "pregnancyrate1819"],template =  "plotly_dark" ,color="cluster_id")

fig2.show()
# %%
#scatter Plot matrix
import plotly.express as px
fig = px.scatter_matrix(df2.iloc[:,1:5].join(cluster_id),dimensions=["income", "State.Minimum.Wage.2020.Dollars", "Party", "pregnancyrate1517"],color="cluster_id")
fig.update_traces(diagonal_visible=False)
fig.show()
# %%
#PCA Scatter plot
n_components = 4

pca = PCA(n_components=n_components)
components = pca.fit_transform(df2)

total_var = pca.explained_variance_ratio_.sum() * 100
labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'income'

fig = px.scatter_matrix(
    components,
    color = df2.income,
    dimensions=range(n_components),
    labels = labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()
# %%
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
labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'income'

# %%

fig = px.scatter_matrix(
    components,
    color = df2['income'],
    dimensions=range(n_components),
    labels = labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()
# %%

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)*10

fig = px.scatter(components, x=0, y=1, color=scaled_df.join(cluster_id)['cluster_id'])

for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
    if i == 28:
        break
fig.show()
#%%

#Line chart
df = pd.read_csv('complete_data.csv')
df2 = df.fillna(0)
df2['Party'].replace(['I','D', 'R'],
                        [0, 1,2], inplace=True)

fig = px.line(df2.loc[df2["State"]=='NY'], x="Year", y="income")
fig.show()
# %%
dfpcp = df2.copy()
dfpcp['Party'].replace([0, 1,2],
                        ['I','D', 'R'], inplace=True)
# %%
