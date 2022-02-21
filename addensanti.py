import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




st.set_page_config(
     page_title="Addensanti",
     page_icon='📈',
     layout="centered",
     initial_sidebar_state="collapsed",
     menu_items={
         'About': "## arturofrisina@hotmail.it"})

st.title('Addensanti e gelato')
st.header('Effetti sulla viscosità e sull\'overrun')
st.subheader('Sono state preparati 10 diversi tipi di miscele ternarie contenenti farina di semi di carrube (FSC), gomma di tara e gomma di guar. \nCome approccio per la raccolta dei dati è stato utilizzato un mixture design di tipo simplex lattice')


df=pd.read_excel('Addensanti_py.xlsx')


st.title('')
with st.expander('Guarda i dati'):
	st.dataframe(df)

fsc=np.array(df['FSC'])
tara=np.array(df['TARA'])
guar=np.array(df['GUAR'])
viscosità=np.array(df['viscosità'])
overrun=np.array(df['overrun'])
nomi=np.array(df['ID'])

visco_plot = ff.create_ternary_contour(
    np.array([fsc,tara,guar]), viscosità,
    pole_labels=['FSC', 'TARA', 'GUAR'],
    ncontours=50,showscale=True,showmarkers=True)

st.title('')
st.subheader('In questo grafico l\'interazione tra gli addensanti e la viscosità (misurata in cP)')

st.plotly_chart(visco_plot, use_container_width=True)


st.title('')
st.title('')
st.subheader('Qui invece l\'interazione tra gli addensanti e l\'overrun (in %)')

overr_plot = ff.create_ternary_contour(
    np.array([fsc,tara,guar]), overrun,
    pole_labels=['FSC', 'TARA', 'GUAR'],
    ncontours=50,showscale=True,colorscale='Viridis',showmarkers=True)

st.plotly_chart(overr_plot, use_container_width=True)

#Creo un dataframe con i dati autoscalati

pca_array=df.loc[:,('viscosità','overrun')]
pca_df=pd.DataFrame(pca_array)
pca_df=preprocessing.scale(pca_df)
pca_ = pd.DataFrame(pca_df,index=nomi,columns=('viscosità','overrun'))


st.title('')
st.subheader('Dopodiché i dati sono stati elaborati secondo un modello di analisi delle componenti principali (PCA)')
st.title('')

with st.expander('Dati per la PCA autoscalati'):
	st.dataframe(pca_)


pca=PCA()
pca.fit(pca_df)
pca_data = pca.transform(pca_df)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC'+ str(x) for x in range (1, len(per_var)+1)]

 #creo lo scree plot

plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
scree=plt.show()


with st.expander('Visualizza lo scree plot'):
	st.pyplot(plt)

pca_df = pd.DataFrame(pca_df,index=nomi,columns=labels)


features=['viscosità','overrun']


#PREPARO IL GRAFICO DEI LOADING

loading=pca.components_*np.sqrt(pca.explained_variance_)

load=pd.DataFrame(loading)

fig=plt.figure()
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('Scores + Loadings')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.xlim(-2,3)
plt.ylim(-2,3)

#plt.scatter(load[0],load[1])
plt.annotate('viscosità',load.loc[1])
plt.annotate('overrun',load.loc[0])
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.xlim(-2,3)
plt.ylim(-2,3)
plt.arrow(0,0,-0.81,0.611,color='red')
plt.arrow(0,0,0.813,0.611,color='red')


X=pca_df.loc[:,'PC1']
Y=pca_df.loc[:,'PC2']

for i,label in enumerate(nomi):
    plt.annotate(label, (X[i], Y[i]))   


st.subheader('Grafico riassuntivo')

st.pyplot(fig)

st.title('')
st.title('')
st.title('')
st.title('')
st.markdown('### Arturo Frisina')




