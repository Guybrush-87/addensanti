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
st.subheader('Sono state preparati 10 diversi tipi di miscele ternarie contenenti farina di semi di carrube (FSC), gomma di tara e gomma di guar. Come approccio per la raccolta dei dati è stato utilizzato un mixture design di tipo simplex lattice')


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

st.plotly_chart(pca_)


