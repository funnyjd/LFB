import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import joblib
import os
import itertools

import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, classification_report, confusion_matrix, accuracy_score

st.markdown(
    """
    <style>
    /* Changer la couleur des titres */
    h1 {
        color: #C00000;
        font-weight: bold; /* Mettre les titres en gras */
    }

    /* Changer la couleur des titres */
    h2 {
        color: #0f52ba;
        font-weight: bold; /* Mettre les titres en gras */
    }

    /* Changer la couleur des titres */
    h3 {
        color: #0f52ba;
        font-weight: normal; /* Mettre les titres h2 sans gras */
    }
    """,
    unsafe_allow_html=True)

LFB=pd.read_csv("FINAL_LFB Mobilisation & Incident data from 2018 - 2023.zip", compression='zip', encoding='ISO-8859-1', sep=',', on_bad_lines='skip')

@st.cache_data
def load_data():
    Inc1824_head=pd.read_csv("1_1_Inc1824_Head.csv")
    Inc1824_describe=pd.read_csv("1_2_Inc1824_Describe.csv")
    Inc1824_NaN=pd.read_csv("1_3_Inc1824_Nan.csv")
    Mob1520_head=pd.read_csv("2_1_Mob1520_Head.csv")
    Mob1520_describe=pd.read_csv("2_2_Mob1520_Describe.csv")
    Mob1520_NaN=pd.read_csv("2_3_Mob1520_Nan.csv")
    Mob2124_head=pd.read_csv("3_1_Mob2124_Head.csv")
    Mob2124_describe=pd.read_csv("3_2_Mob2124_Describe.csv")
    Mob2124_NaN=pd.read_csv("3_3_Mob2124_Nan.csv")
    Final_head=pd.read_csv("4_1_Final_Head.csv")
    Final_describe=pd.read_csv("4_2_Final_Describe.csv")
    Final_NaN=pd.read_csv("4_3_Final_Nan.csv")
    Distrib_RT=pd.read_csv("5_Distrib_ResponseTime.csv")
    Results_Reg=pd.read_csv("Results_regressor.csv", encoding="latin1", sep=";")
    return Results_Reg, Distrib_RT, Inc1824_head, Inc1824_describe, Inc1824_NaN, Mob1520_head, Mob1520_describe, Mob1520_NaN, Mob2124_head, Mob2124_describe, Mob2124_NaN, Final_head, Final_describe, Final_NaN

Results_Reg, Distrib_RT, Inc1824_head, Inc1824_describe, Inc1824_NaN, Mob1520_head, Mob1520_describe, Mob1520_NaN, Mob2124_head, Mob2124_describe, Mob2124_NaN, Final_head, Final_describe, Final_NaN = load_data()

st.title("Projet de prédiction du temps de réponse de la Brigade des Pompiers de Londres")
st.sidebar.title("Sommaire")
pages=["Introduction ⛑️", "Exploration des données 🔎", "DataVizualization 📊", "Modélisation par Régression 🛠️", 
       "Modélisation par Classification 🛠️","Conclusion 📝"]
page=st.sidebar.radio("Aller vers", pages)

st.sidebar.title("Cursus")

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Data Analyst</p>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Bootcamp - Septembre 2024</p>
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Participants au projet")

# URL de l'image du logo LinkedIn
linkedin_logo_url = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"

# URL vers votre profil LinkedIn Julie Adeline
linkedin_profile_julieA_url = "https://www.linkedin.com/in/julie-adeline/"  # Remplacez par votre URL LinkedIn
# Utiliser Markdown pour afficher le logo et le nom sur la même ligne
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Julie ADELINE</p>
    <a href="{linkedin_profile_julieA_url}" target="_blank" style="margin-left: 8px;">
        <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:20px; height:20px; position: relative; top: -2px;">
    </a>
</div>
""", unsafe_allow_html=True)

# URL vers votre profil LinkedIn Julie Deng
linkedin_profile_JulieD_url = "https://www.linkedin.com/in/jjdeng/"  # Remplacez par votre URL LinkedIn
# Utiliser Markdown pour afficher le logo et le nom sur la même ligne
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Julie DENG</p>
    <a href="{linkedin_profile_JulieD_url}" target="_blank" style="margin-left: 8px;">
        <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:20px; height:20px; position: relative; top: -2px;">
    </a>
</div>
""", unsafe_allow_html=True)

# URL vers votre profil LinkedIn Elena Stratan
linkedin_profile_Elena_url = "https://www.linkedin.com/in/elena-stratan-34667b147/"  # Remplacez par votre URL LinkedIn
# Utiliser Markdown pour afficher le logo et le nom sur la même ligne
st.sidebar.markdown(f"""
<div style="display: flex; align-items: center;">
    <p style="margin: 0;">Elena STRATAN</p>
    <a href="{linkedin_profile_Elena_url}" target="_blank" style="margin-left: 8px;">
        <img src="{linkedin_logo_url}" alt="LinkedIn" style="width:20px; height:20px; position: relative; top: -2px;">
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Données")
st.sidebar.markdown("""
        [Ville de Londres](https://data.london.gov.uk/)
    """)

if page==pages[0]:
    st.image("lfb.png")
    st.header("Introduction ⛑️")
    
    st.write("Portant sur les interventions des Pompiers de Londres entre 2018 et 2023, notre projet\
        cherche à révéler les facteurs déterminant le temps de réponse des pompiers\
        et à réaliser des prédictions à partir de ces variables explicatives.")
    st.subheader("Variable cible: le temps de réponse de la LFB")
    st.image('response_time.png')
   
if page==pages[1] :
    st.image("lfb.png")
    st.header("Exploration des données🔎")
    st.subheader("Aperçu général des données")
    st.write("Cette section présente un aperçu des différentes données utilisées pour l'analyse.")
    
# Sélection du jeu de données
    dataset_choice = st.radio("Choisissez un jeu de données à explorer", 
                              ["Incidents (2018-2024)", "Mobilisation (2015-2020)", "Mobilisation (2021-2024)", "Jeu de données final : 2018 - 2023"])
    
    if dataset_choice == "Incidents (2018-2024)":
        st.subheader("1. Source")
        st.markdown("Le jeu de données provient du site du gouvernement de Londres : [London Fire Brigade Incident Records](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)")
        
        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2018 à 2024.")
        
        st.subheader("3. Exploration des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 759359 lignes x 39 colonnes")
        st.markdown("**Les premières lignes de ce jeu de données :**")
        if 'Unnamed: 0' in Inc1824_head.columns:
            Inc1824_head.set_index('Unnamed: 0', inplace=True)
        Inc1824_head.index.name=None
        st.dataframe(Inc1824_head)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        if 'Unnamed: 0' in Inc1824_describe.columns:
            Inc1824_describe.set_index('Unnamed: 0', inplace=True)
        Inc1824_describe.index.name=None
        st.dataframe(Inc1824_describe)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valeurs manquantes :**")
            if 'Unnamed: 0' in Inc1824_NaN.columns:
                Inc1824_NaN.set_index('Unnamed: 0', inplace=True)
            Inc1824_NaN.index.name=None
            st.dataframe(Inc1824_NaN)
        with col2:
            st.markdown("**Informations :**")
            st.image("Inc2018.png")

    elif dataset_choice == "Mobilisation (2015-2020)":
        st.subheader("1. Source")
        st.markdown("Le jeu de données provient du site du gouvernement de Londres : [London Fire Brigade Mobilisation Records](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")

        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2015 à 2020.")
        
        st.subheader("3. Exploration des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 883641  lignes x 22 colonnes")
        st.markdown("**Les premières lignes de ce jeu de données :**")
        if 'Unnamed: 0' in Mob1520_head.columns:
            Mob1520_head.set_index('Unnamed: 0', inplace=True)
        Mob1520_head.index.name=None
        st.dataframe(Mob1520_head)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        if 'Unnamed: 0' in Mob1520_describe.columns:
            Mob1520_describe.set_index('Unnamed: 0', inplace=True)
        Mob1520_describe.index.name=None
        st.dataframe(Mob1520_describe)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valeurs manquantes :**")
            if 'Unnamed: 0' in Mob1520_NaN.columns:
                Mob1520_NaN.set_index('Unnamed: 0', inplace=True)
            Mob1520_NaN.index.name=None
            st.dataframe(Mob1520_NaN)
        with col2:
            st.markdown("**Informations :**")
            st.image("Mob2015.png")
        
    elif dataset_choice == "Mobilisation (2021-2024)":
        st.subheader("1. Source")
        st.markdown("Le jeu de données provient du site du gouvernement de Londres : [London Fire Brigade Mobilisation Records](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")
        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2021 à 2024.")
        
        st.subheader("3. Exploration des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 659200  lignes x 24 colonnes")
        st.markdown("**Les premières lignes de ce jeu de données :**")
        Mob2124_head.set_index('Unnamed: 0', inplace=True)
        Mob2124_head.index.name=None
        st.dataframe(Mob2124_head)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        if 'Unnamed: 0' in Mob2124_describe.columns:
            Mob2124_describe.set_index('Unnamed: 0', inplace=True)
        Mob2124_describe.index.name=None
        st.dataframe(Mob2124_describe)

        col1, col2 = st.columns(2)
        with col1: 
            st.markdown("**Valeurs manquantes :**")
            if 'Unnamed: 0' in Mob2124_NaN.columns:
                Mob2124_NaN.set_index('Unnamed: 0', inplace=True)
            Mob2124_NaN.index.name=None
            st.dataframe(Mob2124_NaN)
        with col2:
            st.markdown("**Informations :**")
            st.image("Mob2021.png")
        
    elif dataset_choice == "Jeu de données final : 2018 - 2023":
        st.subheader("1. Source")
        st.markdown("Le jeu de données est une fusion des 3 jeux de données précédents regroupant les incidents et les mobilisations.")        
        st.subheader("2. Période")
        st.markdown("Les données couvrent la période de 2018 à 2023.")
        st.subheader("3. Feature Engineering")
        st.markdown("Afin de continuer notre étude sur le temps de réponse de la Brigade des Pompiers de Londres, nous avons réalisé des modifications sur nos jeux de données pour obtenir un fichier final exploitable et pertinent.")
        st.markdown("Pour cela, nous avons :")
        st.markdown("- Supprimé les colonnes non nécessaire à notre étude,")
        st.markdown("- Créé la variable cible 'ResponseTimeSeconds'")

        st.subheader("4. Fusion des données")
        st.markdown("**Langage utilisé** : Python")
        st.markdown(f"**Taille du DataFrame** : 634025  lignes x 22 colonnes")
        st.markdown("**Les premières lignes de ce jeu de données :**")
        Final_head.set_index('Unnamed: 0', inplace=True)
        Final_head.index.name=None
        st.dataframe(Final_head)
        st.markdown("**Résumé statistique de tout le jeu de données :**")
        if 'Unnamed: 0' in Final_describe.columns:
            Final_describe.set_index('Unnamed: 0', inplace=True)
        Final_describe.index.name=None
        st.dataframe(Final_describe)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valeurs manquantes :**")
            if 'Unnamed: 0' in Final_NaN.columns:
                Final_NaN.set_index('Unnamed: 0', inplace=True)
            Final_NaN.index.name=None
            st.dataframe(Final_NaN)
        with col2:
            st.markdown("**Informations :**")
            st.image("final.png")

if page==pages[2] :
    st.image("lfb.png")
    st.header("DataVizualization 📊")
    
    categories=["Analyses Univariées", "Analyses Multivariées", "Analyses Statistiques"]
    categorie=st.selectbox("Types d'analyses", categories)

    if categories[0] in categorie :

        types=["Incidents par types", "Incidents par périodes", "Incidents par localisation"]
        type=st.multiselect("Sélection", types)

        if types[0] in type : 
            
            #Création d'un groupe avec le décompte par type d'incident par années
            incident_counts = LFB.groupby(['CalYear'])['IncidentGroup'].value_counts().unstack()
        
            #Création d'un graphique en barre empilée pour visualisier la répartition des incidents par type par années
            fig=plt.figure(figsize=(5,4))
            incident_counts.plot(kind='bar', stacked=True, ax=fig.gca())
            plt.xlabel("Années")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par années par type d'incident")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=360)
            st.pyplot(fig)

            #Création d'un graphique pour visualiser la répartition des types d'incidents
            fig=plt.figure(figsize=(7,4))
            sns.countplot(x='IncidentGroup',data=LFB)
            plt.xlabel("Types d'incidents")
            plt.ylabel("Nombre d'incidents")
            plt.xticks(["Special Service","Fire","False Alarm"],["Special Service","Feu","Fausse alarme"])
            plt.title("Types d'incidents")
            st.pyplot(fig)

            #Ajout des explications des deux graphiques
            st.write("Nous constatons que la répartition des types d’incident par année est sensiblement la même avec environ 50% de déplacements pour fausses alertes, 30% de déplacements pour causes de services spéciaux et 20% de déplacements pour des incendies.")
            st.write("De plus, il ne semble pas y avoir de différences entre le nombre d'incidents par années.")
            st.write("Il semble y avoir eu un léger recul du nombre d'incidents en 2020 pendant le COVID suivie d'une légère hausse des incidents en 2021,2022 et 2023.")

            # Retirer de la colonne SpecialService les entrées qui ne sont pas des SpecialService
            Special_Service=LFB.loc[LFB['SpecialServiceType']!='Not a special service']

            #Faire le décompte de toutes les valeurs de SpecialService
            counts = Special_Service['SpecialServiceType'].value_counts().reset_index()

            #Trier cette liste dans l'ordre décroissant
            counts.columns = ['Special_Service', 'nombre']
            counts = counts.sort_values(by='nombre', ascending=False)

            #Création d'un graphique pour visualiser les types de special service triés par ordre d'importance
            fig=plt.figure(figsize=(7,5))
            sns.barplot(y='Special_Service', x='nombre', data=counts, order=counts['Special_Service'])
            plt.xlabel('Types de Special Service')
            plt.ylabel('Nombre')
            plt.title("Nombre d'incidents par type de Special Service")
            st.pyplot(fig)

            #Ajouter des explications
            st.write("Nous avons pu constater qu'environ 30% des activités des pompiers de la Brigade de Londres concernent des services spéciaux autre que la gestion des incendies.")
            st.write("On peut constater que la majorité de ces services spéciaux concernent des ouvertures de portes (hors incendies et urgences vitales) des inondations, des accidents de la route (RTC) et de l'assistance aux personnes bloqués dans des ascenseurs. ")

            #Création d'une variable pour visualiser le TOP5 des lieux avec le plus d'incidents
            Top_PropertyCategory=LFB['PropertyCategory'].value_counts().head()

            #Création d'un graphique pour visualiser le TOP5 des lieux avec le plus d'incidents
            fig=plt.figure(figsize=(8,5))
            sns.barplot(x=Top_PropertyCategory.values,y=Top_PropertyCategory.index)
            plt.ylabel("Types de lieux")
            plt.xlabel("Nombre d'incidents")
            plt.title("Top 5 des catégories de lieux avec le plus d'incidents")
            st.pyplot(fig)

            #Ajout des explications
            st.write("On constate dans cette variable une disparité importante avec une très forte majorité des incidents ayant lieu dans les habitations résidentielles.")

        if types[1] in type : 
            #Création d'un graphique en barre pour visualisier la distribution des incidents par mois
            fig=plt.figure(figsize=(10,6))
            sns.countplot(x='CalMonth',data=LFB,hue='CalMonth',legend=False,palette='rainbow')
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par mois")
            plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11],["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Décembre"],
                    rotation=45, ha='right')
            st.pyplot(fig)

            #Création d'un graphique en barre pour visualisier la distribution des incidents par jours de la semaine
            fig=plt.figure(figsize=(8,4))
            sns.countplot(x='CalWeekday',data=LFB,hue='CalWeekday',legend=False,palette='rainbow')
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par jours de la semaine")
            plt.xticks([0,1,2,3,4,5,6],["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],rotation=45,ha='right')
            st.pyplot(fig)

            #Création d'un graphique en barre pour visualisier la distribution des incidents par heures de la journée
            fig=plt.figure(figsize=(8,5))
            sns.countplot(x='HourOfCall',data=LFB,hue='HourOfCall',legend=False,palette='rainbow')
            plt.xlabel("Mois")
            plt.ylabel("Nombre d'incidents")
            plt.title("Nombre d'incidents par heures de la journée")
            plt.xticks([0,6,12,18,23],['Minuit','6h','Midi','18h','23h'])
            st.pyplot(fig)

            #Ajout des explications des graphiques temporels
            st.write("Le nombre d'incident semble également répartis sur l'ensemble des mois de l'année à part un léger pic en été.")
            st.write("Le nombre d'incident semble également répartis sur l'ensemble de la semaine.")
            st.write("On constate que une disparité de la répartition du nombre d'incidents selon l'heure de la journée.")
            st.write("Il semble y avoir plus d'incidents en journée que la nuit, avec un pic aux heures de pointes entre 17h et 20h.")
        
        if types[2] in type :
           #Création d'une variable pour visualiser le TOP10 des quartiers avec le plus d'incidents
            Top_Borough=LFB['IncGeo_BoroughName'].value_counts().head(10)

            #Création d'un graphique pour visualiser le TOP10 des lieux avec le plus d'incidents
            fig=plt.figure(figsize=(8,5))
            sns.barplot(x=Top_Borough.values,y=Top_Borough.index)
            plt.ylabel("Noms de quartier")
            plt.xlabel("Nombre d'incidents")
            plt.title("Top 10 des quartiers de Londres avec le plus d'incidents")
            st.pyplot(fig)

            #Création d'une variable avec les lignes du quartier de Westminster
            West=LFB[(LFB['IncGeo_BoroughName']=='WESTMINSTER')]

            #Création d'une variable pour visualiser le TOP5 des ward avec le plus d'incidents dans le quartier de Westminster
            West_no_other=West[West['IncGeo_WardName']!='OTHERS']
            Top_Ward=West_no_other['IncGeo_WardName'].value_counts().head(5)

            #Création d'un graphique pour visualiser le TOP5 des ward de westminster avec le plus d'incidents
            fig=plt.figure(figsize=(7,5))
            sns.barplot(x=Top_Ward.values,y=Top_Ward.index)
            plt.ylabel("Noms des ward")
            plt.xlabel("Nombre d'incidents")
            plt.title("Top 5 des wards de Westminster avec le plus d'incidents")
            st.pyplot(fig)

            #Ajout des explications 
            st.write("Une dernière observation intéressante est que les incidents sont répartis de manière équitable entre les différents quartiers, à l'exception du quartier de Westminster qui recense près du double d’incidents en plus que les autres quartiers.")
            st.write("Nous sommes allées plus loin et dans le quartier de Westminster, il y a deux wards (ou circonscriptions) qui ont une plus forte densité d'incidents : West End et St James's.")
            st.write("Cela peut s'expliquer par le fait que St James's et West End sont deux zones de Londres qui concentrent beaucoup de commerces et de lieux touristiques, politiques et culturels. En effet, on peut y trouver le Parlement britannique, Soho, Mayfair, la National Gallery et l'abbaye de Westminster.")

    if categories[1] in categorie :
        
        types=["Distribution du temps de réponse", "Temps de réponse par périodes", "Temps de réponse par lieux de déploiements"]
        type=st.multiselect("Sélection", types)

        if types[0] in type :   
            
            #Visualisation de la distribution du temps de réponse en secondes
            fig=plt.figure(figsize=(10,5))
            sns.boxplot(x=LFB['ResponseTimeSeconds'])
            plt.xticks([0,100,200,300,400,500,600,700,800,900,1000,1100,1200])
            plt.xlabel("Temps de réponse en secondes")
            plt.title("Boxplot des temps de réponse")
            st.pyplot(fig)

            #Information sur la donnée
            if 'Unnamed: 0' in Distrib_RT.columns:
                Distrib_RT.set_index('Unnamed: 0', inplace=True)
            Distrib_RT.index.name=None
            st.dataframe(Distrib_RT)

            #Ajout des explications
            st.write("Les valeurs au-delà de 500 semblent plutôt des valeurs extrêmes lors d'opérations plus longues. La distribution du temps de réponse pose une moyenne à 313 secondes et une médiane à 300 secondes. Soit environ 5 minutes entre le moment où les pompiers sont mobilisés et le moment où ils arrivent sur les lieux de l'incident.")

        if types[1] in type : 

            #Création d'un graphique permettant de visualiser l'évolution du temps de réponse par mois et par années
            fig=plt.figure(figsize=(10,6))
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2018],errorbar=None,color='blue',alpha=0.8,label=2018)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2019],errorbar=None,color='green',alpha=0.8,label=2019)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2020],errorbar=None,color='red',alpha=0.8,label=2020)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2021],errorbar=None,color='violet',alpha=0.8,label=2021)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2022],errorbar=None,color='yellow',alpha=0.8,label=2022)
            sns.lineplot(x='CalMonth',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2023],errorbar=None,color='orange',alpha=0.8,label=2023)
            plt.legend()
            plt.ylim(260,360)
            plt.xlim(1,12)
            plt.xlabel("Mois")
            plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],
                       ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Aout","Septembre","Octobre","Novembre","Décemnbre"],rotation=45,ha='right')
            plt.title("Evolution du temps de réponse moyen par mois et par années")
            st.pyplot(fig)

            #Ajout des explications
            st.write("On constate que le temps de réponse est généralement situé entre 300 et 320 secondes (soit entre 5 minutes et 5 minutes trente).")
            st.write("Ce temps de réponse augmente lors des mois d'été, et en particulier les mois de Juillet avec des pics jusqu'à 340 secondes.")
            st.write("L'année 2020 est visiblement une anomalie avec, entre Avril et Aout ,des temps de réponses bien inférieurs à la moyenne des autres années.")
            st.write("Cela peut s'expliquer par l'impact des restrictions sanitaires du COVID. Les gens étant confinés chez eux, le nombre d'incidents a baissé sur cette période.")

            #Création d'un graphique permettant de visualiser l'évolution du temps de réponse par jours de la semaine et par années
            fig=plt.figure(figsize=(10,6))
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2018],errorbar=None,color='blue',alpha=0.8,label=2018)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2019],errorbar=None,color='green',alpha=0.8,label=2019)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2020],errorbar=None,color='red',alpha=0.8,label=2020)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2021],errorbar=None,color='violet',alpha=0.8,label=2021)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2022],errorbar=None,color='yellow',alpha=0.8,label=2022)
            sns.lineplot(x='CalWeekday',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2023],errorbar=None,color='orange',alpha=0.8,label=2023)
            plt.legend()
            plt.ylim(290,340)
            plt.xlim(0,6)
            plt.ylabel("Temps de réponse en secondes")
            plt.xlabel("Jours de la semaine")
            plt.xticks([0,1,2,3,4,5,6],["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],rotation=360)
            plt.title("Evolution du temps de réponse moyen par jours de la semaine")
            st.pyplot(fig)

            #Ajout des explications
            st.write("Comme sur le graphique précédent, on constate que le temps de réponse de 2020 est bien inférieur à ceux des autres années.")
            st.write("Cependant, on distingue une tendance avec un temps de réponse en hausse le vendredi et en baisse le weekend.")
            st.write("Le dimanche semble être le jour de la semaine avec le temps de réponse le plus rapide. Probablement car c'est un jour de repos où les gens restent chez eux.")

            #Création d'un graphique permettant de visualiser l'évolution du temps de réponse par heures de la journée et par années
            fig=plt.figure(figsize=(10,6))
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2018],errorbar=None,color='blue',alpha=0.8,label=2018)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2019],errorbar=None,color='green',alpha=0.8,label=2019)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2020],errorbar=None,color='red',alpha=0.8,label=2020)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2021],errorbar=None,color='violet',alpha=0.8,label=2021)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2022],errorbar=None,color='yellow',alpha=0.8,label=2022)
            sns.lineplot(x='HourOfCall',y='ResponseTimeSeconds',data=LFB[LFB['CalYear']==2023],errorbar=None,color='orange',alpha=0.8,label=2023)
            plt.legend()
            plt.ylim(260,360)
            plt.xlim(0,23)
            plt.xlabel("Heures")
            plt.xticks([0,3,6,9,12,15,18,21,23],['Minuit','3h','6h','9h','Midi','15h','18h','21h','23h'])
            plt.title("Evolution du temps de réponse moyen par heures")
            st.pyplot(fig)

            #Ajout des explications
            st.write("Le temps de réponse varie selon le moment de la journée.")
            st.write("En effet, on constate qu'il est en moyenne entre 300 et 340 secondes entre 1h et 6h du matin puis de 11h à 18h. Il baisse drastiquement autour de 9h et de 22h.")
            st.write("Sur l'année 2020, le temps de réponse semble similaire à celui des autres années à la différence de la journée entre 10h et 21h où il est bien inférieur.")

        if types[2] in type :

            #Création d'un graphique pour visualiser la relation entre la caserne de déploiement et le temps de réponse
            fig=plt.figure(figsize=(4,4))
            sns.boxplot(x='DeployedFromLocation',y='ResponseTimeSeconds',data=LFB)
            plt.ylim(0,1300)
            plt.ylabel("Temps de réponse en secondes")
            plt.title("Relation entre la caserne de déploiement et le temps de réponse")
            st.pyplot(fig)

            #Ajout des explications
            st.write("La médiane du temps de réponse ne semble pas fluctuer selon le lieu de déploiement des pompiers. Cependant, les valeurs sont plus dispersées lorsque les pompiers sont déployés depuis une caserne qui n'est pas la leur.")

            #Création d'un graphique pour visualiser la relation entre le quartier de l'incident et le temps de réponse
            fig=plt.figure(figsize=(20,4))
            sns.boxplot(x='IncGeo_BoroughName',y='ResponseTimeSeconds',data=LFB)
            plt.ylim(0,1300)
            plt.xticks(rotation=45,ha='right')
            plt.ylabel("Temps de réponse en secondes")
            plt.xlabel("Quarties de Londres")
            plt.title("Relation entre les types d'incident et le temps de réponse")
            st.pyplot(fig)

            #Ajout des explications
            st.write("On constate que la médiane du temps de réponse semble être assez similaire dans la plupart des quartiers à l'exception de Bromley,Enfield,Tower Hamlets, Kensington and Chealsea.")
            st.write("Cependant, on constate qu'il existe des disparités au niveau des quartiles plus ou moins importants.")
    
    if categories[2] in categorie :
        
        # Vérification si le fichier ANOVA existe déjà
        anova_file_path = 'anova_file'

        # Si le fichier existe, charger les données
        if os.path.exists(anova_file_path):
            anova_table = joblib.load(anova_file_path)
        
        else :
            # Liste des variables catégorielles
            var_cat = ['CalMonth', 'CalWeekday', 'HourOfCall', 'DeployedFromLocation', 'IncGeo_BoroughName']

            # Supression des NaNs
            LFB_anova = LFB.dropna(subset=['ResponseTimeSeconds'] + var_cat)

            # Formule du modèle
            formule  = 'ResponseTimeSeconds ~ C(CalMonth) + C(CalWeekday) + C(HourOfCall) + C(DeployedFromLocation) + C(IncGeo_BoroughName)'
            model = ols(formule, data=LFB_anova).fit()

            # ANOVA
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Sauvegarde du tableau ANOVA dans un fichier
            joblib.dump(anova_table, 'anova_file')

        # Formatage des valeurs de la table en notation scientifique
        anova_table['sum_sq'] = anova_table['sum_sq'].apply(lambda x: f"{x:.3e}")
        anova_table['PR(>F)'] = anova_table['PR(>F)'].apply(lambda x: f"{x:.3e}")

        #Affichage du tableau ANOVA
        st.table(anova_table)

        #Ajout des explications
        st.write("On remarque que pour l'ensemble des variables catégorielles, la p-value est inférieure à 5%.")
        st.write("Nous pouvons en déduire que chaque variable a un effet significatif sur le temps de réponse.")

if page==pages[3] :
    st.image("lfb.png")    
    st.header("Modélisation par Régression 🛠️")

    st.subheader("Objectif")
    st.write("Prédire le temps de réponse de la Brigade des Pompiers de Londres")
 
    st.subheader("1. Étapes de Preprocessing & Modélisation")
    st.markdown("""
                📌 Séparation en jeux de d'entraînement (75%) et de test (25%)\n
                📌 Gestion des valeurs nulles_\n
                📌 Standardisation des données numériques\n
                📌 Encodage des valeurs catégorielles avec OneHotEncoder\n
                📌 Transformation des variables circulaires (CalMonth, CalHour, CalWeekday)\n
                📌 Instanciation & entraînement des modèles\n
                📌 Prédictions de chaque modèle sur le jeu de test\n
                📌 Calcul des métriques de performance
                """)
    
    st.subheader("2 .Comparaison des modèles")
    if 'Unnamed: 0' in Results_Reg.columns:
        Results_Reg.set_index('Unnamed: 0', inplace=True)
    Results_Reg.index.name=None
    Results_Reg.rename(columns={"?R²": "∆ R²"}, inplace=True)
    st.dataframe(Results_Reg.head(9))    

if page==pages[4] :
    st.image("lfb.png")
    st.header("Modélisation par Classification 🛠️")
    
    st.subheader("Objectif")
    st.write("Prédire l'intervalle de temps de réponse de la Brigade des Pompiers de Londres")

    st.subheader("1. Classification de la variable cible")
    st.write("📌 Distribution des valeurs avant la classification")
    st.image('distri_cible.png')

    st.write("📌 Répartition des classes après la classification")
    df1 = pd.DataFrame(
        {"Dataset": ["y_train", "y_test"],
        "Très Lente\n(plus de 500 sec)": [32865, 11011],
        "Lente\n(400-500 sec)": [69645, 23374],
        "Modérée\n(300-400 sec)": [75821, 25249],
        "Rapide\n(200-300 sec)": [207009, 69026],
        "Très Rapide\n(0-200 sec)": [90178, 29847]}
        )
    st.dataframe(df1, use_container_width=True, hide_index=True)

    st.subheader("2. Calcul du poids des classes ")
    df2 = pd.DataFrame(
        {"Très Lente": [2.8939049995435595],
        "Lente": [1.365530906741331],
        "Modérée": [1.254301578718297],
        "Rapide": [0.45941190962711764],
        "Très Rapide": [1.0546543349524253]}
    )
    st.dataframe(df2, use_container_width=True, hide_index=True)

    # Créer une "select box" permettant de choisir le modèle de classification
    st.subheader("3. Entraînement des modèles")
    choix = ['Random Forest Classifier', 'Decision Tree Classifier', 'Logistic Regression']
    option = st.selectbox('📌 Choisissez votre modèle', choix)
    st.write('Le modèle choisi est :', option)

    # Afficher des options à choisir pour scruter la performance
    display = st.radio("📌 Choisissez l'indicateur de performance", ('Accuracy', 'Confusion Matrix'))

    if display == 'Accuracy' and option == 'Random Forest Classifier':
        st.write('54,7%')
    elif display == 'Accuracy' and option == 'Decision Tree Classifier':
        st.write('47,6%')
    elif display == 'Accuracy' and option == 'Logistic Regression':
        st.write('40,2%')
    elif display == 'Confusion Matrix' and option == 'Random Forest Classifier':
        st.image('rf_clf_matrix.png')
    elif display == 'Confusion Matrix' and option == 'Decision Tree Classifier':
        st.image('tree_clf_matrix.png')
    elif display == 'Confusion Matrix' and option == 'Logistic Regression':
        st.image('logistic_reg_matrix.png')

if page==pages[5] :
    st.image("lfb.png")
    st.header("Conclusion 📌")

    # Section 1: Bilan
    st.subheader("1. Bilan")
    st.markdown("""
    Malgré les efforts fournis pour équilibrer les classes et ajuster les hyperparamètres des modèles, les résultats obtenus restent insuffisants pour offrir des recommandations qualitatives et précises quant au temps de réponse des pompiers de Londres. 
    Nous avons constaté que les variables explicatives disponibles ne capturent pas les facteurs clés influençant la variable cible.
    """)

    # Section 2: Problèmes rencontrés
    st.subheader("2. Problèmes rencontrés")
    st.markdown("""
    **Jeux de données :**  
    - Séparation initiale des jeux de données en plusieurs fichiers couvrant diverses périodes, nécessitant de nombreuses itérations pour le nettoyage et le traitement des valeurs nulles.

    **Problèmes liés à l'IT :**  
    - La volumétrie importante du jeu de données a limité nos options d'encodage, nécessitant des regroupements dans certaines catégories.

    **Prévisionnel :**  
    - Le temps de traitement et de nettoyage a été conséquent, réduisant la phase d'optimisation et de finalisation.
    """)

    st.subheader("3. Suite du projet")
    st.markdown("""
    Pour améliorer la modélisation du temps de réponse, nous suggérons :  
    - **Ajout de nouvelles variables :** Les conditions météorologiques et la distance entre la caserne et le lieu de l'incident pourraient affiner les prédictions.
    - **Données GPS :** Intégrer des données GPS des camions pour mesurer les distances tout en respectant l'anonymisation des lieux d'incidents.
    """)

    # Section 4: Bibliographie
    st.subheader("4. Bibliographie")

    bibliographie_choice = st.radio(
        "Quelle bibliographie souhaitez-vous consulter ?",
        ["Données", "Publications, articles et études consultées"]
    )

    # Contenu affiché en fonction du choix
    if bibliographie_choice == "Données":
        st.markdown("""
        - [London Fire Brigade Incident Records - Ville de Londres](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)
        - [Mobilisation Data - Ville de Londres](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)
        """)

    elif bibliographie_choice == "Publications, articles et études consultées":
        st.markdown("""
        - [Review and comparison of prediction algorithms for the estimated time of arrival using geospatial transportation data.](https://doi.org/10.1016/j.procs.2021.11.003)
        - [GPS is (finally) coming to the London Fire Department.](https://www.cbc.ca/news/canada/london/london-fire-department-gps-computerized-dispatch-system-field-tested-2021-1.5801119)
        - [Guide relatif aux opérations des services de sécurité incendie.](https://cdn-contenu.quebec.ca/cdn-contenu/adm/min/securitepublique/publications-adm/publications-secteurs/securite-incendie/services-securite-incendie/guides-reference-ssi/guide_operations_2024_2_01.pdf)
        - [Modelling residential fire incident response times: A spatial analytic approach.](https://doi.org/10.1016/j.apgeog.2017.03.004)
        - [Scalable Real-time Prediction and Analysis of San Francisco Fire Department Response Times.](https://doi.org/10.1109/SmartWorld-UIC-ATC-SCALCOM-IOP-SCI.2019.00154)
        - [Survey of ETA prediction methods in public transport networks.](http://arxiv.org/abs/1904.05037)
        - [Impact of weather conditions on macroscopic urban travel times.](https://doi.org/10.1016/j.jtrangeo.2012.11.003)
        """)

