
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão de Jogos de Futebol", layout="wide")

st.title("⚽ Previsão de Resultados de Futebol - Premier League")
st.markdown("Este app utiliza Machine Learning (XGBoost) para prever o resultado de partidas com base em odds e forma recente dos times.")

# Carregar dados
@st.cache_data
def carregar_dados():
    url = "https://www.football-data.co.uk/mmz4281/2223/E0.csv"
    df = pd.read_csv(url)
    df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','B365H','B365D','B365A']]
    df.columns = ['Date','HomeTeam','AwayTeam','HomeGoals','AwayGoals','Result','OddsH','OddsD','OddsA']
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = carregar_dados()

# Calcular forma recente
def calcular_forma(time, idx, df):
    ultimos = df.loc[:idx-1]
    jogos = ultimos[(ultimos['HomeTeam'] == time) | (ultimos['AwayTeam'] == time)].tail(5)
    pontos = 0
    for _, jogo in jogos.iterrows():
        if jogo['HomeTeam'] == time:
            if jogo['HomeGoals'] > jogo['AwayGoals']:
                pontos += 3
            elif jogo['HomeGoals'] == jogo['AwayGoals']:
                pontos += 1
        elif jogo['AwayTeam'] == time:
            if jogo['AwayGoals'] > jogo['HomeGoals']:
                pontos += 3
            elif jogo['AwayGoals'] == jogo['HomeGoals']:
                pontos += 1
    return pontos / 15

df['HomeForm'] = 0.0
df['AwayForm'] = 0.0
for i in range(len(df)):
    df.at[i, 'HomeForm'] = calcular_forma(df.at[i, 'HomeTeam'], i, df)
    df.at[i, 'AwayForm'] = calcular_forma(df.at[i, 'AwayTeam'], i, df)

# Codificar times
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
team_map = {team: idx for idx, team in enumerate(teams)}
df['HomeTeam_enc'] = df['HomeTeam'].map(team_map)
df['AwayTeam_enc'] = df['AwayTeam'].map(team_map)

# Treinar modelo
X = df[['HomeTeam_enc','AwayTeam_enc','HomeForm','AwayForm','OddsH','OddsD','OddsA']]
y = df['Result']
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X, y)

# Interface para o usuário
st.sidebar.header("⚙️ Parâmetros da Partida")
home = st.sidebar.selectbox("Time da Casa", sorted(team_map.keys()))
away = st.sidebar.selectbox("Time Visitante", sorted(team_map.keys()))

home_odds = st.sidebar.number_input("Odds - Vitória Casa", min_value=1.01, value=1.50)
draw_odds = st.sidebar.number_input("Odds - Empate", min_value=1.01, value=3.50)
away_odds = st.sidebar.number_input("Odds - Vitória Fora", min_value=1.01, value=5.00)

# Calcular forma atual com base histórica
ultimo_jogo_idx = len(df) - 1
home_form = calcular_forma(home, ultimo_jogo_idx, df)
away_form = calcular_forma(away, ultimo_jogo_idx, df)

# Predição
input_data = pd.DataFrame([{
    'HomeTeam_enc': team_map[home],
    'AwayTeam_enc': team_map[away],
    'HomeForm': home_form,
    'AwayForm': away_form,
    'OddsH': home_odds,
    'OddsD': draw_odds,
    'OddsA': away_odds
}])
proba = model.predict_proba(input_data)[0]
classes = model.classes_

# Mostrar resultado
st.subheader("🔮 Previsão do Resultado")
st.write(f"**{home} vs {away}**")
st.write(f"Forma Recente - {home}: {home_form*100:.0f}%, {away}: {away_form*100:.0f}%")

fig, ax = plt.subplots()
bars = ax.bar(classes, proba, color=['green', 'gray', 'red'])
ax.set_ylabel("Probabilidade")
ax.set_ylim([0,1])
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + 0.25, yval + 0.02, f"{yval:.2f}")
st.pyplot(fig)
