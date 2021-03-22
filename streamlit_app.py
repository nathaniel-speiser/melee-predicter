import streamlit as st
from slippi import Game
import pandas as pd
from utils.GameStats import GameStats
st.set_page_config(layout='wide')

import os

for root, dirs, files in os.walk("."):
    for filename in files:
        print(filename)
dir = os.path.dirname(__file__)
f1 = os.path.join(dir, 'saved_data/streamlit_model.pkl')
f2 = os.path.join(dir, 'saved_data/features.pkl')
##############################################################
# Importing data/models
#streamlit_model = pd.read_pickle('saved_data/streamlit_model.pkl')
#features = pd.read_pickle('saved_data/features.pkl')

streamlit_model = pd.read_pickle(f1)
features = pd.read_pickle(f2)

##################################################################

st.title('Melee win probability predictor')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is None:
    st.write('## Please upload a file')
    st.stop()

game=Game(uploaded_file)
GS = GameStats(game, streamlit_model, features)






show_stocks = st.checkbox('Show stock losses on graph')
win_prob = GS.plot_odds(show_stocks=show_stocks, ret=True)
st.plotly_chart(win_prob, use_container_width=True)
if show_stocks:
    st.markdown('*Colored vertical lines correspond to player of same color losing a stock*')




st.write(f"""## Game Stats:""")
st.write(f"""### Winner: P{GS.winner_port+1} ({GS.winner_str})""")
st.write(f"""### Lead changes: {GS.lead_changes()}""")
st.write(f"""### Closeness factor: {GS.closeness_factor()}/100""")
st.write(f"""### Comeback factor: {GS.comeback_factor()}/100""")
st.write(f"""### Turning point: {GS.turning_point()} seconds""")
if GS.clutch():
    st.write(f"""### This game was clutch""")
else:
    st.write(f"""### This game was not clutch""")

with st.beta_expander("See explanation of stats"):
    st.markdown("""
        * Lead changes: The number of times the lead in *odds of winning* changed
        * Closeness factor: Percent of the game when either players' odds of winning were between 35 and 65 percent
        * Comeback factor: 100 - 2 * (minimum odds of winning player), minimum of 0. The higher the score, the lower the minimum winning odds at any of the ultimate winner. Doesn't look at the first 10 seconds, because those can be noisy
        * Turning point: Time when the winner took the lead in win probability and never lost it.
        * Clutch: A game was clutch if the winner's odds of winning in the last 20 seconds was less than 0.5
    """)



col1, col2 = st.beta_columns(2)
with col1:
    p1_icon = 'icons/'+GS.get_final_stat('p1_char') + '.png'
    p1_ground = int(GS.get_final_stat('p1_ground_hits'))
    p1_smash  = int(GS.get_final_stat('p1_smash_hits'))
    p1_aerial = int(GS.get_final_stat('p1_aerial_hits'))
    p1_other  = int(GS.get_final_stat('p1_total_hits') - p1_ground-p1_smash-p1_aerial)
    p1_grabs  = int(GS.get_final_stat('p1_grabs'))
    p1_shield =     GS.get_final_stat('p1_shield_frames')/60

    st.write(f"""## P{GS.port1+1} stats""")
    st.image(p1_icon)
    st.markdown(f"""* Ground attacks (Jabs, dash attacks, tilts) landed: {p1_ground}""")
    st.markdown(f"""* Smashes landed: {p1_smash}""")
    st.markdown(f"""* Aerials landed: {p1_aerial}""")
    st.markdown(f"""* Other attacks (specials, getup, etc.) landed: {p1_other}""")
    st.markdown(f"""* Successful grabs: {p1_grabs}""")
    st.markdown(f"""* Seconds in shield: {p1_shield:.1f}""")
with col2:
    p2_icon = 'icons/'+GS.get_final_stat('p2_char') + '.png'
    p2_ground = int(GS.get_final_stat('p2_ground_hits'))
    p2_smash  = int(GS.get_final_stat('p2_smash_hits'))
    p2_aerial = int(GS.get_final_stat('p2_aerial_hits'))
    p2_other  = int(GS.get_final_stat('p2_total_hits') - p1_ground-p1_smash-p1_aerial)
    p2_grabs  = int(GS.get_final_stat('p2_grabs'))
    p2_shield =     GS.get_final_stat('p2_shield_frames')/60

    st.write(f"""## P{GS.port2+1} stats""")
    st.image(p2_icon)
    st.markdown(f"""* Ground attacks (Jabs, dash attacks, tilts) landed: {p2_ground}""")
    st.markdown(f"""* Smashes landed: {p2_smash}""")
    st.markdown(f"""* Aerials landed: {p2_aerial}""")
    st.markdown(f"""* Other attacks (specials, getup, etc.) landed: {p2_other}""")
    st.markdown(f"""* Successful grabs: {p2_grabs}""")
    st.markdown(f"""* Seconds in shield: {p2_shield:.1f}""")
