# Predicting in-game win probabilities in Super Smash Bros Melee
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/nathaniel-speiser/melee-predicter/main)

## Description

In this project I created a pipeline to create in-game win proability graphs (examples for basketball [here](https://live.numberfire.com/ncaab)) for games of Super Smash Bros. Melee, a competitive fighting game released by Nintendo in 2001.

The data I used to create a model than can predict these win probabilites were 18,000 games from 5 tournaments in 2019, sourced from the [Slippi Discord](https://discord.com/channels/328261477372919811/652736425997107220), specifically the bot-commands channel. These games came in the form of .slp replay files, a file format unique to the community-written tool [Slippi](https://slippi.appspot.com/). To parse the games with Python I used another community written tool, the [py-slippi package](https://github.com/hohav/py-slippi).

I used py-slippi to take take snapshots of each game every 5 seconds and extract features at these times. These features included basic information, such as time, stock count, and character selection, but I used more advanced logic to extract in game statistics such as the number of times each player landed certain moves. I then processed this data to one-hot encode categorical features as well as create features that measured the relative performance of the two players, such as the difference in the total number of hits. I fed this processed data into several models, the best of which was a sklearn ExtraTrees Classifier, whose predict_proba method I then used to predict win probabilities.

The final model acheived an average log loss of 0.25, an accuracy of 0.93, and an ROC AUC of 0.98 on a held-out test set. Overall this means the model performs quite well, and is especially good at distinguishing whether a player is winning or losing. The most predictive features, according to the classifier, were stock based features, including a scaled difference in stock count and the stock counts themselves, as well as hit-based features, such as the difference in the total hits landed by each player and the number of grabs each player got.

Finally, I also crated a streamlit app so that others can upload their own replays and get statistics based off of them. The app, linked at the top of this README, allows users to upload a .slp file nd see their win probabilities over time as well as other statistics about the overall game and each players' performance. I will continue updating this app with new features as time allows.

## Tools

* py-slippi
* pandas
* numpy
* scikit-learn
* Tensorflow/Keras
* plot.ly
* matplotlib/seaborn
* streamlit


## Future directions

Theoretically games can be streamed into py-slippi as the replay file is created, so I'd like to explore that in the future as live win probabilites could be an interesting addition to tournament streams. I'd also want to incorporate player data, not just in game statistics into the model (maybe as some kind of ELO difference feature, as is used in models like FiveThirtyEight's), to allow the model to make more informed guesses at the beginning of games.
