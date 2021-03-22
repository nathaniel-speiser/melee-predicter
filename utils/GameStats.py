import slippi
from slippi import Game
import pickle
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss,accuracy_score, roc_curve, roc_auc_score
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier




import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt

from utils.data_utils import construct_df, ohe_chars_stage, characters, stages, stage_dict, construct_df_igs,\
                        process_df_igs, process_df_igs_final, make_rows_igs, get_ports, get_characters, get_winner
from utils.modeling_utils import plot_log_loss, plot_multiple_log_loss, CV_model, plot_feature_importances,\
                            plot_win_probs, get_log_losses, validate_model

import plotly.graph_objects as go



class GameStats():
    def __init__(self, game, model,features):
        self.game = game
        self.model = model
        self.features = features
        self.port1, self.port2 = get_ports(self.game)
        self.char1, self.char2 = get_characters(self.game, self.port1, self.port2)
        self.char1, self.char2 = self.char1.title().replace('_', ' '), self.char2.title().replace('_', ' ')
        self.stage = stage_dict[self.game.start.stage.name]
        self.winner = get_winner(self.game, self.port1, self.port2)
        self.winner_str = self.char1 if self.winner==1 else self.char2
        self.winner_port = self.port1 if self.winner==1 else self.port2
        self.game_df,f = self.make_game_df()
        self.p1_odds = self.get_p1_odds()
        self.p2_odds = 1-self.p1_odds

    def make_game_df(self):
        df_list = []
        rows = make_rows_igs(self.game, 1, '')
        df_list.append(rows)
        return process_df_igs_final(pd.concat(df_list).reset_index(drop=True))

    def get_p1_odds(self):
        """Return odds over time of p1 winning"""
        return  self.model.predict_proba(self.game_df[self.features])[:,1]

    def get_final_stat(self,feature):
        try:
            return self.game_df.iloc[-1][feature]
        except:
            print('Feature not in feature list')
            return None

    def lead_changes(self):
        leads = np.rint(self.p1_odds)
        changes = 0
        for i in range(10,len(leads)):
            if leads[i] != leads[i-1]:
                changes +=1
        return changes

    def closeness_factor(self):
        """
        Percent of seconds spent betwen 35 and 65 percent odds
        """
        num_seconds_close = len(self.p1_odds[(self.p1_odds >=.35)&(self.p1_odds <=.65)])
        return round(num_seconds_close/len(self.p1_odds)*100)


    def comeback_factor(self):
        lowest_odds = 0
        if self.winner == 1:
            lowest_odds = np.min(self.p1_odds[10:])
        if self.winner == 0:
            lowest_odds = np.min(self.p2_odds[10:])
        return max(100 - 2*round(lowest_odds *100),0)



    def turning_point(self):
        """
        Returns time when winner passed 50% odds and never went below 50%
        If the winner wasn't above 50% will return something else
        """
        if self.winner == 1:
            for i, odds in enumerate(self.p1_odds[::-1]):
                if odds <.5:
                    return len(self.p1_odds)-i
        if self.winner == 0 :
            for i, odds in enumerate(self.p2_odds[::-1]):
                if odds <.5:
                    return len(self.p2_odds)-i
        return 0

    def clutch(self,window=20):
        """
        Returns True if the average win probability of the winner of the game was less than .5 in the last window
        seconds of the game
        """
        if self.winner == 1:
            return np.mean(self.p1_odds[-window:])<.5
        else:
            return np.mean(self.p2_odds[-window:])<.5

    def get_stock_loss_times(self):
        p1_losses = []
        p2_losses = []
        p1_stocks = 4
        p2_stocks = 4
        for i,frame in enumerate(self.game.frames):
            if frame.ports[self.port1].leader.post.stocks < p1_stocks:
                p1_losses.append(i/60)
                p1_stocks -=1
            if frame.ports[self.port2].leader.post.stocks < p2_stocks:
                p2_losses.append(i/60)
                p2_stocks -=1
        return p1_losses, p2_losses

    def plot_odds(self,show_stocks=True, ret = False):
        time = self.game_df['frames_elapsed']/60
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=time, y=self.p1_odds,name = f'P{self.port1+1} ({self.char1})'))

        fig.add_trace(
            go.Scatter(x=time, y=self.p2_odds,name = f'P{self.port2+1} ({self.char2})'))

        # Set title
        title_str = f' <b>{self.char1} vs {self.char2} on {self.stage} <br> Winner: '
        if self.winner == 1:
            title_str += self.char1
        else:
            title_str+= self.char2
        fig.update_layout(
            title_text=title_str, title_x = 0.5,
            title_font_size=24
        )
        fig.update_layout(height=600)
        fig.update_layout(xaxis = dict(title='Time (s)'))
        fig.update_layout(yaxis = dict(title='Predicted Win Probability'))

        fig.update_traces(hovertemplate='%{y:.2f}')
        fig.update_layout(hovermode="x unified")
        p1_losses, p2_losses = self.get_stock_loss_times()
        if show_stocks:
            for loss in p1_losses:
                fig.add_vline(x=loss, line_dash = 'dash', line_color='blue')
            for loss in p2_losses:
                fig.add_vline(x=loss, line_dash = 'dash', line_color='red')
        fig.add_hline(y=.5, line_width=.75, line_dash = 'dash', line_color= 'black')
        if ret:
            return fig
        else:
            fig.show()
