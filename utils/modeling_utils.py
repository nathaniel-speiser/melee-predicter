import slippi
from slippi import Game
import pickle
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss,accuracy_score
from sklearn.base import clone


import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_utils import construct_df, ohe_chars_stage


def get_log_losses(X_test, y_test, model):
    """
    Return array of log losses for a given set of data points and labels

    Args:

    X_test: 2D array/dataframe of points fittable by model
    y_test: Labels for points in X_test
    model: Model that implements predict_proba that fits X_test to compare with y_test
    """
    y_pred_prob = np.clip(model.predict_proba(X_test)[:,1],1e-15,1-1e-15)
    y_test = np.array(y_test)
    log_loss = -(y_test*np.log(y_pred_prob) + (1-y_test)*np.log(1-y_pred_prob))
    return log_loss


def plot_log_loss(X_test, y_test,model,title=''):
    """
    Plot log losses, grouped by time then averaged, for a given set of data points and labels

    Args:

    X_test: 2D array/dataframe of points fittable by model
    y_test: Labels for points in X_test
    model: Model that implements predict_proba that fits X_test to compare with y_test
    title: Title for plot
    """
    Xt_copy = X_test.copy()
    Xt_copy['log_loss'] = get_log_losses(X_test,y_test, model)
    df = Xt_copy.groupby(['frames_elapsed'])[['log_loss']].mean().reset_index()
    plt.figure(figsize=(12,8))
    sns.lineplot(x=df['frames_elapsed']/60, y=df['log_loss'] )
    plt.ylabel('Average Log Loss')
    plt.xlabel('Time elapsed (s)')
    plt.title(title)
    plt.xlim((0,480))
    plt.show()


def plot_multiple_log_loss(X_test, y_test,model_list, model_names):
    """
    Plot log losses, grouped by time then averaged, for a given set of data points and labels and a set of different models

    Args:

    X_test: 2D array/dataframe of points fittable by model
    y_test: Labels for points in X_test
    model_list: List of models that implements predict_proba that fits X_test to compare with y_test
    model_names: List of model names as strings
    """
    Xt_copy = X_test.copy()
    col_names = [name + ' log loss' for name in model_names]
    for model,name in zip(model_list, col_names):
        Xt_copy[name] = get_log_losses(X_test,y_test, model)
    df = Xt_copy.groupby(['frames_elapsed'])[col_names].mean().reset_index()
    plt.figure(figsize=(12,8))
    for name in col_names:
        sns.lineplot(x=df['frames_elapsed']/60, y=df[name], label = name )
    plt.ylabel('Average Log Loss at time')
    plt.xlabel('Time elapsed (s)')
    plt.xlim((0,480))
    plt.legend()
    plt.show()


def CV_model(model, X, y, title = ''):
    """
    Cross-validate model, plotting validation log loss for folds and printing out train/validation metrics

    Args:
    Model: Sklearn/XGboost model
    X: Points that will be split into train and validation set
    y: Labels for points in X
    title: Title for plot



    """
    val_accuracies = []
    val_avg_loglosses = []

    train_accuracies = []
    train_avg_loglosses = []
    kf = KFold(random_state=42, shuffle=True)
    plt.figure(figsize=(12,8))
    for train_index, val_index in kf.split(X):
        cv_model = clone(model)
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.to_numpy()[train_index], y.to_numpy()[val_index]

        cv_model.fit(X_train,y_train)

        y_pred = cv_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        val_accuracies.append(acc)

        y_pred_prob = cv_model.predict_proba(X_val)
        ll = log_loss(y_val, y_pred_prob)
        val_avg_loglosses.append(ll)

        y_pred_t = cv_model.predict(X_train)
        acc_t = accuracy_score(y_train, y_pred_t)
        train_accuracies.append(acc_t)

        y_pred_prob_t = cv_model.predict_proba(X_train)
        ll_t = log_loss(y_train, y_pred_prob_t)
        train_avg_loglosses.append(ll_t)


        df = X_val.copy()
        df['log_loss'] = get_log_losses(X_val,y_val, cv_model)
        df = df.groupby(['frames_elapsed'])[['log_loss']].mean().reset_index()
        sns.lineplot(x=df['frames_elapsed']/60, y=df['log_loss'])

    print('Average training accuracy: {:.3f}'.format(np.mean(train_accuracies)))
    print('Average training log loss: {:.3f}'.format(np.mean(train_avg_loglosses)))

    print('Average CV accuracy: {:.3f}'.format(np.mean(val_accuracies)))
    print('Average CV log loss: {:.3f}'.format(np.mean(val_avg_loglosses)))
    plt.ylabel('Average Log Loss at time')
    plt.xlabel('Time elapsed (s)')
    plt.xlim((0,480))
    plt.title(title)
    plt.show()

    model.fit(X,y)
    return model

def validate_model(model, X_train, y_train, X_val, y_val, xgb=False, title = ''):
    """
    Validate model, plotting validation log loss and printing out train/validation metrics

    Args:
    Model: Sklearn/XGboost model
    X_train: Training points
    y_train: Training labels
    X_val: Validation points
    y_val: Validation labels
    xgb: Set to True if using xgboost model to run fit with extra parameters to make it go faster
    title: Title for plot
    """
    if xgb:
        eval_set = [(X_val, y_val)]
        model.fit( X_train, y_train, eval_set=eval_set,
                   eval_metric='error', early_stopping_rounds=50,verbose=False)
    else:
        model.fit(X_train,y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    y_pred_prob = model.predict_proba(X_val)
    ll = log_loss(y_val, y_pred_prob)


    y_pred_t = model.predict(X_train)
    acc_t = accuracy_score(y_train, y_pred_t)

    y_pred_prob_t = model.predict_proba(X_train)
    ll_t = log_loss(y_train, y_pred_prob_t)

    df = X_val.copy()
    df['log_loss'] = get_log_losses(X_val,y_val, model)
    df = df.groupby(['frames_elapsed'])[['log_loss']].mean().reset_index()

    plt.figure(figsize=(12,8))
    sns.lineplot(x=df['frames_elapsed']/60, y=df['log_loss'])

    print('Training accuracy: {:.3f}'.format(acc_t))
    print('Training log loss: {:.3f}'.format(ll_t))

    print('Validation accuracy: {:.3f}'.format(acc))
    print('Validation log loss: {:.3f}'.format(ll))
    plt.ylabel('Validation Log Loss at time')
    plt.xlabel('Time elapsed (s)')
    plt.xlim((0,480))
    plt.title(title)
    plt.show()

    return model

def plot_feature_importances(model, features, num):
    """
    Plot most predictive features for model that has parameter feature_importances_ (Tree models, xgboost, etc.)

    Args:

    model: Model that has already been fit on train data
    features: List of names for features (list of strings)
    num: Number of features to show

    """
    feat_importances = model.feature_importances_
    df = pd.DataFrame(data = {'feature':features, 'importances':feat_importances})
    df = df.sort_values(by='importances', ascending=False)
    plt.figure(figsize=(12,8))
    sns.barplot(data = df[:num], x='importances', y='feature',dodge=False, color='Cornflowerblue' )
    plt.show()


def plot_win_probs(game_df, model,feature_list, title=''):
    """
    Plot win probabilities for player with lower port in game

    Args:

    game_df: Complete pandas DataFrame for a game (ie output from process_df_igs_final), contains for for every sampled time in game
    model: Trained model used to predict probabilities
    Feature_list: List of features used by model, subset of game_df's columns
    title: Title for plot

    """
    X = game_df[feature_list]
    p1_win_probs = model.predict_proba(X)[:,1]
    plt.figure(figsize=(12,8))
    sns.lineplot(x=game_df['frames_elapsed']/60, y=p1_win_probs)
    plt.xlabel('Time (s)')
    plt.ylim((0,1))
    plt.title('P1 Win probability '+title)
    plt.show()
