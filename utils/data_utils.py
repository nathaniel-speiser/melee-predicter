import slippi
from slippi import Game
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

characters = ['FOX', 'FALCO', 'MARTH', 'SHEIK','JIGGLYPUFF', 'PEACH', 'ICE_CLIMBERS','CAPTAIN_FALCON',
                 'PIKACHU','SAMUS','DR_MARIO','YOSHI','LUIGI','GANONDORF','MARIO','YOUNG_LINK','DONKEY_KONG',
                 'LINK','GAME_AND_WATCH','ROY','MEWTWO','ZELDA','NESS','PICHU','BOWSER','KIRBY']


stages = ['BATTLEFIELD', 'FINAL_DESTINATION',
               'DREAM_LAND_N64', 'YOSHIS_STORY', 'FOUNTAIN_OF_DREAMS','POKEMON_STADIUM']

stage_dict = {'BATTLEFIELD':'Battlefield', 'FINAL_DESTINATION':'Final Destination',
               'DREAM_LAND_N64':'Dream Land', 'YOSHIS_STORY': 'Yoshi\'s Story',
                'FOUNTAIN_OF_DREAMS': 'Fountain of Dreams','POKEMON_STADIUM':'Pokemon Stadium'}
def get_ports(game):
    """
    Returns tuple of ports occupied by players in Slippi Game

    Args:

    game: PySlippi Game

    """
    player_tup = game.start.players
    ports = tuple([i for i in range(4) if player_tup[i] is not None])
    return ports

def get_characters(game, port1, port2):
    """
    Returns tuple of characters (Fox, Falco, etc.) being played by players in occupied ports

    Args:

    game: PySlippi Game
    port1 (int): First occupied port
    port2 (int): Second occupied port

    """
    player_tup = game.start.players
    chars = (player_tup[port1].character.name,player_tup[port2].character.name )
    return chars

def get_stage(game):
    """
    Returns stage Slippi Game is being played on

    Args:

    game: PySlippi Game

    """
    return game.start.stage.name

def get_winner(game, port1, port2):
    """
    Returns if player with lower port is winner

    Args:
    game: PySlippi Game
    port1 (int): First occupied port
    port2 (int): Second occupied port
    """
    lras = game.end.lras_initiator
    if lras is not None:
        if lras == port1:
            return 0
        else:
            return 1

    p1_stocks = game.frames[-1].ports[port1].leader.post.stocks
    p2_stocks = game.frames[-1].ports[port2].leader.post.stocks

    if p1_stocks == p2_stocks:
        p1_damage = game.frames[-1].ports[port1].leader.post.damage
        p2_damage = game.frames[-1].ports[port2].leader.post.damage
        return p1_damage > p2_damage
    return int(p1_stocks > p2_stocks)


def is_valid_game(game):
    """
    Returns true only if game is valid
    Valid games defined as games more than 15 seconds long, ended conclusively, isn't teams, and is on a legal stage
    """


    if game.metadata.duration < 15*60:
        return False
    if game.end.method.value != 3 and game.end.method.value != 1 and game.end.method.value!= 2:
        return False
    if game.start.is_teams is True:
        return False
    if game.start.stage.value not in [2,3,8,28,31,32]:
        return False
    if len(get_ports(game))!=2:
        return False
    return True

def make_rows(game, num_seconds, game_id):
    """
    Make a basic df for each game. Rows are timeslices, every num_seconds of the game, as well as the first frame.
    Columns are game_id, characters, stage, time, stocks, damage, and winner

    Args:

    num_seconds: How often to sample the game, in seconds
    game_id: Identifier for game

    """

    port1, port2 = get_ports(game)
    p1_char, p2_char = get_characters(game, port1, port2)
    stage = get_stage(game)
    winner = get_winner(game, port1, port2)
    duration = game.metadata.duration

    times = np.array(list(range(0,duration, num_seconds*60)))
    num_times = len(times)
    p1_stocks, p2_stocks = np.zeros(num_times),np.zeros(num_times)
    p1_damage, p2_damage = np.zeros(num_times),np.zeros(num_times)

    for i, frame_num in enumerate(times):
        p1_stocks[i] = game.frames[frame_num].ports[port1].leader.post.stocks
        p2_stocks[i] = game.frames[frame_num].ports[port2].leader.post.stocks
        p1_damage[i] = game.frames[frame_num].ports[port1].leader.post.damage
        p2_damage[i] = game.frames[frame_num].ports[port2].leader.post.damage

    game_id_list = [game_id]*num_times
    p1_char_list = [p1_char]*num_times
    p2_char_list = [p2_char]*num_times
    stage_list = [stage]*num_times
    winner_list = [winner] * num_times
    return pd.DataFrame(data = {
            'id'     : game_id_list,
            'p1_char': p1_char_list,
            'p2_char': p2_char_list,
            'stage' : stage_list,
            'frames_elapsed'   : times,
            'p1_stocks': p1_stocks,
            'p2_stocks': p2_stocks,
            'p1_damage': p1_damage,
            'p2_damage': p2_damage,
            'winner' : winner_list
        })


def construct_df(file_list, num_seconds):
    """
    Make a df out of all the games in file_list, using make_rows

    Args:

    file_list: List of .slp files
    num_seconds: How often each game is sampled, in seconds
    """
    df_list = []
    for slp_file in tqdm(file_list,desc='Games parsed'):
        try:
            game = Game(slp_file)
            if is_valid_game(game):
                game_id = slp_file.split('/')[-1][:-4]
                rows = make_rows(game, num_seconds,game_id)
                df_list.append(rows)
        except:
            continue
    return pd.concat(df_list).reset_index(drop=True)


GROUND_ATTACK_STATES = list(range(44,58))
SMASH_ATTACK_STATES = list(range(58,65))
AERIAL_ATTACK_STATES = [65,66,67,68,69]
GRAB_STATES = [213,215]

ROLL_STATES = [233,234]
SHIELD_STATES = [178,179,180,181]

def get_ingame_stats(game, times, port1,port2):
    """
    Returns tuple of lists of game stats for each player
    Each entry in each list is that stat at the corresponding index in time
    Stats (for each player): Number of hits landed, number of ground attacks landed, number of smashes landed, number of aerials landed, number of successful grabs, number of frames in shield, number of rolls, stocks lost before 50%, number of frames since last stock was lost

    Args:

    game: PySlippi game
    times: List of int, times to sample statistics at
    port1: Port number of lower port player
    port2: Port number of higher port player

    """



    #Total number of hits that player 1 landed
    p1_hits_landed = 0
    #Jab, dash attack, tilts
    p1_ground_attacks = 0
    #Smashes
    p1_smashes = 0
    #Aerials
    p1_aerials = 0
    #Number of successful graphs p1 got
    p1_grabs=0
    #Number of frames that p1 spends in shield
    p1_frames_shielding = 0
    #Number of rolls - I wanted to add spotdodging too but there seems to be something weird with that state and
    #advanced defensive play - wouldn't want to get those confused
    p1_rolls = 0
    #Number of stocks p1 LOST before 50%
    p1_stocks_before_50 = 0
    #Number of frames since p1 lost their last stock
    p1_time_since_stock = 0
    p1_hit_list    = np.zeros(len(times))
    p1_gnd_list    = np.zeros(len(times))
    p1_smash_list  = np.zeros(len(times))
    p1_aerial_list = np.zeros(len(times))
    p1_grab_list   = np.zeros(len(times))
    p1_fs_list     = np.zeros(len(times))
    p1_roll_list   = np.zeros(len(times))
    p1_sb50_list   = np.zeros(len(times))
    p1_tss_list    = np.zeros(len(times))

    #Total number of hits that player 2 landed
    p2_hits_landed = 0
    #Jab, dash attack, tilts
    p2_ground_attacks = 0
    #Smashes
    p2_smashes = 0
    #Aerials
    p2_aerials = 0
    #Number of successful graphs p2 got
    p2_grabs=0
    p2_frames_shielding = 0
    p2_rolls = 0
    p2_stocks_before_50 = 0
    p2_time_since_stock = 0
    p2_hit_list    = np.zeros(len(times))
    p2_gnd_list    = np.zeros(len(times))
    p2_smash_list  = np.zeros(len(times))
    p2_aerial_list = np.zeros(len(times))
    p2_grab_list   = np.zeros(len(times))
    p2_fs_list     = np.zeros(len(times))
    p2_roll_list   = np.zeros(len(times))
    p2_sb50_list   = np.zeros(len(times))
    p2_tss_list    = np.zeros(len(times))

    stat_index = 1
    for frame_num in range(1,len(game.frames)):
        #Current frame information
        p1_cur = game.frames[frame_num].ports[port1].leader.post
        p2_cur = game.frames[frame_num].ports[port2].leader.post

        #Current action state
        p1_state = p1_cur.state.value
        p2_state = p2_cur.state.value

        #Previous frame information
        p1_prev = game.frames[frame_num-1].ports[port1].leader.post
        p2_prev = game.frames[frame_num-1].ports[port2].leader.post

        #If a players damage value has changed, update the other players hits
        if p2_cur.damage > p2_prev.damage:
            p1_hits_landed +=1
            if p1_state   in GROUND_ATTACK_STATES: p1_ground_attacks += 1
            elif p1_state in SMASH_ATTACK_STATES:  p1_smashes += 1
            elif p1_state in AERIAL_ATTACK_STATES: p1_aerials +=1
        if p1_cur.damage > p1_prev.damage:
            p2_hits_landed +=1
            if p2_state   in GROUND_ATTACK_STATES: p2_ground_attacks += 1
            elif p2_state in SMASH_ATTACK_STATES:  p2_smashes += 1
            elif p2_state in AERIAL_ATTACK_STATES: p2_aerials +=1


        #Grab or dash grab pull states
        if entered_state(p1_cur, p1_prev, GRAB_STATES): p1_grabs+=1
        if entered_state(p2_cur, p2_prev, GRAB_STATES): p2_grabs+=1

        #Shielding states
        if p1_state in SHIELD_STATES: p1_frames_shielding+=1
        if p2_state in SHIELD_STATES: p2_frames_shielding+=1

        #Roll states
        if entered_state(p1_cur, p1_prev, ROLL_STATES): p1_rolls+=1
        if entered_state(p2_cur, p2_prev, ROLL_STATES): p2_rolls+=1

        #If a player has lost a stock, check if they lost it under 50%, and reset the timer for time since they
        #lost a stock
        if p1_cur.stocks < p1_prev.stocks:
            p1_time_since_stock = 0
            if p1_prev.damage <=50:
                p1_stocks_before_50 +=1
        else: p1_time_since_stock +=1

        if p2_cur.stocks < p2_prev.stocks:
            p2_time_since_stock = 0
            if p2_prev.damage <=50:
                p2_stocks_before_50 +=1
        else: p2_time_since_stock +=1

        #If this is a frame where we want to capture the stats, record the stats to their relevant list
        if frame_num in times:
            p1_hit_list   [stat_index] = p1_hits_landed
            p1_gnd_list   [stat_index] = p1_ground_attacks
            p1_smash_list [stat_index] = p1_smashes
            p1_aerial_list[stat_index] = p1_aerials
            p1_grab_list  [stat_index] = p1_grabs
            p1_fs_list    [stat_index] = p1_frames_shielding
            p1_roll_list  [stat_index] = p1_rolls
            p1_sb50_list  [stat_index] = p1_stocks_before_50
            p1_tss_list   [stat_index] = p1_time_since_stock

            p2_hit_list   [stat_index] = p2_hits_landed
            p2_gnd_list   [stat_index] = p2_ground_attacks
            p2_smash_list [stat_index] = p2_smashes
            p2_aerial_list[stat_index] = p2_aerials
            p2_grab_list  [stat_index] = p2_grabs
            p2_fs_list    [stat_index] = p2_frames_shielding
            p2_roll_list  [stat_index] = p2_rolls
            p2_sb50_list  [stat_index] = p2_stocks_before_50
            p2_tss_list   [stat_index] = p2_time_since_stock

            stat_index+=1

    return (p1_hit_list,p1_gnd_list,p1_smash_list,p1_aerial_list,p1_grab_list,p1_fs_list,p1_roll_list,p1_sb50_list,p1_tss_list,
            p2_hit_list,p2_gnd_list,p2_smash_list,p2_aerial_list,p2_grab_list,p2_fs_list,p2_roll_list,p2_sb50_list,p2_tss_list)

igs_col_names = ['p1_total_hits', 'p1_ground_hits','p1_smash_hits','p1_aerial_hits','p1_grabs',
                 'p1_shield_frames','p1_rolls','p1_early_stocks_lost','p1_frames_since_lost',
                 'p2_total_hits', 'p2_ground_hits','p2_smash_hits','p2_aerial_hits','p2_grabs',
                 'p2_shield_frames','p2_rolls','p2_early_stocks_lost','p2_frames_since_lost']
def entered_state(p_cur, p_prev, state_list):
    """
    Returns true if player entered state in state_list in the current frame

    Args:

    p_cur: post object for current frame, ie game. ... .leader.post
    p_prev: post object for previous frame, ie game. ... .leader.post
    state_list: List of ints representing action states

    """

    cur_state = p_cur.state.value
    prev_state = p_prev.state.value

    return cur_state in state_list and prev_state not in state_list






def make_rows_igs(game, num_seconds, game_id):
    """
    Make a df, with both in basic features and in game stats colums, for a game. Rows are timeslices, every num_seconds of the game, as well as the first frame.
    Columns are game_id, characters, stage, time, stocks, damage, winner, and ingame stats for each player from get_ingame_stats

    Args:

    game: PySlippi Game
    num_seconds: How often to sample the game, in seconds
    game_id: Identifier for game

    """

    port1, port2 = get_ports(game)
    p1_char, p2_char = get_characters(game, port1, port2)
    stage = get_stage(game)
    winner = get_winner(game, port1, port2)
    duration = game.metadata.duration

    times = np.array(list(range(0,duration, num_seconds*60)))
    num_times = len(times)
    p1_stocks, p2_stocks = np.zeros(num_times),np.zeros(num_times)
    p1_damage, p2_damage = np.zeros(num_times),np.zeros(num_times)

    for i, frame_num in enumerate(times):
        p1_stocks[i] = game.frames[frame_num].ports[port1].leader.post.stocks
        p2_stocks[i] = game.frames[frame_num].ports[port2].leader.post.stocks
        p1_damage[i] = game.frames[frame_num].ports[port1].leader.post.damage
        p2_damage[i] = game.frames[frame_num].ports[port2].leader.post.damage
    ingame_stats = get_ingame_stats(game, times, port1,port2)
    game_id_list = [game_id]*num_times
    p1_char_list = [p1_char]*num_times
    p2_char_list = [p2_char]*num_times
    stage_list = [stage]*num_times
    winner_list = [winner] * num_times
    info_df=pd.DataFrame(data = {
            'id'     : game_id_list,
            'p1_char': p1_char_list,
            'p2_char': p2_char_list,
            'stage' : stage_list,
            'frames_elapsed'   : times,
            'p1_stocks': p1_stocks,
            'p2_stocks': p2_stocks,
            'p1_damage': p1_damage,
            'p2_damage': p2_damage,
            'winner' : winner_list
        })
    igs_df = pd.DataFrame(ingame_stats).T
    igs_df.columns = igs_col_names
    return pd.concat([info_df,igs_df],axis=1)


def construct_df_igs(file_list, num_seconds):
    """
    Make a df with ingame stats columns out of all the games in file_list, using make_rows_igs

    Args:

    file_list: List of .slp files
    num_seconds: How often each game is sampled, in seconds
    """
    df_list = []
    for slp_file in tqdm(file_list,desc='Games parsed'):
        try:
            game = Game(slp_file)
            if is_valid_game(game):
                game_id = slp_file.split('/')[-1][:-4]
                rows = make_rows_igs(game, num_seconds,game_id)
                df_list.append(rows)
        except:
            continue
    return pd.concat(df_list).reset_index(drop=True)


def process_df_igs(df):
    """
    Add difference features to df (output from construct_df_igs), then OHE characters and stages.
    Return the new df and all the features

    Args:

    df: Pandas DataFrame, output from construct_df_igs
    """
    df2 = df.copy()
    df2 = ohe_chars_stage(df2)
    df2['hit_diff'] = df['p1_total_hits'] - df['p2_total_hits']
    df2['shield_diff'] = df['p1_shield_frames']-df['p2_shield_frames']
    df2['early_stock_diff'] = df['p1_early_stocks_lost'] - df['p2_early_stocks_lost']
    df2['ground_diff'] = df['p1_ground_hits'] - df['p2_ground_hits']
    df2['smash_diff'] = df['p1_smash_hits']-df['p2_aerial_hits']
    df2['aerial_diff'] = df['p1_aerial_hits'] - df['p1_aerial_hits']
    df2['grab_diff'] = df['p1_grabs'] - df['p2_grabs']
    df2['roll_diff'] = df['p1_rolls']-df['p2_rolls']
    df2['stock_diff_sc'] = df['p1_stocks']**4 - df['p2_stocks']**4
    features = list(df2.columns[4:])
    features.remove('winner')
    return df2, features

def process_df_igs_final(df):
    """
    Add selected difference features to df (output from construct_df_igs), then OHE characters and stages.
    Return the new df and all the features

    Args:

    df: Pandas DataFrame, output from construct_df_igs
    """
    df2 = df.copy()
    df2 = ohe_chars_stage(df2)
    df2['hit_diff'] = df['p1_total_hits'] - df['p2_total_hits']
    df2['shield_diff'] = df['p1_shield_frames']-df['p2_shield_frames']
    df2['early_stock_diff'] = df['p1_early_stocks_lost'] - df['p2_early_stocks_lost']
    df2['grab_diff'] = df['p1_grabs'] - df['p2_grabs']
    df2['stock_diff_sc'] = df['p1_stocks']**4 - df['p2_stocks']**4
    features = list(df2.columns[4:])
    features.remove('winner')
    return df2, features


def ohe_chars_stage(df):
    """
    One-hot encode characters and stage

    Args:

    df: Pandas DataFrame with colums p1_char, p2_char, and stage
    """
    characters = ['FOX', 'FALCO', 'MARTH', 'SHEIK','JIGGLYPUFF', 'PEACH', 'ICE_CLIMBERS','CAPTAIN_FALCON',
                 'PIKACHU','SAMUS','DR_MARIO','YOSHI','LUIGI','GANONDORF','MARIO','YOUNG_LINK','DONKEY_KONG',
                 'LINK','GAME_AND_WATCH','ROY','MEWTWO','ZELDA','NESS','PICHU','BOWSER','KIRBY']

    stages = ['BATTLEFIELD', 'FINAL_DESTINATION',
               'DREAM_LAND_N64', 'YOSHIS_STORY', 'FOUNTAIN_OF_DREAMS','POKEMON_STADIUM']
    ohe = OneHotEncoder(categories=[characters,characters,stages], sparse=False)
    ohe_cols = ohe.fit_transform(df[['p1_char','p2_char','stage']])
    ohe_df = pd.DataFrame(ohe_cols, columns = ohe.get_feature_names(['p1','p2','stage']))
    return  pd.concat([df,ohe_df],axis=1)

