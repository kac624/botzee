import pandas as pd
import numpy as np
from classes import ScoreSheet, TurnData

# dice decision functions
def pick_random_dice(hand):
    return np.random.choice([0,1], size = 5)

def pick_frequent_dice(hand):
    dice_picks = []
    for idx, die in enumerate(hand):
        rest_of_hand = np.delete(hand, idx)
        if die in rest_of_hand:
            dice_picks.append(1)
        else:
            dice_picks.append(0)
    return dice_picks    

# score decision functions
def pick_random_score(potential_scores):
    eligible_scores = [x[0] for x in potential_scores.items() if x[1] >= 0]
    chosen_score_type = np.random.choice(eligible_scores)
    return chosen_score_type

def pick_max_score(potential_scores):
    return max(potential_scores, key = potential_scores.get)

# picking dice and scores
def pick_dice(hand, decision_function, **kwargs):
    dice_picks = decision_function(hand, **kwargs)
    keepers = hand[np.array(dice_picks, dtype = bool)]
    return dice_picks, keepers

def pick_score(potential_scores, decision_function, **kwargs):
    chosen_score_type = decision_function(potential_scores, **kwargs)
    turn_score = potential_scores[chosen_score_type]
    return chosen_score_type, turn_score

# simulation
def simulate_game(game_number, dice_decision_function, score_decision_function):
        
    game_number = game_number
    
    # to capture game data
    game_data = []

    # instantiate a new score sheet and game variables
    score_sheet = ScoreSheet()
    score_sheet.initialize_score_types()

    # 13 rounds per game
    for turn_number in range(1,14):

        # instantiate a new game data object
        remaining_score_types = [x[0] for x in score_sheet.scores.items() if x[1] == None]
        turn = TurnData(game_number, turn_number, score_sheet.total)

        # first roll
        turn.hand_1 = np.sort(np.random.randint(1, 7, 5))[::-1]

        # choose dice to keep
        turn.dice_picks_1, hand_1_keepers = pick_dice(turn.hand_1, dice_decision_function)

        # second roll
        roll = np.random.randint(1, 7, 5 - len(hand_1_keepers))
        turn.hand_2 = np.sort(np.append(hand_1_keepers, roll))[::-1]

        # choose dice to keep
        turn.dice_picks_2, hand_2_keepers = pick_dice(turn.hand_2, dice_decision_function)

        # third roll
        roll = np.random.randint(1, 7, 5 - len(hand_2_keepers))
        turn.hand_3 = np.sort(np.append(hand_2_keepers, roll))[::-1]

        # loop through all remaining scores, check which are eligible and calculate potential score
        potential_scores = {}
        for score_type, score in zip(score_sheet.score_types, score_sheet.scores.items()):
            if score_type.check_condition(turn.hand_3) and score[1] == None:
                value = score_type.calculate_score(turn.hand_3)
            elif score[1] != None:
                value = -1
            else:
                value = 0
            potential_scores[score_type.name] = value

        # choose score type
        turn.chosen_score_type, turn.turn_score = pick_score(potential_scores, score_decision_function)

        # mark score
        score_sheet.mark_score(turn.chosen_score_type, turn.turn_score)
        turn.post_total_score = score_sheet.total

        # capture data
        turn_data = turn.capture_data()

        remaining_score_types = dict(zip(
            [f'{x}_score' for x in score_sheet.scores.keys()],
            score_sheet.scores.values()
        ))
        turn_data.update(remaining_score_types)

        for old_key in list(potential_scores.keys()):
            potential_scores[f'{old_key}_potential'] = potential_scores.pop(old_key)
        turn_data.update(potential_scores)

        game_data.append(turn_data)
    
    return game_data