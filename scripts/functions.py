import numpy as np

import torch
import torch.nn.functional as F

from scripts.classes import ScoreSheet, TurnData


######################
### DICE FUNCTIONS ###
######################

def model_pick_dice(inputs, model, roll):
    # model = model.eval()
    # with torch.no_grad():
    probabilities = model(inputs)[roll-1][0]

    distribution = torch.distributions.Bernoulli(probabilities)
    dice_picks = distribution.sample() 
    log_probs = distribution.log_prob(dice_picks)

    # dice_picks = []
    #     if pick >= 0.5:
    #         dice_picks.append(1)
    #     else:
    #         dice_picks.append(0)

    dice_picks = [int(x) for x in distribution.sample()]
    return dice_picks, log_probs

def pick_random_dice(inputs, model, roll):
    return np.random.choice([0,1], size = 5)

def pick_frequent_dice(inputs, model, roll):
    hand = [x[1] for x in inputs.items() if f'hand_{roll}' in x[0]]
    dice_picks = []
    for idx, die in enumerate(hand):
        rest_of_hand = np.delete(hand, idx)
        if die in rest_of_hand:
            dice_picks.append(1)
        else:
            dice_picks.append(0)
    return dice_picks    

#######################
### SCORE FUNCTIONS ###
#######################

def model_pick_score(inputs, model, potential_scores, score_sheet):
    # model = model.eval()
    # with torch.no_grad():
    probabilities = model(inputs)[2][0]
    
    # ineligible_idx = [idx for idx, value in enumerate(list(potential_scores.values())) if value < 0]
    # for idx in range(len(probabilities)):
    #     if idx in ineligible_idx:
    #         probabilities[idx] = 0
    # score_pick_idx = predictions.argmax()

    # Create a mask for eligible scores
    mask = torch.ones_like(probabilities, dtype = torch.bool)
    for idx, value in enumerate(list(potential_scores.values())):
        if value < 0:
            mask[idx] = False

    # Apply the mask (set ineligible probabilities to -inf)
    masked_probabilities = probabilities.masked_fill(~mask, float('-inf'))

    # Apply softmax to the masked probabilities
    distribution = torch.distributions.Categorical(torch.softmax(masked_probabilities, dim=0))
    score_pick_idx = distribution.sample()
    log_prob = distribution.log_prob(score_pick_idx)

    score_pick = list(score_sheet.scores)[score_pick_idx]
    return score_pick, log_prob

def pick_random_score(inputs, model, potential_scores, score_sheet):
    eligible_scores = [x[0] for x in potential_scores.items() if x[1] >= 0]
    score_pick = np.random.choice(eligible_scores)
    return score_pick

def pick_max_score(inputs, model, potential_scores, score_sheet):
    return max(potential_scores, key = potential_scores.get)

#######################
###### GAME LOOP ######
#######################

def prep_data_for_model(turn_data, model):
    model_inputs = turn_data
    pad_length = model.input_sizes[2] - len(model_inputs)
    model_inputs = F.pad(
        torch.tensor([x for x in model_inputs.values()], dtype = torch.float32), 
        (0, pad_length), 'constant', 0
    )
    return model_inputs

def pick_dice(decision_function, inputs, model, roll):
    dice_picks = decision_function(inputs, model, roll)
    return dice_picks

def pick_score(decision_function, inputs, model, potential_scores, score_sheet):
    score_pick = decision_function(inputs, model, potential_scores, score_sheet)
    return score_pick

def play_botzee(game_number, dice_function, score_function, model = None, rl = False):
            
    game_number = game_number

    # to capture game data
    game_data = []
    dice_probs_1_all = []
    dice_probs_2_all = []
    score_pick_probs = []

    # instantiate a new score sheet and game variables
    score_sheet = ScoreSheet()
    score_sheet.initialize_score_types()

    # 13 rounds per game
    for turn_number in range(1,14):

        # instantiate a new game data object
        turn = TurnData(game_number, turn_number, score_sheet.total)
        for score in score_sheet.get_current_scores().items():
            turn.__dict__.update([score])

        # FIRST ROLL
        hand = np.sort(np.random.randint(1, 7, 5))[::-1]
        for idx, die in enumerate(hand):
            turn.__dict__.update([(f'hand_1_dice_{idx+1}', die)])

        # calculate potential score based on current hand
        potential_scores = score_sheet.get_potential_scores(hand)
        for score_type, score in potential_scores.items():
            turn.__dict__.update([(f'{score_type}_potential_1', score)])
        
        # choose dice to keep
        if model is not None:
            inputs = prep_data_for_model(turn.capture_data(), model)
            dice_picks, dice_probs_1 = pick_dice(dice_function, inputs, model, roll = 1)
        else:
            inputs = turn.capture_data()
            dice_picks = pick_dice(dice_function, inputs, model, roll = 1)
        for idx, pick in enumerate(dice_picks):
            turn.__dict__.update([(f'picks_1_dice_{idx+1}', pick)])
        hand = hand[np.array(dice_picks, dtype = bool)]

        # SECOND ROLL
        roll = np.random.randint(1, 7, 5 - len(hand))
        hand = np.sort(np.append(hand, roll))[::-1]
        for idx, die in enumerate(hand):
            turn.__dict__.update([(f'hand_2_dice_{idx+1}', die)])

        # calculate potential score based on current hand
        potential_scores = score_sheet.get_potential_scores(hand)
        for score_type, score in potential_scores.items():
            turn.__dict__.update([(f'{score_type}_potential_2', score)])
        
        # prep data for decision
        if model is not None:
            inputs = prep_data_for_model(turn.capture_data(), model)
            dice_picks, dice_probs_2 = pick_dice(dice_function, inputs, model, roll = 2)
        else:
            inputs = turn.capture_data()
            dice_picks = pick_dice(dice_function, inputs, model, roll = 2)
        for idx, pick in enumerate(dice_picks):
            turn.__dict__.update([(f'picks_2_dice_{idx+1}', pick)])
        hand = hand[np.array(dice_picks, dtype = bool)]

        # THIRD ROLL
        roll = np.random.randint(1, 7, 5 - len(hand))
        hand = np.sort(np.append(hand, roll))[::-1]
        for idx, die in enumerate(hand):
            turn.__dict__.update([(f'hand_3_dice_{idx+1}', die)])

        # calculate potential score based on current hand
        potential_scores = score_sheet.get_potential_scores(hand)
        for score_type, score in potential_scores.items():
            turn.__dict__.update([(f'{score_type}_potential_3', score)])

        # prep data for decision
        if model is not None:
            inputs = prep_data_for_model(turn.capture_data(), model)
            score_pick, score_pick_prob = pick_score(score_function, inputs, model, potential_scores, score_sheet)
        else:
            inputs = turn.capture_data()
            score_pick = pick_score(score_function, inputs, model, potential_scores, score_sheet)

        # CHOOSE SCORE
        turn.score_pick = score_pick
        turn.turn_score = potential_scores[turn.score_pick]

        # mark score
        score_sheet.mark_score(turn.score_pick, turn.turn_score)
        turn.post_total_score = score_sheet.total

        game_data.append(turn.capture_data())

        if rl:
            dice_probs_1_all.append(dice_probs_1)
            dice_probs_2_all.append(dice_probs_2)
            score_pick_probs.append(score_pick_prob)

    if rl:
        return (game_data, dice_probs_1_all, dice_probs_2_all, score_pick_probs)
    else:
        return game_data


# OLD
# def simulate_game(game_number, dice_decision_function, score_decision_function):
        
#     game_number = game_number
    
#     # to capture game data
#     game_data = []

#     # instantiate a new score sheet and game variables
#     score_sheet = ScoreSheet()
#     score_sheet.initialize_score_types()

#     # 13 rounds per game
#     for turn_number in range(1,14):

#         # instantiate a new game data object
#         remaining_score_types = [x[0] for x in score_sheet.scores.items() if x[1] == None]
#         turn = TurnData(game_number, turn_number, score_sheet.total)

#         # first roll
#         turn.hand_1 = np.sort(np.random.randint(1, 7, 5))[::-1]

#         # choose dice to keep
#         turn.dice_picks_1, hand_1_keepers = pick_dice(turn.hand_1, dice_decision_function)

#         # second roll
#         roll = np.random.randint(1, 7, 5 - len(hand_1_keepers))
#         turn.hand_2 = np.sort(np.append(hand_1_keepers, roll))[::-1]

#         # choose dice to keep
#         turn.dice_picks_2, hand_2_keepers = pick_dice(turn.hand_2, dice_decision_function)

#         # third roll
#         roll = np.random.randint(1, 7, 5 - len(hand_2_keepers))
#         turn.hand_3 = np.sort(np.append(hand_2_keepers, roll))[::-1]

#         # loop through all remaining scores, check which are eligible and calculate potential score
#         potential_scores = {}
#         for score_type, score in zip(score_sheet.score_types, score_sheet.scores.items()):
#             if score_type.check_condition(turn.hand_3) and score[1] == None:
#                 value = score_type.calculate_score(turn.hand_3)
#             elif score[1] != None:
#                 value = -1
#             else:
#                 value = 0
#             potential_scores[score_type.name] = value

#         # choose score type
#         turn.chosen_score_type, turn.turn_score = pick_score(potential_scores, score_decision_function)

#         # mark score
#         score_sheet.mark_score(turn.chosen_score_type, turn.turn_score)
#         turn.post_total_score = score_sheet.total

#         # capture data
#         turn_data = turn.capture_data()

#         remaining_score_types = dict(zip(
#             [f'{x}_score' for x in score_sheet.scores.keys()],
#             score_sheet.scores.values()
#         ))
#         turn_data.update(remaining_score_types)

#         for old_key in list(potential_scores.keys()):
#             potential_scores[f'{old_key}_potential'] = potential_scores.pop(old_key)
#         turn_data.update(potential_scores)

#         game_data.append(turn_data)
    
#     return game_data

# # picking dice and scores
# def pick_dice(hand, decision_function, **kwargs):
#     dice_picks = decision_function(hand, **kwargs)
#     keepers = hand[np.array(dice_picks, dtype = bool)]
#     return dice_picks, keepers

# def pick_score(potential_scores, decision_function, **kwargs):
#     chosen_score_type = decision_function(potential_scores, **kwargs)
#     turn_score = potential_scores[chosen_score_type]
#     return chosen_score_type, turn_score

# def pick_random_dice(hand):
#     return np.random.choice([0,1], size = 5)

# def pick_random_score(potential_scores):
#     eligible_scores = [x[0] for x in potential_scores.items() if x[1] >= 0]
#     chosen_score_type = np.random.choice(eligible_scores)
#     return chosen_score_type

# def simple_sim(game_number):

#     game_number = game_number

#     game_data = []

#     score_sheet = ScoreSheet()
#     score_sheet.initialize_score_types()

#     for turn_number in range(1,14):
        
#         turn = TurnData(game_number, turn_number, score_sheet.total)

#         # first roll
#         turn.hand_1 = np.sort(np.random.randint(1, 7, 5))[::-1]

#         # choose dice to keep
#         turn.dice_picks_1 = np.random.choice([0,1], size = 5)
#         hand_1_keepers = turn.hand_1[np.array(turn.dice_picks_1, dtype = bool)]

#         # second roll
#         roll = np.random.randint(1, 7, 5 - len(hand_1_keepers))
#         turn.hand_2 = np.sort(np.append(hand_1_keepers, roll))[::-1]

#         # choose dice to keep
#         turn.dice_picks_2 = np.random.choice([0,1], size = 5)
#         hand_2_keepers = turn.hand_2[np.array(turn.dice_picks_2, dtype = bool)]

#         # third roll
#         roll = np.random.randint(1, 7, 5 - len(hand_2_keepers))
#         turn.hand_3 = np.sort(np.append(hand_2_keepers, roll))[::-1]

#         turn.chosen_score_type = np.random.choice(list(score_sheet.scores.keys()))
#         turn.turn_score = 10

#         # mark score
#         score_sheet.mark_score(turn.chosen_score_type, turn.turn_score)
#         turn.post_total_score = score_sheet.total

#         game_data.append(turn.capture_data())

#         # capture data
#         turn_data = turn.capture_data()

#         remaining_score_types = dict(zip(
#             [f'{x}_score' for x in score_sheet.scores.keys()],
#             score_sheet.scores.values()
#         ))
#         turn_data.update(remaining_score_types)

#         game_data.append(turn_data)
    
#     return game_data


def reinforce_by_turn(_, model, optimizer, baseline):

    # Play game
    game_data, dice_probs_1, dice_probs_2, score_pick_probs = play_botzee(
        game_number = 0, 
        dice_function = model_pick_dice, 
        score_function = model_pick_score, 
        model = model,
        rl = True
    )
    score = game_data[-1]['post_total_score']
    rewards = [x['turn_score'] for x in game_data]
    advantages = [reward - baseline for reward in rewards]

    # Initialize lists for concatenated log probabilities
    concatenated_log_probs = []

    # Iterate through each turn and concat log probs
    for dp1, dp2, sp in zip(dice_probs_1, dice_probs_2, score_pick_probs):
        turn_log_probs = torch.cat([dp1, dp2, sp.unsqueeze(0)])
        concatenated_log_probs.append(turn_log_probs)

    # Calculate policy gradient loss
    loss = []
    for log_probs, advantage in zip(concatenated_log_probs, advantages):
        for log_prob in log_probs:
            loss.append(-log_prob * advantage)  # Negative because we're doing gradient ascent
    loss = torch.sum(torch.stack(loss))

    # Update the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # detach loss
    loss = loss.detach().clone().item()

    return score, loss, advantages