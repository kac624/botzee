import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# define class
class ScoreType:
    def __init__(self, name, condition_function, scoring_function):
        self.name = name
        self.condition_function = condition_function
        self.scoring_function = scoring_function

    # method for checking whether the current hand 
    # meets the conditions for a certain score type
    def check_condition(self, hand):
        return self.condition_function(hand)
    
    # method for calculating score based on score 
    # type's unique logic
    def calculate_score(self, hand):
        return self.scoring_function(hand)
    
class TurnData:
    def __init__(self, game, turn, pre_total_score):
        self.game = game
        self.turn = turn
        self.pre_total_score = pre_total_score

    def capture_data(self):
        return self.__dict__
    
class ScoreSheet:
    def __init__(self, player_name = None, scores = None, score_types = None, total = 0):
        self.player_name = player_name
        self.total = 0

    # get current scores ? and probabilities?
    def get_current_scores(self):
        current_scores = dict(zip(
            [f'{x}_score' for x in self.scores.keys()],
            [x if x != None else -1 for x in self.scores.values()]
        ))
        return current_scores
    
    def get_potential_scores(self, hand):    
        potential_scores = {}
        for score_type, score in zip(self.score_types, self.scores.items()):
            if score_type.check_condition(hand) and score[1] == None:
                value = score_type.calculate_score(hand)
            elif score[1] != None:
                value = -1
            else:
                value = 0
            potential_scores[score_type.name] = value
        return potential_scores

    # update the scores attribute with the score score / type
    def mark_score(self, chosen_score_type, chosen_score):
        self.scores[chosen_score_type] = chosen_score
        self.total += chosen_score

    # establish each score type
    def initialize_score_types(self):
        chance = ScoreType(
            name = 'chance',
            condition_function = lambda _: True,
            scoring_function = lambda hand: hand.sum()
        )

        ones = ScoreType(
            name = 'ones',
            condition_function = lambda hand: 1 in hand,
            scoring_function = lambda hand: list(hand).count(1) * 1
        )

        twos = ScoreType(
            name = 'twos',
            condition_function = lambda hand: 2 in hand,
            scoring_function = lambda hand: list(hand).count(2) * 2
        )

        threes = ScoreType(
            name = 'threes',
            condition_function = lambda hand: 3 in hand,
            scoring_function = lambda hand: list(hand).count(3) * 3
        )

        fours = ScoreType(
            name = 'fours',
            condition_function = lambda hand: 4 in hand,
            scoring_function = lambda hand: list(hand).count(4) * 4
        )

        fives = ScoreType(
            name = 'fives',
            condition_function = lambda hand: 5 in hand,
            scoring_function = lambda hand: list(hand).count(5) * 5
        )

        sixes = ScoreType(
            name = 'sixes',
            condition_function = lambda hand: 6 in hand,
            scoring_function = lambda hand: list(hand).count(6) * 6
        )

        three_kind = ScoreType(
            name = 'three_kind',
            condition_function = lambda hand: np.unique(hand, return_counts = True)[1].max() >= 3,
            scoring_function = lambda hand: hand.sum()
        )

        four_kind = ScoreType(
            name = 'four_kind',
            condition_function = lambda hand: np.unique(hand, return_counts = True)[1].max() >= 4,
            scoring_function = lambda hand: hand.sum()
        )

        full_house = ScoreType(
            name = 'full_house',
            condition_function = lambda hand: (
                (np.unique(hand, return_counts = True)[1].max() == 3) &
                (np.unique(hand, return_counts = True)[1].min() == 2)
            ),
            scoring_function = lambda _: 25
        )

        small_straight = ScoreType(
            name = 'small_straight',
            condition_function = lambda hand: (
                all(number in hand for number in [1,2,3,4]) or
                all(number in hand for number in [2,3,4,5]) or
                all(number in hand for number in [3,4,5,6])
            ),
            scoring_function = lambda _: 30
        )

        large_straight = ScoreType(
            name = 'large_straight',
            condition_function = lambda hand: (
                all(number in hand for number in [1,2,3,4,5]) or
                all(number in hand for number in [2,3,4,5,6])
            ),
            scoring_function = lambda _: 40
        )

        yahtzee = ScoreType(
            name = 'yahtzee',
            condition_function = lambda hand: list(hand).count(hand[0]) == 5,
            scoring_function = lambda _: 50
        )

        # collect all types in tuple
        score_types = (
            chance, ones, twos, threes, fours, fives, sixes,
            three_kind, four_kind, full_house,
            small_straight, large_straight, yahtzee
        )

        self.score_types = score_types

        self.scores = dict(zip(
            [type.name for type in score_types],
            [None] * len(score_types)
        ))


class Botzee(nn.Module):
    def __init__(self, input_sizes, lstm_sizes, dice_output_size, score_output_size, masks):
        super(Botzee, self).__init__()
        self.input_sizes = input_sizes
        self.lstm_sizes = lstm_sizes
        self.dice_output_size = dice_output_size
        self.score_output_size = score_output_size
        self.masks = masks

        # branch 1 - binary dice pick 1
        self.lstm1 = nn.LSTM(input_sizes[0], lstm_sizes[0])
        self.fc1 = nn.Linear(lstm_sizes[0], lstm_sizes[0])
        self.branch1 = nn.Linear(lstm_sizes[0], dice_output_size)

        # branch 2 - binary dice pick 2
        self.lstm2 = nn.LSTM(input_sizes[1], lstm_sizes[1])
        self.fc2 = nn.Linear(lstm_sizes[1], lstm_sizes[1])
        self.branch2 = nn.Linear(lstm_sizes[1], dice_output_size)

        # branch 3 - multilabel score pick
        self.lstm3 = nn.LSTM(input_sizes[2], lstm_sizes[2])
        self.fc3 = nn.Linear(lstm_sizes[2], lstm_sizes[2])
        self.branch3 = nn.Linear(lstm_sizes[2], score_output_size)

    def forward(self, x):
        # in case of 2D input, add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # branch 1 - binary
        out1, _ = self.lstm1(x[:, self.masks[0]])
        out1 = F.relu(self.fc1(out1))
        out1 = self.branch1(out1)
        dice1_output = torch.sigmoid(out1)

        # branch 2 - binary
        out2, _ = self.lstm2(x[:, self.masks[1]])
        out2 = F.relu(self.fc2(out2))
        out2 = self.branch2(out2)
        dice2_output = torch.sigmoid(out2)

        # branch 3 - multilabel
        out3, _ = self.lstm3(x)
        out3 = F.relu(self.fc3(out3))
        out3 = self.branch3(out3)
        score_output = F.softmax(out3, dim = 1)

        return dice1_output, dice2_output, score_output
    
class BotzeeDataset(Dataset):
    def __init__(self, features, targets_branch1, targets_branch2, targets_branch3):
        self.features = features
        self.targets_branch1 = targets_branch1
        self.targets_branch2 = targets_branch2
        self.targets_branch3 = targets_branch3

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to tensors
        x = torch.tensor(self.features.iloc[idx, :].values, dtype=torch.float32)
        y1 = torch.tensor(self.targets_branch1.iloc[idx, :].values, dtype = torch.float32)
        y2 = torch.tensor(self.targets_branch2.iloc[idx, :].values, dtype = torch.float32)
        y3 = torch.tensor(self.targets_branch3.iloc[idx, :].values, dtype = torch.float32)
        return x, (y1, y2, y3)