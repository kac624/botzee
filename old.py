# define class for score sheet
class ScoreSheet:
    def __init__(self, player_name = None, sheet = None):
        self.player_name = player_name
        self.sheet = dict(zip(
            [type.name for type in score_types],
            [None] * len(score_types)
        ))

    # update the sheet attribute with the score score / type
    def mark_score(self, chosen_score_type, chosen_score):
        self.sheet[chosen_score_type] = chosen_score

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
        
# establish each score type
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
    chance,
    ones,
    twos,
    threes,
    fours,
    fives,
    sixes,
    three_kind,
    four_kind,
    full_house,
    small_straight,
    large_straight,
    yahtzee
)



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
    
# define class for score sheet
class ScoreSheet:
    def __init__(self, player_name = None, sheet = None):
        self.player_name = player_name
        self.sheet = dict(zip(
            [type.name for type in score_types],
            [None] * len(score_types)
        ))

    # update the sheet attribute with the score score / type
    def mark_score(self, chosen_score_type, chosen_score):
        self.sheet[chosen_score_type] = chosen_score

# establish each score type
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
    chance,
    ones,
    twos,
    threes,
    fours,
    fives,
    sixes,
    three_kind,
    four_kind,
    full_house,
    small_straight,
    large_straight,
    yahtzee
)

class TurnData:
    def __init__(
            self, game, turn, pre_total_score, # remaining_score_types, 
            hand_1 = None, dice_picks_1 = None, 
            hand_2 = None, dice_picks_2 = None, hand_3 = None, 
            chosen_score_type = None, # eligible_score_types = None, 
            post_total_score = None, turn_score = None
    ):
        self.game = game
        self.turn = turn
        self.pre_total_score = pre_total_score
        # self.remaining_score_types = remaining_score_types
        self.turn_score = turn_score

    def capture_data(self):
        return self.__dict__