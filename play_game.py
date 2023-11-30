import numpy as np

for round in range(13):

    hand = []

    for turn in range(3):

        roll = np.random.randint(1, 6, 5 - len(hand))
        hand = hand + roll

        keepers = np.random.choice(
            a = hand,
            size = np.random.randint(1,5)
        )
        hand = [x for x in hand if x in keepers]

        print(hand)

