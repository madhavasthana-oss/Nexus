#define a pokemon class with attributes like name, type, level, and moves

class Pokemon:
    #have poketype as list 

    def __init__(self, name: str, poketype: list[str], level: int = 1, moves: list[str] = None):
        ...
    def level_up(self, amount: int):
        ...  # Increase the Pokemon's level by 1

    def add_move(self, move: str):
        ...  # Add a move to the Pokemon's move list

    def description(self)-> str:
        ...
    
    def pokemonAttacks(self, other: 'Pokemon', move: str)-> attack:
        ...

    class attack:
        def __init__(self, move: str, damage: int):
            ...

        def attackDescription(self):
            ...