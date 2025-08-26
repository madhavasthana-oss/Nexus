# this is a package to initialize custom objects like students, employees, pokemon, fighters 
# etc. you name it and it will be added

import pandas as pd
from typing import List, Union
import random

pokemon_data = pd.read_csv('pokemon.csv')

pokemon_type_advantages = {
    "Normal": [],
    "Fire": ["Grass", "Ice", "Bug", "Steel"],
    "Water": ["Fire", "Ground", "Rock"],
    "Electric": ["Water", "Flying"],
    "Grass": ["Water", "Ground", "Rock"],
    "Ice": ["Grass", "Ground", "Flying", "Dragon"],
    "Fighting": ["Normal", "Ice", "Rock", "Dark", "Steel"],
    "Poison": ["Grass", "Fairy"],
    "Ground": ["Fire", "Electric", "Poison", "Rock", "Steel"],
    "Flying": ["Grass", "Fighting", "Bug"],
    "Psychic": ["Fighting", "Poison"],
    "Bug": ["Grass", "Psychic", "Dark"],
    "Rock": ["Fire", "Ice", "Flying", "Bug"],
    "Ghost": ["Psychic", "Ghost"],
    "Dragon": ["Dragon"],
    "Dark": ["Psychic", "Ghost"],
    "Steel": ["Ice", "Rock", "Fairy"],
    "Fairy": ["Fighting", "Dragon", "Dark"]
}

# Null effect - key types are immune to (take 0x damage from) value types
pokemon_type_null_effects = {
    "Normal": ["Ghost"],
    "Fire": [],
    "Water": [],
    "Electric": [],
    "Grass": [],
    "Ice": [],
    "Fighting": ["Ghost"],
    "Poison": [],
    "Ground": ["Electric"],
    "Flying": ["Ground"],
    "Psychic": ["Ghost"],
    "Bug": [],
    "Rock": [],
    "Ghost": ["Normal", "Fighting"],
    "Dragon": [],
    "Dark": ["Psychic"],
    "Steel": ["Poison"],
    "Fairy": ["Dragon"]
}

class Pokemon:
    def __init__(self, name: str, level: int = 1, moves: list['Pokemon.Attack'] = None):
        # Check if name exists in dataset
        if name not in pokemon_data['Name'].values:
            raise ValueError(f"{name} is not a valid Pokemon name.")
        
        self.name = name
        self.level = level if isinstance(level, int) and level > 0 else 1
        self.moves = moves if moves is not None else []
        
        # Get Pokemon data from dataset
        row_idx = pokemon_data[pokemon_data['Name'] == name].index[0]  # Get first index
        self.type1 = pokemon_data.loc[row_idx, 'Type 1']
        self.type2 = pokemon_data.loc[row_idx, 'type 2'] if pd.notna(pokemon_data.loc[row_idx, 'Type 2']) else None
        
        # Fix HP calculation logic
        self.hp = 4 * self.level if self.level > 10 else 40
        self.special_attack = pokemon_data.loc[row_idx, 'Sp. Atk']
        self.special_defense = pokemon_data.loc[row_idx, 'Sp. Def']
            
    def level_up(self, amount: int = 1):
        print(f"{self.name} has levelled up!")
        self.level += amount
        # Update HP when leveling up
        self.hp = 4 * self.level if self.level > 10 else 40

    def add_move(self, move: 'Pokemon.Attack'):
        print(f"{self.name} has learned {move.name}!")
        # Note: You need to check if 'moves' column exists in your CSV
        # if move not in pokemon_data['moves'].values:
        #     raise ValueError(f"{move} is not a valid move.")
        self.moves.append(move)

    def description(self):
        type_str = f"{self.type1}/{self.type2}" if self.type2 else self.type1
        print(f"Pokemon name: {self.name}")
        print(f"Level: {self.level}")
        print(f"Type: {type_str}")
        print(f"HP: {self.hp}")
        print(f"Special Attack: {self.special_attack}")
        print(f"Special Defense: {self.special_defense}")
        print(f"Moves: {', '.join([move.name for move in self.moves]) if self.moves else 'None'}")

    def pokemon_attacks(self, target_pokemon: 'Pokemon', move: 'Pokemon.Attack') -> 'Pokemon.Attack':
        if not isinstance(target_pokemon, Pokemon):
            raise ValueError("The target must be a Pokemon")
        if move not in self.moves:
            raise ValueError(f"{self.name} does not know the move {move.name}.")
    
        # Check for null effect (immunity)s
        if move.type and (move.type in pokemon_type_null_effects and target_pokemon.type1 in pokemon_type_null_effects[move.type] or (target_pokemon.type2 and target_pokemon.type2 in pokemon_type_null_effects[move.type])):
            print(f"It's ineffective! {target_pokemon.name} is immune to {move.type}-type attacks.")
        # Check for type advantage (super effective)
        elif move.type and ((target_pokemon.type1 in pokemon_type_advantages.get(move.type, [])) or (target_pokemon.type2 and target_pokemon.type2 in pokemon_type_advantages.get(move.type, []))):
            print("It's super effective!")
            damage = int(move.damage * 2)
        # Check for type disadvantage (not very effective)
        elif move.type and ((target_pokemon.type1 in pokemon_type_advantages and move.type in pokemon_type_advantages[target_pokemon.type1]) or (target_pokemon.type2 and target_pokemon.type2 in pokemon_type_advantages and move.type in pokemon_type_advantages[target_pokemon.type2])):
            print("It's not very effective...")
            damage = int(move.damage * 0.5)
        print(f"{self.name} attacks {target_pokemon.name} with {move.name} for {damage} damage!")
        target_pokemon.hp -= move.damage
        if target_pokemon.hp<=0:
            target_pokemon.hp=0
            print(f'{target_pokemon.name} has been defeated!! {self.name} wins!!')
        # Return the move (Attack instance) used
    def restore_hp(self):
        if self.hp==0:
            self.hp= 4 * self.level if self.level > 10 else 40
        else:
            print(f'{self.name} needs to be defeated first|| current hp={self.hp}')

    class Attack:  # Moved outside Pokemon class and capitalized
        def  _init__(self, name: str, damage: int, type: str = None):
            self.name = name
            self.damage = damage
            self.type = type

        def attack_description(self):  # Fixed method name
            print(f"{self.name} deals {self.damage} damage!")

# Example usage (assuming you have the CSV data):
# pikachu = Pokemon("Pikachu", level=5, moves=["Thunderbolt"])
# charizard = Pokemon("Charizard", level=10, moves=["Flamethrower"])
# 
# attack = pikachu.pokemon_attacks(charizard, "Thunderbolt")
# attack.attack_description()

class Trainer:
    def __init__(self, name: str, pokemon: List[Pokemon], pokemon_chosen: List[Pokemon]):
        if len(pokemon_chosen) != 7:
            raise ValueError("You can choose only 7 pokemon")
        self.name = name
        self.pokemon = pokemon
        self.pokemon_chosen = pokemon_chosen
        if not set(self.pokemon_chosen).issubset(set(self.pokemon)):
            raise ValueError("you don't have the pokemon you are choosing")
    
    def capture_pokemon(self,pokemon_to_capture: Pokemon):
        wild_battle=Battle(Challenger=self, Opponent=pokemon_to_capture)

class Battle:
    def __init__(self, Challenger: Trainer, Opponent:Union[Pokemon,Trainer]):
        self.Challenger=Challenger
        self.Opponent=Opponent
        if isinstance(Opponent, Pokemon):
            print(f'A wild {Opponent.name} has appeared!!')
            while(Opponent.hp>=0):
                pokemon_found=False
                print('Choose your pokemon!!')
                for P in Trainer.pokemon:
                    P.description()
                    print('\n')
                pokemon_str=input('enter the name of your chosen pokemon')
                if pokemon_str.lower() in Trainer.pokemon.name.lower() :
                    print(f'You have chosen {pokemon_str}')
                    pokemon_found = True
                    # Find the chosen Pokemon object
                    chosen_pokemon = next((p for p in Trainer.pokemon if p.name.lower() == pokemon_str.lower()), None)
                    if chosen_pokemon:
                        print(f"{chosen_pokemon.name} enters the battle!")
                        # Simple battle loop
                        while chosen_pokemon.hp > 0 and Opponent.hp > 0:
                            print(f"\n{chosen_pokemon.name} HP: {chosen_pokemon.hp} | {Opponent.name} HP: {Opponent.hp}")
                            print("Choose a move:")
                            for idx, move in enumerate(chosen_pokemon.moves):
                                print(f"{idx+1}. {move.name} (Type: {move.type}, Damage: {move.damage})")
                            move_choice = input("Enter move number: ")
                            try:
                                move_idx = int(move_choice) - 1
                                move = chosen_pokemon.moves[move_idx]
                            except (ValueError, IndexError):
                                print("Invalid move. Try again.")
                                continue
                            chosen_pokemon.pokemon_attacks(Opponent, move)
                            if Opponent.hp <= 0:
                                print(f"{Opponent.name} fainted!")
                                break
                            # Wild Pokemon attacks back (random move if available)
                            if Opponent.moves:
                                wild_move = random.choice(Opponent.moves)
                                Opponent.pokemon_attacks(chosen_pokemon, wild_move)
                                if chosen_pokemon.hp <= 0:
                                    print(f"{chosen_pokemon.name} fainted!")
                                    break
                            else:
                                print(f"{Opponent.name} has no moves left!")
                        break
                    else:
                        print("Pokemon not found in your team.")
                if not pokemon_found:
                    print("Please choose a valid Pokemon from your team.")
