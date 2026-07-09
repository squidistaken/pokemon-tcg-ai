# Pokémon TCG AI Battle Challenge — Simulator (`ptcgProgram`)

`ptcgProgram` is the battle engine (the "cabt Engine") that simulates and scores Pokémon TCG matches
in the competition. It implements the game state, card effects, and the JSON I/O the competition
environment uses.
Please do not delete or alter the contents of this README.

## Using it

This package is shared only for taking part in the competition, and is not open-source software.
Please use it just to build and test your entries, don't share or republish it, and delete it when
the competition ends. The license is in the `LICENSES/` folder, so please be sure to read it.

## Competition rules

The official Competition Rules are published on Kaggle and are the binding terms for this package:
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules . You agreed to them when you entered;
the version on Kaggle (the version you accepted) governs.

## Copyright and intellectual property

© Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
The engine is provided "as is", without warranties of any kind, express or implied. To the extent permitted by law, The Pokémon Company accepts no liability for any loss or damage arising from its use.

The contents of this package — the code, card names, card text, and characters — together with any
derivative work of them, shall be deemed "Pokémon Elements" under the Competition Rules and belong to
Pokémon/Nintendo/Creatures/GAME FREAK.
No trademark or patent license is granted, and no rights are granted beyond those set out in the Rules.

## Build notes

- C++20 (uses `<ranges>`, `<bit>`, etc.), header-only style; entry points in `Export.cpp` / `All.h`.
- Visual Studio 2022 (v17.12) solution included (`game.sln`).
- No third-party or open-source dependencies — the C++ standard library only.

## Contents

- `LICENSES/` — the license (what you may do with this package)
- `*.h`, `Export.cpp`, `game.sln`, `game.vcxproj` — the C++ source
- `REUSE.toml` — license metadata (REUSE)

## Questions

For anything about the competition or this package, use the competition Discussion forum:
[competition discussion URL](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion?sort=hotness).
