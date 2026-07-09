// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "State.h"

struct EnergyTable {
	std::array<int, 10> energies;
	int colorless;
	int sum;

	void set(const AttackEnergies& src) {
		for (EnergyType e : src) {
			if (e == EnergyType::Colorless) {
				colorless++;
			} else {
				energies.at(EnergyTypeIndex(e))++;
			}
			sum++;
		}
	}
};

inline int InsufficientEnergyCount(const State& state, const AttackEnergies& required, const std::vector<EnergyType>& energies, const Card& card, const Attack& attack, bool returnExtra = false) {
	EnergyTable requiredTable = {};
	bool fixEnergy = false;
	if (attack.noEnergyIfSpecialCondition && state.players[card.playerIndex].isSpecialCondition()) {
		fixEnergy = true;
	} else if ((card.attackEnergyColoressOne || card.attackEnergyPsychicOne) && card.getMaster().hasAttack(&attack)) {
		fixEnergy = true;
		if (card.attackEnergyColoressOne) {
			requiredTable.colorless++;
		} else {
			requiredTable.energies.at(EnergyTypeIndex(EnergyType::Psychic))++;
		}
		requiredTable.sum++;
	} else if (attack.darkness1IfDamaged && card.damage > 0) {
		fixEnergy = true;
		requiredTable.energies.at(EnergyTypeIndex(EnergyType::Darkness))++;
		requiredTable.sum++;
	} else {
		requiredTable.set(required);
	}


	int allCount = 0;
	int pdCount = 0;
	int colorlessCount = 0;
	for (EnergyType e : energies) {
		if (e == EnergyType::All) {
			allCount++;
		} else if (e == EnergyType::Colorless) {
			colorlessCount++;
		} else if (e == (EnergyType::Psychic | EnergyType::Darkness)) {
			pdCount++;
		} else {
			int& num = requiredTable.energies.at(EnergyTypeIndex(e));
			if (num > 0) {
				num--;
				requiredTable.sum--;
			} else {
				colorlessCount++;
			}
		}
	}

	if (!fixEnergy) {
		allCount += card.attackCostDown;

		int change = card.attackCostChangeColorless + card.thisTurn.attackCostChange;
		if (change < 0) {
			colorlessCount -= change;
		} else {
			requiredTable.colorless += change;
			requiredTable.sum += change;
		}

		if (card.attackCostDownColorlessOwnAttack > 0) {
			if (card.getMaster().hasAttack(&attack)) {
				colorlessCount += card.attackCostDownColorlessOwnAttack;
			}
		}
	}

	for (int i : range(pdCount)) {
		{
			int& num = requiredTable.energies.at(EnergyTypeIndex(EnergyType::Psychic));
			if (num > 0) {
				num--;
				requiredTable.sum--;
				continue;
			}
		}
		{
			int& num = requiredTable.energies.at(EnergyTypeIndex(EnergyType::Darkness));
			if (num > 0) {
				num--;
				requiredTable.sum--;
				continue;
			}
		}
		colorlessCount++;
	}

	if (returnExtra) {
		return requiredTable.sum - allCount - colorlessCount;
	} else {
		int needCount = requiredTable.sum - allCount - std::min(colorlessCount, requiredTable.colorless);
		return std::max(needCount, 0);
	}
}

inline bool EnoughEnergy(const State& state, const AttackEnergies& required, const std::vector<EnergyType>& energies, const Card& card, const Attack& attack) {
	return InsufficientEnergyCount(state, required, energies, card, attack) <= 0;
}


inline void SetAttackEnergy(const State& state, const Card& card, const std::vector<EnergyType>& energyList, bool extract, bool ownOnly = false) {
	std::vector<AttackEnergy>& result = state.game->attackEnergyList;
	result.clear();
	int playerIndex = card.playerIndex;
	const CardMaster& master = card.getMaster();
	for (const Attack* attack : master.attacks) {
		int count = InsufficientEnergyCount(state, attack->energies, energyList, card, *attack);
		if (count <= 0) {
			if (extract) {
				if (attack->asActiveEnemyPokemonAttack) {
					const CardMaster& enemy = state.getCard(state.players[1 - playerIndex].getActive()).getMaster();
					for (const Attack* enemyAttack : enemy.attacks) {
						result.push_back({ enemyAttack, 0, attack->attackId });
					}
				} else if (attack->asActiveEnemyTerastalPokemonAttack) {
					const CardMaster& enemy = state.getCard(state.players[1 - playerIndex].getActive()).getMaster();
					if (enemy.tera) {
						for (const Attack* enemyAttack : enemy.attacks) {
							result.push_back({ enemyAttack, 0, attack->attackId });
						}
					}
				} else if(attack->asMyBenchNPokemonAttack){
					for (CardRef ref : state.players.at(card.playerIndex).bench) {
						const Card& c = state.getCard(ref);
						const CardMaster& m = c.getMaster();
						if (m.n) {
							for (const Attack* a : m.attacks) {
								result.push_back({ a, 0, attack->attackId });
							}
						}
					}
				} else {
					result.push_back({ attack, 0 });
				}
			} else {
				result.push_back({ attack, 0 });
			}
		} else {
			result.push_back({ attack, count });
		}
	}
	if (!ownOnly) {
		if (card.canUsePreEvolutionAttack) {
			for (CardRef preEvolutionRef : state.players.at(playerIndex).preEvolution) {
				const Card& preEvolutionCard = state.getCard(preEvolutionRef);
				if (card.moveCounter == preEvolutionCard.attachMoveCounter) {
					for (const Attack* attack : preEvolutionCard.getMaster().attacks) {
						int count = InsufficientEnergyCount(state, attack->energies, energyList, card, *attack);
						result.push_back({ attack, count });
					}
				}
			}
		}
		if (card.technicalMachine) {
			for (CardRef ref : state.getAttachedToolRef(card)) {
				const Card& tool = state.getCard(ref);
				const CardMaster& toolMaster = tool.getMaster();
				if (toolMaster.name == u8"コアメモリ" && master.name != u8"メガジガルデex") {
					continue;
				}
				for (const Attack* attack : toolMaster.attacks) {
					int count = InsufficientEnergyCount(state, attack->energies, energyList, card, *attack);
					result.push_back({ attack, count });
				}
			}
		}
	}
}
