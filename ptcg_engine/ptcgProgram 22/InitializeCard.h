// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Card.h"

inline void InitializeCard() {
	std::unordered_map<std::u8string, std::u8string> evolveMap;
	for (const auto& entry : CardTable) {
		const CardMaster& card = entry.second;
		if (card.evolutionType == EvolutionType::Stage1) {
			evolveMap[card.name] = card.evolvesFrom;
		}
	}
	for (auto& entry : CardTable) {
		CardMaster& card = entry.second;
		if (card.evolutionType == EvolutionType::Stage2) {
			card.evolvesFrom2 = evolveMap.at(card.evolvesFrom);
		}

		for (Attack* attack : card.attacks) {
			if (attack->lastCancelFailAttack) {
				Effect& ef = attack->preEffects.emplace_back();
				ef = {};
				ef.attack = attack;
				ef.effectType = EffectType::CancelFailAttack;
			}
		}
	}
}
