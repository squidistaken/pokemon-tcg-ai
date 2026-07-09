// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "State.h"

inline void AddOptionEnd(State& state) {
	state.addOption(SelectOptionType::End);
}

// YesとNo両方追加
inline void AddOptionYesAndNo(State& state) {
	state.addOption(SelectOptionType::Yes);
	state.addOption(SelectOptionType::No);
}

inline void AddOptionNumber(State& state, int number) {
	SelectOption& option = state.addOption(SelectOptionType::Number);
	option.param0 = number;
}

inline void AddOptionCard(State& state, AreaType area, int index, int playerIndex) {
	SelectOption& option = state.addOption(SelectOptionType::Card);
	option.param0 = (short)area;
	option.param1 = index;
	option.param2 = playerIndex;
}

inline void AddOptionToolCard(State& state, AreaType area, int index, int playerIndex, int toolIndex) {
	SelectOption& option = state.addOption(SelectOptionType::ToolCard);
	option.param0 = (short)area;
	option.param1 = index;
	option.param2 = playerIndex;
	option.param3 = toolIndex;
}

inline void AddOptionEnergyCard(State& state, AreaType area, int index, int playerIndex, int energyIndex) {
	SelectOption& option = state.addOption(SelectOptionType::EnergyCard);
	option.param0 = (short)area;
	option.param1 = index;
	option.param2 = playerIndex;
	option.param3 = energyIndex;
}

inline void AddOptionEnergy(State& state, AreaType area, int index, int playerIndex, int energyIndex, int count) {
	SelectOption& option = state.addOption(SelectOptionType::Energy);
	option.param0 = (short)area;
	option.param1 = index;
	option.param2 = playerIndex;
	option.param3 = energyIndex;
	option.param4 = count;
}

inline void AddOptionPlay(State& state, int index) {
	SelectOption& option = state.addOption(SelectOptionType::Play);
	option.param0 = index;
}

inline void AddOptionAttach(State& state, AreaType area, int index, AreaType inPlayArea, int inPlayIndex) {
	SelectOption& option = state.addOption(SelectOptionType::Attach);
	option.param0 = (short)area;
	option.param1 = index;
	option.param2 = (short)inPlayArea;
	option.param3 = inPlayIndex;
}

inline void AddOptionEvolve(State& state, AreaType area, int index, AreaType inPlayArea, int inPlayIndex) {
	SelectOption& option = state.addOption(SelectOptionType::Evolve);
	option.param0 = (short)area;
	option.param1 = index;
	option.param2 = (short)inPlayArea;
	option.param3 = inPlayIndex;
}

inline void AddOptionAbility(State& state, AreaType area, int index) {
	assert(area == AreaType::Active || area == AreaType::Bench || area == AreaType::Stadium);
	SelectOption& option = state.addOption(SelectOptionType::Ability);
	option.param0 = (short)area;
	option.param1 = index;
}

inline void AddOptionDiscard(State& state, AreaType area, int index) {
	assert(area == AreaType::Active || area == AreaType::Bench || area == AreaType::Stadium);
	SelectOption& option = state.addOption(SelectOptionType::Discard);
	option.param0 = (short)area;
	option.param1 = index;
}

inline void AddOptionRetreat(State& state) {
	state.addOption(SelectOptionType::Retreat);
}

inline void AddOptionAttack(State& state, int attackId, int srcAttackId, int benchIndex = -1) {
	SelectOption& option = state.addOption(SelectOptionType::Attack);
	option.param0 = attackId;
	option.param1 = srcAttackId;
	option.param2 = benchIndex;
}

inline void AddOptionSpecialCondition(State& state, SelectSpecialConditionType type) {
	SelectOption& option = state.addOption(SelectOptionType::SpecialCondition);
	option.param0 = (short)type;
}


// カード効果でない場合はref0
inline void AddOptionSkillOrder(State& state, CardRef ref) {
	SelectOption& option = state.addOption(SelectOptionType::Skill);
	if (!ref.isNull()) {
		option.param0 = state.getCardId(ref);
		option.param1 = ref.cardIndex;
	}
}

