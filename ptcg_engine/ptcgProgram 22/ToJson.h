// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "ApiJson.h"


inline void CardJson(const State& state, JsonBuilder& j, CardRef ref, bool canReverse, bool addName = false) {
	const Card& card = state.getCard(ref);
	if (canReverse && card.reverse) {
		j.appendNull();
	} else {
		const CardMaster& master = card.getMaster();
		j.append('{');
		j.appendKeyValue("id", master.cardId);
		j.appendCommaKeyValue("serial", ref.cardIndex);
		j.appendCommaKeyValue("playerIndex", card.playerIndex);
		if (addName) {
			j.appendCommaKeyValue("name", master.nameEn);
		}
		j.append('}');
	}
}

template<typename T>
inline void CardListJson(const State& state, JsonBuilder& j, const T& list, bool canReverse, bool addName) {
	j.append('[');
	for (int i = 0; i < list.size(); i++) {
		j.comma(i);
		CardJson(state, j, list[i], canReverse, addName);
	}
	j.append(']');
}

inline void PokemonJson(const State& state, JsonBuilder& j, CardRef ref, bool addName) {
	const Card& card = state.getCard(ref);
	if (card.reverse && !addName) {
		j.appendNull();
	} else {
		const CardMaster& master = card.getMaster();
		int playerIndex = card.playerIndex;
		j.append('{');

		j.appendKeyValue("id", master.cardId);
		j.appendCommaKeyValue("serial", ref.cardIndex);
		j.appendCommaKeyValue("playerIndex", card.playerIndex);
		if (addName) {
			j.appendCommaKeyValue("name", master.nameEn);
		}
		j.appendCommaKeyValue("hp", state.getHp(card));
		j.appendCommaKeyValue("maxHp", state.getMaxHp(card));
		j.appendCommaKeyValue("appearThisTurn", card.appear);

		auto& energies = state.game->energyList;
		state.getEnergies(playerIndex, ref, energies);
		j.appendCommaKey("energies");
		j.append('[');
		for (int i : range(energies)) {
			j.comma(i);
			j.append(EnergyTypeIndex(energies[i]));
		}
		j.append(']');

		auto& cardList = state.game->cardList;
		state.getEnergyCards(ref, cardList);
		j.appendCommaKey("energyCards");
		CardListJson(state, j, cardList, false, addName);

		auto tools = state.getAttachedToolRef(card);
		j.appendCommaKey("tools");
		CardListJson(state, j, tools, false, addName);

		auto preEvolutions = state.getPreEvolutions(card);
		j.appendCommaKey("preEvolution");
		CardListJson(state, j, preEvolutions, false, addName);

		j.append('}');
	}
}

template<typename T>
inline void PokemonListJson(const State& state, JsonBuilder& j, const T& list, bool addName) {
	j.append('[');
	for (int i = 0; i < list.size(); i++) {
		j.comma(i);
		PokemonJson(state, j, list[i], addName);
	}
	j.append(']');
}

inline void PlayerJson(const State& state, JsonBuilder& j, int pi, int myPlayerIndex) {
	bool addName = (myPlayerIndex == 2);
	const PlayerState& ps = state.players[pi];

	j.append('{');

	j.appendKey("active");
	PokemonListJson(state, j, ps.active, addName);

	j.appendCommaKey("bench");
	PokemonListJson(state, j, ps.bench, addName);

	j.appendCommaKeyValue("benchMax", state.benchCapacity(pi));
	j.appendCommaKeyValue("deckCount", ps.deck.size());

	j.appendCommaKey("discard");
	CardListJson(state, j, ps.trash, false, addName);

	j.appendCommaKey("prize");
	CardListJson(state, j, ps.prize, myPlayerIndex != 2, addName);

	j.appendCommaKeyValue("handCount", ps.hand.size());

	j.appendCommaKey("hand");
	if (pi == myPlayerIndex || myPlayerIndex == 2) {
		CardListJson(state, j, ps.hand, false, addName);
	} else {
		j.appendNull();
	}

	if (state.game->config.sendDeck || myPlayerIndex == 2) {
		j.appendCommaKey("deck");
		CardListJson(state, j, ps.deck, false, addName);
	}

	j.appendCommaKeyValue("poisoned", ps.isPoisoned());
	j.appendCommaKeyValue("burned", ps.burned);
	j.appendCommaKeyValue("asleep", ps.badStatus == BadStatusType::Asleep);
	j.appendCommaKeyValue("paralyzed", ps.badStatus == BadStatusType::Paralyzed);
	j.appendCommaKeyValue("confused", ps.badStatus == BadStatusType::Confused);

	j.append('}');
}

inline void Current(const State& state, JsonBuilder& j, int playerIndex, bool web) {
	bool addName = (playerIndex == 2);
	j.append('{');

	j.appendKeyValue("turn", state.turn);
	j.appendCommaKeyValue("turnActionCount", state.turnActionCount);
	if (playerIndex == 2) {
		j.appendCommaKeyValue("yourIndex", state.selectPlayer);
	} else {
		j.appendCommaKeyValue("yourIndex", playerIndex);
	}
	j.appendCommaKeyValue("firstPlayer", state.firstPlayer);
	j.appendCommaKeyValue("supporterPlayed", state.supporterPlayed);
	j.appendCommaKeyValue("stadiumPlayed", state.stadiumPlayed);
	j.appendCommaKeyValue("energyAttached", state.energyPlayed);
	j.appendCommaKeyValue("retreated", state.retreated);
	j.appendCommaKeyValue("result", state.apiResult());

	j.appendCommaKey("stadium");
	CardListJson(state, j, state.stadium, false, addName);

	if (web) {
		j.appendCommaKeyValue("lookingCount", state.looking.size());
	}

	j.appendCommaKey("looking");
	if (state.looking.size() == 0 || (state.lookingPlayer != playerIndex && state.lookingPlayer != 2 && playerIndex != 2)) {
		if(state.looking.size() != 0 && state.lookingPlayer >= 3 && state.lookingPlayer == playerIndex + 3){
			j.append('[');
			for (int i = 0; i < state.looking.size(); i++) {
				j.comma(i);
				j.appendNull();
			}
			j.append(']');
		} else {
			j.appendNull();
		}
	} else {
		CardListJson(state, j, state.looking, false, addName);
	}

	j.appendCommaKey("players");
	j.append('[');
	for (int i = 0; i < 2; i++) {
		j.comma(i);
		PlayerJson(state, j, i, playerIndex);
	}
	j.append(']');

	j.append('}');
}

inline bool NeedContextCard(SelectContext context) {
	return context == SelectContext::Activate;
}

inline void SelectJson(const State& state, JsonBuilder& j, bool web) {
	j.append('{');

	if (web) {
		j.appendKey("type");
		j.appendDoubleQuote(SelectTypeStr[(int)state.selectType]);

		j.appendCommaKey("context");
		j.appendDoubleQuote(SelectContextStr[(int)state.selectContext]);
	} else {
		j.appendKeyValue("type", std::max(0, (int)state.selectType - 1));
		j.appendCommaKeyValue("context", std::max(0, (int)state.selectContext - 1));
	}

	j.appendCommaKeyValue("minCount", state.selectMin);
	j.appendCommaKeyValue("maxCount", state.selectMax);
	j.appendCommaKeyValue("remainDamageCounter", state.remainDamageCounter);
	j.appendCommaKeyValue("remainEnergyCost", state.remainEnergyCost);

	j.appendCommaKey("option");
	j.append('[');
	for (int i :range(state.options)) {
		j.comma(i);
		SelectOptionJson(j, state.options[i], web);
	}
	j.append(']');

	j.appendCommaKey("deck");
	if (state.selectDeck) {
		CardListJson(state, j, state.players[state.selectPlayer].deck, false, false);
	} else {
		j.appendNull();
	}

	j.appendCommaKey("contextCard");
	if (state.contextCard.isNull()) {
		j.appendNull();
	} else {
		CardJson(state, j, state.contextCard, false);
	}

	j.appendCommaKey("effect");
	if (state.onEffect()) {
		CardJson(state, j, state.getEffectCard().card, false);
	} else {
		j.appendNull();
	}

	j.append('}');
}

inline void LogsJson(const State& state, JsonBuilder& j, int playerIndex, int startLogIndex, bool web) {
	auto& logList = state.logs;
	int logSize = (int)logList.size() - startLogIndex;
	int counter = 0;
	j.append('[');
	for (int i : range(logSize)) {
		const Log& log = logList[i + startLogIndex];
		if (log.logType > LogType::Result) {
			continue;
		}
		j.comma(counter);
		counter++;
		LogJson(j, log, playerIndex, web);
	}
	j.append(']');
}

inline void ToJsonWeb(const State& state, JsonBuilder& j, int playerIndex, int startLogIndex, int responseIndex) {
	j.clear();
	j.append('{');

	j.appendKeyValue("index", responseIndex);

	j.appendCommaKey("select");
	if (state.selectType == SelectType::None || state.selectPlayer != playerIndex) {
		j.appendNull();
	} else {
		SelectJson(state, j, true);
	}

	j.appendCommaKey("logs");
	LogsJson(state, j, playerIndex, startLogIndex, true);

	j.appendCommaKey("current");
	Current(state, j, playerIndex, true);

	j.append('}');
}

inline void ToJsonApi(const State& state, JsonBuilder& j, int startLogIndex) {
	assert(state.selectType != SelectType::None || state.isFinish());
	int playerIndex = state.selectPlayer;

	j.append('{');

	j.appendKey("select");
	SelectJson(state, j, false);

	j.appendCommaKey("logs");
	LogsJson(state, j, playerIndex, startLogIndex, false);

	j.appendCommaKey("current");
	Current(state, j, playerIndex, false);

	j.append('}');
}

inline void ToJsonSearch(State* state, JsonBuilder& j, int searchId, int error) {
	j.clear();

	j.append('{');

	j.appendKey("state");
	if (error == 0) {
		j.append('{');

		j.appendKey("observation");
		ToJsonApi(*state, j, state->nextLogStart());

		j.appendCommaKeyValue("searchId", searchId);

		j.append('}');
	} else {
		j.appendNull();
	}

	j.appendCommaKeyValue("error", error);

	j.append('}');
}

inline void ToJsonVis(const State& state, JsonBuilder& j, int startLogIndex, const std::vector<int>* selected) {
	j.clear();

	j.append('{');

	j.appendKey("select");
	SelectJson(state, j, true);

	j.appendCommaKey("logs");
	LogsJson(state, j, 2, startLogIndex, true);

	j.appendCommaKey("current");
	Current(state, j, 2, true);

	j.appendCommaKey("selected");
	if (selected == nullptr) {
		j.appendNull();
	} else {
		j.append('[');
		for (int i : range(*selected)) {
			j.comma(i);
			j.append(selected->at(i));
		}
		j.append(']');
	}

	j.append('}');
}
