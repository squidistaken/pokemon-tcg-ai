// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "GameProc.h"

inline void ResetupActivePokemon(State& state, int playerIndex);
inline void SetupActivePokemon(State& state, int playerIndex);
inline void PreSetupActivePokemon(State& state, int playerIndex);
inline void BothSetupActivePokemon(State& state);


inline void MoveToBenchSelected(State& state, int playerIndex) {
	PlayerState& ps = state.players[playerIndex];
	for (const AreaRef& ref : state.targetList) {
		int handIndex = state.currentAreaIndex(ps.hand, ref.card);
		MoveListCard(ps.hand, ps.bench, handIndex);
		state.cardMoved(ref.card, AreaType::Bench, true);
		LogMoveCard(state, playerIndex, ref.card, AreaType::Hand, AreaType::Bench, true);
	}
}

inline void SelectedSetupBenchPokemon(State& state, int playerIndex) {
	if (playerIndex == state.firstPlayer) {
		state.setSelectedCardTarget();
	} else {
		MoveToBenchSelected(state, state.firstPlayer);
		state.setSelectedCardTarget();
		MoveToBenchSelected(state, playerIndex);

		state.setupDone = {};
		for (int i : state.basicPlayerOrder()) {
			PlayerState& ps = state.players[i];
			state.getCard(ps.active.at(0)).reverse = false;
			for (CardRef ref : ps.bench) {
				state.getCard(ref).reverse = false;
			}
		}

		Card& card0 = state.getCard(state.players[state.firstPlayer].getActive());
		Card& card1 = state.getCard(state.players[1 - state.firstPlayer].getActive());
		if (card0.moveCounter > card1.moveCounter) {
			std::swap(card0.moveCounter, card1.moveCounter);
		}



		TurnStart(state);
	}
}

inline void SetupBenchPokemon(State& state, int playerIndex) {
	state.setSelect(SelectType::Card, SelectContext::SetupBenchPokemon, playerIndex);
	auto& hand = state.players[playerIndex].hand;
	for (int i : range(hand)) {
		if (state.getCard(hand[i]).getMaster().canSetup()) {
			AddOptionCard(state, AreaType::Hand, i, playerIndex);
		}
	}
	state.selectMin = 0;
	state.selectMax = std::min((int)state.options.size(), state.remainingBench(playerIndex));
	state.pushFunction(SelectedSetupBenchPokemon, playerIndex);
}


inline void SelectedDrawCount(State& state, int playerIndex) {
	int count = state.selectedNumber();
	state.clearSelect();
	Draw(state, playerIndex, count);
}

inline void StartSetupBench(State& state) {
	for (int i : state.reversePlayerOrder()) {
		state.pushFunction(SetupBenchPokemon, i);
	}
	for (int i : state.basicPlayerOrder()) {
		int mulliganCount = state.mulliganCount[i];
		if (mulliganCount > 0) {
			int selectPlayer = 1 - i;
			state.setSelect(SelectType::Count, SelectContext::DrawCount, selectPlayer);
			for (int j = 0; j <= mulliganCount; j++) {
				AddOptionNumber(state, j);
			}
			state.pushFunction(SelectedDrawCount, selectPlayer);
		}
	}
}

inline void SetupPrize(State& state, int playerIndex) {
	DeckToPrize(state, playerIndex, PRIZE_SIZE);
}

inline void AfterResetupActivePokemon(State& state, int playerIndex) {
	if (state.setupDone[playerIndex]) {
		SetupPrize(state, playerIndex);
	} else {
		ResetupActivePokemon(state, playerIndex);
	}
}

inline void ResetupActivePokemon(State& state, int playerIndex) {
	if (state.mulliganCount[playerIndex] < DECK_SIZE - FIRST_HAND - PRIZE_SIZE) {
		state.mulliganCount[playerIndex] += 1;
	}
	OpenReturnAndShuffle(state, playerIndex);
	Draw(state, playerIndex, FIRST_HAND);
	state.pushFunction(AfterResetupActivePokemon, playerIndex);
	state.pushFunction(SetupActivePokemon, playerIndex);
	PreSetupActivePokemon(state, playerIndex);
}

inline void AfterSetupActivePokemon(State& state) {
	if (state.setupDone[0]) {
		if (state.setupDone[1]) {
			for (int i : state.basicPlayerOrder()) {
				SetupPrize(state, i);
			}
		} else {
			SetupPrize(state, 0);
			ResetupActivePokemon(state, 1);
		}
	} else {
		if (state.setupDone[1]) {
			SetupPrize(state, 1);
			ResetupActivePokemon(state, 0);
		} else {
			for (int i : state.basicPlayerOrder()) {
				OpenReturnAndShuffle(state, i);
				Draw(state, i, FIRST_HAND);
			}
			BothSetupActivePokemon(state);
		}
	}
}

inline void SelectedSetupActivePokemon(State& state, int playerIndex) {
	SelectOption option = state.firstSelected();
	state.clearSelect();
	PlayerState& ps = state.players[playerIndex];
	int handIndex = option.getCardPosition().areaIndex;
	CardRef ref = MoveListCard(ps.hand, ps.active, handIndex);
	state.cardMoved(ref, AreaType::Active, true);
	LogMoveCard(state, playerIndex, ref, AreaType::Hand, AreaType::Active, true);
	state.setupDone[playerIndex] = true;
}

// [hasBasic, hasDoll]
inline std::pair<bool, bool> HasBasic(State& state, int playerIndex) {
	bool hasBasic = false;
	bool hasDoll = false;
	for (CardRef ref : state.players[playerIndex].hand) {
		const CardMaster& master = state.getCard(ref).getMaster();
		if (master.cardType == CardType::Pokemon && master.evolutionType == EvolutionType::Basic) {
			hasBasic = true;
		} else if (master.toBattleFieldOnlySetup || master.toActiveOnlySetup) {
			hasDoll = true;
		}
	}
	return { hasBasic, hasDoll };
}


inline void SetupActivePokemon(State& state, int playerIndex) {
	auto [hasBasic, hasDoll] = HasBasic(state, playerIndex);
	if (!hasBasic && !hasDoll) {
		state.mulligan[playerIndex] = true;
	}

	if (state.mulligan[playerIndex]) {
		bool noBasicDeck = true;
		for (CardRef ref : state.players[playerIndex].deck) {
			const CardMaster& master = state.getCard(ref).getMaster();
			if (master.cardType == CardType::Pokemon && master.evolutionType == EvolutionType::Basic) {
				noBasicDeck = false;
			}
		}
		if (noBasicDeck && !hasBasic) {
			Exception("No Basic Pokemon.");
		}
	} else {
		state.setSelect(SelectType::Card, SelectContext::SetupActivePokemon, playerIndex);
		auto& hand = state.players[playerIndex].hand;
		for (int i : range(hand)) {
			if (state.getCard(hand[i]).getMaster().canSetupActive()) {
				AddOptionCard(state, AreaType::Hand, i, playerIndex);
			}
		}
		state.pushFunction(SelectedSetupActivePokemon, playerIndex);
	}
}

inline void BothSetupActivePokemon(State& state) {
	state.pushFunction(AfterSetupActivePokemon);
	state.pushFunction(SetupActivePokemon, 1 - state.firstPlayer);
	state.pushFunction(SetupActivePokemon, state.firstPlayer);
	state.pushFunction(PreSetupActivePokemon, 1 - state.firstPlayer);
	PreSetupActivePokemon(state, state.firstPlayer);
}

inline void SelectedMulligan(State& state, int playerIndex) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	LogHasBasicPokemon(state, playerIndex, !isYes);
	state.mulligan[playerIndex] = isYes;
}

inline void PreSetupActivePokemon(State& state, int playerIndex) {
	auto [hasBasic, hasDoll] = HasBasic(state, playerIndex);
	if (hasBasic) {
		LogHasBasicPokemon(state, playerIndex, true);
		state.mulligan[playerIndex] = false;
	} else if (hasDoll) {
		state.setSelect(SelectType::YesNo, SelectContext::Mulligan, playerIndex);
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedMulligan, playerIndex);
	} else {
		LogHasBasicPokemon(state, playerIndex, false);
		state.mulligan[playerIndex] = true;
	}
}

// 先攻か後攻か選択後
inline void SelectedIsFirst(State& state) {
	int selectPlayer = state.selectPlayer;
	if (state.selectedYes()) {
		state.firstPlayer = selectPlayer;
	} else {
		state.firstPlayer = 1 - selectPlayer;
	}
	state.clearSelect();

	for (int i : state.basicPlayerOrder()) {
		Draw(state, i, FIRST_HAND);
	}
	state.pushFunction(StartSetupBench);
	BothSetupActivePokemon(state);
}

// ゲーム開始時処理
inline void SetupGame(State& state) {
	for (int i : range(2)) {
		ShuffleDeck(state, i, true);
	}

	SetYesNoSelect(state, SelectContext::IsFirst, 0);
	state.pushFunction(SelectedIsFirst);
}

