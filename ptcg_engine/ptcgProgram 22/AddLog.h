// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "State.h"

inline void LogShuffle(State& state, int playerIndex) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Shuffle);
		log.add(playerIndex);
	}
}

inline void LogHasBasicPokemon(State& state, int playerIndex, bool hasBasicPokemon) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::HasBasicPokemon);
		log.add(playerIndex);
		log.add(hasBasicPokemon);
	}
}

inline void LogTurnStart(State& state, int playerIndex) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::TurnStart);
		log.add(playerIndex);
	}
}

inline void LogTurnEnd(State& state, int playerIndex) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::TurnEnd);
		log.add(playerIndex);
	}
}

inline void LogDraw(State& state, int playerIndex, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Draw);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogDrawReverse(State& state, int playerIndex) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::DrawReverse);
		log.add(playerIndex);
	}
}

inline void LogMoveCard(State& state, int playerIndex, CardRef ref, AreaType fromArea, AreaType toArea, int openType) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::MoveCard);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(fromArea);
		log.add(toArea);
		log.add(openType);
	}
}

inline void LogMoveCardReverse(State& state, int playerIndex, AreaType fromArea, AreaType toArea) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::MoveCardReverse);
		log.add(playerIndex);
		log.add(fromArea);
		log.add(toArea);
	}
}

inline void LogSwitch(State& state, int playerIndex, CardRef refActive, CardRef refBench) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Switch);
		log.add(playerIndex);
		log.add(state.getCardId(refActive));
		log.add(refActive.cardIndex);
		log.add(state.getCardId(refBench));
		log.add(refBench.cardIndex);
	}
}

inline void LogChange(State& state, int playerIndex, CardRef refBefore, CardRef refAfter) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Change);
		log.add(playerIndex);
		log.add(state.getCardId(refBefore));
		log.add(refBefore.cardIndex);
		log.add(state.getCardId(refAfter));
		log.add(refAfter.cardIndex);
	}
}

inline void LogPlay(State& state, int playerIndex, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Play);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogAttach(State& state, int playerIndex, CardRef ref, CardRef refTarget) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Attach);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(state.getCardId(refTarget));
		log.add(refTarget.cardIndex);
	}
}

inline void LogEvolve(State& state, int playerIndex, CardRef ref, CardRef refTarget) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Evolve);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(state.getCardId(refTarget));
		log.add(refTarget.cardIndex);
	}
}

inline void LogDevolve(State& state, int playerIndex, CardRef ref, CardRef refTarget) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Devolve);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(state.getCardId(refTarget));
		log.add(refTarget.cardIndex);
	}
}

inline void LogMoveAttached(State& state, int playerIndex, CardRef ref, CardRef refBefore, CardRef refAfter) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::MoveAttached);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(state.getCardId(refBefore));
		log.add(refBefore.cardIndex);
		log.add(state.getCardId(refAfter));
		log.add(refAfter.cardIndex);
	}
}

inline void LogAttack(State& state, int playerIndex, CardRef ref, int attackId) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Attack);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(attackId);
	}
}

inline void LogHpChange(State& state, int playerIndex, CardRef ref, int value, bool putDamageCounter) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::HpChange);
		log.add(playerIndex);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
		log.add(value);
		log.add(putDamageCounter);
	}
}

inline void LogPoisoned(State& state, int playerIndex, bool isRecover, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Poisoned);
		log.add(playerIndex);
		log.add(isRecover);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogBurned(State& state, int playerIndex, bool isRecover, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Burned);
		log.add(playerIndex);
		log.add(isRecover);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogAsleep(State& state, int playerIndex, bool isRecover, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Asleep);
		log.add(playerIndex);
		log.add(isRecover);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogParalyzed(State& state, int playerIndex, bool isRecover, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Paralyzed);
		log.add(playerIndex);
		log.add(isRecover);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogConfused(State& state, int playerIndex, bool isRecover, CardRef ref) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Confused);
		log.add(playerIndex);
		log.add(isRecover);
		log.add(state.getCardId(ref));
		log.add(ref.cardIndex);
	}
}

inline void LogCoin(State& state, int playerIndex, bool head) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Coin);
		log.add(playerIndex);
		log.add(head);
	}
}

inline void LogResult(State& state, int result, int reason) {
	if (state.isAddLog()) {
		Log& log = state.addLog(LogType::Result);
		log.add(result);
		log.add(reason);
	}
}

