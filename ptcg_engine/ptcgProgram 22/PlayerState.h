// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Card.h"

// 次の自分の番、自分は～/次の相手の番、相手は～
union PlayerNextTurnState {
	unsigned value;
	struct {
		short metalDamageChange;
		bool cannotAttackLessEqualEnergy2 : 1;
		bool cannotPlayItem : 1;
		bool cannotPlaySupporter : 1;
		bool cannotPlayStadium : 1;
		bool cannotPlaySpecialEnergy : 1;
		bool cannotEvolve : 1;
		bool cannotRetreatPoison : 1; // ポケモン効果扱い
	};
};

struct RefPosition {
	CardRef ref;
	AreaType area;
	int index;
};

using CardList = FixedList<CardRef, DECK_SIZE + 1>;

// 各プレイヤーの状態
struct PlayerState {
	ByteFixedList<CardRef, 1> active;
	ByteFixedList<CardRef, BENCH_SIZE_MAX> bench;
	CardList prize;
	CardList hand;
	CardList deck;
	CardList trash;
	CardList energy; // Attached Energy
	CardList tool; // Attached Tool
	CardList preEvolution;
	CardList temporary; // 一時エリア

	signed char playerIndex;
	bool koPrizeOnceChanged;

	PlayerNextTurnState thisTurn;
	PlayerNextTurnState nextTurn;

	union { // バトルポケモンの状態
		unsigned activeState;
		struct {
			signed char poisonDamageCounter; // どくでのせるダメカン
			BadStatusType badStatus;
			bool burned; // やけど
		};
	};

	union {
		unsigned reserved; // 予約領域
		struct {
			unsigned char reserved0;
		};
	};

	union {
		unsigned long long continualState;
		struct {
			short poisonDamageChange;
			short burnDamageChange;
			signed char poisonDamageChangeNotDarkness;
			unsigned char benchCapacity : 4; // ベンチ最大数を変更
			bool cannotPlayItem : 1;
			bool cannotPlayStadium : 1;
			bool cannotPlayTool : 1;
			bool cannotPlayAceSpec : 1;
			bool cannotPlayAbilityPokemonNotRocket : 1;
			bool cannotTrashToHandAbilityOrTrainers : 1;
		};
	};
	union { // この番
		unsigned long long turnState;
		struct {
			short playerDamageChange; // ワザのダメージ変化
			short playerDamageChangeEx;
			short playerDamageChangeMyFighting;
			signed char takePrizeCountChangeTerastalAttackKoActive;
			signed char takePrizeCountChangeNAttackKoActive;
		};
	};

	void turnStart(int activePlayerIndex) {
		if (playerIndex == activePlayerIndex) {
			thisTurn = nextTurn;
			nextTurn.value = 0;
		}
	}

	void turnEnd() {
		turnState = 0;
		thisTurn.value = 0;
	}

	void clearActiveState() {
		activeState = 0;
	}

	bool isPoisoned() const {
		return poisonDamageCounter > 0;
	}

	void clearPoison() {
		poisonDamageCounter = 0;
	}

	bool isBurned() const {
		return burned;
	}

	bool isSpecialCondition() const {
		return badStatus != BadStatusType::None || isPoisoned() || burned;
	}

	void clearSpecialCondition() {
		badStatus = BadStatusType::None;
		clearPoison();
		burned = false;
	}

	CardRef getActive() const {
		return active.at(0);
	}

	ByteFixedList<RefPosition, BENCH_SIZE_MAX + 1> getInPlayPokemon() const {
		ByteFixedList<RefPosition, BENCH_SIZE_MAX + 1> result;
		for (int i : range(active)) {
			result.push_back({ active[i], AreaType::Active, i });
		}
		for (int i : range(bench)) {
			result.push_back({ bench[i], AreaType::Bench, i });
		}
		return result;
	}
};
