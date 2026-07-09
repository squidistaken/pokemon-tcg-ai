// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Card.h"
#include "JsonBuilder.h"

struct Log {
	LogType logType;
	FixedList<int, 7> param;

	Log() = default;
	explicit Log(LogType logType) : logType(logType) {}

	void add(int val) {
		param.push_back(val);
	}

	void add(AreaType area) {
		param.push_back((int)area);
	}
};

struct NpcOption {
	int isNpc;
};

struct GameConfig {
	std::array<Deck, 2> decks;
	std::array<std::string, 2> deckNames;
	uint32_t seed;
	int timeLimit; // 時間制限(秒)、0なら制限無し
	bool recordLog;
	bool manualCoin;
	bool sendDeck;
	bool deviceRand;
};

struct AttackEnergy {
	const Attack* attack;
	int insufficientEnergy; // 足りないエネルギー数
	int srcAttackId = 0; // このワザとして扱う系の元のワザ
};

struct CardEffect {
	CardRef ref;
	signed char priority;
	int skillOrder;
	int moveCounter;
};

struct Game {
	bool selecting = false;
	int actionCount = 0;
	GameConfig config = {};

	std::mt19937 rng;

	std::array<double, 2> remainingTime; // 秒単位
	std::chrono::high_resolution_clock::time_point startTime;

	// 一時利用
	std::vector<EnergyType> energyList;
	std::vector<EnergyType> energyList2;
	std::vector<CardRef> cardList;
	std::vector<CardEffect> cardEffectList;
	std::vector<AreaRef> targetList;
	std::vector<AttackEnergy> attackEnergyList;
	std::array<std::unordered_set<int>, 2> abilitySet;

	std::function<void(void)> pushResponseFunc; // クライアントへ送るレスポンスを追加

	JsonBuilder jsonBuilder;

	void init(const GameConfig& config) {
		this->config = config;
		if (this->config.seed == 0) {
			this->config.seed = std::random_device()();
		}
		rng = std::mt19937(this->config.seed);
	}

	bool isAddLog() const {
		return !selecting && config.recordLog;
	}

	void pushResponse() {
		if (pushResponseFunc) {
			pushResponseFunc();
		}
	}

	void timerStart() {
		if (config.timeLimit > 0) {
			startTime = std::chrono::high_resolution_clock::now();
		}
	}

	bool timerStop(int playerIndex) {
		if (config.timeLimit > 0) {
			auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
			std::chrono::duration<double> elapsedSeconds = elapsed;
			double& time = remainingTime.at(playerIndex);
			time -= elapsedSeconds.count();
			return time <= 0;
		} else {
			return false;
		}
	}
};
