// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "State.h"

inline bool IsActiveNull(const State& state, int playerIndex) {
	auto& active = state.players.at(playerIndex).active;
	if (active.size() > 0) {
		return active[0].isNull();
	}
	return false;
}

struct SearchStartConfig {
	bool manualCoin = false;
	std::vector<int> myDeck;
	std::vector<int> myPrize;
	std::vector<int> enemyDeck;
	std::vector<int> enemyPrize;
	std::vector<int> enemyHand;
	std::vector<int> enemyActive;
};

struct SearchState {
	State* state;
};

struct SearchInfo {
	State* state;
	int searchId;
	int errorCode;

	static SearchInfo error(int errorCode) {
		SearchInfo si = {};
		si.errorCode = errorCode;
		return si;
	}
};

class Search {
public:

	using StateArray = std::array<State, 128>;

	~Search() {
		for (StateArray* sa : stateMemory) {
			delete sa;
		}
	}

	void free() {
		for (StateArray* sa : stateMemory) {
			delete sa;
		}
		searchList.clear();
		stateMemory.clear();
		freeStateList.clear();
	}

	void clear() {
		searchList.clear();
		freeStateList.clear();
		for (StateArray* sa : stateMemory) {
			for (State& state : *sa) {
				freeStateList.push_back(&state);
			}
		}
	}

	// 0以外を返したらエラー
	int clearSingle(long long id) {
		if (std::ssize(searchList) <= id) {
			return 1;
		}
		SearchState& ss = searchList[id];
		if (ss.state == nullptr) {
			return 2;
		}
		freeStateList.push_back(ss.state);
		ss.state = nullptr;
		return 0;
	}

	SearchInfo start(const SearchStartConfig& config, const State& state) {
		for (int id : config.enemyActive) {
			const CardMaster& master = CardTable.at(id);
			if (master.cardType != CardType::Pokemon && !master.toBattleFieldOnlySetup && !master.toActiveOnlySetup) {
				return SearchInfo::error(2);
			}
		}
		if (IsActiveNull(state, 1 - state.selectPlayer) && config.enemyActive.size() == 0) {
			return SearchInfo::error(98);
		}

		SearchState ss = alloc(state);
		State& s = *ss.state;
		
		int index = 3;
		auto set = [&](const std::vector<int>& ids, CardList& out, int playerIndex, AreaType area) {
			for (int i : range(ids)) {
				int id = ids[i];
				while (index < std::ssize(s.allCard)) {
					Card& card = s.allCard.at(index);
					if (card.cardId == 0) {
						out.at(i) = CardRef(index);
						card = {};
						card.init(id, s.moveCounter++, playerIndex);
						card.area = area;
						break;
					}
					index++;
				}
			}
		};

		auto setActive = [&]() {
			int playerIndex = 1 - s.selectPlayer;
			auto& active = s.players[playerIndex].active;
			if (active.size() != 0 && active[0].isNull()) {
				int id = config.enemyActive.at(0);
				while (index < std::ssize(s.allCard)) {
					Card& card = s.allCard.at(index);
					if (card.cardId == 0) {
						active[0] = CardRef(index);
						card = {};
						card.init(id, s.moveCounter++, playerIndex);
						card.area = AreaType::Active;
						card.reverse = true;
						break;
					}
					index++;
				}
			}
		};

		if (s.selectPlayer == 0) {
			if (!state.selectDeck) {
				set(config.myDeck, s.players[0].deck, 0, AreaType::Deck);
			}
			set(config.myPrize, s.players[0].prize, 0, AreaType::Prize);
			set(config.enemyDeck, s.players[1].deck, 1, AreaType::Deck);
			set(config.enemyPrize, s.players[1].prize, 1, AreaType::Prize);
			set(config.enemyHand, s.players[1].hand, 1, AreaType::Hand);
			setActive();
		} else {
			set(config.enemyDeck, s.players[0].deck, 0, AreaType::Deck);
			set(config.enemyPrize, s.players[0].prize, 0, AreaType::Prize);
			set(config.enemyHand, s.players[0].hand, 0, AreaType::Hand);
			setActive();
			if (!state.selectDeck) {
				set(config.myDeck, s.players[1].deck, 1, AreaType::Deck);
			}
			set(config.myPrize, s.players[1].prize, 1, AreaType::Prize);
		}
		s.game->config.manualCoin = config.manualCoin;

		int lastId = lastSearchId();
		return { ss.state, lastId, 0 };
	}

	SearchInfo step(long long id, const std::vector<int>& selected) {
		if (std::ssize(searchList) <= id) {
			return SearchInfo::error(1);
		}
		const SearchState& src = searchList[id];
		if (src.state == nullptr) {
			return SearchInfo::error(2);
		}
		if (src.state->isFinish()) {
			return SearchInfo::error(3);
		}

		SearchState ss = alloc(*src.state);
		State& state = *ss.state;
		state.selected = selected;
		int error = state.checkPlayerSelect();
		if (error) {
			return SearchInfo::error(error);
		}

		state.step();
		while (!state.isFinish() && state.selectMax == 0) {
			state.selected.clear();
			state.step();
		}

		int lastId = lastSearchId();
		return { ss.state, lastId, 0 };
	}

	SearchInfo shuffle(long long id, int playerIndex) {
		if (std::ssize(searchList) <= id) {
			return SearchInfo::error(1);
		}
		const SearchState& src = searchList[id];
		if (src.state == nullptr) {
			return SearchInfo::error(2);
		}

		SearchState ss = alloc(*src.state);
		auto& deck = ss.state->players.at(playerIndex).deck;
		std::shuffle(deck.begin(), deck.end(), ss.state->game->rng);

		int lastId = lastSearchId();
		return { ss.state, lastId, 0 };
	}

	int lastSearchId() const {
		return (int)searchList.size() - 1;
	}

	const State& lastState() const {
		return *searchList.back().state;
	}

private:

	std::vector<SearchState> searchList;
	std::vector<State*> freeStateList;
	std::vector<StateArray*> stateMemory;

	SearchState alloc(const State& src) {
		SearchState& ss = searchList.emplace_back();
		if (freeStateList.empty()) {
			StateArray* sa = new StateArray();
			stateMemory.push_back(sa);
			for (State& state : *sa) {
				freeStateList.push_back(&state);
			}
		}
		State* state = freeStateList.back();
		freeStateList.pop_back();
		*state = src;
		ss.state = state;
		return ss;
	}
};
