// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "CardMove.h"

inline void SelectPokemonEnergyLoop(State& state, int playerIndex);
inline void SelectPrize(State& state, int playerIndex, int count, int coinPrizePlus);


inline void SetSelectedAttachedCard(State& state, bool isEnergy) {
	state.targetList.clear();
	for (int i : state.selected) {
		const SelectOption& o = state.options[i];
		CardPosition pos = o.getCardPosition();
		CardRef pokemonRef = state.getCardRef(pos.area, pos.areaIndex, pos.playerIndex);
		int moveCounter = state.getCard(pokemonRef).moveCounter;
		int index = 0;
		PlayerState& ps = state.players[pos.playerIndex];
		CardList& list = (isEnergy ? ps.energy : ps.tool);
		for (CardRef ref : list) {
			if (state.getCard(ref).attachMoveCounter == moveCounter) {
				if (index == o.param3) {
					state.targetList.push_back(state.makeAreaRef(ref));
					break;
				} else {
					index++;
				}
			}
		}
	}
	assert(state.targetList.size() == state.selected.size());
	state.clearSelect();
}

inline void SelectedCoinSingle(State& state, int playerIndex) {
	bool head = state.selectedYes();
	state.clearSelect();
	if (head) {
		state.coinHeadCount++;
	}
	LogCoin(state, playerIndex, head);
}

// コインの回数をリセットしないので、単体では呼ばない
inline void SelectCoinSingle(State& state, int playerIndex) {
	if (state.game->config.manualCoin) {
		state.setSelect(SelectType::YesNo, SelectContext::CoinHead, playerIndex);
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedCoinSingle, playerIndex);
	} else {
		auto r = (state.game->config.deviceRand ? std::random_device()() : state.game->rng());
		bool head = (r % 2 == 0);
		if (head) {
			state.coinHeadCount++;
		}
		LogCoin(state, playerIndex, head);
	}
}

inline void SelectCoin(State& state, int playerIndex, int count) {
	state.coinHeadCount = 0;
	state.pushFunction(SelectCoinSingle, playerIndex);
	state.setLastFunctionCallCount(count);
}

inline void SelectedCoinUntilTail(State& state, int playerIndex) {
	bool head = state.selectedYes();
	state.clearSelect();
	LogCoin(state, playerIndex, head);
	if (head) {
		state.coinHeadCount++;
		if (state.coinHeadCount >= 10'000'000) {
			return;
		}
		state.setSelect(SelectType::YesNo, SelectContext::CoinHead, playerIndex);
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedCoinUntilTail, playerIndex);
	}
}

inline void SelectCoinUntilTail(State& state, int playerIndex) {
	state.coinHeadCount = 0;
	if (state.game->config.manualCoin) {
		state.setSelect(SelectType::YesNo, SelectContext::CoinHead, playerIndex);
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedCoinUntilTail, playerIndex);
	} else {
		while (true) {
			auto r = (state.game->config.deviceRand ? std::random_device()() : state.game->rng());
			bool head = (r % 2 == 0);
			LogCoin(state, playerIndex, head);
			if (head) {
				state.coinHeadCount++;
			} else {
				break;
			}
		}
	}
}

inline void SelectedSwitchPokemon(State& state) {
	CardPosition pos = state.firstSelected().getCardPosition();
	state.clearSelect();

	SwitchPokemon(state, pos.playerIndex, pos.areaIndex);
}

inline void SelectSwitchPokemon(State& state, int switchPlayer, int selectPlayer) {
	const PlayerState& ps = state.players[switchPlayer];
	if (ps.bench.empty()) {
		return;
	}

	state.setSelect(SelectType::Card, SelectContext::Switch, selectPlayer);
	for (int i : range(ps.bench)) {
		AddOptionCard(state, AreaType::Bench, i, switchPlayer);
	}
	state.pushFunction(SelectedSwitchPokemon);
}

inline void SelectActivePokemon(State& state, int selectPlayer) {
	const PlayerState& ps = state.players[selectPlayer];
	if (ps.bench.empty()) {
		return;
	}

	state.setSelect(SelectType::Card, SelectContext::ToActive, selectPlayer);
	for (int i : range(ps.bench)) {
		AddOptionCard(state, AreaType::Bench, i, selectPlayer);
	}
	state.pushFunction(SelectedSwitchPokemon);
}

inline void SelectedBenchMaxTrash(State& state) {
	state.setSelectedCardTarget();
	for (const AreaRef& ref : state.targetList) {
		const Card& card = state.getCard(ref.card);
		MoveRefCard(state, ref.card, card, AreaType::Trash);
	}
}

inline void SelectBenchMaxTrash(State& state, int playerIndex) {
	const PlayerState& ps = state.players.at(playerIndex);
	int trashCount = -state.remainingBench(playerIndex);
	state.setSelect(SelectType::Card, SelectContext::Discard, playerIndex, trashCount, trashCount);
	for (int i : range(ps.bench)) {
		AddOptionCard(state, AreaType::Bench, i, playerIndex);
	}

	state.pushFunction(SelectedBenchMaxTrash);
}

inline void SelectedToolMaxTrash(State& state) {
	SetSelectedAttachedCard(state, false);
	for (const AreaRef& ref : state.targetList) {
		const Card& card = state.getCard(ref.card);
		MoveRefCard(state, ref.card, card, AreaType::Trash);
	}
	RefreshEffect(state);
}

inline void SelectToolMaxTrash(State& state, int playerIndex, int cardIndex) {
	const PlayerState& ps = state.players.at(playerIndex);
	CardRef ref(cardIndex);
	Card& card = state.getCard(ref);
	int remain = state.remainingToolCapacity(card);
	if (remain < 0) {
		AttachedCardList list = state.getAttachedList(ps, ps.tool);
		CardPosition pos = state.currentCardPosition(ref);
		int trashCount = -remain;
		state.setSelect(SelectType::AttachedCard, SelectContext::DiscardToolCard, playerIndex, trashCount, trashCount);
		for (const AttachedCard& ac : list) {
			if (ref == ac.targetRef) {
				AddOptionToolCard(state, pos.area, pos.areaIndex, playerIndex, ac.attachedIndex);
			}
		}

		state.pushFunction(SelectedToolMaxTrash);
	}
}

inline void SetSelectedPokemonEnergy(State& state) {
	SetSelectedAttachedCard(state, true);
}

inline void SelectedPokemonEnergy(State& state) {
	state.energyCost = 0;
	state.remainEnergyCost = 0;
	state.selectedEnergyCardCount = 0;
	state.selectingEnergyPokemonRef = {};
	state.targetList.clear();
}

inline void AfterEnergyDiscard(State& state, CardRef energyRef, const Card& energyCard, int attachMoveCounter) {
	const CardMaster& master = energyCard.getMaster();
	if (master.cardId == BOOMERANG_ENERGY || master.cardId == NITRO_FIRE_ENERGY) {
		if (state.onAttack() && state.onEffect()) {
			const Card& attacker = state.getCard(state.attacker);
			const Card& effectCard = state.getCard(state.getEffectCard().card);
			if (attachMoveCounter == attacker.moveCounter && attachMoveCounter == effectCard.moveCounter) {
				if (attacker.area == AreaType::Active || attacker.area == AreaType::Bench) {
					if (master.cardId == NITRO_FIRE_ENERGY) {
						if (!ContainsEnergyType(state.getEnergyType(attacker), EnergyType::Fire)) {
							return;
						}
					}
					TriggeredAbility ta = {};
					ta.trigger.type = TriggerType::Attach;
					ta.trigger.subject = state.makeAreaRef(state.attacker);
					ta.activateInfo = ActivateAbilityInfo(master.ability->skillId, state.makeAreaRef(energyRef), energyCard.playerIndex);
					state.temporaryTriggerStack.push_back(ta);
				}
			}
		}
	}
}

inline void SelectedPokemonEnergyLoop(State& state, int playerIndex) {
	SelectContext context = state.selectContext;
	if (state.selected.size() > 0) {
		SelectOption o = state.firstSelected();
		state.clearSelect();
		RefPosition rp = state.attachedEnergyPositionFromOption(o);
		const Card& energyCard = state.getCard(rp.ref);
		state.selectedList.push_back(rp.ref);
		if (!state.onEffect() || !state.isPreventEffect(energyCard)) {
			if (state.selectContext == SelectContext::DiscardEnergy) {
				int moveCounter = energyCard.attachMoveCounter;
				MoveCard(state, energyCard.playerIndex, AreaType::Energy, rp.index, AreaType::Trash);
				AfterEnergyDiscard(state, rp.ref, energyCard, moveCounter);
			} else if (state.selectContext == SelectContext::ToDeckEnergy) {
				MoveCard(state, energyCard.playerIndex, AreaType::Energy, rp.index, AreaType::Deck);
			} else if (state.selectContext == SelectContext::ToHandEnergy) {
				MoveCard(state, energyCard.playerIndex, AreaType::Energy, rp.index, AreaType::Hand);
			} else if (state.selectContext == SelectContext::SwitchEnergy) {
				// empty
			} else {
				Exception("not implemented");
			}
		}
		state.remainEnergyCost -= o.param4;
		state.selectedEnergyCardCount++;
		if (state.energyCost > state.selectedEnergyCardCount) {
			state.selectContext = context;
			state.pushFunction(SelectPokemonEnergyLoop, playerIndex);
		}
	} else {
		state.clearSelect();
	}
}

inline void SelectPokemonEnergyLoop(State& state, int playerIndex) {
	int selectMin = (state.remainEnergyCost <= 0 ? 0 : 1);
	if (state.onEffect() && state.getEffect().energyMaxSelect) {
		if (state.remainEnergyCost < state.energyCost) {
			selectMin = 0;
		} else if (state.onAttackEffect()) {
			selectMin = 0;
		}
	}
	state.setSelect(SelectType::Energy, state.selectContext, playerIndex, selectMin, 1);
	for (const AreaRef& tref : state.targetList) {
		if (!Contains(state.selectedList, tref.card)) {
			const Card& energyCard = state.getCard(tref.card);
			RefPosition rp = state.attachedCardPosition(energyCard);
			EnergyInfo ei = state.getEnergyInfo(energyCard, rp.ref);
			int index = state.energyIndex(tref.card, rp.ref);
			if (state.onEffect() && state.isPreventEffect(energyCard)) {
				continue;
			}
			AddOptionEnergy(state, rp.area, rp.index, energyCard.playerIndex, index, ei.count);
		}
	}

	if (state.options.empty()) {
		state.clearSelect();
	} else {
		state.pushFunction(SelectedPokemonEnergyLoop, playerIndex);
	}
}

inline void SelectPokemonEnergy(State& state, int playerIndex, const AttachedCardList& energies, int cost, SelectContext context) {
	state.selectContext = context;
	state.energyCost = cost;
	state.remainEnergyCost = cost;
	state.selectedEnergyCardCount = 0;
	state.targetList.clear();
	state.selectedList.clear();
	state.pushFunction(SelectedPokemonEnergy);
	if (energies.size() > 0) {
		state.changed = true;
		int energySum = 0;
		for (const AttachedCard& c : energies) {
			state.targetList.push_back(state.makeAreaRef(c.ref));

			EnergyInfo ei = state.getEnergyInfo(state.getCard(c.ref), c.targetRef);
			energySum += ei.count;
		}
		if (state.energyCost > energySum) {
			state.energyCost = energySum;
			state.remainEnergyCost = energySum;
		}
		SelectPokemonEnergyLoop(state, playerIndex);
	}
}

inline void SelectTrashSinglePokemonEnergy(State& state, int playerIndex, CardRef targetRef, int cost) {
	AttachedCardList energies;
	int moveCounter = state.getCard(targetRef).moveCounter;
	for (CardRef ref : state.players[playerIndex].energy) {
		if (state.getCard(ref).attachMoveCounter == moveCounter) {
			energies.push_back({ ref, targetRef, energies.size() });
		}
	}
	SelectPokemonEnergy(state, playerIndex, energies, cost, SelectContext::DiscardEnergy);
}

inline void CoinLuckyBonus(State& state, int playerIndex) {
	if (state.coinHeadCount > 0) {
		state.pushFunction(SelectPrize, playerIndex, 1, 0);
	}
}

inline void SelectedPrizeToHand(State& state, int cardIndex) {
	CardRef ref(cardIndex);
	CardPosition pos = state.currentCardPosition(ref);
	MoveCard(state, pos.playerIndex, pos.area, pos.areaIndex, AreaType::Hand, 1);
}

inline void SelectedLuckyBonus(State& state, int cardIndex) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	CardRef ref(cardIndex);
	const Card& card = state.getCard(ref);
	int index = state.currentAreaIndex(card.playerIndex, AreaType::Temporary, ref);
	if (isYes) {
		MoveCard(state, card.playerIndex, AreaType::Temporary, index, AreaType::Bench, 0);
		state.pushFunction(CoinLuckyBonus, card.playerIndex);
		SelectCoin(state, card.playerIndex, 1);
	} else {
		MoveCard(state, card.playerIndex, AreaType::Temporary, index, AreaType::Hand, 1);
	}
}

inline void SelectLuckyBonus(State& state, int cardIndex) {
	CardRef ref(cardIndex);
	const Card& card = state.getCard(ref);
	if (state.remainingBench(card.playerIndex) <= 0) {
		int index = state.currentAreaIndex(card.playerIndex, AreaType::Temporary, ref);
		MoveCard(state, card.playerIndex, AreaType::Temporary, index, AreaType::Hand, 1);
	} else {
		state.contextCard = ref;
		state.setSelect(SelectType::YesNo, SelectContext::Activate, card.playerIndex);
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedLuckyBonus, cardIndex);
	}
}

inline void SelectedPrizeTarget(State& state) {
	FixedList<int, 6> rest;
	for (int i = (int)state.targetList.size() - 1; i >= 0; i--) {
		const AreaRef& ref = state.targetList.at(i);
		const Card& card = state.getCard(ref.card);
		const Skill* ability = state.getAbility(card, card.getMaster());
		if (card.reverse && ability != nullptr && ability->luckyBonus && state.remainingBench(card.playerIndex) > 0 && state.isPlayerTurn() && state.activePlayerIndex() == card.playerIndex) {
			int index = state.currentAreaIndex(ref.card, card);
			PlayerState& ps = state.players[card.playerIndex];
			ps.prize.remove(index);
			ps.temporary.push_back(ref.card);
			state.pushFunction(SelectLuckyBonus, ref.card.cardIndex);
		} else {
			rest.push_back(i);
		}
	}
	for (int i = rest.size() - 1; i >= 0; i--) {
		const AreaRef& ref = state.targetList.at(rest[i]);
		SelectedPrizeToHand(state, ref.card.cardIndex);
	}
}

inline void SelectedPrize(State& state) {
	state.setSelectedCardTarget();
	SelectedPrizeTarget(state);
}

inline void SelectPrize2(State& state, int playerIndex, int count) {
	const PlayerState& ps = state.players.at(playerIndex);
	int selectCount = std::min(count, ps.prize.size());
	state.setSelect(SelectType::Card, SelectContext::ToHand, playerIndex, selectCount, selectCount);
	for (int i : range(ps.prize)) {
		AddOptionCard(state, AreaType::Prize, i, playerIndex);
	}
	state.pushFunction(SelectedPrize);
}

inline void CoinPrize(State& state, int playerIndex, int count) {
	if (state.coinHeadCount > 0) {
		count++;
	}
	SelectPrize2(state, playerIndex, count);
}

inline void SelectPrize(State& state, int playerIndex, int count, int coinPrizePlus) {
	const PlayerState& ps = state.players.at(playerIndex);
	if (ps.prize.empty()) {
		return;
	}

	if (coinPrizePlus) {
		state.pushFunction(CoinPrize, playerIndex, count);
		SelectCoin(state, playerIndex, 1);
	} else {
		SelectPrize2(state, playerIndex, count);
	}
}

