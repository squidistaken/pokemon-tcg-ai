// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "SelectProc.h"

inline void DamageCounterSwitchAny(State& state, int targetPlayerIndex);

inline void SelectCard(State& state) {
	bool clear = true;
	if (state.onEffect()) {
		const Effect& effect = state.getEffect();
		if (effect.loopCount > 0) {
			if (state.effectState.selectedListIndex > 0) {
				clear = false;
			}
		}
		if (effect.notClearSelectedList) {
			clear = false;
		}
	}
	if (clear) {
		state.selectedList.clear();
	}
	for (const AreaRef& ref : state.targetList) {
		const Card& card = state.getCard(ref.card);
		if (state.isPreventEffect(card)) {
			continue;
		}
		state.selectedList.push_back(ref.card);
	}
}

inline void SelectedDamageCounterAny(State& state) {
	for (int i : state.selected) {
		const SelectOption& o = state.options[i];
		CardRef ref = state.getCardRef(o);
		Card& card = state.getCard(ref);
		int damage = 10;
		if (state.isPreventEffect(card) || state.isPreventDamageCounter(card)) {
			damage = 0;
		}
		if (damage > 0) {
			state.changed = true;
			card.damage += damage;
		}
		LogHpChange(state, card.playerIndex, ref, -damage, true);
	}
	state.clearSelect();
	state.remainDamageCounter--;
}

inline void SelectDamageCounterAny(State& state) {
	state.setSelect(SelectType::Card, SelectContext::DamageCounterAny, state.effectPlayerIndex());
	for (const AreaRef& ref : state.targetList) {
		Card* card = state.checkGetCard(ref);
		if (card != nullptr) {
			AddOptionCard(state, card->area, state.currentAreaIndex(ref.card, *card), card->playerIndex);
		}
	}
	state.pushFunction(SelectedDamageCounterAny);
}

inline void SelectedRemoveDamageCounter(State& state) {
	CardRef contextCard = state.contextCard;
	assert(!contextCard.isNull());
	int count = state.selectedNumber();
	state.clearSelect();
	if (state.isPreventEffect(state.getCard(contextCard))) {
		count = 0;
	}
	Heal(state, contextCard, count * 10, false);
	state.removedDamageCounter = count;
}

inline void SelectedAttackDamageChangePutDamageCounter(State& state) {
	int count = state.selectedNumber();
	state.clearSelect();

	CardRef contextCard = state.getEffectCard().card;
	Card& card = state.getCard(contextCard);
	AddDamage(state, contextCard, card, count * 10, false, contextCard, true);
	state.attackDamageChange = state.getEffect().values[1] * count;
}


inline void SelectedAddDamageCounterSwitchAny(State& state, int targetPlayerIndex) {
	SelectOption option = state.firstSelected();
	state.clearSelect();
	CardRef ref = state.getCardRef(option);
	Card& card = state.getCard(ref);
	int damage = 10;
	if (state.isPreventEffect(card) || state.isPreventDamageCounter(card)) {
		damage = 0;
	}
	card.damage += damage;
	LogHpChange(state, card.playerIndex, ref, -damage, true);

	DamageCounterSwitchAny(state, targetPlayerIndex);
}

inline void SelectedRemoveDamageCounterSwitchAny(State& state, int targetPlayerIndex) {
	if (state.selected.empty()) {
		state.clearSelect();
		return;
	}
	SelectOption option = state.firstSelected();
	state.clearSelect();
	CardRef removeRef = state.getCardRef(option);
	Heal(state, removeRef, 10, false);

	state.setSelect(SelectType::Card, SelectContext::DamageCounter, state.effectPlayerIndex());
	for (RefPosition rp : state.players[targetPlayerIndex].getInPlayPokemon()) {
		if (rp.ref != removeRef) {
			AddOptionCard(state, rp.area, rp.index, targetPlayerIndex);
		}
	}
	state.pushFunction(SelectedAddDamageCounterSwitchAny, targetPlayerIndex);
}

inline void DamageCounterSwitchAny(State& state, int targetPlayerIndex) {
	state.effectActionCount++;
	if (state.effectActionCount > 10000) {
		return;
	}
	PlayerState& ps = state.players[targetPlayerIndex];
	if (ps.bench.empty()) {
		return;
	}

	state.setSelect(SelectType::Card, SelectContext::RemoveDamageCounter, state.effectPlayerIndex(), 0, 1);
	for (RefPosition rp : ps.getInPlayPokemon()) {
		Card& card = state.getCard(rp.ref);
		if (!state.isPreventEffect(card) && !card.cannotMoveDamageCounter && card.damage > 0) {
			AddOptionCard(state, rp.area, rp.index, targetPlayerIndex);
		}
	}
	if (state.options.empty()) {
		state.clearSelect();
	} else {
		state.pushFunction(SelectedRemoveDamageCounterSwitchAny, targetPlayerIndex);
	}
}


inline void AfterAttackDamageChangeCoin(State& state) {
	state.attackDamageChange = state.getSecondEffectValue() * state.coinHeadCount;
}

inline void AfterAttackDamageChangeCoinUntilTail(State& state) {
	state.attackDamageChange = state.getEffectValue() * state.coinHeadCount;
}

inline void AfterAttackDamageChangeTargetCountEnemyCoin(State& state) {
	state.attackDamageChange = state.getEffectValue() * ((int)state.targetList.size() - state.coinHeadCount);
}

inline void AfterBreakIfCoinHead(State& state) {
	if (state.coinHeadCount != 0) {
		state.breakEffect();
	}
}

inline void AfterBreakIfCoinTail(State& state) {
	if (state.coinHeadCount == 0) {
		state.breakEffect();
	}
}

inline void AfterBreakIfCoinTailMulti(State& state, int count) {
	if (state.coinHeadCount < count) {
		state.breakEffect();
	}
}

inline void AfterSkipIfCoinTail(State& state) {
	if (state.coinHeadCount == 0) {
		state.effectJump = state.getEffectValue();
	}
}

inline void SelectedActivate(State& state) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	if (!isYes) {
		state.breakEffect();
	}
}

inline void SelectedFirstEffect(State& state) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	if (!isYes) {
		state.effectJump = state.getEffectValue();
	}
}

inline void SelectedBurn(State& state) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	if (isYes) {
		EffectBurn(state, 1 - state.effectPlayerIndex());
	} else {
		EffectConfuse(state, 1 - state.effectPlayerIndex());
	}
}

inline void SelectedSpecialCondition(State& state) {
	SelectOption o = state.firstSelected();
	state.clearSelect();
	SelectSpecialConditionType type = (SelectSpecialConditionType)o.param0;
	int playerIndex = 1 - state.effectPlayerIndex();
	switch (type)
	{
	case SelectSpecialConditionType::Poison:
		EffectPoison(state, playerIndex);
		break;
	case SelectSpecialConditionType::Burn:
		EffectBurn(state, playerIndex);
		break;
	case SelectSpecialConditionType::Sleep:
		EffectSleep(state, playerIndex);
		break;
	case SelectSpecialConditionType::Paralyze:
		EffectParalyze(state, playerIndex);
		break;
	case SelectSpecialConditionType::Confuse:
		EffectConfuse(state, playerIndex);
		break;
	default:
		assert(false);
		break;
	}
}

inline void SelectedRecoverSpecialCondition(State& state, int playerIndex) {
	SelectOption o = state.firstSelected();
	state.clearSelect();
	SelectSpecialConditionType type = (SelectSpecialConditionType)o.param0;
	PlayerState& ps = state.players.at(playerIndex);
	switch (type)
	{
	case SelectSpecialConditionType::Poison:
		if (ps.isPoisoned()) {
			ps.clearPoison();
			LogPoisoned(state, playerIndex, true, ps.getActive());
		}
		break;
	case SelectSpecialConditionType::Burn:
		if (ps.isBurned()) {
			ps.burned = false;
			LogBurned(state, playerIndex, true, ps.getActive());
		}
		break;
	case SelectSpecialConditionType::Sleep:
	case SelectSpecialConditionType::Paralyze:
	case SelectSpecialConditionType::Confuse:
		ClearSleepParalyzeConfuse(state, playerIndex);
		break;
	default:
		assert(false);
		break;
	}
}


inline void TransformProc(State& state, bool toDeck) {
	if (state.selectedList.empty()) {
		return;
	}
	for (const AreaRef& preRef : state.targetList) {
		Card* c = state.checkGetCard(preRef);
		if (c != nullptr) {
			Card& preCard = *c;
			if (preCard.area == AreaType::Active || preCard.area == AreaType::Bench) {
				state.changed = true;
				CardRef ref = state.selectedList[0];
				Card& afterCard = state.getCard(ref);
				assert(preCard.playerIndex == afterCard.playerIndex);

				LogChange(state, preCard.playerIndex, preRef.card, ref);

				PlayerState& ps = state.players[preCard.playerIndex];
				auto prePs = ps.activeState;
				int areaIndex = state.currentAreaIndex(ref, afterCard);
				CardRef afterRef = state.removeCardRef(afterCard.area, areaIndex, afterCard.playerIndex);
				assert(afterRef == ref);
				int ownerAreaIndex = state.currentAreaIndex(preRef.card, preCard);
				CardRef& inPlayRef = (preCard.area == AreaType::Active ? ps.active.at(0) : ps.bench.at(ownerAreaIndex));
				assert(preRef.card == inPlayRef);
				afterCard.copyState(preCard);

				if (toDeck) {
					ps.deck.push_back(inPlayRef);
					state.cardMoved(inPlayRef, AreaType::Deck);
				} else {
					ps.trash.push_back(inPlayRef);
					state.cardMoved(inPlayRef, AreaType::Trash);
				}
				inPlayRef = afterRef;
				ps.activeState = prePs;

				for (TriggeredAbility& ta : state.triggerStack) {
					TriggerInfo& ti = ta.trigger;
					if (ti.subject.card == preRef.card) {
						ti.subject.card = afterRef;
					}
					if (ti.object.card == preRef.card) {
						ti.object.card = afterRef;
					}
				}
				break;
			}
		}
	}
}

inline void EffectAttackDamage2(State& state, int cardIndex, int damage) {
	CardRef ref(cardIndex);
	Card& card = state.getCard(ref);
	CardRef effectCardRef = state.getEffectCard().card;
	const Card& effectCard = state.getCard(effectCardRef);
	const Attack& attack = state.getAttack();
	AddDamage(state, ref, card, damage, true, effectCardRef, false, &attack);
	AfterDamage(state, ref, card, effectCardRef, effectCard);
}

inline void EffectAttackNoDamageCoin(State& state, int cardIndex, int damage) {
	if (state.coinHeadCount == 0 || state.getAttack().noTargetEffect) {
		EffectAttackDamage2(state, cardIndex, damage);
	}
	state.game->pushResponse();
}

inline void EffectAttackDamage(State& state, int cardIndex, int baseDamage) {
	CardRef ref(cardIndex);
	Card& card = state.getCard(ref);
	CardRef effectCardRef = state.getEffectCard().card;
	const Card& effectCard = state.getCard(effectCardRef);
	int damage = CalcDamage(state, baseDamage, ref, card, effectCardRef, effectCard, card.area == AreaType::Active, &state.getAttack());
	state.changed = true;
	if (card.noDamageCoin && damage > 0) {
		state.pushFunction(EffectAttackNoDamageCoin, cardIndex, damage);
		SelectCoin(state, card.playerIndex, 1);
	} else {
		EffectAttackDamage2(state, cardIndex, damage);
	}
}

inline void AfterAttackDamageCoin(State& state, int cardIndex) {
	if (state.coinHeadCount > 0) {
		EffectAttackDamage(state, cardIndex, state.getEffectValue());
	}
}

inline void AttackDamageCoin(State& state, int cardIndex) {
	state.pushFunction(AfterAttackDamageCoin, cardIndex);
	SelectCoin(state, state.effectPlayerIndex(), 1);
}

inline void SelectedDamageMultiAll(State& state) {
	int playerIndex = 1 - state.effectPlayerIndex();
	for (int i = (int)state.selectCounts.size() - 1; i >= 0; i--) {
		int count = state.selectCounts[i];
		if (count > 0) {
			int baseDamage = state.getSecondEffectValue() * count;
			AreaType area = (i == 0 ? AreaType::Active : AreaType::Bench);
			int areaIndex = (i == 0 ? 0 : i - 1);
			CardRef ref = state.getCardRef(area, areaIndex, playerIndex);
			state.pushFunction(EffectAttackDamage, ref.cardIndex, baseDamage);
		}
	}
}

inline void SelectedDamageMulti(State& state) {
	SelectOption option = state.firstSelected();
	state.clearSelect();
	CardPosition p = option.getCardPosition();
	if (p.area == AreaType::Active) {
		state.selectCounts[0]++;
	} else {
		state.selectCounts[1 + p.areaIndex]++;
	}
}

inline void SelectDamageMulti(State& state) {
	state.setSelect(SelectType::Card, SelectContext::Damage, state.effectPlayerIndex());
	for (const AreaRef& ref : state.targetList) {
		CardPosition pos = state.currentCardPosition(ref);
		if (pos.areaIndex >= 0) {
			AddOptionCard(state, pos.area, pos.areaIndex, pos.playerIndex);
		}
	}
	state.pushFunction(SelectedDamageMulti);
}

inline void SelectedMoreDevolve(State& state) {
	bool isYes = state.selectedYes();
	CardRef ref = state.contextCard;
	state.clearSelect();
	if (isYes) {
		RefreshEffect(state);
		DevolveProc(state, ref, state.getCard(ref), AreaType::Hand);
	}
}

inline void SelectedDisableAttack(State& state, int cardIndex) {
	int attackId = state.firstSelected().param0;
	state.clearSelect();
	CardRef ref(cardIndex);
	state.getCard(ref).nextTurn.cannotUseAttackId2 = attackId;
}

inline void AfterDeckToTrashCoinUntilTail(State& state) {
	int count = state.coinHeadCount;
	state.targetList.clear();
	for (int i : state.effectTargetPlayer(state.getEffect())) {
		PlayerState& ps = state.players[i];
		for (int _ : range(count)) {
			if (ps.deck.empty()) {
				break;
			}
			state.changed = true;
			CardRef ref = MoveCard(state, i, AreaType::Deck, ps.deck.size() - 1, AreaType::Trash);
			state.targetList.push_back(state.makeAreaRef(ref));
		}
	}
}

inline void EffectInstatnt(State& state) {
	const Effect& effect = state.getEffect();
	switch (effect.effectType) {
	case EffectType::NoEffect:
		break;
	case EffectType::SelectCard:
		SelectCard(state);
		break;
	case EffectType::ForEach:
		state.eachList.clear();
		for (const AreaRef& ref : state.targetList) {
			state.eachList.push_back(ref.card);
		}
		break;
	case EffectType::Ko:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				card->damage = state.getMaxHp(*card);
				card->ko = true;
				state.changed = true;
			}
		}
		break;
	case EffectType::ToHand:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Hand);
			}
		}
		break;
	case EffectType::PrizeToHand:
		SelectedPrizeTarget(state);
		break;
	case EffectType::ToHandReverse:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Hand, 1);
			}
		}
		break;
	case EffectType::ToHandWithAttach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Hand, 0, false, true);
			}
		}
		break;
	case EffectType::ToTrash:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				int attach = card->attachMoveCounter;
				MoveRefCard(state, ref.card, *card, AreaType::Trash);
				AfterEnergyDiscard(state, ref.card, *card, attach);
			}
		}
		break;
	case EffectType::ToDeck:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Deck, 0);
			}
		}
		break;
	case EffectType::ToDeckWithAttach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Deck, 0, false, true);
			}
		}
		break;
	case EffectType::ToDeckReverse:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Deck, 1);
			}
		}
		break;
	case EffectType::LookToDeckReverse:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Deck, state.effectPlayerIndex() + 3);
			}
		}
		break;
	case EffectType::ToDeckAndShuffle:
	{
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Deck);
			}
		}
		for (int i : state.effectTargetPlayer(effect)) {
			ShuffleDeck(state, i);
		}
		break;
	}
	case EffectType::ToDeckReverseAndShuffle:
	{
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Deck, 1);
			}
		}
		for (int i : state.effectTargetPlayer(effect)) {
			ShuffleDeck(state, i);
		}
		break;
	}
	case EffectType::ToDeckBottom:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::DeckBottom);
			}
		}
		break;
	case EffectType::ToDeckBottomReverse:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::DeckBottom, 1);
			}
		}
		break;
	case EffectType::ToDeckBottomClose:
		if (state.targetList.size() >= 2) {
			if (state.game->config.deviceRand) {
				std::shuffle(state.targetList.begin(), state.targetList.end(), std::random_device());
			} else {
				std::shuffle(state.targetList.begin(), state.targetList.end(), state.game->rng);
			}
		}
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::DeckBottom, 2);
			}
		}
		break;
	case EffectType::ToActiveAndTrashActive:
	{
		if (state.targetList.size() == 0) {
			break;
		}
		assert(state.targetList.size() == 1);
		const AreaRef& ref = state.targetList[0];
		Card* card = state.checkGetCard(ref);
		if (card != nullptr) {
			if (state.transformOnly(*card)) {
				break;
			}
			state.changed = true;
			PlayerState& ps = state.players.at(card->playerIndex);
			if (ps.active.size() > 0) {
				MoveCard(state, card->playerIndex, AreaType::Active, 0, AreaType::Trash);
			}
			MoveRefCard(state, ref.card, *card, AreaType::Active);
		}
		break;
	}
	case EffectType::ToBench:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				if (card->area != AreaType::Active && state.transformOnly(*card)) {
					continue;
				}
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Bench);
			}
		}
		break;
	case EffectType::ToPrize:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				assert(card->area == AreaType::Hand || card->area == AreaType::Looking);
				int openType;
				if (card->area == AreaType::Looking && state.lookingPlayer >= 3) {
					openType = 2;
				} else {
					openType = 1;
				}
				MoveRefCard(state, ref.card, *card, AreaType::Prize, openType);
				card->reverse = true;
			}
		}
		break;
	case EffectType::ToLooking:
		state.lookingPlayer = state.effectLookingPlayerIndex();
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				MoveRefCard(state, ref.card, *card, AreaType::Looking, state.lookingOpenType());
			}
		}
		break;
	case EffectType::ToPlayingFirst:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				MoveRefCard(state, ref.card, *card, AreaType::Playing);
				break;
			}
		}
		break;
	case EffectType::Switch:
	{
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				CardRef activeRef = state.players.at(card->playerIndex).getActive();
				CardRef effectedRef;
				const Effect& effect = state.getEffect();
				if (effect.effectTargetBench) {
					effectedRef = ref.card;
				} else if (effect.effectTargetActive || (state.effectPlayerIndex() != card->playerIndex && state.selectPlayer != card->playerIndex)) {
					effectedRef = activeRef; // バトルポケモンが受けた扱い
				} else {
					effectedRef = ref.card;
				}
				if (state.isPreventEffect(state.getCard(effectedRef))) {
					continue;
				}

				if (card->area != AreaType::Bench) {
					continue;
				}
				state.changed = true;
				int index = state.currentAreaIndex(ref.card, *card);
				SwitchPokemon(state, card->playerIndex, index);
				if (state.getEffect().setTargetSwitchBench) {
					state.targetList.clear();
					state.targetList.push_back(state.makeAreaRef(activeRef));
					break;
				}
			}
		}
		break;
	}
	case EffectType::SwitchDeck:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				auto& deck = state.players.at(card->playerIndex).deck;
				if (deck.empty()) {
					continue;
				}
				state.changed = true;
				Draw(state, card->playerIndex, 1);
				MoveRefCard(state, ref.card, *card, AreaType::Deck, 1);
			}
		}
		break;
	case EffectType::NotMove:
		SelectCard(state);
		break;
	case EffectType::LookDeck:
	{
		assert(state.looking.size() == 0);
		int count = state.getEffectValue();
		int playerIndex = state.getEffectTargetPlayerIndex();
		assert(playerIndex <= 1);

		PlayerState& ps = state.players[playerIndex];
		count = std::min(count, ps.deck.size());
		if (count <= 0) {
			return;
		}
		state.lookingPlayer = state.effectLookingPlayerIndex();
		for (int i = 0; i < count; i++) {
			state.changed = true;
			int index = ps.deck.size() - 1;
			MoveCard(state, playerIndex, AreaType::Deck, index, AreaType::Looking, state.lookingOpenType());
		}
		break;
	}
	case EffectType::LookDeckReverse:
	{
		assert(state.looking.size() == 0);
		int count = state.getEffectValue();
		int playerIndex = state.getEffectTargetPlayerIndex();
		assert(playerIndex <= 1);

		PlayerState& ps = state.players[playerIndex];
		count = std::min(count, ps.deck.size());
		if (count <= 0) {
			return;
		}
		state.lookingPlayer = state.effectLookingPlayerIndex() + 3; // 裏向き
		for (int i = 0; i < count; i++) {
			state.changed = true;
			int index = ps.deck.size() - 1;
			MoveCard(state, playerIndex, AreaType::Deck, index, AreaType::Looking, 2);
		}
		break;
	}
	case EffectType::LookDeckBottom:
	{
		int count = state.getEffectValue();
		int playerIndex = state.getEffectTargetPlayerIndex();

		PlayerState& ps = state.players[playerIndex];
		count = std::min(count, ps.deck.size());
		if (count <= 0) {
			return;
		}
		state.lookingPlayer = state.effectLookingPlayerIndex();
		for (int i = 0; i < count; i++) {
			state.changed = true;
			MoveCard(state, playerIndex, AreaType::Deck, 0, AreaType::Looking, state.lookingOpenType());
		}
		break;
	}
	case EffectType::LookAndReturn:
	{
		int openType = state.effectPlayerIndex() + 3;
		if (state.targetList.empty()) {
			return;
		} else {
			CardPosition pos = state.currentCardPosition(state.targetList[0].card);
			if (pos.area == AreaType::Hand) {
				openType = 0;
			}
		}
		for (const AreaRef& ref : state.targetList) {
			CardPosition pos = state.currentCardPosition(ref.card);
			LogMoveCard(state, pos.playerIndex, ref.card, pos.area, AreaType::Looking, openType);
		}
		if (state.getEffectValue() == 1) {
			openType = 0;
		}
		for (const AreaRef& ref : state.targetList) {
			CardPosition pos = state.currentCardPosition(ref.card);
			LogMoveCard(state, pos.playerIndex, ref.card, AreaType::Looking, pos.area, openType);
			if (state.getEffectValue() == 1) {
				state.getCard(ref.card).reverse = false;
			}
		}
		break;
	}
	case EffectType::DamageCounter:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int damage = state.getEffectValue() * 10 + state.effectState.damageChange;
				if (state.isPreventEffect(*card)) {
					damage = 0;
				}
				if (state.isPreventDamageCounter(*card)) {
					damage = 0;
				}
				AddDamage(state, ref.card, *card, damage, false, state.getEffectCard().card, true);
				if (damage > 0) {
					state.changed = true;
				}
			}
		}
		break;
	case EffectType::DamageCounterRemoved:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int damage = state.removedDamageCounter * 10;
				if (state.isPreventEffect(*card)) {
					damage = 0;
				}
				if (state.isPreventDamageCounter(*card)) {
					damage = 0;
				}
				AddDamage(state, ref.card, *card, damage, false, state.getEffectCard().card, true);
				if (damage > 0) {
					state.changed = true;
				}
			}
		}
		break;
	case EffectType::DamageCounterDamaged:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int damage = state.triggerInfo.value;
				if (state.isPreventEffect(*card)) {
					damage = 0;
				}
				if (state.isPreventDamageCounter(*card)) {
					damage = 0;
				}
				AddDamage(state, ref.card, *card, damage, false, state.getEffectCard().card, true);
				if (damage > 0) {
					state.changed = true;
				}
			}
		}
		break;
	case EffectType::DamageCounterAny:
	{
		if (state.targetList.empty()) {
			break;
		}
		int count = state.getEffectValue();
		if (count <= 0) {
			break;
		}
		state.remainDamageCounter = count;
		state.pushFunction(SelectDamageCounterAny);
		state.setLastFunctionCallCount(count);
		break;
	}
	case EffectType::DamageCounterDouble:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int damage = card->damage;
				if (state.isPreventEffect(*card)) {
					damage = 0;
				}
				if (state.isPreventDamageCounter(*card)) {
					damage = 0;
				}
				AddDamage(state, ref.card, *card, damage, false, state.getEffectCard().card, true);
				if (damage > 0) {
					state.changed = true;
				}
			}
		}
		break;
	case EffectType::DamageCounterHp:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int damage = state.getHp(*card) - state.getEffectValue();
				if (state.isPreventEffect(*card)) {
					damage = 0;
				}
				if (state.isPreventDamageCounter(*card)) {
					damage = 0;
				}
				AddDamage(state, ref.card, *card, damage, false, state.getEffectCard().card, true);
				if (damage > 0) {
					state.changed = true;
				}
			}
		}
		break;
	case EffectType::DamageCounterSwitchAny:
		state.effectActionCount = 0;
		for (int i : state.effectTargetPlayer(effect)) {
			DamageCounterSwitchAny(state, i);
		}
		break;
	case EffectType::DamageCounterTypeEnergyCountMe:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int count = state.typeEnergyCount(state.effectPlayerIndex(), state.getEffectCard().card, (EnergyType)state.getEffectValue());
				int damage = state.getSecondEffectValue() * count * 10;
				if (state.isPreventEffect(*card)) {
					damage = 0;
				}
				if (state.isPreventDamageCounter(*card)) {
					damage = 0;
				}
				AddDamage(state, ref.card, *card, damage, false, state.getEffectCard().card, true);
				if (damage > 0) {
					state.changed = true;
				}
			}
		}
		break;
	case EffectType::AttackDamage:
		assert(state.onAttack());
		for (int i = (int)state.targetList.size() - 1; i >= 0; i--) {
			const AreaRef& ref = state.targetList[i];
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int baseDamage = state.getEffectValue() + state.effectState.damageChange;
				state.pushFunction(EffectAttackDamage, ref.card.cardIndex, baseDamage);
			}
		}
		break;
	case EffectType::AttackDamageMulti: {
		assert(state.onAttack());
		state.selectCounts = {};
		int count = state.getEffectValue();
		state.pushFunction(SelectedDamageMultiAll);
		for (int i : range(count)) {
			state.pushFunction(SelectDamageMulti);
		}
		break;
	}
	case EffectType::AttackDamageCoin:
		assert(state.onAttack());
		for (int i = (int)state.targetList.size() - 1; i >= 0; i--) {
			const AreaRef& ref = state.targetList[i];
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.pushFunction(AttackDamageCoin, ref.card.cardIndex);
			}
		}
		break;
	case EffectType::RemoveDamageCounter:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				if (state.isPreventEffect(*card) || card->cannotMoveDamageCounter) {
					Heal(state, ref.card, 0, false);
					continue;
				}
				int maxCount = std::min(state.getEffectValue(), card->damage / 10);
				if (maxCount >= 2) {
					state.changed = true;
					state.contextCard = ref.card;
					state.setSelect(SelectType::Count, SelectContext::RemoveDamageCounterCount, state.effectPlayerIndex());
					for (int i = 1; i <= maxCount; i++) {
						AddOptionNumber(state, i);
					}
					state.pushFunction(SelectedRemoveDamageCounter);
				} else if(maxCount == 1) {
					state.changed = true;
					Heal(state, ref.card, 10, false);
					state.removedDamageCounter = 1;
				}
			}
		}
		if (!state.changed) {
			state.breakEffect();
		}
		break;
	case EffectType::RemoveDamageCounterAll:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				if (state.isPreventEffect(*card) || card->cannotMoveDamageCounter) {
					Heal(state, ref.card, 0, false);
					continue;
				}
				state.removedDamageCounter = card->damage / 10;
				if (state.removedDamageCounter > 0) {
					state.changed = true;
					Heal(state, ref.card, card->damage, false);
				}
			}
		}
		if (!state.changed) {
			state.breakEffect();
		}
		break;
	case EffectType::Heal:
	{
		int putIndex = 0;
		for (int i : range(state.targetList)) {
			const AreaRef& ref = state.targetList[i];
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int heal = state.getEffectValue();
				if (state.isPreventEffect(*card)) {
					heal = 0;
				}
				int healed = Heal(state, ref.card, heal, true);
				if (healed > 0) {
					state.changed = true;
					if (effect.removeEffectedIfNoEffect) {
						if (i != putIndex) {
							state.targetList[putIndex] = ref;
						}
						putIndex++;
					}
				}
			}
		}
		if (effect.removeEffectedIfNoEffect) {
			state.targetList.resize(putIndex);
		}
		break;
	}
	case EffectType::HealAll:
	{
		int putIndex = 0;
		for (int i : range(state.targetList)) {
			const AreaRef& ref = state.targetList[i];
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int heal = card->damage;
				if (state.isPreventEffect(*card)) {
					heal = 0;
				}
				int healed = Heal(state, ref.card, heal, true);
				if (healed > 0) {
					state.changed = true;
					if (effect.removeEffectedIfNoEffect) {
						if (i != putIndex) {
							state.targetList[putIndex] = ref;
						}
						putIndex++;
					}
				}
			}
		}
		if (effect.removeEffectedIfNoEffect) {
			state.targetList.resize(putIndex);
		}
		break;
	}
	case EffectType::HealSand:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				int heal = 30;
				if (card->getMaster().arven) {
					heal = 100;
				}
				int healed = Heal(state, ref.card, heal, true);
			}
		}
		break;
	case EffectType::ResetHp:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				int damage = state.getMaxHp(*card) - state.getEffectValue();
				if (damage < card->damage) {
					state.changed = true;
					card->damage = damage;
					card->ko = false;
					card->koAttackDamage = false;
					card->koEnemyAttackDamage = false;
					card->koEnemyAttackDamageActive = false;
					card->koEnemyTerastalAttackDamage = false;
					card->koEnemyNAttackDamage = false;
					card->koFull = false;
					card->koPrizePlus1 = false;
					card->koPrizeDecreaseOnce = false;
					card->koPrizeZero = false;
					card->koNoDamageAndEffectAttackNextEnemyTurn = false;
				}
			}
		}
		break;
	case EffectType::Drain:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				Heal(state, ref.card, state.lastAttackDamage, true);
			}
		}
		break;
	case EffectType::Devolve: {
		AreaType toArea = (AreaType)state.getEffectValue();
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				DevolveProc(state, ref.card, *card, toArea);
			}
		}
		break;
	}
	case EffectType::DevolveAny: {
		if (state.targetList.empty()) {
			break;
		}
		AreaRef preRef = state.targetList[0];
		Card* card = state.checkGetCardNotPreventEffect(preRef);
		if (card == nullptr) {
			break;
		}
		CardPosition pos = state.currentCardPosition(preRef.card);
		DevolveProc(state, preRef.card, *card, AreaType::Hand);

		CardRef ref = state.getCardRef(pos);
		if (state.isEvolved(ref)) {
			state.contextCard = ref;
			state.setSelect(SelectType::YesNo, SelectContext::MoreDevolve, state.effectPlayerIndex());
			AddOptionYesAndNo(state);
			state.pushFunction(SelectedMoreDevolve);
		}
		break;
	}
	case EffectType::TransformDeck:
		TransformProc(state, true);
		break;
	case EffectType::TransformTrash:
		TransformProc(state, false);
		break;
	case EffectType::ExchangeSelected:
		if (!state.checkList.empty() && !state.targetList.empty()) {
			CardPosition pos1 = state.currentCardPosition(state.checkList[0]);
			CardPosition pos2 = state.currentCardPosition(state.targetList[0]);
			CardRef& ref1 = state.players[pos1.playerIndex].prize[pos1.areaIndex];
			CardRef& ref2 = state.players[pos2.playerIndex].hand[pos2.areaIndex];
			Card& card1 = state.getCard(ref1);
			Card& card2 = state.getCard(ref2);
			LogMoveCard(state, pos1.playerIndex, ref1, pos1.area, pos2.area, 0);
			LogMoveCard(state, pos2.playerIndex, ref2, pos2.area, pos1.area, 0);
			AreaType a1 = card1.area;
			state.cardMoved(ref1, card2.area);
			state.cardMoved(ref2, a1);
			std::swap(ref1, ref2);
			card1.reverse = false;
			card2.reverse = false;
		}
		break;
	case EffectType::KoPrizeChangeAlways:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				card->koPrizeChangeAlways += state.getEffectValue();
			}
		}
		break;
	case EffectType::KoPrizeChange:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				card->koPrizeChange += state.getEffectValue();
			}
		}
		break;
	case EffectType::KoPrizeDecreaseOnce:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				PlayerState& ps = state.players.at(card->playerIndex);
				if (!ps.koPrizeOnceChanged) {
					ps.koPrizeOnceChanged = true;
					card->koPrizeDecreaseOnce = true;
				}
			}
		}
		break;
	case EffectType::SelectEvolvesFrom:
		SelectCard(state);
		break;
	case EffectType::EvolvesToEach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				CardRef contextRef = state.selectedList.at(state.effectState.selectedListIndex);
				const Card& contextCard = state.getCard(contextRef);
				int index = state.currentAreaIndex(contextRef, contextCard);
				CardPosition pos = state.currentCardPosition(ref);
				EvolveProc(state, pos.area, pos.areaIndex, contextCard.area, index, pos.playerIndex);
			}
		}
		break;
	case EffectType::SelectEvolvesTo:
		SelectCard(state);
		break;
	case EffectType::EvolvesFromEach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				CardRef contextRef = state.selectedList.at(state.effectState.selectedListIndex);
				CardPosition contextPos = state.currentCardPosition(contextRef);
				CardPosition pos = state.currentCardPosition(ref);
				EvolveProc(state, contextPos.area, contextPos.areaIndex, pos.area, pos.areaIndex, pos.playerIndex);
				break;
			}
		}
		break;
	case EffectType::SelectAttachFrom:
		SelectCard(state);
		break;
	case EffectType::AttachToEach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				CardRef targetRef = state.selectedList.at(state.effectState.selectedListIndex);
				CardPosition pos = state.currentCardPosition(ref.card);
				AttachProc(state, pos.area, pos.areaIndex, pos.playerIndex, targetRef, true);
				if (state.getCard(targetRef).area == AreaType::Active) {
					state.attachActive = true;
				}
			}
		}
		break;
	case EffectType::SelectAttachTo:
		SelectCard(state);
		break;
	case EffectType::AttachEnergyMe:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				assert(IsEnergy(card->getMaster().cardType));
				CardRef effectCardRef = state.getEffectCard().card;
				CardPosition pos = state.currentCardPosition(ref.card);
				AttachProc(state, pos.area, pos.areaIndex, pos.playerIndex, effectCardRef, true);
			}
		}
		break;
	case EffectType::AttachSelectedCard:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				for (CardRef attachRef : state.selectedList) {
					state.changed = true;
					CardPosition pos = state.currentCardPosition(attachRef);
					AttachProc(state, pos.area, pos.areaIndex, pos.playerIndex, ref.card, true);
					if (state.getCard(ref.card).area == AreaType::Active) {
						state.attachActive = true;
					}
				}
				break;
			}
		}
		break;
	case EffectType::SwitchSelectedCard:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				for (CardRef energyRef : state.selectedList) {
					state.changed = true;
					if (state.isPreventEffect(*card)) {
						MoveRefCard(state, energyRef, state.getCard(energyRef), AreaType::Trash);
					} else {
						SwitchEnergyProc(state, energyRef, ref.card);
					}
				}
				break;
			}
		}
		break;
	case EffectType::AttachFromEach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				CardRef attachRef = state.selectedList.at(state.effectState.selectedListIndex);
				CardPosition pos = state.currentCardPosition(attachRef);
				AttachProc(state, pos.area, pos.areaIndex, pos.playerIndex, ref.card, true);
				if (state.getCard(ref.card).area == AreaType::Active) {
					state.attachActive = true;
				}
			}
		}
		break;
	case EffectType::SelectSwitchEnergyCard:
		SelectCard(state);
		break;
	case EffectType::EnergySwitchEach:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.changed = true;
				CardRef energyRef = state.selectedList.at(state.effectState.selectedListIndex);
				if (state.isPreventEffect(*card)) {
					MoveRefCard(state, energyRef, state.getCard(energyRef), AreaType::Trash);
				} else {
					SwitchEnergyProc(state, energyRef, ref.card);
				}
			}
		}
		break;
	case EffectType::DelayEffect:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				state.changed = true;
				AreaRef effectCardRef = state.getEffectCard();
				const Card& effectCard = state.getCard(effectCardRef.card);
				const Skill* delay;
				if (state.onAttack()) {
					const Attack& attack = state.getAttack();
					delay = CardTable.at(attack.cardId).delay;
				} else {
					const Skill& skill = state.getSkill();
					delay = CardTable.at(skill.cardId).delay;
				}
				assert(delay != nullptr);
				TriggeredAbility ta = {};
				ta.trigger.type = delay->triggers.at(0).triggerType;
				ta.trigger.subject = ref;
				ta.activateInfo = ActivateAbilityInfo(delay->skillId, effectCardRef, effectCard.playerIndex);
				state.delayTriggerStack.push_back(ta);
			}
		}
		break;
	case EffectType::Coin:
		SelectCoin(state, state.effectPlayerIndex(), state.getEffectValue());
		break;
	case EffectType::CoinUntilTail:
		if (state.getEffectValue() == 1) {
			state.pushFunction(AfterBreakIfCoinTail);
		}
		SelectCoinUntilTail(state, state.effectPlayerIndex());
		break;
	case EffectType::AttackDamageChange:
		state.attackDamageChange = state.getEffectValue();
		break;
	case EffectType::AttackDamageChangeTargetCount:
		state.attackDamageChange = state.getEffectValue() * (int)state.targetList.size();
		break;
	case EffectType::EffectDamageChangeTargetCount:
		state.effectState.damageChange = state.getEffectValue() * (int)state.targetList.size();
		break;
	case EffectType::AttackDamageChangeEnergyCount:
		state.attackDamageChange = state.getEffectValue() * state.targetListEnergyCount();
		break;
	case EffectType::EffectDamageChangeEnergyCount:
		state.effectState.damageChange = state.getEffectValue() * state.targetListEnergyCount();
		break;
	case EffectType::AttackDamageChangeTypeEnergyCount:
	{
		int count = state.targetListTypeEnergyCount((EnergyType)state.getEffectValue());
		state.attackDamageChange = state.getSecondEffectValue() * count;
		break;
	}
	case EffectType::EffectDamageChangeTypeEnergyCount:
	{
		int count = state.targetListTypeEnergyCount((EnergyType)state.getEffectValue());
		state.effectState.damageChange = state.getSecondEffectValue() * count;
		break;
	}
	case EffectType::AttackDamageChangeEnergyCountCoin:
		state.pushFunction(AfterAttackDamageChangeCoinUntilTail);
		SelectCoin(state, state.effectPlayerIndex(), state.targetListEnergyCount());
		break;
	case EffectType::AttackDamageChangeTypeEnergyCountCoin:
	{
		int count = state.targetListTypeEnergyCount((EnergyType)state.getEffectValue());
		state.pushFunction(AfterAttackDamageChangeCoin);
		SelectCoin(state, state.effectPlayerIndex(), count);
		break;
	}
	case EffectType::AttackDamageChangeCoin:
		state.pushFunction(AfterAttackDamageChangeCoin);
		SelectCoin(state, state.effectPlayerIndex(), state.getEffectValue());
		break;
	case EffectType::AttackDamageChangeCoinUntilTail:
		state.pushFunction(AfterAttackDamageChangeCoinUntilTail);
		SelectCoinUntilTail(state, state.effectPlayerIndex());
		break;
	case EffectType::AttackDamageChangeTargetCountCoin:
		state.pushFunction(AfterAttackDamageChangeCoinUntilTail);
		SelectCoin(state, state.effectPlayerIndex(), (int)state.targetList.size());
		break;
	case EffectType::AttackDamageChangeTargetCountEnemyCoin:
		state.pushFunction(AfterAttackDamageChangeTargetCountEnemyCoin);
		SelectCoin(state, 1 - state.effectPlayerIndex(), (int)state.targetList.size());
		break;
	case EffectType::AttackDamageChangeTakenPrize:
	{
		int count = 0;
		for (int i : state.effectTargetPlayer(effect)) {
			count += state.takenPrizeCount(i);
		}
		state.attackDamageChange = state.getEffectValue() * count;
		break;
	}
	case EffectType::EffectDamageChangeTakenPrize:
	{
		int count = 0;
		for (int i : state.effectTargetPlayer(effect)) {
			count += state.takenPrizeCount(i);
		}
		state.effectState.damageChange = state.getEffectValue() * count;
		break;
	}
	case EffectType::AttackDamageChangeDamageCounter:
		state.attackDamageChange = 0;
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.attackDamageChange += state.getEffectValue() * card->damage / 10;
			}
		}
		break;
	case EffectType::EffectDamageChangeDamageCounter:
		state.effectState.damageChange = 0;
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.effectState.damageChange += state.getEffectValue() * card->damage / 10;
			}
		}
		break;
	case EffectType::AttackDamageChangePutDamageCounter:
		state.setSelect(SelectType::Count, SelectContext::DamageCounterCount, state.effectPlayerIndex());
		for (int i = 0; i <= state.getEffectValue(); i++) {
			AddOptionNumber(state, i);
		}
		state.pushFunction(SelectedAttackDamageChangePutDamageCounter);
		break;
	case EffectType::AttackDamageChangeRetreatCost:
		state.attackDamageChange = 0;
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				int cost = state.retreatCost(*card);
				state.attackDamageChange += state.getEffectValue() * cost;
			}
		}
		break;
	case EffectType::AttackDamageChangeTypeCount:
	{
		unsigned type = 0;
		for (const AreaRef& ref : state.targetList) {
			Card& card = state.getCard(ref.card);
			type |= (unsigned)card.getMaster().energyType;
		}
		state.attackDamageChange = state.getEffectValue() * std::popcount(type);
		break;
	}
	case EffectType::AttackDamageChangeSpecialConditionCount:
	{
		int count = 0;
		for (int i : state.effectTargetPlayer(effect)) {
			const PlayerState& ps = state.players[i];
			if (ps.isPoisoned()) {
				count++;
			}
			if (ps.isBurned()) {
				count++;
			}
			if (ps.badStatus != BadStatusType::None) {
				count++;
			}
		}
		state.attackDamageChange = state.getEffectValue() * count;
		break;
	}
	case EffectType::AttackDamageChangeTakeAttackDamagePreTurn:
		state.attackDamageChange = 0;
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				state.attackDamageChange += card->takeAttackDamagePreTurn;
			}
		}
		break;
	case EffectType::AttackDamageChangePreTurnTakePrizeCount:
		state.attackDamageChange = state.getEffectValue() * state.turnHistories[1].takePrizeCountTurnPlayer;
		break;
	case EffectType::Burn:
		if (state.targetList.size() > 0) {
			Card* card = state.checkGetCard(state.targetList[0]);
			if (card != nullptr && card->area == AreaType::Active) {
				EffectBurn(state, card->playerIndex);
			}
		} else {
			for (int i : state.effectTargetPlayer(effect)) {
				EffectBurn(state, i);
			}
		}
		break;
	case EffectType::Poison:
		if (state.targetList.size() > 0) {
			Card* card = state.checkGetCard(state.targetList[0]);
			if (card != nullptr && card->area == AreaType::Active) {
				EffectPoison(state, card->playerIndex);
			}
		} else {
			for (int i : state.effectTargetPlayer(effect)) {
				EffectPoison(state, i);
			}
		}
		break;
	case EffectType::Poison8:
		for (int i : state.effectTargetPlayer(effect)) {
			EffectPoison(state, i, 8);
		}
		break;
	case EffectType::Poison16:
		for (int i : state.effectTargetPlayer(effect)) {
			EffectPoison(state, i, 16);
		}
		break;
	case EffectType::Sleep:
		if (state.targetList.size() > 0) {
			Card* card = state.checkGetCard(state.targetList[0]);
			if (card != nullptr && card->area == AreaType::Active) {
				EffectSleep(state, card->playerIndex);
			}
		} else {
			for (int i : state.effectTargetPlayer(effect)) {
				EffectSleep(state, i);
			}
		}
		break;
	case EffectType::Confuse:
		for (int i : state.effectTargetPlayer(effect)) {
			EffectConfuse(state, i);
		}
		break;
	case EffectType::Paralyze:
		for (int i : state.effectTargetPlayer(effect)) {
			EffectParalyze(state, i);
		}
		break;
	case EffectType::RecoverSpecialCondition:
		for (int i : state.effectTargetPlayer(effect)) {
			if (state.isPreventEffectActive(i)) {
				continue;
			}
			state.changed = true;
			ClearSpecialCondition(state, i);
		}
		break;
	case EffectType::RecoverSpecialConditionSingle:
		for (int i : state.effectTargetPlayer(effect)) {
			if (state.isPreventEffectActive(i)) {
				continue;
			}
			const PlayerState& ps = state.players[i];
			state.changed = true;
			state.setSelect(SelectType::SpecialCondition, SelectContext::RecoverSpecialCondition, state.effectPlayerIndex());
			if (ps.isPoisoned()) {
				AddOptionSpecialCondition(state, SelectSpecialConditionType::Poison);
			}
			if (ps.isBurned()) {
				AddOptionSpecialCondition(state, SelectSpecialConditionType::Burn);
			}
			if (ps.badStatus == BadStatusType::Asleep) {
				AddOptionSpecialCondition(state, SelectSpecialConditionType::Sleep);
			} else if (ps.badStatus == BadStatusType::Paralyzed) {
				AddOptionSpecialCondition(state, SelectSpecialConditionType::Paralyze);
			} else if (ps.badStatus == BadStatusType::Confused) {
				AddOptionSpecialCondition(state, SelectSpecialConditionType::Confuse);
			}
			if (state.options.size() > 0) {
				state.pushFunction(SelectedRecoverSpecialCondition, i);
			} else {
				state.clearSelect();
			}
			break;
		}
		break;
	case EffectType::Draw:
		for (int i : state.effectTargetPlayer(effect)) {
			Draw(state, i, state.getEffectValue());
		}
		break;
	case EffectType::DrawTargetCount:
		for (int i : state.effectTargetPlayer(effect)) {
			Draw(state, i, (int)state.targetList.size());
		}
		break;
	case EffectType::DrawPrizeCount:
		for (int i : state.effectTargetPlayer(effect)) {
			Draw(state, i, state.players[i].prize.size());
		}
		break;
	case EffectType::DrawUntil:
		for (int i : state.effectTargetPlayer(effect)) {
			Draw(state, i, state.getEffectValue() - state.players[i].hand.size());
		}
		break;
	case EffectType::DrawUntilPsychic:
		for (int i : state.effectTargetPlayer(effect)) {
			int count = 0;
			auto inPlay = state.players[i].getInPlayPokemon();
			for (RefPosition& rp : inPlay) {
				const Card& card = state.getCard(rp.ref);
				if (ContainsEnergyType(state.getEnergyType(card), EnergyType::Psychic)) {
					count++;
				}
			}
			Draw(state, i, count - state.players[i].hand.size());
		}
		break;
	case EffectType::DrawMirror:
		for (int i : state.effectTargetPlayer(effect)) {
			Draw(state, i, state.players[1 - i].hand.size() - state.players[i].hand.size());
		}
		break;
	case EffectType::DeckToTrash:
		state.targetList.clear();
		for (int i : state.effectTargetPlayer(effect)) {
			int count = state.getEffectValue();
			PlayerState& ps = state.players[i];
			for (int _ : range(count)) {
				if (ps.deck.empty()) {
					break;
				}
				state.changed = true;
				CardRef ref = MoveCard(state, i, AreaType::Deck, ps.deck.size() - 1, AreaType::Trash);
				state.targetList.push_back(state.makeAreaRef(ref));
			}
		}
		break;
	case EffectType::DeckToTrashCoinUntilTail:
		state.pushFunction(AfterDeckToTrashCoinUntilTail);
		SelectCoinUntilTail(state, state.effectPlayerIndex());
		break;
	case EffectType::DeckBottomToTrash:
		state.targetList.clear();
		for (int i : state.effectTargetPlayer(effect)) {
			int count = state.getEffectValue();
			PlayerState& ps = state.players[i];
			for (int _ : range(count)) {
				if (ps.deck.empty()) {
					break;
				}
				state.changed = true;
				CardRef ref = MoveListCard(ps.deck, ps.trash, 0);
				state.cardMoved(ref, AreaType::Trash);
				LogMoveCard(state, i, ref, AreaType::Deck, AreaType::Trash, 0);
				state.targetList.push_back(state.makeAreaRef(ref));
			}
		}
		break;
	case EffectType::DeckToPrize:
		for (int i : state.effectTargetPlayer(effect)) {
			state.changed = true;
			int count = state.getEffectValue();
			DeckToPrize(state, i, count);
		}
		break;
	case EffectType::Shuffle:
		for (int i : state.effectTargetPlayer(effect)) {
			ShuffleDeck(state, i);
		}
		break;
	case EffectType::EffectWin:
		for (int i : state.effectTargetPlayer(effect)) {
			state.changed = true;
			state.setResult(1 - i, FinishReason::Effect);
			break;
		}
		break;
	case EffectType::FailRetreat:
		state.failRetreat = true;
		break;
	case EffectType::TurnEnd:
		state.changed = true;
		state.turnEnd = true;
		break;
	case EffectType::DamageChangeThisTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->damageChangeThisTurn += state.getEffectValue();
			}
		}
		break;
	case EffectType::DamageChangeExThisTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->damageChangeExThisTurn += state.getEffectValue();
			}
		}
		break;
	case EffectType::PlayerDamageChange:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].playerDamageChange += state.getEffectValue();
		}
		break;
	case EffectType::PlayerDamageChangeEx:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].playerDamageChangeEx += state.getEffectValue();
		}
		break;
	case EffectType::PlayerDamageChangeMyFighting:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].playerDamageChangeMyFighting += state.getEffectValue();
		}
		break;
	case EffectType::TakePrizeCountChangeTerastalAttackKoActive:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].takePrizeCountChangeTerastalAttackKoActive += state.getEffectValue();
		}
		break;
	case EffectType::TakePrizeCountChangeNAttackKoActive:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].takePrizeCountChangeNAttackKoActive += state.getEffectValue();
		}
		break;
	case EffectType::TakeDamageChangeNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->takeDamageChangeNextEnemyTurn += state.getEffectValue();
			}
		}
		break;
	case EffectType::NoDamageLessEqualAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				int val = state.getEffectValue();
				if (card->noDamageLessEqualAttackNextEnemyTurn < val) {
					card->noDamageLessEqualAttackNextEnemyTurn = val;
				}
			}
		}
		break;
	case EffectType::NoDamageAndEffectAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageAndEffectAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoDamageAndEffectEnemyAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageAndEffectEnemyAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoDamageAndEffectEnemyExAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageAndEffectEnemyExAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoDamageAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoDamageBasicAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageBasicAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoDamageBasicColorAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageBasicColorAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoDamageAbilityAttackNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noDamageAbilityAttackNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::NoWeaknessNextEnemyTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->noWeaknessNextEnemyTurn = true;
			}
		}
		break;
	case EffectType::CannotUseThisAttackNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.cannotUseAttackId = state.getAttack().attackId;
			}
		}
		break;
	case EffectType::DamageChangeMyAttackNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.damageChangeMyAttack += state.getEffectValue();
			}
		}
		break;
	case EffectType::DamageChangeActiveNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.damageChangeActive += state.getEffectValue();
			}
		}
		break;
	case EffectType::DamageChangeNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.damageChange += state.getEffectValue();
			}
		}
		break;
	case EffectType::AttackCostChangeNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.attackCostChange += state.getEffectValue();
			}
		}
		break;
	case EffectType::RetreatCostChangeNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.retreatCostChange += state.getEffectValue();
			}
		}
		break;
	case EffectType::CannotRetreatNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.cannotRetreat = true;
			}
		}
		break;
	case EffectType::CannotHandAttachEnergyNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.cannotHandAttachEnergy = true;
			}
		}
		break;
	case EffectType::CannotAttackNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.cannotAttack = true;
			}
		}
		break;
	case EffectType::CannotAttackLessEqualEnergy2NextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.cannotAttackLessEqualEnergy2 = true;
			}
		}
		break;
	case EffectType::AttackCoinNextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.attackCoin = true;
			}
		}
		break;
	case EffectType::AttackCoin2NextTurn:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurn.attackCoin2 = true;
			}
		}
		break;
	case EffectType::CannotUseSelectedAttack:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				const CardMaster& master = card->getMaster();
				if (master.attacks.size() > 0) {
					state.setSelect(SelectType::Attack, SelectContext::DisableAttack, state.effectPlayerIndex());
					for (const Attack* attack : master.attacks) {
						AddOptionAttack(state, attack->attackId, attack->attackId);
					}
					state.pushFunction(SelectedDisableAttack, ref.card.cardIndex);
				}
			}
		}
		break;
	case EffectType::CannotAttackLessEqualEnergy2NextTurnPlayer:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotAttackLessEqualEnergy2 = true;
		}
		break;
	case EffectType::CannotPlayItemNextTurn:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotPlayItem = true;
		}
		break;
	case EffectType::CannotPlaySupporterNextTurn:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotPlaySupporter = true;
		}
		break;
	case EffectType::CannotPlayStadiumNextTurn:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotPlayStadium = true;
		}
		break;
	case EffectType::CannotPlaySpecialEnergyNextTurn:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotPlaySpecialEnergy = true;
		}
		break;
	case EffectType::CannotEvolveNextTurn:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotEvolve = true;
		}
		break;
	case EffectType::CannotRetreatPoison:
		for (int i : state.effectTargetPlayer(effect)) {
			state.players[i].nextTurn.cannotRetreatPoison = true;
		}
		break;
	case EffectType::MetalDamageChangeNextTurn:
		for (int i : state.effectTargetPlayer(effect)) {
			ClampShort(state.players[i].nextTurn.metalDamageChange, state.getEffectValue());
		}
		break;
	case EffectType::TakeDamageChangeNextMyTurnEnemy:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->nextTurnEnemy.takeDamageChange = state.getEffectValue();
			}
		}
		break;
	case EffectType::CannotUseThisAttackNonActive:
		for (const AreaRef& ref : state.targetList) {
			Card* card = state.checkGetCardNotPreventEffect(ref);
			if (card != nullptr) {
				card->cannotUseAttackIdNonActive = state.getAttack().attackId;
			}
		}
		break;
	case EffectType::SelectActivate:
		state.setSelect(SelectType::YesNo, SelectContext::Activate, state.effectPlayerIndex());
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedActivate);
		break;
	case EffectType::SelectEffect:
		state.setSelect(SelectType::YesNo, SelectContext::FirstEffect, state.effectPlayerIndex());
		AddOptionYesAndNo(state);
		state.pushFunction(SelectedFirstEffect);
		break;
	case EffectType::FailAttack:
		state.failAttack = true;
		break;
	case EffectType::CancelFailAttack:
		state.failAttack = false;
		break;
	case EffectType::BreakIfCoinHead:
		state.pushFunction(AfterBreakIfCoinHead);
		SelectCoin(state, state.effectPlayerIndex(), 1);
		break;
	case EffectType::BreakIfCoinTail:
		state.pushFunction(AfterBreakIfCoinTail);
		SelectCoin(state, state.effectPlayerIndex(), 1);
		break;
	case EffectType::BreakIfCoinTailMulti:
	{
		int count = state.getEffectValue();
		state.pushFunction(AfterBreakIfCoinTailMulti, count);
		SelectCoin(state, state.effectPlayerIndex(), count);
		break;
	}
	case EffectType::SkipIfCoinTail:
		state.pushFunction(AfterSkipIfCoinTail);
		SelectCoin(state, state.getSecondEffectValue() == 1 ? (1 - state.effectPlayerIndex()) : state.effectPlayerIndex(), 1);
		break;
	case EffectType::PostEffectActivate:
		state.postEffectActivate = true;
		break;
	case EffectType::BreakIfNotPostEffectActivated:
		if (!state.postEffectActivate) {
			state.breakEffect();
		}
		break;
	case EffectType::SelectPoisonBurnConfuse:
		state.setSelect(SelectType::SpecialCondition, SelectContext::AffectSpecialCondition, state.effectPlayerIndex());
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Poison);
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Burn);
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Confuse);
		state.pushFunction(SelectedSpecialCondition);
		break;
	case EffectType::SelectSpecialCondition:
		state.setSelect(SelectType::SpecialCondition, SelectContext::AffectSpecialCondition, state.effectPlayerIndex());
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Poison);
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Burn);
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Sleep);
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Paralyze);
		AddOptionSpecialCondition(state, SelectSpecialConditionType::Confuse);
		state.pushFunction(SelectedSpecialCondition);
		break;
	default:
		assert(false);
		break;
	}
}
