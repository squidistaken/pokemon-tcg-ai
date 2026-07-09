// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "EffectInstant.h"
#include "EffectContinual.h"

inline void ResolveTriggerStack(State& state, int depth);
inline void KOProc(State& state);
inline void BenchCheck(State& state);
inline void Refresh(State& state);
inline void AfterMoveTriggerStack(State& state, int depth);


inline void ParalyzeProc(State& state) {
	int playerIndex = state.activePlayerIndex();
	PlayerState& ps = state.players[playerIndex];
	if (ps.active.empty()) {
		return;
	}
	if (ps.badStatus == BadStatusType::Paralyzed) {
		LogParalyzed(state, playerIndex, true, ps.getActive());
		ps.badStatus = BadStatusType::None;
	}
}

inline void AfterSleepProc(State& state, int playerIndex) {
	if (state.coinHeadCount > 0) {
		PlayerState& ps = state.players[playerIndex];
		LogAsleep(state, playerIndex, true, ps.getActive());
		ps.badStatus = BadStatusType::None;
	}
}

inline void SleepProc(State& state, int playerIndex) {
	PlayerState& ps = state.players[playerIndex];
	if (ps.active.empty()) {
		return;
	}
	if (ps.badStatus == BadStatusType::Asleep) {
		state.pushFunction(AfterSleepProc, playerIndex);
		SelectCoin(state, playerIndex, 1);
	}
}

inline void AfterBurnProc(State& state, int playerIndex) {
	if (state.coinHeadCount > 0) {
		PlayerState& ps = state.players[playerIndex];
		if (ps.burned) {
			LogBurned(state, playerIndex, true, ps.getActive());
			ps.burned = false;
		}
	}
}

inline void BurnProc(State& state, int playerIndex) {
	PlayerState& ps = state.players[playerIndex];
	if (ps.active.empty()) {
		return;
	}
	if (ps.burned) {
		CardRef ref = ps.getActive();
		int damage = 20 + ps.burnDamageChange;
		AddDamage(state, ref, state.getCard(ref), damage, false, ref);
		state.pushFunction(AfterBurnProc, playerIndex);
		SelectCoin(state, playerIndex, 1);
	}
}

inline void SpecialConditionProc(State& state) {
	for (int i : state.basicPlayerOrder()) {
		PlayerState& ps = state.players[i];
		if (ps.active.empty()) {
			continue;
		}
		if (ps.isPoisoned()) {
			CardRef ref = ps.getActive();
			int damage = ps.poisonDamageCounter * 10 + ps.poisonDamageChange;
			if (!ContainsEnergyType(state.getEnergyType(state.getCard(ref)), EnergyType::Darkness)) {
				damage += ps.poisonDamageChangeNotDarkness;
			}
			if (damage > 0) {
				AddDamage(state, ref, state.getCard(ref), damage, false, ref);
			}
		}
	}

	state.pushFunction(ParalyzeProc);
	for (int i : state.reversePlayerOrder()) {
		state.pushFunction(SleepProc, i);
	}
	for (int i : state.reversePlayerOrder()) {
		state.pushFunction(BurnProc, i);
	}
}


inline void StaticEffect(State& state, CardRef ref) {
	Card& card = state.getCard(ref);
	const CardMaster& master = card.getMaster();
	int skillOrder = card.skillOrder;
	card.skillOrder = INT_MAX;
	if (state.noToolEffect && master.cardType == CardType::Tool) {
		return;
	}
	const Skill* ability = state.getAbility(card, master);
	if (ability == nullptr) {
		return;
	}
	const Skill& skill = *ability;
	if (skill.notStack && state.game->abilitySet[card.playerIndex].contains(skill.skillId)) {
		return;
	}

	std::vector<AreaRef>& targetList = state.game->targetList;
	targetList.clear();

	AreaRef areaRef = state.makeAreaRef(ref);
	for (int i = 0; i < (int)skill.effects.size(); i++) {
		const Effect& effect = skill.effects[i];
		if (effect.isCondition) {
			if (SatisfyCondition(state, skill.effects, i, ref, card.playerIndex)) {
				continue;
			} else {
				if (effect.failSkip) {
					i += effect.failSkip;
					continue;
				} else {
					return;
				}
			}
		}

		if (skillOrder == INT_MAX) {
			card.skillOrder = state.currentSkillOrder++;
		} else {
			card.skillOrder = skillOrder;
		}
		if (skill.notStack) {
			state.game->abilitySet[card.playerIndex].insert(skill.skillId);
		}
		TargetList(state, effect.target, targetList, areaRef, card.playerIndex);
		for (int i = (int)targetList.size() - 1; i >= 0; i--) {
			const Card& c = state.getCard(targetList[i].card);
			if(c.noEnemyAbility && c.playerIndex != card.playerIndex && card.area != AreaType::Stadium){
				targetList.erase(targetList.begin() + i);
			}
		}
		EffectContinual(state, effect, targetList, card.playerIndex, ref);
	}
}

inline void RefreshEffect(State& state, int depth) {
	state.continualState = 0;
	for (int i = 0; i < 2; i++) {
		state.game->abilitySet[i].clear();
		PlayerState& ps = state.players[i];
		ps.continualState = {};
		for (CardRef ref : ps.active) {
			state.getCard(ref).continualState = {};
		}
		for (CardRef ref : ps.bench) {
			state.getCard(ref).continualState = {};
		}
		for (CardRef ref : ps.hand) {
			state.getCard(ref).continualState = {};
		}

		for (CardRef ref : ps.tool) {
			state.getCard(ref).continualState = {};
		}
		for (CardRef ref : ps.energy) {
			state.getCard(ref).continualState = {};
		}
	}

	std::vector<CardEffect>& cardList = state.game->cardEffectList;
	cardList.clear();

	auto func = [&state, &cardList](CardRef ref) {
		Card& card = state.getCard(ref);
		const CardMaster& master = card.getMaster();
		const Skill* ability = state.getAbility(card, master);
		if (card.noAbility) {
			card.skillOrder = 0;
			return;
		}
		if (ability == nullptr) {
			return;
		}
		if (!ability->hasContinual() || !ability->isAreaMatch(card.area)) {
			return;
		}
		cardList.push_back({ ref, ability->priority, card.skillOrder, card.moveCounter });
	};

	for (int i = 0; i < 2; i++) {
		PlayerState& ps = state.players[i];
		ps.continualState = 0;
		for (CardRef ref : ps.active) {
			func(ref);
		}
		for (CardRef ref : ps.bench) {
			func(ref);
		}
		for (CardRef ref : ps.energy) {
			func(ref);
		}
		for (CardRef ref : ps.tool) {
			func(ref);
		}
		for (CardRef ref : ps.hand) {
			func(ref);
		}
	}
	for (CardRef ref : state.stadium) {
		func(ref);
	}

	state.updateOrder = false;
	std::sort(cardList.begin(), cardList.end(), [](const CardEffect& left, const CardEffect& right) {
		if (left.priority != right.priority) {
			return left.priority > right.priority;
		} else if (left.skillOrder != right.skillOrder) {
			return left.skillOrder < right.skillOrder;
		} else {
			return left.moveCounter < right.moveCounter;
		}
	});

	for (int i = 0; i < std::ssize(cardList); i++) {
		state.currentCardEffectIndex = i;
		const CardEffect& ce = cardList[i];
		StaticEffect(state, ce.ref);
		if (state.updateOrder) {
			break;
		}
	}

	if (state.updateOrder) {
		if (depth < 10) {
			RefreshEffect(state, depth + 1);
		}
	} else {
		for (int i = 0; i < 2; i++) {
			PlayerState& ps = state.players[i];
			for (CardRef ref : ps.active) {
				const Card& card = state.getCard(ref);
				if (card.noSpecialCondition) {
					ClearSpecialCondition(state, i);
				}
				if (card.noSleepParalyzeConfuse) {
					ClearSleepParalyzeConfuse(state, i);
				}
			}
		}
	}
}


inline void AfterAbility(State& state) {
	state.clearAbility();
}

// Effect発動後
inline void AfterEffect(State& state) {
	state.effectState.onEffect = false;
	state.game->pushResponse();
}

inline void ActivateEffect2(State& state) {
	auto& targetList = state.targetList;
	if (state.getEffect().loopCount > 0 && targetList.size() == 0) {
		state.effectLoopStop = true;
		return;
	}
	if (state.getEffect().addCheckList) {
		for (const AreaRef& ref : targetList) {
			state.checkList.push_back(ref.card);
		}
	}

	state.pushFunction(AfterEffect);
	EffectInstatnt(state);
}

// 効果対象選択後
inline void SelectedEvolveTarget(State& state) {
	if (state.selected.empty()) {
		state.clearSelect();
		return;
	}
	SelectOption o = state.options.at(state.selected[0]);
	int playerIndex = state.selectPlayer;
	state.clearSelect();
	state.changed = true;
	EvolveProc(state, (AreaType)o.param0, o.param1, (AreaType)o.param2, o.param3, playerIndex);
}

// 効果対象選択後
inline void SelectedEnergyTarget(State& state) {
	SetSelectedPokemonEnergy(state);
	ActivateEffect2(state);
}

// 効果対象選択後
inline void SelectedToolTarget(State& state) {
	SetSelectedAttachedCard(state, false);
	ActivateEffect2(state);
}

// 効果対象選択後
inline void SelectedCardOrAttachedCard(State& state) {
	state.targetList.clear();
	for (int i : state.selected) {
		const SelectOption& o = state.options[i];
		if (o.type == SelectOptionType::Card) {
			state.targetList.push_back(state.makeAreaRef(o));
		} else {
			CardPosition pos = o.getCardPosition();
			CardRef pokemonRef = state.getCardRef(pos.area, pos.areaIndex, pos.playerIndex);
			int moveCounter = state.getCard(pokemonRef).moveCounter;
			int index = 0;
			PlayerState& ps = state.players[pos.playerIndex];
			CardList& list = (o.type == SelectOptionType::EnergyCard ? ps.energy : ps.tool);
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
	}
	state.clearSelect();
	ActivateEffect2(state);
}

// 効果対象選択後
inline void SelectedEffectTarget(State& state) {
	state.setSelectedCardTarget();
	ActivateEffect2(state);
}

inline void ActivateEffect(State& state, const Effect& effect, int effectIndex) {
	int firstCheckIndex = -1;
	if (state.onSkill()) {
		firstCheckIndex = state.getSkill().firstConditionCount;
	}

	if (effect.isCondition) {
		// 事前条件チェック
		if (effectIndex >= firstCheckIndex) {
			if (!SatisfyCondition(state, state.getEffects(), effectIndex, state.getEffectCard().card, state.effectPlayerIndex())) {
				if (effect.failSkip) {
					state.lastFunction().calledCount += effect.failSkip;
				} else {
					state.isBreak = true;
				}
			}
			AfterEffect(state);
			return;
		} else {
			return;
		}
	}

	ActivateAbilityInfo& info = state.effectState.ability;
	std::vector<AreaRef>& targetList = state.targetList;

	int selectCount = effect.selectCount;
	if (effect.selectTargetCount) {
		selectCount = (int)targetList.size();
	} else if (effect.selectCoinHeadCount) {
		selectCount = state.coinHeadCount;
		if (selectCount == 0) {
			AfterEffect(state);
			return;
		}
	} else if (effect.selectCoinHeadCount2) {
		selectCount = state.coinHeadCount * 2;
		if (selectCount == 0) {
			AfterEffect(state);
			return;
		}
	} else if (effect.selectEnemyEnergyCount) {
		selectCount = 0;
		int playerIndex = 1 - state.effectPlayerIndex();
		for (const RefPosition& rp : state.players[playerIndex].getInPlayPokemon()) {
			selectCount += state.getEnergyCount(playerIndex, rp.ref);
		}
		if (selectCount == 0) {
			AfterEffect(state);
			return;
		}
	}

	if (!effect.notUpdateTarget) {
		if (effect.target.areas.size() == 0 || effect.target.areas[0] != AreaType::EffectedPreTarget) {
			state.preTargetList = targetList;
		}
		TargetList(state, effect.target, targetList, state.getEffectCard(), state.effectPlayerIndex());
		if (targetList.size() == 0) {
			if (effect.loopCount > 0) {
				state.effectLoopStop = true;
				return;
			}
			if (effect.target.areas.size() == 1) {
				if (effect.target.areas[0] == AreaType::TriggerSubject || effect.target.areas[0] == AreaType::TriggerObject) {
					AfterEffect(state);
					return;
				}
			}
		}
	}

	if (effect.notPreTarget) {
		// 重複を削除
		for (int i = 0; i < targetList.size(); i++) {
			const AreaRef& r = targetList[i];
			for (const AreaRef& ref : state.preTargetList) {
				if (r.card == ref.card) {
					targetList.erase(targetList.begin() + i);
					i--;
					break;
				}
			}
		}
	}
	if (effect.skipNoTarget && targetList.empty()) {
		return;
	}

	if (effect.effectSelectType != EffectSelectType::All) {
		if (Contains(effect.target.areas, AreaType::Deck)) {
			state.selectDeck = true;
		}
	}
	if (effect.seeingDeck) {
		state.selectDeck = true;
	}

	int selectPlayerIndex = info.usePlayerIndex;
	if (effect.enemySelect) {
		selectPlayerIndex = 1 - selectPlayerIndex;
	}

	if (effect.effectType == EffectType::Switch) {
		auto targetPlayer = state.effectTargetPlayer(effect);
		if (targetPlayer.size() == 1) {
			int index = targetPlayer[0];
			if (effect.effectTargetActive || (state.effectPlayerIndex() != index && selectPlayerIndex != index)) {
				if (!effect.effectTargetBench) {
					// バトルポケモンが受けた扱い
					const PlayerState& ps = state.players[index];
					if (ps.active.size() > 0) {
						CardRef ref = ps.active[0];
						if (state.isPreventEffect(state.getCard(ref))) {
							return;
						}
					} else {
						return;
					}
				}
			}
		}
	}

	if (effect.effectSelectType == EffectSelectType::Evolve || effect.effectSelectType == EffectSelectType::Evolve2) {
		auto inPlayPokemon = state.players[selectPlayerIndex].getInPlayPokemon();
		state.setSelect(SelectType::Evolve, SelectContext::Evolve, selectPlayerIndex);
		for (const AreaRef& ref : targetList) {
			const Card& card = state.getCard(ref.card);
			const CardMaster& master = card.getMaster();
			for (const RefPosition& rp : inPlayPokemon) {
				bool canEvolve;
				if (effect.effectSelectType == EffectSelectType::Evolve) {
					canEvolve = state.canEvolveEffect(state, card, master, rp.ref);
				} else {
					canEvolve = state.canEvolve2(card, master, rp.ref);
				}
				if (canEvolve) {
					int index = state.currentAreaIndex(ref.card, card);
					AddOptionEvolve(state, card.area, index, rp.area, rp.index);
				}
			}
		}

		state.pushFunction(SelectedEvolveTarget);
	} else if (effect.effectSelectType == EffectSelectType::Energy || effect.effectSelectType == EffectSelectType::MaxEnergyCard) {

		std::array<AttachedCardList, 2> list;
		for (int i : range(2)) {
			const PlayerState& ps = state.players[i];
			list[i] = state.getAttachedList(ps, ps.energy);
		}

		int playerIndex = -1;
		AttachedCardList selectList;
		for (const AreaRef& ref : targetList) {
			const Card& card = state.getCard(ref.card);
			playerIndex = card.playerIndex;
			const AttachedCardList& cardList = list[playerIndex];
			for (int i : range(cardList)) {
				const AttachedCard& c = cardList[i];
				if (c.ref == ref.card) {
					selectList.push_back(c);
					break;
				}
			}
		}

		if (playerIndex >= 0) {
			SelectContext context = effect.selectContext;
			if (effect.effectSelectType == EffectSelectType::MaxEnergyCard) {
				if (context == SelectContext::Discard) {
					context = SelectContext::DiscardEnergyCard;
				}

				int selectMax = std::min((int)targetList.size(), selectCount);
				int selectMin = 1;
				if (state.onAttackEffect()) {
					selectMin = 0;
				}
				if (effect.canNoSelect) {
					selectMin = 0;
				}
				if (effect.cannotNoSelect) {
					selectMin = 1;
				}
				state.setSelect(SelectType::AttachedCard, context, selectPlayerIndex, selectMin, selectMax);
				for (const AttachedCard& ac : selectList) {
					const Card& energyCard = state.getCard(ac.ref);
					RefPosition target = state.attachedCardPosition(energyCard);
					AddOptionEnergyCard(state, target.area, target.index, playerIndex, ac.attachedIndex);
				}

				state.pushFunction(SelectedEnergyTarget);
			} else {
				if (context == SelectContext::Discard) {
					context = SelectContext::DiscardEnergy;
				} else if (context == SelectContext::ToDeck) {
					context = SelectContext::ToDeckEnergy;
				} else if (context == SelectContext::ToHand) {
					context = SelectContext::ToHandEnergy;
				}
				int cost = selectCount;

				//state.pushFunction(ActivateEffect2);
				SelectPokemonEnergy(state, selectPlayerIndex, selectList, cost, context);
			}
		}
	} else if (effect.effectSelectType == EffectSelectType::ToolCard) {
		std::array<AttachedCardList, 2> list;
		for (int i : range(2)) {
			const PlayerState& ps = state.players[i];
			list[i] = state.getAttachedList(ps, ps.tool);
		}

		AttachedCardList selectList;
		for (const AreaRef& ref : targetList) {
			const Card& card = state.getCard(ref.card);
			const AttachedCardList& cardList = list[card.playerIndex];
			for (int i : range(cardList)) {
				const AttachedCard& c = cardList[i];
				if (c.ref == ref.card) {
					selectList.push_back(c);
					break;
				}
			}
		}

		SelectContext context = effect.selectContext;
		if (context == SelectContext::Discard) {
			context = SelectContext::DiscardToolCard;
		}

		int selectMax = std::min((int)targetList.size(), selectCount);
		int selectMin = 1;
		if (effect.canNoSelect) {
			selectMin = 0;
		}
		state.setSelect(SelectType::AttachedCard, context, selectPlayerIndex, selectMin, selectMax);
		for (const AttachedCard& ac : selectList) {
			const Card& toolCard = state.getCard(ac.ref);
			RefPosition target = state.attachedCardPosition(toolCard);
			AddOptionToolCard(state, target.area, target.index, toolCard.playerIndex, ac.attachedIndex);
		}

		state.pushFunction(SelectedToolTarget);
	} else if (effect.effectSelectType == EffectSelectType::CardOrAttachedCardCount) {
		SelectContext context = effect.selectContext;
		if (context == SelectContext::Discard) {
			context = SelectContext::DiscardCardOrAttachedCard;
		}

		int selectMax = std::min((int)targetList.size(), selectCount);
		int selectMin = 1;
		state.setSelect(SelectType::CardOrAttachedCard, context, selectPlayerIndex, selectMin, selectMax);
		for (const AreaRef& ref : targetList) {
			const Card& card = state.getCard(ref.card);
			int areaIndex = state.currentAreaIndex(card.playerIndex, card.area, ref.card);
			assert(areaIndex >= 0);

			if (card.area == AreaType::Energy) {
				CardList& list = state.players[card.playerIndex].energy;
				int index = 0;
				for (CardRef r : list) {
					if (r == ref.card) {
						break;
					} else if (card.attachMoveCounter == state.getCard(r).attachMoveCounter) {
						index++;
					}
				}
				RefPosition target = state.attachedCardPosition(card);
				AddOptionEnergyCard(state, target.area, target.index, card.playerIndex, index);
			} else if (card.area == AreaType::Tool) {
				CardList& list = state.players[card.playerIndex].tool;
				int index = 0;
				for (CardRef r : list) {
					if (r == ref.card) {
						break;
					} else if(card.attachMoveCounter == state.getCard(r).attachMoveCounter){
						index++;
					}
				}
				RefPosition target = state.attachedCardPosition(card);
				AddOptionToolCard(state, target.area, target.index, card.playerIndex, index);
			} else {
				AddOptionCard(state, card.area, areaIndex, card.playerIndex);
			}
		}
		state.pushFunction(SelectedCardOrAttachedCard);
	} else if (effect.effectSelectType != EffectSelectType::All) {
		// 効果対象選択
		if (effect.effectSelectType == EffectSelectType::CardUntil ) {
			selectCount = (int)targetList.size() - effect.selectCount;
			if (selectCount <= 0) {
				return;
			}
		} else if (effect.effectSelectType == EffectSelectType::MaxCardUntil) {
			selectCount = (int)targetList.size() - effect.selectCount;
			if (selectCount < 0) {
				selectCount = 0;
			}
		}

		int selectMax = std::min((int)targetList.size(), selectCount);
		if (effect.effectType == EffectType::ToBench) {
			int remain = state.remainingBench(state.getEffectTargetPlayerIndex());
			if (remain <= 0) {
				return;
			}
			selectMax = std::min(selectMax, remain);
		} else if (effect.effectType == EffectType::AttachSelectedCard || effect.effectType == EffectType::SwitchSelectedCard) {
			if (state.selectedList.size() == 0) {
				return;
			}
		}

		int selectMin = selectMax;
		if (effect.effectSelectType == EffectSelectType::MaxCardCount) {
			if (state.onAttackEffect()) {
				selectMin = 0;
			}
			if (selectMin > 1) {
				selectMin = 1;
			}
		} else if (effect.effectSelectType == EffectSelectType::MaxCardUntil) {
			selectMax = (int)targetList.size();
		}
		if (effect.canNoSelect) {
			selectMin = 0;
		}
		if (effect.canNoSelectIfExistPreTarget) {
			if (state.preTargetList.size() >= 1) {
				selectMin = 0;
			}
		}
		if (effect.isNotOpenSelectNoCondition()) {
			selectMin = 0;
		}
		if (selectMax >= 1 && selectMin == 0 && effect.cannotNoSelect) {
			if (!state.onAttackEffect() || effect.skillId == 0) {
				selectMin = 1;
			}
		}

		if (effect.randomSelect) {
			std::shuffle(targetList.begin(), targetList.end(), state.game->rng);
			targetList.resize(selectMax);
			ActivateEffect2(state);
		} else {
			state.setSelect(SelectType::Card, effect.selectContext, selectPlayerIndex, selectMin, selectMax);

			for (const AreaRef& ref : targetList) {
				const Card& card = state.getCard(ref.card);
				int areaIndex = state.currentAreaIndex(card.playerIndex, card.area, ref.card);
				assert(areaIndex >= 0);

				AddOptionCard(state, card.area, areaIndex, card.playerIndex);
			}

			state.pushFunction(SelectedEffectTarget);
		}
	} else {
		ActivateEffect2(state);
	}
}

inline void ActivateEffectEachSelected(State& state, int effectIndex) {
	state.effectState.selectedListIndex = state.calledCount();
	state.contextCard = state.selectedList.at(state.effectState.selectedListIndex);

	state.effectState.effectIndex = effectIndex;
	state.effectState.onEffect = true;
	const Effect& effect = state.getEffect();

	ActivateEffect(state, effect, effectIndex);
}

inline void ActivateEffectForEach(State& state, int effectIndex) {
	state.effectState.eachListIndex = state.calledCount();
	state.contextCard = state.eachList.at(state.effectState.eachListIndex);

	state.effectState.effectIndex = effectIndex;
	state.effectState.onEffect = true;
	const Effect& effect = state.getEffect();

	ActivateEffect(state, effect, effectIndex);
}

inline void ActivateEffectMultiple(State& state, int effectIndex) {
	if (state.effectLoopStop) {
		state.effectLoopStop = false;
		state.isBreak = true;
		return;
	}
	state.effectState.selectedListIndex = state.calledCount();

	state.effectState.effectIndex = effectIndex;
	state.effectState.onEffect = true;
	const Effect& effect = state.getEffect();

	ActivateEffect(state, effect, effectIndex);
}

inline void SeparatorProc(State& state) {
	if (!state.changed) {
		state.breakEffect();
	} else {
		BenchCheck(state);
	}
}

// 効果発動
inline void ActivateEffectLoop(State& state, int effectIndex) {
	if (state.effectJump > 0) {
		state.effectJump--;
		return;
	}
	if (effectIndex >= 100) {
		return;
	}
	state.effectState.effectIndex = effectIndex;
	state.effectState.onEffect = true;
	const Effect& effect = state.getEffect();
	if (effect.separator) {
		state.pushFunction(SeparatorProc);
	}

	if (effect.eachSelectedList) {
		if (state.selectedList.size() > 0) {
			state.pushFunction(ActivateEffectEachSelected, effectIndex);
			state.setLastFunctionCallCount(state.selectedList.size());
		}
	} else if (effect.eachList) {
		if (state.eachList.size() > 0) {
			state.pushFunction(ActivateEffectForEach, effectIndex);
			state.setLastFunctionCallCount(state.eachList.size());
		}
	} else if (effect.loopCount > 0) {
		state.effectLoopStop = false;
		state.pushFunction(ActivateEffectMultiple, effectIndex);
		state.setLastFunctionCallCount(effect.loopCount);
	} else {
		ActivateEffect(state, effect, effectIndex);
	}
}

inline void ActivateSkillEffect(State& state) {
	int effectIndex = state.calledCount();
	ActivateEffectLoop(state, effectIndex);
}

inline void ActivateAbility2(State& state) {
	const Skill& skill = state.getSkill();
	if (skill.onceTurn) {
		Card& card = state.getCard(state.getEffectCard().card);
		CardId cardId = card.getMaster().cardId;
		if (Contains(card.abilityUsed, cardId)) {
			return;
		}
		if (card.abilityUsed.isFull()) {
			card.abilityUsed.remove(0);
		}
		card.abilityUsed.push_back(cardId);
	}
	state.turnUsedSkill.push_back(skill.skillId);

	state.pushFunction(ActivateSkillEffect);
	state.setLastFunctionCallCount((int)skill.effects.size());
	if (skill.triggerStartIndex > 0) {
		state.lastFunction().calledCount = skill.triggerStartIndex;
	}
}

inline void ActivateSelectedEffect(State& state, int index) {
	ActivateAbility2(state);
	if (index == 1) {
		state.lastFunction().calledCount = state.getSkill().secondEffectStartIndex;
	}
}

inline void SelectedWhichEffect(State& state) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	if (isYes) {
		ActivateSelectedEffect(state, 0);
	} else {
		ActivateSelectedEffect(state, 1);
	}
}

inline void ActivateEnemySelectedEffect(State& state, int index) {
	ActivateAbility2(state);
	if (index == 1) {
		state.lastFunction().calledCount = state.getSkill().secondEffectStartIndexEnemy;
	}
}

inline void EnemySelectedWhichEffect(State& state) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	if (isYes) {
		ActivateEnemySelectedEffect(state, 0);
	} else {
		ActivateEnemySelectedEffect(state, 1);
	}
}

inline void SelectedActivateAbility(State& state) {
	bool isYes = state.selectedYes();
	state.clearSelect();
	if (isYes) {
		ActivateAbility2(state);
	}
}

// 能力発動
inline void ActivateAbility(State& state) {
	state.targetList.clear();
	state.changed = false;
	const Skill& skill = state.getSkill();
	if (skill.canSelectActivate) {
		CardRef ref = state.getEffectCard().card;
		SetYesNoSelect(state, SelectContext::Activate, state.effectState.usePlayerIndex());
		state.contextCard = ref;
		state.pushFunction(SelectedActivateAbility);
	} else if (skill.secondEffectStartIndex > 0) {
		CardRef ref = state.getEffectCard().card;
		int playerIndex = state.effectState.usePlayerIndex();
		if (!SatisfySkillCondition(state, skill, ref, playerIndex)) {
			ActivateSelectedEffect(state, 1);
		} else {
			SetYesNoSelect(state, SelectContext::FirstEffect, playerIndex);
			state.contextCard = ref;
			state.pushFunction(SelectedWhichEffect);
		}
	} else if (skill.secondEffectStartIndexEnemy > 0) {
		CardRef ref = state.getEffectCard().card;
		int playerIndex = 1 - state.effectState.usePlayerIndex();
		SetYesNoSelect(state, SelectContext::Activate, playerIndex);
		state.contextCard = ref;
		state.pushFunction(EnemySelectedWhichEffect);
	} else {
		ActivateAbility2(state);
	}
}

inline bool SatisfyFirstSkillCondition(State& state) {
	ActivateAbilityInfo& info = state.effectState.ability;
	const Skill& skill = SkillTable.at(info.skillId);
	return SatisfySkillCondition(state, skill, info.effectCard.card, info.usePlayerIndex, skill.triggerStartIndex);
}

inline void ActivateTriggerAbility(State& state) {
	const Card* card = state.checkGetCard(state.getEffectCard());
	if (card != nullptr && card->noAbility) {
		return;
	}

	if (!SatisfyFirstSkillCondition(state)) {
		return;
	}

	ActivateAbility(state);
}

inline int AreaTypeTriggerOrder(AreaType areaType) {
	return -(int)areaType;
}

inline void AfterTriggerAbility(State& state, int depth) {
	AfterAbility(state);
	if (depth != 0) {
		RefreshEffect(state);
		AfterMoveTriggerStack(state, depth);
	}
}

inline void AfterMoveTriggerStack(State& state, int depth) {
	if (state.triggerStack.size() >= 1) {
		state.stateChanged = true;

		TriggeredAbility ap = state.triggerStack.back();
		if (ap.trigger.depth < depth) {
			return;
		}

		state.triggerStack.pop_back();

		state.pushFunction(AfterTriggerAbility, depth);
		if (ap.activateInfo.isSpecialCondition) {
			SpecialConditionProc(state);
		} else {
			const Card& card = state.getCard(ap.activateInfo.effectCard.card);
			if (card.area == AreaType::Trash || card.area == AreaType::Deck || card.area == AreaType::Hand || card.area == AreaType::Prize) {
				const Skill& skill = SkillTable.at(ap.activateInfo.skillId);
				if (skill.canActivateTrash && card.area == AreaType::Trash) {
					// empty
				} else {
					return;
				}
			}
			state.setTriggeredAbility(ap);

			ActivateTriggerAbility(state);
		}
	}
}

inline int FirstTriggerIndex(State& state, int depth) {
	for (int i : range(state.temporaryTriggerStack)) {
		const TriggeredAbility& ta = state.temporaryTriggerStack[i];
		if (ta.trigger.depth == depth) {
			return i;
		}
	}
	return 0;
}

inline void SelectedSkillOrder(State& state, int depth) {
	int first = FirstTriggerIndex(state, depth);
	for (int i = (int)state.selected.size() - 1; i >= 0; i--) {
		state.triggerStack.push_back(state.temporaryTriggerStack.at(state.selected[i] + first));
	}
	state.clearSelect();
	state.temporaryTriggerStack.resize(first);

	AfterMoveTriggerStack(state, depth);
}


inline void ResolveTriggerStack(State& state, int depth) {
	int first = FirstTriggerIndex(state, depth);
	int count = (int)state.temporaryTriggerStack.size() - first;
	if (count > 0) {
		if (count >= 2) {
			int selectPlayer = state.activePlayerIndex();
			TriggerType type = {};
			for (int i = first; i < std::ssize(state.temporaryTriggerStack); i++) {
				const TriggeredAbility& ta = state.temporaryTriggerStack[i];
				type = ta.trigger.type;
			}
			if (state.phase != GamePhase::Main || type == TriggerType::DamagedEnemyAttackActive || type == TriggerType::DamagedEnemyAttack) {
				selectPlayer = 1 - state.activePlayerIndex();
			}

			state.setSelect(SelectType::Skill, SelectContext::SkillOrder, selectPlayer, count, count);
			for (int i = first; i < std::ssize(state.temporaryTriggerStack); i++) {
				const TriggeredAbility& ta = state.temporaryTriggerStack[i];
				AddOptionSkillOrder(state, ta.activateInfo.effectCard.card);
			}
			state.pushFunction(SelectedSkillOrder, depth);
			return;
		} else {
			state.triggerStack.push_back(state.temporaryTriggerStack[first]);
			state.temporaryTriggerStack.resize(first);
		}
	}
	AfterMoveTriggerStack(state, depth);
}

inline void ActiveCheckPush(State& state) {
	for (int i : state.activePlayerOrder()) {
		if (state.players[i].active.empty()) {
			state.stateChanged = true;
			state.pushFunction(SelectActivePokemon, i);
		}
	}
}

inline void ActiveCheck(State& state) {
	if (state.finishCheck()) {
		return;
	}
	state.pushFunction(ResolveTriggerStack, 0);
	ActiveCheckPush(state);
}

inline void KOProc3(State& state) {
	for (const AreaRef& ref : state.koList) {
		const Card* card = state.checkGetCard(ref);
		if (card != nullptr) {
			const CardMaster& master = card->getMaster();
			int playerIndex = card->playerIndex;
			int prize = state.getPrizeCount(*card);
			if (!master.noPrize) {
				const PlayerState& ps = state.players[1 - playerIndex];
				if (prize > 0) {
					state.pushFunction(SelectPrize, 1 - playerIndex, prize, 0);
				}
			}
			if (state.phase == GamePhase::Main) {
				if (state.isPlayerTurn() && state.activePlayerIndex() != card->playerIndex) {
					state.thisTurnHistory().ko = true;
					if (master.teamRocket) {
						state.thisTurnHistory().koTeamRocket = true;
					}
					if (card->koEnemyAttackDamage) {
						state.thisTurnHistory().koAttackDamage = true;
						if (master.ethan) {
							state.thisTurnHistory().koAttackDamageEthan = true;
						}
						if (master.hop) {
							state.thisTurnHistory().koAttackDamageHop = true;
						}
					}
				}
			}
		}
	}
	for (int i = (int)state.koList.size() - 1; i >= 0; i--) {
		// 逆順で入っているので逆からMoveCard
		const AreaRef& ref = state.koList[i];
		const Card* card = state.checkGetCard(ref);
		if (card != nullptr) {
			int playerIndex = card->playerIndex;
			int index = state.currentAreaIndex(playerIndex, card->area, ref.card);
			AreaType toArea = AreaType::Trash;
			if (card->koByDamageToHand && card->koEnemyAttackDamage) {
				toArea = AreaType::Hand;
			}
			MoveCard(state, playerIndex, card->area, index, toArea, 0, false, false, false, true);
		}
	}
	state.koList.clear();
}

inline void KOProc2(State& state) {
	for (int i : state.reversePlayerOrder()) {
		auto inPlayPokemon = state.players[i].getInPlayPokemon();
		for (int j = inPlayPokemon.size() - 1; j >= 0; j--) {
			const RefPosition& rp = inPlayPokemon[j];
			Card& card = state.getCard(rp.ref);
			if (card.ko) {
				state.koList.push_back(state.makeAreaRef(rp.ref));
			}
		}
	}

	if (state.koList.empty()) {
		ActiveCheck(state);
	} else {
		state.stateChanged = true;
		state.pushFunction(KOProc3);

		for (const AreaRef& ref : state.koList) {
			const Card* card = state.checkGetCard(ref);
			if (card != nullptr) {
				PullTrigger(state, TriggerType::Ko, ref.card, CardRef(card->koCauseRef), 1);
				if (card->koEnemyAttackDamage) {
					PullTrigger(state, TriggerType::KoEnemyAttackDamage, ref.card, CardRef(card->koCauseRef), 1);
				}
				if (card->koEnemyExAttackDamage) {
					PullTrigger(state, TriggerType::KoEnemyExAttackDamage, ref.card, CardRef(card->koCauseRef), 1);
				}
				if (card->koEnemyAttackDamageActive) {
					PullTrigger(state, TriggerType::KoEnemyAttackDamageActive, ref.card, CardRef(card->koCauseRef), 1);
				}
				if (card->koNoDamageAndEffectAttackNextEnemyTurn) {
					CardRef ref(card->koCauseRef);
					if (!ref.isNull()) {
						Card& c = state.getCard(ref);
						if (c.area == AreaType::Active) {
							c.noDamageAndEffectAttackNextEnemyTurn = true;
						}
					}
				}
			}
		}

		ResolveTriggerStack(state, 1);
	}
}

inline void KOProc(State& state) {
	state.pushFunction(KOProc2);
	for (int i : state.basicPlayerOrder()) {
		auto inPlayPokemon = state.players[i].getInPlayPokemon();
		for (int j = inPlayPokemon.size() - 1; j >= 0; j--) {
			const RefPosition& rp = inPlayPokemon[j];
			Card& card = state.getCard(rp.ref);
			if (state.isKo(card)) {
				card.ko = true;
			}
			if (card.koAttackDamage) {
				PullTrigger(state, TriggerType::PreKo, rp.ref, {}, 1);
				if (card.koFull) {
					PullTrigger(state, TriggerType::PreKoFull, rp.ref, {}, 1);
					if (card.koEnemyAttackDamage) {
						PullTrigger(state, TriggerType::PreKoFullEnemy, rp.ref, {}, 1);
					}
				}
			}
		}
	}
	ResolveTriggerStack(state, 1);
}

inline void ToolCountProc(State& state) {
	state.pushFunction(KOProc);

	for (int i : state.basicPlayerOrder()) {
		const PlayerState& ps = state.players[i];
		if (ps.tool.size() >= 2) {
			auto inPlayPokemon = ps.getInPlayPokemon();
			for (const RefPosition& rp : inPlayPokemon) {
				const Card& card = state.getCard(rp.ref);
				int remain = state.remainingToolCapacity(card);
				if (remain < 0) {
					state.stateChanged = true;
					state.pushFunction(SelectToolMaxTrash, i, rp.ref.cardIndex);
				}
			}
		}

		for (int i = ps.energy.size() - 1; i >= 0; i--) {
			CardRef ref = ps.energy[i];
			const Card& card = state.getCard(ref);
			const CardMaster& master = card.getMaster();
			if (master.onlyTeamRocket) {
				RefPosition rp = state.attachedCardPosition(card);
				if (!state.getCard(rp.ref).getMaster().teamRocket) {
					state.stateChanged = true;
					MoveCard(state, state.activePlayerIndex(), AreaType::Energy, i, AreaType::Trash);
				}
			}
		}
	}
}

inline void AfterRefresh(State& state) {
	if (state.stateChanged) {
		Refresh(state);
	} else {
		if (state.finishCheck()) {
			return;
		}
	}
}

inline void BenchCheck(State& state) {
	for (int i : {1, 0}) {
		int playerIndex = (i ^ state.lastStadiumPlayer);
		if (state.remainingBench(playerIndex) < 0) {
			state.stateChanged = true;
			state.pushFunction(SelectBenchMaxTrash, playerIndex);
		}
	}
}

// 状態をアップデート
inline void Refresh(State& state) {
	state.stateChanged = false;
	state.pushFunction(AfterRefresh);
	RefreshEffect(state);
	state.game->pushResponse();
	if (state.phase == GamePhase::PokemonCheckup) {
		// きぜつ処理はスキップ
		state.pushFunction(ResolveTriggerStack, 0);
	} else {
		state.pushFunction(ToolCountProc);
	}

	BenchCheck(state);
}




inline void SetAndActivateSkill(State& state, const Skill& skill, const Card& card, CardRef ref, int usePlayerIndex) {
	ActivateAbilityInfo info{ skill.skillId, AreaRef(ref, card.moveCounter), usePlayerIndex };
	state.setActivateAbility(info);
	ActivateAbility(state);
}

