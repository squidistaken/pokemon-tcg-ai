// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "EffectProc.h"

inline void ToMain(State& state);
inline void TurnStart(State& state);
inline void SelectedAttackId(State& state, int srcAttackId);
inline void SelectedAttack2(State& state);

inline void PokemonCheckupEnd(State& state) {
	state.phase = GamePhase::PokemonCheckupEnd;
	state.pushFunction(TurnStart);
	Refresh(state);
}

inline void PokemonCheckup(State& state) {
	if (state.finishCheck()) {
		return;
	}
	state.phase = GamePhase::PokemonCheckup;

	PullTrigger(state, TriggerType::PokemonCheckup, {});

	for (int i : range(2)) {
		const PlayerState& ps = state.players[i];
		if (ps.badStatus == BadStatusType::Asleep
			|| ps.badStatus == BadStatusType::Paralyzed
			|| ps.isPoisoned()
			|| ps.isBurned()) {
			TriggeredAbility& ta = state.temporaryTriggerStack.emplace_back();
			ta.trigger.type = TriggerType::None;
			ta.activateInfo.isSpecialCondition = true;
			break;
		}
	}

	state.pushFunction(PokemonCheckupEnd);
	Refresh(state);
}

inline void TurnEnd2(State& state) {
	if (state.finishCheck()) {
		return;
	}
	int playerIndex = state.activePlayerIndex();
	state.turnState = 0;
	for (CardRef ref : state.stadium) {
		state.getCard(ref).turnEnd(playerIndex);
	}
	for (int i : state.basicPlayerOrder()) {
		PlayerState& ps = state.players[i];
		ps.turnEnd();
		for (CardRef ref : ps.active) {
			state.getCard(ref).turnEnd(playerIndex);
		}
		for (CardRef ref : ps.bench) {
			state.getCard(ref).turnEnd(playerIndex);
		}
	}

	{
		PlayerState& ps = state.players[state.activePlayerIndex()];
		if (!state.noToolEffect) {
			for (int i = ps.tool.size() - 1; i >= 0; i--) {
				CardRef ref = ps.tool[i];
				const Card& card = state.getCard(ref);
				const CardMaster& master = card.getMaster();
				if (master.trashMyTurnEnd) {
					MoveCard(state, state.activePlayerIndex(), AreaType::Tool, i, AreaType::Trash);
				}
			}
		}
		for (int i = ps.energy.size() - 1; i >= 0; i--) {
			CardRef ref = ps.energy[i];
			const Card& card = state.getCard(ref);
			const CardMaster& master = card.getMaster();
			if (master.trashMyTurnEnd) {
				MoveCard(state, state.activePlayerIndex(), AreaType::Energy, i, AreaType::Trash);
			}
		}
	}

	state.pushFunction(PokemonCheckup);
	Refresh(state);
}

inline void TurnEnd(State& state) {
	if (state.finishCheck()) {
		return;
	}

	int playerIndex = state.activePlayerIndex();
	LogTurnEnd(state, playerIndex);

	for (int i = (int)state.delayTriggerStack.size() - 1; i >= 0; i--) {
		const TriggeredAbility& ta = state.delayTriggerStack[i];
		const Card& effectCard = state.getCard(ta.activateInfo.effectCard.card);
		if (effectCard.playerIndex != playerIndex) {
			if (ta.trigger.type == TriggerType::TurnEnd) {
				state.temporaryTriggerStack.push_back(ta);
			}
			state.delayTriggerStack.erase(state.delayTriggerStack.begin() + i);
		} else {
			if (ta.trigger.type == TriggerType::TurnEnd && ta.trigger.subject.card == state.playerCardRef(playerIndex)) {
				state.temporaryTriggerStack.push_back(ta);
				state.delayTriggerStack.erase(state.delayTriggerStack.begin() + i);
			}
		}
	}
	PullTrigger(state, TriggerType::TurnEnd, state.playerCardRef(state.activePlayerIndex()));
	state.pushFunction(TurnEnd2);
	Refresh(state);
}

inline void AfterPlay(State& state) {
	AfterAbility(state);
	while (state.playing.size() > 0) {
		MoveCard(state, state.activePlayerIndex(), AreaType::Playing, 0, AreaType::Trash, false, true);
	}
}

inline void SelectedPlay(State& state, int handIndex) {
	int playerIndex = state.activePlayerIndex();
	const PlayerState& ps = state.players[playerIndex];
	CardRef ref = ps.hand.at(handIndex);
	const Card& card = state.getCard(ref);
	const CardMaster& master = card.getMaster();
	LogPlay(state, playerIndex, ref);
	if (master.cardType == CardType::Pokemon) {
		MoveCard(state, playerIndex, AreaType::Hand, handIndex, AreaType::Bench, 0, true);
	} else if (master.cardType == CardType::Stadium) {
		state.stadiumPlayed = true;
		MoveCard(state, playerIndex, AreaType::Hand, handIndex, AreaType::Stadium, 0, true);
	} else {
		if (master.play == nullptr) {
			Exception("no play skill");
		}
		if (master.cardType == CardType::Supporter) {
			state.supporterPlayed = true;
			state.turnPlay.push_back(ref);
		}

		if (master.toBench) {
			MoveCard(state, playerIndex, AreaType::Hand, handIndex, AreaType::Bench, 0, true);
		} else {
			MoveCard(state, playerIndex, AreaType::Hand, handIndex, AreaType::Playing, 0, true);

			state.pushFunction(AfterPlay);
			SetAndActivateSkill(state, *master.play, card, ref, card.playerIndex);
		}
	}
}

inline void SelectedAttach(State& state, int handIndex, AreaType area, int index) {
	int playerIndex = state.activePlayerIndex();
	CardRef targetRef = state.getCardRef(area, index, playerIndex);
	AttachProc(state, AreaType::Hand, handIndex, playerIndex, targetRef, false);
}

inline void SelectedEvolve(State& state, int handIndex, AreaType area, int index) {
	EvolveProc(state, AreaType::Hand, handIndex, area, index, state.activePlayerIndex());
}

inline void SelectedAbility(State& state, AreaType area, int index) {
	int playerIndex = state.activePlayerIndex();
	CardRef ref = state.getCardRef(area, index, playerIndex);
	Card& card = state.getCard(ref);
	const CardMaster& master = card.getMaster();
	state.pushFunction(AfterAbility);
	SetAndActivateSkill(state, *master.ability, card, ref, playerIndex);
}

inline void SelectedDiscard(State& state, AreaType area, int index) {
	int playerIndex = state.activePlayerIndex();
	MoveCard(state, playerIndex, area, index, AreaType::Trash);
}

inline void AfterRetreat(State& state) {
	state.selectedList.clear();
	state.targetList.clear();
}

inline void SelectedRetreat2(State& state) {
	if (state.failRetreat) {
		return;
	}
	int playerIndex = state.activePlayerIndex();
	int cost = state.retreatCost(playerIndex);

	state.pushFunction(AfterRetreat);
	state.pushFunction(SelectSwitchPokemon, playerIndex, playerIndex);
	if (cost > 0) {
		SelectTrashSinglePokemonEnergy(state, playerIndex, state.players[playerIndex].getActive(), cost);
	}
}

inline void SelectedRetreat(State& state) {
	state.failRetreat = false;
	state.retreated = true;
	int playerIndex = state.activePlayerIndex();
	PullTrigger(state, TriggerType::PreRetreat, state.players[playerIndex].getActive());

	state.pushFunction(SelectedRetreat2);
	Refresh(state);
}

// エネルギーが足りているかと、Attackを所持しているかは確認しない
inline bool CanUseAttack(const State& state, CardRef ref, const Card& card, const AttackEnergy& ae) {
	if (!state.canAttack(ref, card)) {
		return false;
	}

	const Attack& attack = *ae.attack;
	if (attack.damage == 0 && !attack.noCheckCondition) {
		if (!SatisfyAttackCondition(state, attack, ref, card)) {
			return false;
		}
	}

	return true;
}

inline void AfterAttack5(State& state) {
	state.currentAttackId = 0;
	state.attacker = {};
	TurnEnd(state);
}

inline void SelectedSecondAttack(State& state) {
	if (state.selected.empty()) {
		state.clearSelect();
		AfterAttack5(state);
	} else {
		state.secondAttack = true;
		SelectedAttackId(state, 0);
	}
}

inline void AfterAttack4(State& state) {
	if (state.secondAttack) {
		state.secondAttack = false;
	} else {
		int playerIndex = state.activePlayerIndex();
		const PlayerState& ps = state.players.at(playerIndex);
		if (ps.active.size() > 0) {
			CardRef active = ps.getActive();
			if (state.attacker == active) {
				Card& attacker = state.getCard(active);
				if (attacker.doubleAttack) {
					if (ps.badStatus != BadStatusType::Asleep && ps.badStatus != BadStatusType::Paralyzed) {
						if (attacker.getMaster().hasAttack(&state.getAttack())) {
							std::vector<EnergyType>& energyList = state.game->energyList;
							state.getEnergies(playerIndex, active, energyList);
							SetAttackEnergy(state, attacker, energyList, false, true);
							state.setSelect(SelectType::Attack, SelectContext::Attack, playerIndex, 0, 1);
							for (const AttackEnergy& ae : state.game->attackEnergyList) {
								if (ae.insufficientEnergy <= 0) {
									if (CanUseAttack(state, active, attacker, ae)) {
										AddOptionAttack(state, ae.attack->attackId, ae.srcAttackId);
									}
								}
							}
							if (state.options.empty()) {
								state.clearSelect();
							} else {
								state.pushFunction(SelectedSecondAttack);
								return;
							}
						}
					}
				}
			}
		}
	}

	AfterAttack5(state);
}

inline void AfterAttack3(State& state) {
	state.pushFunction(AfterAttack4);
	Refresh(state);
}

inline void AfterAttack2(State& state) {
	state.pushFunction(AfterAttack3);
	RefreshEffect(state);
	ActiveCheckPush(state);
}

inline void AfterAttackTrigger(State& state) {
	if (state.triggerStack.size() + state.temporaryTriggerStack.size() > 0) {
		state.pushFunction(AfterAttackTrigger);
		RefreshEffect(state);
		ResolveTriggerStack(state, 0);
	}
}

inline void AfterAttack(State& state) {
	state.attackDamageChange = 0;
	state.postAttackEffect = false;
	state.postEffectActivate = false;
	state.failAttack = false;
	state.clearAbility();

	state.pushFunction(AfterAttack2);
	AfterAttackTrigger(state);
}

inline void AttackDamage2(State& state, int damage) {
	const PlayerState& es = state.players[1 - state.activePlayerIndex()];
	const Attack& attack = state.getAttack();
	CardRef targetRef = es.getActive();
	Card& target = state.getCard(targetRef);
	CardRef attackerRef = state.getAttacker();
	Card& attacker = state.getCard(attackerRef);
	state.lastAttackDamage = damage;
	AddDamage(state, targetRef, target, damage, true, attackerRef, false, &attack);
	AfterDamage(state, targetRef, target, attackerRef, attacker);
}

inline void AttackNoDamageCoin(State& state, int damage) {
	if (state.coinHeadCount == 0 || state.getAttack().noTargetEffect) {
		AttackDamage2(state, damage);
	}
	state.game->pushResponse();
}

inline void AttackDamage(State& state) {
	if (state.failAttack) {
		return;
	}
	const PlayerState& es = state.players[1 - state.activePlayerIndex()];
	if (es.active.empty()) {
		return;
	}

	RefreshEffect(state);

	const Attack& attack = state.getAttack();
	CardRef targetRef = es.getActive();
	Card& target = state.getCard(targetRef);
	CardRef attackerRef = state.getAttacker();
	Card& attacker = state.getCard(attackerRef);
	int baseDamage = attack.damage + state.attackDamageChange;
	bool calcDamage = (baseDamage > 0);
	if (!calcDamage) {
		for (const Effect& effect : attack.preEffects) {
			if (effect.effectType == EffectType::AttackDamageChange
				|| effect.effectType == EffectType::AttackDamageChangeTargetCount
				|| effect.effectType == EffectType::AttackDamageChangeEnergyCount
				|| effect.effectType == EffectType::AttackDamageChangeTypeEnergyCount
				|| effect.effectType == EffectType::AttackDamageChangeEnergyCountCoin
				|| effect.effectType == EffectType::AttackDamageChangeTypeEnergyCountCoin
				|| effect.effectType == EffectType::AttackDamageChangeCoin
				|| effect.effectType == EffectType::AttackDamageChangeCoinUntilTail
				|| effect.effectType == EffectType::AttackDamageChangeTargetCountCoin
				|| effect.effectType == EffectType::AttackDamageChangeTargetCountEnemyCoin
				|| effect.effectType == EffectType::AttackDamageChangeTakenPrize
				|| effect.effectType == EffectType::AttackDamageChangeDamageCounter
				|| effect.effectType == EffectType::AttackDamageChangePutDamageCounter
				|| effect.effectType == EffectType::AttackDamageChangeRetreatCost
				|| effect.effectType == EffectType::AttackDamageChangeTypeCount
				|| effect.effectType == EffectType::AttackDamageChangeSpecialConditionCount
				|| effect.effectType == EffectType::AttackDamageChangeTakeAttackDamagePreTurn
				|| effect.effectType == EffectType::AttackDamageChangePreTurnTakePrizeCount) {
				calcDamage = true;
				break;
			}
		}
	}
	if (calcDamage) {
		if (attacker.thisTurn.damageChangeMyAttack != 0) {
			for (const Attack* a : attacker.getMaster().attacks) {
				if (attack.attackId == a->attackId) {
					baseDamage += attacker.thisTurn.damageChangeMyAttack;
					break;
				}
			}
		}
		int damage = CalcDamage(state, baseDamage, targetRef, target, attackerRef, attacker, true, &attack);
		if (target.noDamageCoin && damage > 0) {
			state.pushFunction(AttackNoDamageCoin, damage);
			SelectCoin(state, target.playerIndex, 1);
		} else {
			AttackDamage2(state, damage);
		}
	}
}

inline void AttackEffect(State& state) {
	int effectIndex = state.calledCount();
	ActivateEffectLoop(state, effectIndex);
}

inline void AttackEffects(State& state, bool preAttack) {
	if (!preAttack) {
		state.pushFunction(AfterAttack);
		state.postAttackEffect = true;
	}
	if (state.failAttack) {
		return;
	}
	state.changed = false;
	const Attack& attack = state.getAttack();
	const std::vector<Effect>& effects = (preAttack ? attack.preEffects : attack.postEffects);
	if (effects.size() > 0) {
		state.pushFunction(AttackEffect);
		state.setLastFunctionCallCount((int)effects.size());
	}
}

inline void SelectedAttack3(State& state) {
	int playerIndex = state.activePlayerIndex();
	CardRef active = state.getAttacker();
	state.thisTurnHistory().turnAttackId = state.srcAttackId;
	state.thisTurnHistory().turnAttackCard = active;

	EffectState& es = state.effectState;
	es.init();
	es.ability.effectCard = state.makeAreaRef(active);
	es.ability.usePlayerIndex = playerIndex;

	state.pushFunction(AttackEffects, false);
	state.pushFunction(AttackDamage);
	AttackEffects(state, true);
}

inline void SelectedAttackId(State& state, int srcAttackId) {
	if (state.selected.empty()) {
		state.clearSelect();
		AfterAttack(state);
		return;
	}
	SelectOption o = state.firstSelected();
	state.clearSelect();

	state.currentAttackId = o.param0;

	CardRef attacker = state.getAttacker();
	const Card& card = state.getCard(attacker);
	LogAttack(state, card.playerIndex, attacker, state.currentAttackId);

	const Attack& attack = state.getAttack();
	if (!state.satisfyAttackCondition(card, attack, srcAttackId)) {
		AfterAttack(state);
		return;
	}

	SelectedAttack2(state);
}

inline void SpecialAttackProc(State& state) {
	const Attack& srcAttack = state.getAttack();
	if (srcAttack.deckTopAttack) {
		int playerIndex = state.activePlayerIndex();
		PlayerState& ps = state.players[playerIndex];
		if (ps.deck.empty()) {
			AfterAttack(state);
			return;
		}
		CardRef ref = MoveCard(state, playerIndex, AreaType::Deck, ps.deck.size() - 1, AreaType::Trash);
		const Card& card = state.getCard(ref);
		const CardMaster& master = card.getMaster();
		if (master.isNotRulePokemonCard() && master.attacks.size() > 0) {
			state.setSelect(SelectType::Attack, SelectContext::Attack, playerIndex);
			for (const Attack* attack : master.attacks) {
				AddOptionAttack(state, attack->attackId, srcAttack.attackId);
			}
			state.pushFunction(SelectedAttackId, srcAttack.attackId);
		} else {
			AfterAttack(state);
		}
	} else if (srcAttack.asMyBenchNPokemonAttack) {
		const Card& card = state.getCard(state.getAttacker());
		state.setSelect(SelectType::Attack, SelectContext::Attack, card.playerIndex);
		for (CardRef ref : state.players.at(card.playerIndex).bench) {
			const Card& c = state.getCard(ref);
			const CardMaster& m = c.getMaster();
			if (m.n) {
				for (const Attack* a : m.attacks) {
					if (!a->asMyBenchNPokemonAttack) {
						AddOptionAttack(state, a->attackId, srcAttack.attackId);
					}
				}
			}
		}
		if (state.options.size() == 0) {
			state.clearSelect();
			AfterAttack(state);
			return;
		} else {
			state.pushFunction(SelectedAttackId, srcAttack.attackId);
		}
	} else if (srcAttack.asActiveEnemyPokemonAttack || srcAttack.asActiveEnemyPokemonAttackIfCoinHead || srcAttack.asActiveEnemyTerastalPokemonAttack) {
		const Card& card = state.getCard(state.getAttacker());
		const CardMaster& enemy = state.getCard(state.players[1 - card.playerIndex].getActive()).getMaster();
		if (srcAttack.asActiveEnemyTerastalPokemonAttack) {
			if (!enemy.tera) {
				AfterAttack(state);
				return;
			}
		}
		state.setSelect(SelectType::Attack, SelectContext::Attack, card.playerIndex);
		for (const Attack* a : enemy.attacks) {
			if (srcAttack.asActiveEnemyPokemonAttack && a->asActiveEnemyPokemonAttack) {
				continue;
			}
			AddOptionAttack(state, a->attackId, srcAttack.attackId);
		}
		if (state.options.size() == 0) {
			state.clearSelect();
			AfterAttack(state);
			return;
		} else {
			state.pushFunction(SelectedAttackId, srcAttack.attackId);
		}
	} else if (srcAttack.asEnemyDeckTop10Attack) {
		int playerIndex = 1 - state.activePlayerIndex();
		PlayerState& ps = state.players[playerIndex];
		FixedList<const Attack*, 40> attacks;
		for (int i : range(10)) {
			if (ps.deck.empty()) {
				break;
			}
			CardRef ref = MoveCard(state, playerIndex, AreaType::Deck, ps.deck.size() - 1, AreaType::Looking);
			const CardMaster& master = state.getCard(ref).getMaster();
			if (master.isPokemonCard()) {
				for (const Attack* attack : master.attacks) {
					if (!Contains(attacks, attack)) {
						attacks.push_back(attack);
					}
				}
			}
		}

		if (!state.looking.empty()) {
			do {
				MoveCard(state, playerIndex, AreaType::Looking, 0, AreaType::Deck);
			} while (!state.looking.empty());
			ShuffleDeck(state, playerIndex);
		}

		if (attacks.size() > 0) {
			state.setSelect(SelectType::Attack, SelectContext::Attack, state.activePlayerIndex(), 0);
			for (const Attack* attack : attacks) {
				AddOptionAttack(state, attack->attackId, srcAttack.attackId);
			}
			state.pushFunction(SelectedAttackId, srcAttack.attackId);
		} else {
			AfterAttack(state);
		}
	} else if (srcAttack.deckTopSupporter) {
		int playerIndex = state.activePlayerIndex();
		PlayerState& ps = state.players[playerIndex];
		if (ps.deck.empty()) {
			AfterAttack(state);
			return;
		}
		CardRef ref = MoveCard(state, playerIndex, AreaType::Deck, ps.deck.size() - 1, AreaType::Trash);
		const Card& card = state.getCard(ref);
		const CardMaster& master = card.getMaster();
		if (master.cardType == CardType::Supporter) {
			state.pushFunction(AfterAttack);
			ActivateAbilityInfo info(master.play->skillId, state.makeAreaRef(state.attacker), playerIndex);
			state.setActivateAbility(info);
			if (!SatisfyFirstSkillCondition(state)) {
				return;
			}
			ActivateAbility(state);
		} else {
			AfterAttack(state);
		}
	} else {
		SelectedAttack3(state);
	}
}

inline void AttackCoinProc2(State& state) {
	if (state.coinHeadCount > 0) {
		SpecialAttackProc(state);
	} else {
		AfterAttack(state);
	}
}

inline void SelectedAttack2(State& state) {
	state.turnAttackCount++;
	if (state.turnAttackCount > 10000) {
		AfterAttack(state);
		return;
	}
	const Attack& attack = state.getAttack();
	if (attack.asActiveEnemyPokemonAttackIfCoinHead) {
		state.pushFunction(AttackCoinProc2);
		SelectCoin(state, state.activePlayerIndex(), 1);
	} else {
		SpecialAttackProc(state);
	}
}

inline void ConfuseProc(State& state) {
	if (state.coinHeadCount > 0) {
		SelectedAttack2(state);
	} else {
		CardRef active = state.getAttacker();
		AddDamage(state, active, state.getCard(active), 30, false, active);
		AfterAttack(state);
	}
}

inline void PreConfuseProc(State& state) {
	int playerIndex = state.activePlayerIndex();
	const PlayerState& ps = state.players[playerIndex];
	if (ps.badStatus == BadStatusType::Confused && ps.getActive() == state.attacker) {
		state.pushFunction(ConfuseProc);
		SelectCoin(state, playerIndex, 1);
	} else {
		SelectedAttack2(state);
	}
}

inline void AttackCoinProc(State& state, int coinCount) {
	if (state.coinHeadCount >= coinCount) {
		PreConfuseProc(state);
	} else {
		AfterAttack(state);
	}
}

inline void SelectedAttack(State& state, int attackId, int srcAttackId, int benchIndex) {
	PlayerState& ps = state.players[state.activePlayerIndex()];
	CardRef attacker = ps.getActive();
	if (benchIndex >= 0) {
		attacker = ps.bench.at(benchIndex);
	}
	const Card& card = state.getCard(attacker);

	LogAttack(state, card.playerIndex, attacker, attackId);

	if (!state.canAttack(attacker, card)) {
		AfterAttack(state);
		return;
	}

	const Attack& attack = AttackTable.at(attackId);
	if (!state.satisfyAttackCondition(card, attack, srcAttackId)) {
		AfterAttack(state);
		return;
	}

	state.currentAttackId = attackId;
	if (srcAttackId == 0) {
		state.srcAttackId = attackId;
	} else {
		state.srcAttackId = srcAttackId;
	}
	state.attacker = attacker;
	state.postEffectActivate = false;
	state.failAttack = false;
	state.lastAttackDamage = 0;
	state.turnAttackCount = 0;

	if (card.thisTurn.attackCoin2) {
		state.pushFunction(AttackCoinProc, 2);
		SelectCoin(state, state.activePlayerIndex(), 2);
	} else if (card.thisTurn.attackCoin) {
		state.pushFunction(AttackCoinProc, 1);
		SelectCoin(state, state.activePlayerIndex(), 1);
	} else {
		PreConfuseProc(state);
	}
}

inline void SelectedMain(State& state) {
	SelectOption selected = state.firstSelected();
	state.clearSelect();
	if (selected.type == SelectOptionType::End) {
		TurnEnd(state);
	} else if (selected.type == SelectOptionType::Attack) {
		SelectedAttack(state, selected.param0, selected.param1, selected.param2);
	} else {
		state.pushFunction(ToMain);
		switch (selected.type) {
		case SelectOptionType::Play:
			SelectedPlay(state, selected.param0);
			break;
		case SelectOptionType::Attach:
			SelectedAttach(state, selected.param1, (AreaType)selected.param2, selected.param3);
			break;
		case SelectOptionType::Evolve:
			SelectedEvolve(state, selected.param1, (AreaType)selected.param2, selected.param3);
			break;
		case SelectOptionType::Ability:
			SelectedAbility(state, (AreaType)selected.param0, selected.param1);
			break;
		case SelectOptionType::Discard:
			SelectedDiscard(state, (AreaType)selected.param0, selected.param1);
			break;
		case SelectOptionType::Retreat:
			SelectedRetreat(state);
			break;
		default:
			Exception("unexpected option type " + std::to_string((int)selected.type));
			break;
		}
	}
}

inline bool CanRetreat(State& state, int playerIndex, const std::vector<EnergyType>& energyList) {
	if (state.retreated) {
		return false;
	}
	if (std::ssize(energyList) < state.retreatCost(playerIndex)) {
		return false;
	}
	if (!state.canExchangeActive(playerIndex)) {
		return false;
	}

	PlayerState& ps = state.players[playerIndex];
	if (ps.badStatus == BadStatusType::Asleep || ps.badStatus == BadStatusType::Paralyzed) {
		return false;
	}

	const Card& card = state.getCard(ps.getActive());
	if (card.thisTurn.cannotRetreat || card.cannotRetreat) {
		return false;
	}
	if (card.getMaster().pokemonType == PokemonType::PokemonItem) {
		return false;
	}

	if (ps.thisTurn.cannotRetreatPoison && ps.isPoisoned()) {
		if (!card.noEffectEnemySupporter) {
			return false;
		}
	}

	return true;
}

// メイン選択
inline void MainSelect(State& state) {
	if (state.finishCheck()) {
		return;
	}

	if (state.turnEnd) {
		TurnEnd(state);
		return;
	}
	if (state.turnActionCount >= 10000) {
		TurnEnd(state);
		return;
	}
	if (state.turn >= 10000) {
		state.setResult(2, FinishReason::Other);
		return;
	}

	int playerIndex = state.activePlayerIndex();
	state.setSelect(SelectType::Main, SelectContext::Main, playerIndex);
	PlayerState& ps = state.players[playerIndex];

	auto inPlayPokemon = ps.getInPlayPokemon();
	for (int i : range(ps.hand)) {
		CardRef ref = ps.hand[i];
		const Card& card = state.getCard(ref);
		const CardMaster& master = card.getMaster();
		if (master.cardType == CardType::Pokemon) {
			if (ps.cannotPlayAbilityPokemonNotRocket) {
				if (master.ability != nullptr && !master.teamRocket) {
					continue;
				}
			}
			if (master.evolutionType == EvolutionType::Basic) {
				// empty
			} else {
				if (!ps.thisTurn.cannotEvolve) {
					for (const RefPosition& rp : inPlayPokemon) {
						if (state.canEvolve(state, card, master, rp.ref)) {
							AddOptionEvolve(state, AreaType::Hand, i, rp.area, rp.index);
						}
					}
				}
				if (!card.canPlay) {
					continue;
				}
			}
			if (state.remainingBench(ps.playerIndex) <= 0) {
				continue;
			}
		} else if (IsEnergy(master.cardType)) {
			if (state.energyPlayed) {
				continue;
			}
			if (!state.canPlay(ps, card, master)) {
				continue;
			}
			if (ps.thisTurn.cannotPlaySpecialEnergy && master.cardType == CardType::SpecialEnergy) {
				continue;
			}
			for (const RefPosition& rp : inPlayPokemon) {
				const Card& pokemonCard = state.getCard(rp.ref);
				if (pokemonCard.thisTurn.cannotHandAttachEnergy) {
					continue;
				}
				if (state.canAttachEnergy(card, master, rp.ref)) {
					AddOptionAttach(state, AreaType::Hand, i, rp.area, rp.index);
				}
			}
			continue;
		} else {
			if (!state.canPlay(ps, card, master)) {
				continue;
			}
			if (master.cardType == CardType::Supporter) {
				if (state.turn <= 1 && !master.canPlayFirstTurn) {
					continue;
				}
				if (state.supporterPlayed) {
					continue;
				}
				if (ps.thisTurn.cannotPlaySupporter) {
					continue;
				}
				if (!CanPlaySkill(state, ref, card, master)) {
					continue;
				}
			} else if (master.cardType == CardType::Stadium) {
				if (master.cardId == ANGE_FLOETTE) {
					if (state.stadium.empty()) {
						continue;
					}
					if (state.getCard(state.stadium[0]).getMaster().name != u8"プリズムタワー") {
						continue;
					}
				} else {
					if (state.stadiumPlayed) {
						continue;
					}
				}
				if (ps.cannotPlayStadium || ps.thisTurn.cannotPlayStadium) {
					continue;
				}
				if (state.stadium.size() > 0) {
					const Card& stadium = state.getCard(state.stadium[0]);
					if (master.name == stadium.getMaster().name) {
						continue;
					}
				}
			} else if (master.cardType == CardType::Tool) {
				if (ps.cannotPlayTool) {
					continue;
				}
				for (const RefPosition& rp : inPlayPokemon) {
					if (state.canAttachTool(master, rp.ref)) {
						AddOptionAttach(state, AreaType::Hand, i, rp.area, rp.index);
					}
				}
				continue;
			} else {
				if (ps.thisTurn.cannotPlayItem || ps.cannotPlayItem) {
					continue;
				}
				if (!CanPlaySkill(state, ref, card, master)) {
					continue;
				}
			}
		}
		AddOptionPlay(state, i);
	}

	for (const RefPosition& rp : inPlayPokemon) {
		if (CanActivateMainAbility(state, rp.ref, playerIndex)) {
			AddOptionAbility(state, rp.area, rp.index);
		}

		const Card& card = state.getCard(rp.ref);
		if (card.getMaster().canTrash) {
			AddOptionDiscard(state, rp.area, rp.index);
		}
	}
	for (CardRef ref : state.stadium) {
		if (CanActivateMainAbility(state, ref, playerIndex)) {
			AddOptionAbility(state, AreaType::Stadium, 0);
		}
	}

	std::vector<EnergyType>& energyList = state.game->energyList;
	CardRef active = ps.getActive();
	state.getEnergies(playerIndex, active, energyList);
	if (ps.badStatus != BadStatusType::Asleep && ps.badStatus != BadStatusType::Paralyzed) {
		const Card& card = state.getCard(active);
		if (state.canAttack(active, card)) {
			SetAttackEnergy(state, card, energyList, true);
			for (const AttackEnergy& ae : state.game->attackEnergyList) {
				if (ae.insufficientEnergy <= 0) {
					if (CanUseAttack(state, active, card, ae)) {
						if (ae.srcAttackId == 0 || ae.srcAttackId == ae.attack->attackId) {
							if (!state.satisfyAttackCondition(card, *ae.attack, ae.srcAttackId)) {
								continue;
							}
						} else {
							if (!state.satisfyAttackCondition(card, AttackTable.at(ae.srcAttackId), ae.srcAttackId)) {
								continue;
							}
						}
						if (state.turn >= 2 || ae.attack->canUseFirst || card.canAttackFirst) {
							AddOptionAttack(state, ae.attack->attackId, ae.srcAttackId);
						}
					}
				}
			}
		}
	}

	if (state.turn >= 2) {
		for (int i : range(ps.bench)) {
			CardRef ref = ps.bench[i];
			const Card& card = state.getCard(ref);
			const CardMaster& master = card.getMaster();
			for (const Attack* attack : master.attacks) {
				if (attack->canUseBench) {
					std::vector<EnergyType>& energyList2 = state.game->energyList2;
					state.getEnergies(playerIndex, ref, energyList2);
					if (EnoughEnergy(state, attack->energies, energyList2, card, *attack)) {
						AddOptionAttack(state, attack->attackId, attack->attackId, i);
					}
				}
			}
		}
	}

	if (CanRetreat(state, playerIndex, energyList)) {
		AddOptionRetreat(state);
	}

	AddOptionEnd(state);

	state.pushFunction(SelectedMain);
}

inline void ToMain(State& state) {
	state.clearSelect();
	state.pushFunction(MainSelect);
	Refresh(state);
}

// ターン開始
inline void TurnStart(State& state) {
	if (state.finishCheck()) {
		return;
	}
	state.turn++;
	state.turnActionCount = 0;
	int playerIndex = state.activePlayerIndex();
	LogTurnStart(state, playerIndex);
	state.phase = GamePhase::Main;

	state.turnUsedSkill.clear();
	state.turnPlay.clear();
	state.turnHeal.clear();
	state.turnEvolve.clear();
	state.turnHistories[2] = state.turnHistories[1];
	state.turnHistories[1] = state.turnHistories[0];
	state.turnHistories[0] = {};

	for (CardRef ref : state.stadium) {
		state.getCard(ref).turnStart(playerIndex);
	}
	for (int i : state.basicPlayerOrder()) {
		PlayerState& ps = state.players[i];
		ps.turnStart(playerIndex);
		for (CardRef ref : ps.active) {
			state.getCard(ref).turnStart(playerIndex);
		}
		for (CardRef ref : ps.bench) {
			state.getCard(ref).turnStart(playerIndex);
		}
	}

	if (state.players[playerIndex].deck.empty()) {
		// デッキ切れ負け
		state.setResult(playerIndex, FinishReason::Deck0);
		return;
	}
	Draw(state, playerIndex, 1);

	ToMain(state);
}
