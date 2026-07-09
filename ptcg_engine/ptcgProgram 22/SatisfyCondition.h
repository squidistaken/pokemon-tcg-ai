// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "TargetList.h"

inline bool SatisfyCondition(const State& state, const std::vector<Effect>& effects, int effectIndex, CardRef effectCard, int effectOwner) {
	const Effect& effect = effects.at(effectIndex);
	switch (effect.conditionType) {
	case ConditionType::Always: {
		return BoolCompare(true, effect.comparatorType);
	}
	case ConditionType::AnyTargetAfterEffect: {
		std::vector<AreaRef>& targetList = state.game->targetList;
		bool found = false;
		for (int i = effectIndex + 1; i < std::ssize(effects); i++) {
			const Effect& e = effects[i];
			if (!e.isCondition) {
				TargetList(state, e.target, targetList, state.makeAreaRef(effectCard), effectOwner);
				if (targetList.size() > 0) {
					found = true;
					break;
				}
			}
		}
		return BoolCompare(found, effect.comparatorType);
	}
	case ConditionType::CountTarget: {
		std::vector<AreaRef>& targetList = state.game->targetList;
		TargetList(state, effect.target, targetList, state.makeAreaRef(effectCard), effectOwner);
		return Compare((int)targetList.size(), effect.values[0], effect.comparatorType);
	}
	case ConditionType::CountTarget2: {
		std::vector<AreaRef>& targetList = state.game->targetList;
		TargetList(state, effect.target, targetList, state.makeAreaRef(effectCard), effectOwner);
		int count = (int)targetList.size();

		return BoolCompare(effect.values[0] == count || effect.values[1] == count, effect.comparatorType);
	}
	case ConditionType::CountTargetMeOrEnemy: {
		std::vector<AreaRef>& targetList = state.game->targetList;
		Target target = effect.target;

		target.targetPlayer = TargetPlayer::Me;
		TargetList(state, target, targetList, state.makeAreaRef(effectCard), effectOwner);
		int countMe = (int)targetList.size();

		target.targetPlayer = TargetPlayer::Enemy;
		TargetList(state, target, targetList, state.makeAreaRef(effectCard), effectOwner);
		int countEnemy = (int)targetList.size();

		return Compare(countMe, effect.values[0], effect.comparatorType)
			|| Compare(countEnemy, effect.values[1], effect.comparatorType);
	}
	case ConditionType::CompareCountTargetMeEnemy: {
		std::vector<AreaRef>& targetList = state.game->targetList;
		TargetList(state, effect.target, targetList, state.makeAreaRef(effectCard), effectOwner);
		std::array<int, 2> counts = {};
		for (const AreaRef& ref : targetList) {
			const Card& card = state.getCard(ref.card);
			counts[card.playerIndex]++;
		}
		return Compare(counts[effectOwner], counts[1 - effectOwner], effect.comparatorType);
	}
	case ConditionType::CountEnergy: {
		int count = 0;
		for (int i : state.effectTargetPlayerContinual(effect, effectOwner)) {
			auto inPlay = state.players[i].getInPlayPokemon();
			for (const RefPosition& p : inPlay) {
				count += state.energyCount(i, p.ref);
			}
		}
		return Compare(count, effect.values[0], effect.comparatorType);
	}
	case ConditionType::CountEnergyType: {
		EnergyType type = (EnergyType)effect.target.conditions.at(0).val;
		int count = 0;
		for (int i : state.effectTargetPlayerContinual(effect, effectOwner)) {
			auto inPlay = state.players[i].getInPlayPokemon();
			for (const RefPosition& p : inPlay) {
				count += state.typeEnergyCount(i, p.ref, type);
			}
		}
		return Compare(count, effect.values[0], effect.comparatorType);
	}
	case ConditionType::CompareCountEnergyMeEnemy: {
		std::vector<AreaRef>& targetList = state.game->targetList;
		TargetList(state, effect.target, targetList, state.makeAreaRef(effectCard), effectOwner);
		std::array<int, 2> counts = {};
		for (const AreaRef& ref : targetList) {
			const Card& card = state.getCard(ref.card);
			auto rp = state.attachedCardPosition(card);
			EnergyInfo ei = state.getEnergyInfo(card, rp.ref);
			counts[card.playerIndex] += ei.count;
		}
		return Compare(counts[effectOwner], counts[1 - effectOwner], effect.comparatorType);
	}
	case ConditionType::AttackEnergyExtra: {
		const Card& attacker = state.getCard(state.attacker);
		const Attack& attack = AttackTable.at(state.srcAttackId);
		state.getEnergies(attacker.playerIndex, state.attacker, state.game->energyList);
		int extra = InsufficientEnergyCount(state, attack.energies, state.game->energyList, attacker, attack, true);
		return Compare(-extra, effect.values[0], effect.comparatorType);
	}
	case ConditionType::NotFullBench: {
		int playerIndex = (effect.target.targetPlayer == TargetPlayer::Me ? effectOwner : (1 - effectOwner));
		return BoolCompare(state.remainingBench(playerIndex) >= 1, effect.comparatorType);
	}
	case ConditionType::MyTurn:
		if (state.isPlayerTurn()) {
			const Card& card = state.getCard(effectCard);
			return BoolCompare(card.playerIndex == state.activePlayerIndex(), effect.comparatorType);
		} else {
			return false;
		}
	case ConditionType::Turn:
		return Compare(state.turn, effect.values[0], effect.comparatorType);
	case ConditionType::KoPreEnemyTurn:
		return BoolCompare(state.turnHistories[1].ko, effect.comparatorType);
	case ConditionType::KoPreEnemyTurnTeamRocket:
		return BoolCompare(state.turnHistories[1].koTeamRocket, effect.comparatorType);
	case ConditionType::KoAttackDamagePreEnemyTurn:
		return BoolCompare(state.turnHistories[1].koAttackDamage, effect.comparatorType);
	case ConditionType::KoAttackDamageEthanPreEnemyTurn:
		return BoolCompare(state.turnHistories[1].koAttackDamageEthan, effect.comparatorType);
	case ConditionType::KoAttackDamageHopPreEnemyTurn:
		return BoolCompare(state.turnHistories[1].koAttackDamageHop, effect.comparatorType);
	case ConditionType::NoSameNameSkillThisTurn: {
		bool notUsed = true;
		const Skill& skill = SkillTable.at(effect.skillId);
		for (int id : state.turnUsedSkill) {
			if (id == skill.skillId) {
				notUsed = false;
				break;
			}
		}
		return BoolCompare(notUsed, effect.comparatorType);
	}
	case ConditionType::SameAttackPreMyTurn:
		return BoolCompare(state.turnHistories[2].turnAttackId == state.currentAttackId && state.turnHistories[2].turnAttackCard == state.attacker, effect.comparatorType);
	case ConditionType::CoinHeadCount:
		return Compare(state.coinHeadCount, effect.values[0], effect.comparatorType);
	case ConditionType::AttachActive:
		return BoolCompare(state.attachActive, effect.comparatorType);
	case ConditionType::MysteryGarden: {
		int count = 0;
		const PlayerState& ps = state.players[effectOwner];
		auto inPlay = ps.getInPlayPokemon();
		for (RefPosition& rp : inPlay) {
			const Card& card = state.getCard(rp.ref);
			if (ContainsEnergyType(state.getEnergyType(card), EnergyType::Psychic)) {
				count++;
			}
		}
		return ps.hand.size() <= count;
	}
	case ConditionType::LoveBall: {
		const PlayerState& ps = state.players[effectOwner];
		const PlayerState& es = state.players[1 - effectOwner];
		for (const RefPosition& rp : es.getInPlayPokemon()) {
			int count = 0;
			const CardMaster& master = state.getCard(rp.ref).getMaster();
			if (master.cardType != CardType::Pokemon) {
				continue;
			}
			for (CardRef ref : ps.active) {
				if (state.getCard(ref).getMaster().name == master.name) {
					count++;
				}
			}
			for (CardRef ref : ps.bench) {
				if (state.getCard(ref).getMaster().name == master.name) {
					count++;
				}
			}
			for (CardRef ref : ps.trash) {
				if (state.getCard(ref).getMaster().name == master.name) {
					count++;
				}
			}
			if (count < 4) {
				return true;
			}
		}
		return false;
	}
	default:
		assert(false);
		return false;
	}
}

inline bool SatisfySkillCondition(const State& state, const Skill& skill, CardRef effectCard, int effectOwner, int startIndex = 0) {
	for (int i = startIndex; i < std::ssize(skill.effects); i++) {
		const Effect& effect = skill.effects[i];
		if (!effect.isCondition) {
			break;
		}
		if (!SatisfyCondition(state, skill.effects, i, effectCard, effectOwner)) {
			if (effect.failSkip) {
				break;
			}
			return false;
		}
	}
	if (skill.onceTurn) {
		const Card& card = state.getCard(effectCard);
		if (Contains(card.abilityUsed, card.getMaster().cardId)) {
			return false;
		}
	}
	return true;
}

inline bool CanPlaySkill(const State& state, CardRef ref, const Card& card, const CardMaster& master) {
	const Skill& skill = *master.play;
	if (skill.secondEffectStartIndex > 0) {
		return true;
	}
	if (!SatisfySkillCondition(state, skill, ref, card.playerIndex)) {
		return false;
	}
	return true;
}

inline bool CanActivateMainAbility(const State& state, CardRef ref, int playerIndex) {
	const Card& card = state.getCard(ref);
	const CardMaster& master = card.getMaster();
	const Skill* skill = state.getAbility(card, master);
	if (skill == nullptr) {
		return false;
	}

	if (!skill->mainAbility) {
		return false;
	}

	if (!skill->isAreaMatch(card.area)) {
		return false;
	}

	if (!SatisfySkillCondition(state, *skill, ref, playerIndex)) {
		return false;
	}
	return true;
}

inline bool SatisfyAttackCondition(const State& state, const Attack& attack, CardRef attackerRef, const Card& attacker) {
	if (attack.preEffects.size() > 0 && attack.postEffects.size() > 0) {
		return true;
	}
	for (int i : range(attack.postEffects)) {
		const Effect& effect = attack.postEffects[i];
		if (!effect.isCondition) {
			break;
		}
		if (effect.failSkip) {
			break;
		}
		if (!SatisfyCondition(state, attack.postEffects, i, attackerRef, attacker.playerIndex)) {
			return false;
		}
	}
	return true;
}
