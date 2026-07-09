// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "SelectProc.h"

inline bool IsNoEffectActive(const State& state, int playerIndex, CardRef effectCardRef) {
	const PlayerState& ps = state.players[playerIndex];
	if (ps.active.size() > 0) {
		if (state.getCard(effectCardRef).area != AreaType::Stadium) {
			if (state.getCard(ps.active[0]).noEnemyAbility) {
				return true;
			}
		}
	}
	return false;
}

inline void EffectContinual(State& state, const Effect& effect, const std::vector<AreaRef>& targetList, int effectPlayerIndex, CardRef effectCardRef) {
	switch (effect.effectType) {
	case EffectType::ContinualEffectSeparator:
		break;
	case EffectType::MaxHpChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.hpChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::MaxHpChangeFighting:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			state.getEnergies(card.playerIndex, ref.card, state.game->energyList);
			int change = 0;
			for (EnergyType type : state.game->energyList) {
				if (ContainsEnergyType(type, EnergyType::Fighting)) {
					change += state.getValueByEffect(effect);
				}
			}
			ClampShort(card.hpChange, change);
		}
		break;
	case EffectType::DamageChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.damageChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::DamageChangeActive:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.damageChangeActive, state.getValueByEffect(effect));
		}
		break;
	case EffectType::DamageChangeEx:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.damageChangeEx, state.getValueByEffect(effect));
		}
		break;
	case EffectType::DamageChangeAbility:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.damageChangeAbility, state.getValueByEffect(effect));
		}
		break;
	case EffectType::DamageChangeEvolved:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.damageChangeEvolved, state.getValueByEffect(effect));
		}
		break;
	case EffectType::DamageChangeEnemyTakenPrize:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.damageChangeEnemyTakenPrize, state.getValueByEffect(effect));
		}
		break;
	case EffectType::TakeDamageChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.takeDamageChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::TakeEnemyAttackDamageChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.takeEnemyAttackDamageChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::TakeEnemyAbilityPokemonAttackDamageChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.takeEnemyAbilityPokemonAttackDamageChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::TakeEnemyFireOrWaterPokemonAttackDamageChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.takeEnemyFireOrWaterPokemonAttackDamageChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::TakeEnemy4TypePokemonAttackDamageChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			ClampShort(card.takeEnemy4TypePokemonAttackDamageChange, state.getValueByEffect(effect));
		}
		break;
	case EffectType::NoDamageGreaterEqual:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageGreaterEqual = state.getValueByEffect(effect);
		}
		break;
	case EffectType::RetreatCostChange:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			int val = card.retreatCostChange + state.getValueByEffect(effect);
			card.retreatCostChange = std::clamp(val, -100, 100);
		}
		break;
	case EffectType::AttackCostChangeColorless:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			int val = card.attackCostChangeColorless + state.getValueByEffect(effect);
			card.attackCostChangeColorless = std::clamp(val, -100, 100);
		}
		break;
	case EffectType::AttackCostDown:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			int val = card.attackCostDown + state.getValueByEffect(effect);
			card.attackCostDown = std::clamp(val, 0, 100);
		}
		break;
	case EffectType::AttackCostDownColorlessTargetCount:
	{
		Card& card = state.getCard(effectCardRef);
		int val = card.attackCostChangeColorless - (int)targetList.size();
		card.attackCostChangeColorless = std::clamp(val, -100, 100);
	}
	break;
	case EffectType::AttackCostDownColorlessOwnAttackEnemyTakenPrize:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			int prize = state.players.at(1 - effectPlayerIndex).prize.size();
			int val = card.attackCostDownColorlessOwnAttack + PRIZE_SIZE - prize;
			card.attackCostDownColorlessOwnAttack = std::clamp(val, 0, 100);
		}
		break;
	case EffectType::AddEnergyType:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			int val = state.getValueByEffect(effect);
			card.typeIndex = val;
		}
		break;
	case EffectType::SetWeakness:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			int val = state.getValueByEffect(effect);
			card.weaknessIndex = val;
		}
		break;
	case EffectType::NoAbility:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noAbility = true;
			for (int i = 0; i < state.currentCardEffectIndex; i++) {
				CardEffect& ce = state.game->cardEffectList.at(i);
				if (ce.ref == ref.card) {
					card.skillOrder = state.getCard(state.game->cardEffectList.at(state.currentCardEffectIndex).ref).skillOrder + 1;
					state.updateOrder = true;
				}
			}
		}
		break;
	case EffectType::NoKoMeAbility:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noKoMeAbility = true;
		}
		break;
	case EffectType::NoDamageEnemyAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageEnemyAttack = true;
		}
		break;
	case EffectType::NoDamageEnemyAbilityPokemonAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageEnemyAbilityPokemonAttack = true;
		}
		break;
	case EffectType::NoDamageEnemyExAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageEnemyExAttack = true;
		}
		break;
	case EffectType::NoDamageEnemyBasicExAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageEnemyBasicExAttack = true;
		}
		break;
	case EffectType::NoDamageAndEffectEnemyTerastalAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageAndEffectEnemyTerastalAttack = true;
		}
		break;
	case EffectType::NoDamageAndEffectEnemySpecialEnergyAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageAndEffectEnemySpecialEnergyAttack = true;
		}
		break;
	case EffectType::NoEffectEnemyAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noEffectEnemyAttack = true;
		}
		break;
	case EffectType::NoDamageAndEffectEnemyAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageEnemyAttack = true;
			card.noEffectEnemyAttack = true;
		}
		break;
	case EffectType::NoEffectEnemyItem:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noEffectEnemyItem = true;
		}
		break;
	case EffectType::NoEffectEnemySupporter:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noEffectEnemySupporter = true;
		}
		break;
	case EffectType::NoDamageCounterEnemyAttackAbility:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageCounterEnemyAttackAbility = true;
		}
		break;
	case EffectType::NoEnemyAbility:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noEnemyAbility = true;
		}
		break;
	case EffectType::NoSpecialCondition:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noSpecialCondition = true;
		}
		break;
	case EffectType::NoSleepParalyzeConfuse:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noSleepParalyzeConfuse = true;
		}
		break;
	case EffectType::NoSleep:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noSleep = true;
		}
		break;
	case EffectType::NoRetreatCost:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noRetreatCost = true;
		}
		break;
	case EffectType::NoPrizeEx:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noPrizeEx = true;
		}
		break;
	case EffectType::NotRecoverConfuseEvolve:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.notRecoverConfuseEvolve = true;
		}
		break;
	case EffectType::CanUsePreEvolutionAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.canUsePreEvolutionAttack = true;
		}
		break;
	case EffectType::CanEvolveAppearTurn:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.canEvolveAppearTurn = true;
		}
		break;
	case EffectType::CanEvolveGrassAppearTurn:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.canEvolveGrassAppearTurn = true;
		}
		break;
	case EffectType::CanAttackFirst:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.canAttackFirst = true;
		}
		break;
	case EffectType::CannotRetreat:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.cannotRetreat = true;
		}
		break;
	case EffectType::CannotAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.cannotAttack = true;
		}
		break;
	case EffectType::CannotToHand:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.cannotToHand = true;
		}
		break;
	case EffectType::CannotMoveDamageCounter:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.cannotMoveDamageCounter = true;
		}
		break;
	case EffectType::AttackEnergyColoressOne:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.attackEnergyColoressOne = true;
		}
		break;
	case EffectType::AttackEnergyPsychicOne:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.attackEnergyPsychicOne = true;
		}
		break;
	case EffectType::DoubleGrassEnergy:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.doubleGrassEnergy = true;
		}
		break;
	case EffectType::NoDamageCoin:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.noDamageCoin = true;
		}
		break;
	case EffectType::KoByDamageToHand:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.koByDamageToHand = true;
		}
		break;
	case EffectType::BasicPrizePlus1:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.basicPrizePlus1 = true;
		}
		break;
	case EffectType::DoubleAttack:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.doubleAttack = true;
		}
		break;
	case EffectType::Tool2:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.tool2 = true;
		}
		break;
	case EffectType::Tool4:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.tool4 = true;
		}
		break;
	case EffectType::TechnicalMachine:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.technicalMachine = true;
		}
		break;
	case EffectType::SpecialFlagTool:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.specialFlagTool = true;
		}
		break;
	case EffectType::RainbowDna:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.rainbowDna = true;
		}
		break;
	case EffectType::CanPlay:
		for (const AreaRef& ref : targetList) {
			Card& card = state.getCard(ref.card);
			card.canPlay = true;
		}
		break;
	case EffectType::PoisonDamageChange:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			if (IsNoEffectActive(state, i, effectCardRef)) {
				continue;
			}
			PlayerState& ps = state.players[i];
			ps.poisonDamageChange += state.getValueByEffect(effect);
		}
		break;
	case EffectType::BurnDamageChange:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			if (IsNoEffectActive(state, i, effectCardRef)) {
				continue;
			}
			PlayerState& ps = state.players[i];
			ps.burnDamageChange += state.getValueByEffect(effect);
		}
		break;
	case EffectType::PoisonDamageChangeNotDarkness:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			if (IsNoEffectActive(state, i, effectCardRef)) {
				continue;
			}
			PlayerState& ps = state.players[i];
			ps.poisonDamageChangeNotDarkness += state.getValueByEffect(effect);
		}
		break;
	case EffectType::BenchCapacity:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			int val = state.getValueByEffect(effect);
			PlayerState& ps = state.players[i];
			if (ps.benchCapacity == 0 || ps.benchCapacity > val) {
				ps.benchCapacity = val;
			}
		}
		break;
	case EffectType::CannotPlayItem:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			PlayerState& ps = state.players[i];
			ps.cannotPlayItem = true;
		}
		break;
	case EffectType::CannotPlayStadium:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			PlayerState& ps = state.players[i];
			ps.cannotPlayStadium = true;
		}
		break;
	case EffectType::CannotPlayTool:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			PlayerState& ps = state.players[i];
			ps.cannotPlayTool = true;
		}
		break;
	case EffectType::CannotPlayAceSpec:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			PlayerState& ps = state.players[i];
			ps.cannotPlayAceSpec = true;
		}
		break;
	case EffectType::CannotPlayAbilityPokemonNotRocket:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			PlayerState& ps = state.players[i];
			ps.cannotPlayAbilityPokemonNotRocket = true;
		}
		break;
	case EffectType::CannotTrashToHandAbilityOrTrainers:
		for (int i : state.effectTargetPlayerContinual(effect, effectPlayerIndex)) {
			PlayerState& ps = state.players[i];
			ps.cannotTrashToHandAbilityOrTrainers = true;
		}
		break;
	case EffectType::NoToolEffect:
		state.noToolEffect = true;
		break;
	default:
		assert(false);
		break;
	}
}



