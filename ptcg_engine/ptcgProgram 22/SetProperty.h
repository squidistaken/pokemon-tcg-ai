// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "AddLog.h"
#include "AddOption.h"
#include "GameUtil.h"
#include "PullTrigger.h"

inline CardRef MoveCardBase(State& state, int playerIndex, AreaType fromArea, int fromIndex, AreaType toArea);

inline void ClampShort(short& var, int value) {
	int val = var + value;
	var = std::clamp(val, -30000, 30000);
}

inline void SetYesNoSelect(State& state, SelectContext selectContext, int selectPlayer) {
	state.setSelect(SelectType::YesNo, selectContext, selectPlayer);
	AddOptionYesAndNo(state);
}

inline void EffectPoison(State& state, int playerIndex, int damageCounter = 1) {
	if (state.isPreventEffectActive(playerIndex)) {
		return;
	}
	PlayerState& ps = state.players[playerIndex];
	const Card& card = state.getCard(ps.getActive());
	if (state.noSpecialCondition(card)) {
		return;
	}
	if (ps.poisonDamageCounter == damageCounter) {
		return;
	}
	LogPoisoned(state, card.playerIndex, false, ps.getActive());
	state.changed = true;
	ps.poisonDamageCounter = damageCounter;
}

inline void EffectBurn(State& state, int playerIndex) {
	if (state.isPreventEffectActive(playerIndex)) {
		return;
	}
	PlayerState& ps = state.players[playerIndex];
	const Card& card = state.getCard(ps.getActive());
	if (state.noSpecialCondition(card)) {
		return;
	}
	if (ps.burned) {
		return;
	}
	LogBurned(state, card.playerIndex, false, ps.getActive());
	state.changed = true;
	ps.burned = true;
}

inline void EffectSleep(State& state, int playerIndex) {
	if (state.isPreventEffectActive(playerIndex)) {
		return;
	}
	PlayerState& ps = state.players[playerIndex];
	const Card& card = state.getCard(ps.getActive());
	if (state.noSpecialCondition(card) || card.noSleepParalyzeConfuse || card.noSleep) {
		return;
	}
	if (ps.badStatus == BadStatusType::Asleep) {
		return;
	}
	LogAsleep(state, card.playerIndex, false, ps.getActive());
	state.changed = true;
	ps.badStatus = BadStatusType::Asleep;
}

inline void EffectParalyze(State& state, int playerIndex) {
	if (state.isPreventEffectActive(playerIndex)) {
		return;
	}
	PlayerState& ps = state.players[playerIndex];
	const Card& card = state.getCard(ps.getActive());
	if (state.noSpecialCondition(card) || card.noSleepParalyzeConfuse) {
		return;
	}
	if (ps.badStatus == BadStatusType::Paralyzed) {
		return;
	}
	LogParalyzed(state, card.playerIndex, false, ps.getActive());
	state.changed = true;
	ps.badStatus = BadStatusType::Paralyzed;
}

inline void EffectConfuse(State& state, int playerIndex) {
	if (state.isPreventEffectActive(playerIndex)) {
		return;
	}
	PlayerState& ps = state.players[playerIndex];
	const Card& card = state.getCard(ps.getActive());
	if (state.noSpecialCondition(card) || card.noSleepParalyzeConfuse) {
		return;
	}
	if (ps.badStatus == BadStatusType::Confused) {
		return;
	}
	LogConfused(state, card.playerIndex, false, ps.getActive());
	state.changed = true;
	ps.badStatus = BadStatusType::Confused;
}

inline void ClearSleepParalyzeConfuse(State& state, int playerIndex) {
	PlayerState& ps = state.players.at(playerIndex);
	if (ps.badStatus != BadStatusType::None) {
		if (ps.badStatus == BadStatusType::Asleep) {
			LogAsleep(state, playerIndex, true, ps.getActive());
		} else if (ps.badStatus == BadStatusType::Paralyzed) {
			LogParalyzed(state, playerIndex, true, ps.getActive());
		} else if (ps.badStatus == BadStatusType::Confused) {
			LogConfused(state, playerIndex, true, ps.getActive());
		}
		ps.badStatus = BadStatusType::None;
	}
}

inline void ClearSpecialCondition(State& state, int playerIndex) {
	ClearSleepParalyzeConfuse(state, playerIndex);
	PlayerState& ps = state.players.at(playerIndex);
	if (ps.isPoisoned()) {
		ps.clearPoison();
		LogPoisoned(state, playerIndex, true, ps.getActive());
	}
	if (ps.isBurned()) {
		ps.burned = false;
		LogBurned(state, playerIndex, true, ps.getActive());
	}
}

inline void AddDamage(State& state, CardRef ref, Card& card, int damage, bool isAttackDamage, CardRef cause, bool putDamageCounter = false, const Attack* attack = nullptr) {
	if (damage > 0) {
		int maxHp = state.getMaxHp(card);
		if (card.damage >= maxHp) {
			// すでにHP0
			card.damage += damage;
		} else {
			const Card& causeCard = state.getCard(cause);
			bool isEnemyAttackDamage = (isAttackDamage && causeCard.playerIndex != card.playerIndex);
			if (isEnemyAttackDamage) {
				if (card.damage == 0) {
					if (damage >= maxHp) {
						card.koFull = true;
					}
				}
			}
			card.damage += damage;
			if (card.damage >= maxHp) {
				card.damage = maxHp;
				card.ko = true;
				card.koCauseRef = cause.cardIndex;
				if (isAttackDamage) {
					card.koAttackDamage = true;
				}
				if (isEnemyAttackDamage) {
					const CardMaster& master = causeCard.getMaster();
					card.koEnemyAttackDamage = true;
					if (card.area == AreaType::Active) {
						card.koEnemyAttackDamageActive = true;
					}
					if (master.isEx()) {
						card.koEnemyExAttackDamage = true;
					}
					if (master.tera) {
						card.koEnemyTerastalAttackDamage = true;
					}
					if (master.n) {
						card.koEnemyNAttackDamage = true;
					}
					if (causeCard.basicPrizePlus1 && card.getMaster().evolutionType == EvolutionType::Basic) {
						card.koPrizePlus1 = true;
					}
					if (card.noPrizeEx && master.isEx()) {
						card.koPrizeZero = true;
					}
				}
				if (attack != nullptr) {
					if (attack->prizePlus1) {
						card.koPrizePlus1 = true;
					}
					if (attack->koNoDamageAndEffectAttackNextEnemyTurn) {
						card.koNoDamageAndEffectAttackNextEnemyTurn = true;
					}
				}
			}

			if (isAttackDamage) {
				card.takeAttackDamageThisTurn += damage;
			}

			if (isEnemyAttackDamage) {
				PullTrigger(state, TriggerType::DamagedEnemyAttack, ref, cause, 1);
				if (card.area == AreaType::Active) {
					PullTrigger(state, TriggerType::DamagedEnemyAttackActive, ref, cause, 1);
					for (int i = (int)state.delayTriggerStack.size() - 1; i >= 0; i--) {
						TriggeredAbility ta = state.delayTriggerStack[i];
						if (ta.trigger.type == TriggerType::DamagedEnemyAttackActive && ta.trigger.subject.card == ref) {
							ta.trigger.object = state.makeAreaRef(cause);
							ta.trigger.depth = 1;
							ta.trigger.value = damage;
							state.temporaryTriggerStack.push_back(ta);
						}
					}
				}
			}
		}
	} else {
		damage = 0;
	}

	LogHpChange(state, card.playerIndex, ref, -damage, putDamageCounter);
}

inline void AfterDamage(State& state, CardRef ref, Card& card, CardRef causeRef, const Card& causeCard) {
	if (card.specialFlagTool) {
		EnergyType attackerType = state.getEnergyType(causeCard);
		PlayerState& ps = state.players[card.playerIndex];
		auto tools = state.getAttachedTools(card);
		for (const AttachedToolEnergy& t : tools) {
			auto& name = t.name();
			if (name == u8"ウタンのみ") {
				if (ContainsEnergyType(attackerType, EnergyType::Psychic)) {
					int index = state.currentAreaIndex(ps.tool, t.ref);
					MoveCardBase(state, card.playerIndex, AreaType::Tool, index, AreaType::Trash);
				}
			} else if (name == u8"ハバンのみ") {
				if (ContainsEnergyType(attackerType, EnergyType::Dragon)) {
					int index = state.currentAreaIndex(ps.tool, t.ref);
					MoveCardBase(state, card.playerIndex, AreaType::Tool, index, AreaType::Trash);
				}
			}
		}
	}
}

inline int Heal(State& state, CardRef ref, int heal, bool isHeal) {
	Card& card = state.getCard(ref);
	int srcDamage = card.damage;
	card.damage -= heal;
	if (card.damage < 0) {
		card.damage = 0;
	}
	int healed = srcDamage - card.damage;
	if (healed >= 0) {
		LogHpChange(state, card.playerIndex, ref, healed, false);
		if (isHeal && healed > 0) {
			state.turnHeal.push_back(ref);
		}
	}
	return healed;
}

// ワザのダメージ
inline int CalcDamage(const State& state, int baseDamage, CardRef targetRef, const Card& target, CardRef attackerRef, const Card& attacker, bool calcWeakness, const Attack* attack) {
	int damage = baseDamage;
	if (damage <= 0) {
		return 0;
	}

	// 2. ダメージを与えるポケモンが受けている効果
	const PlayerState& ps = state.players.at(attacker.playerIndex);
	const CardMaster& targetMaster = target.getMaster();
	EnergyType attackerType = state.getEnergyType(attacker);
	damage += attacker.thisTurn.damageChange;
	damage += attacker.damageChange;
	if (target.area == AreaType::Active) {
		damage += attacker.thisTurn.damageChangeActive;
	}
	if (target.area == AreaType::Active && attacker.playerIndex != target.playerIndex) {
		damage += attacker.damageChangeActive;
		damage += attacker.damageChangeThisTurn;
		damage += attacker.damageChangeEnemyTakenPrize * state.takenPrizeCount(1 - attacker.playerIndex);
		damage += ps.playerDamageChange;
		if (targetMaster.isEx()) {
			damage += attacker.damageChangeEx;
			damage += attacker.damageChangeExThisTurn;
			damage += ps.playerDamageChangeEx;
		}
		if (ContainsEnergyType(attackerType, EnergyType::Fighting)) {
			damage += ps.playerDamageChangeMyFighting;
		}
		if (attacker.damageChangeAbility) {
			const Skill* ability = state.getAbility(target, targetMaster);
			if (ability != nullptr) {
				damage += attacker.damageChangeAbility;
			}
		}
		if (attacker.damageChangeEvolved) {
			if (targetMaster.evolutionType != EvolutionType::Basic) {
				damage += attacker.damageChangeEvolved;
			}
		}
	}

	if (damage <= 0) {
		return 0;
	}

	bool calcResistance = calcWeakness;
	bool calcTargetEffect = true;
	if (attack != nullptr) {
		if (attack->noTargetEffect) {
			calcTargetEffect = false;
		}
		if (attack->noTargetWeakness) {
			calcWeakness = false;
		}
		if (attack->noTargetResistance) {
			calcResistance = false;
		}
	}

	if (calcWeakness) {
		// 3. 弱点の計算
		if (!attack->noTargetWeakness) {
			EnergyType weakness = state.getWeakness(target, targetMaster);
			if (ContainsEnergyType(weakness, attackerType)) {
				damage *= 2;
			}
		}
	}
	if (calcResistance) {
		if (!attack->noTargetResistance) {
			// 4. 抵抗力の計算
			EnergyType resistance = targetMaster.resistance;
			if (ContainsEnergyType(resistance, attackerType)) {
				damage -= 30;
				if (damage <= 0) {
					return 0;
				}
			}
		}
	}

	const CardMaster& attackerMaster = attacker.getMaster();

	// 5. ダメージを受けるポケモンが受けている効果
	if (calcTargetEffect) {
		damage += target.takeDamageChange;
		damage += target.takeDamageChangeNextEnemyTurn;
		damage += target.thisTurnEnemy.takeDamageChange;

		const Skill* attackerAbility = state.getAbility(attacker, attackerMaster);
		if (attacker.playerIndex != target.playerIndex) {
			damage += target.takeEnemyAttackDamageChange;

			if (ContainsEnergyType(attackerType, EnergyType::Fire | EnergyType::Water)) {
				damage += target.takeEnemyFireOrWaterPokemonAttackDamageChange;
			}
			if (ContainsEnergyType(attackerType, EnergyType::Fire | EnergyType::Water | EnergyType::Grass | EnergyType::Lightning)) {
				damage += target.takeEnemy4TypePokemonAttackDamageChange;
			}

			if (attackerAbility != nullptr) {
				damage += target.takeEnemyAbilityPokemonAttackDamageChange;
			}

			if (target.specialFlagTool) {
				auto tools = state.getAttachedTools(target);
				for (const AttachedToolEnergy& t : tools) {
					auto& name = t.name();
					if (name == u8"ウタンのみ") {
						if (ContainsEnergyType(attackerType, EnergyType::Psychic)) {
							damage -= 60;
						}
					} else if (name == u8"ハバンのみ") {
						if (ContainsEnergyType(attackerType, EnergyType::Dragon)) {
							damage -= 60;
						}
					}
				}
			}

			if (ps.thisTurn.metalDamageChange) {
				EnergyType targetType = state.getEnergyType(target);
				if (ContainsEnergyType(targetType, EnergyType::Metal)) {
					damage += ps.thisTurn.metalDamageChange;
				}
			}

			if (target.noDamageEnemyAbilityPokemonAttack) {
				if (attackerAbility != nullptr) {
					damage = 0;
				}
			}
			if (target.noDamageEnemyExAttack) {
				if (attackerMaster.isEx()) {
					damage = 0;
				}
			}
			if (target.noDamageEnemyBasicExAttack) {
				if (attackerMaster.evolutionType == EvolutionType::Basic && attackerMaster.isEx()) {
					damage = 0;
				}
			}
			if (target.noDamageAndEffectEnemyTerastalAttack) {
				if (attackerMaster.tera) {
					damage = 0;
				}
			}
			if (target.noDamageAndEffectEnemySpecialEnergyAttack) {
				if (state.isAttachedSpecialEnergy(attackerRef)) {
					damage = 0;
				}
			}
			if (target.noDamageAndEffectEnemyExAttackNextEnemyTurn) {
				if (attackerMaster.isEx()) {
					damage = 0;
				}
			}
			if (target.noDamageEnemyAttack || target.noDamageAndEffectEnemyAttackNextEnemyTurn) {
				damage = 0;
			}
		}
		if (target.noDamageAttackNextEnemyTurn || target.noDamageAndEffectAttackNextEnemyTurn) {
			damage = 0;
		}
		if (target.noDamageBasicAttackNextEnemyTurn) {
			if (attackerMaster.evolutionType == EvolutionType::Basic) {
				damage = 0;
			}
		}
		if (target.noDamageBasicColorAttackNextEnemyTurn) {
			if (attackerMaster.evolutionType == EvolutionType::Basic && attackerType != EnergyType::Colorless) {
				damage = 0;
			}
		}
		if (target.noDamageAbilityAttackNextEnemyTurn) {
			if (attackerAbility != nullptr) {
				damage = 0;
			}
		}
		if (damage <= target.noDamageLessEqualAttackNextEnemyTurn) {
			damage = 0;
		}

		if (target.area == AreaType::Bench) {
			if (targetMaster.tera) {
				damage = 0;
			}
		}


		if (target.noDamageGreaterEqual > 0) {
			if (target.noDamageGreaterEqual <= damage) {
				damage = 0;
			}
		}
	}

	if (damage <= 0) {
		return 0;
	}

	damage = std::min(damage, 100000000);
	return damage;
}
