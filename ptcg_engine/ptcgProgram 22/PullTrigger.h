// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "SatisfyCondition.h"


inline bool IsTriggerTarget(const State& state, CardRef ref, const Target& target, CardRef effectCardRef, const Card& effectCard) {
	if (target.targetPlayer == TargetPlayer::None) {
		return true;
	} else {
		if (ref.isNull()) {
			return false;
		}
		const Card& card = state.getCard(ref);
		if (target.targetPlayer != TargetPlayer::Both) {
			if (target.targetPlayer == TargetPlayer::Me) {
				if (card.playerIndex != effectCard.playerIndex) {
					return false;
				}
			} else {
				if (card.playerIndex == effectCard.playerIndex) {
					return false;
				}
			}
		}
		bool match = false;
		for (AreaType area : target.areas) {
			if (area == AreaType::Me) {
				if (card.moveCounter == effectCard.moveCounter) {
					match = true;
					break;
				}
			} else if (area == AreaType::Attach) {
				assert(effectCard.attachMoveCounter > 0);
				if (card.moveCounter == effectCard.attachMoveCounter) {
					match = true;
					break;
				}
			} else if (area == AreaType::All || card.area == area) {
				match = true;
				break;
			}
		}
		if (match) {
			return IsTarget(state, ref, target, state.makeAreaRef(effectCardRef));
		} else {
			return false;
		}
	}
}

inline void TriggerList(State& state, TriggerType triggerType, CardRef subject, CardRef object, CardRef effectCard, int depth) {
	const Card& card = state.getCard(effectCard);
	const CardMaster& master = card.getMaster();
	const Skill* ability = state.getAbility(card, master);
	if (ability == nullptr) {
		return;
	}

	const Skill& skill = *ability;
	if (!skill.hasTrigger()) {
		return;
	}
	if (!skill.isAreaMatch(card.area)) {
		return;
	}

	std::vector<TriggeredAbility>& triggerStack = state.temporaryTriggerStack;
	for (const Trigger& trigger : skill.triggers) {
		assert(trigger.triggerType != TriggerType::None);
		if (trigger.triggerType != triggerType) {
			continue;
		}
		if (!IsTriggerTarget(state, subject, trigger.subject, effectCard, card)) {
			continue;
		}


		TriggeredAbility ta = {};
		ta.trigger.type = trigger.triggerType;
		ta.trigger.depth = depth;
		if (!subject.isNull()) {
			ta.trigger.subject = state.makeAreaRef(subject);
		}
		if (!object.isNull()) {
			ta.trigger.object = state.makeAreaRef(object);
		}

		if (skill.effects.size() > 0) {
			bool satisfy = true;
			TriggerInfo preInfo = state.triggerInfo;
			state.triggerInfo = ta.trigger;
			const Effect& effect = skill.effects[0];
			if (effect.isCondition) {
				ConditionType type = effect.conditionType;
				if (type == ConditionType::MyTurn || type == ConditionType::Turn || type == ConditionType::KoPreEnemyTurn || type == ConditionType::KoAttackDamagePreEnemyTurn) {
					if (!SatisfyCondition(state, skill.effects, 0, effectCard, card.playerIndex)) {
						satisfy = false;
					}
				}
			}
			state.triggerInfo = preInfo;
			if (!satisfy) {
				break;
			}
		}

		if (skill.onceTurn) {
			AreaRef ref = state.makeAreaRef(effectCard);
			bool found = false;
			for (TriggeredAbility& ta : state.triggerStack) {
				if (ta.activateInfo.skillId == skill.skillId && ta.activateInfo.effectCard == ref) {
					found = true;
					break;
				}
			}
			for (TriggeredAbility& ta : triggerStack) {
				if (ta.activateInfo.skillId == skill.skillId && ta.activateInfo.effectCard == ref) {
					found = true;
					break;
				}
			}
			if (found) {
				break;
			}
		}
		ta.activateInfo = ActivateAbilityInfo(skill.skillId, state.makeAreaRef(effectCard), card.playerIndex);

		if (skill.notStack) {
			bool found = false;
			for (TriggeredAbility& a : triggerStack) {
				if (a.activateInfo.skillId == ta.activateInfo.skillId) {
					if (a.trigger.subject.card == ta.trigger.subject.card) {
						found = true;
						break;
					}
				}
			}
			if (found) {
				break;
			}
		}

		triggerStack.push_back(ta);

		break;
	}
}


inline void PullTrigger(State& state, TriggerType triggerType, CardRef subject, CardRef object = {}, int depth = 0) {
	for (int i = 0; i < 2; i++) {
		PlayerState& ps = state.players[i];

		for (CardRef ref : ps.active) {
			TriggerList(state, triggerType, subject, object, ref, depth);
		}

		for (CardRef ref : ps.bench) {
			TriggerList(state, triggerType, subject, object, ref, depth);
		}

		if (!state.noToolEffect) {
			for (CardRef ref : ps.tool) {
				TriggerList(state, triggerType, subject, object, ref, depth);
			}
		}

		for (CardRef ref : ps.energy) {
			TriggerList(state, triggerType, subject, object, ref, depth);
		}

		for (CardRef ref : ps.trash) {
			TriggerList(state, triggerType, subject, object, ref, depth);
		}
	}

	for (CardRef ref : state.stadium) {
		TriggerList(state, triggerType, subject, object, ref, depth);
	}
}
