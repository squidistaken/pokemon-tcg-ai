// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Card.h"

struct ActivateAbilityInfo {
	int skillId;
	AreaRef effectCard; // 能力を発動したと見なされるカード
	signed char usePlayerIndex; // 使用したプレイヤー。カードの持ち主とは限らない
	bool isEffectStack;
	signed char effectStackIndex;
	bool isSpecialCondition;

	ActivateAbilityInfo() = default;
	ActivateAbilityInfo(int skillId, AreaRef effectCard, int usePlayerIndex) : skillId(skillId), effectCard(effectCard), usePlayerIndex(usePlayerIndex), isEffectStack(false), effectStackIndex(0), isSpecialCondition(false){
	}
};

struct TriggerInfo {
	TriggerType type;
	signed char depth;
	int value;
	AreaRef subject;
	AreaRef object;

	bool isNull() const {
		return type == TriggerType::None;
	}
};

struct TriggeredAbility {
	ActivateAbilityInfo activateInfo;
	TriggerInfo trigger;
};

struct EffectState {
	ActivateAbilityInfo ability;
	signed char effectIndex;
	bool onEffect; // エフェクト処理中ならtrue
	signed char selectedListIndex;
	signed char eachListIndex;
	short effectRate; // 効果倍率
	int damageChange;

	void init() {
		effectRate = 1;
	}

	int usePlayerIndex() const {
		return ability.usePlayerIndex;
	}
};