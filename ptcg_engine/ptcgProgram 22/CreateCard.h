// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Types.h"
#include "Card.h"

class Chain {
public:

	Chain(CardMaster& card) : card(card) {}

	CardMaster& getCard() {
		return card;
	}

	Chain& nameEn(const char8_t* name) {
		card.nameEn = name;
		return *this;
	}

	Chain& textEn(const char8_t* name, const char8_t* text) {
		if (sk == nullptr) {
			at->nameEn = name;
			at->textEn = text;
		} else {
			sk->nameEn = name;
			sk->textEn = text;
		}
		return *this;
	}

	Chain& pokemon(PokemonType pokemonType, EvolutionType evolutionType, EnergyType energyType, int hp, int retreatCost) {
		card.pokemonType = pokemonType;
		card.evolutionType = evolutionType;
		card.energyType = energyType;
		card.hp = hp;
		card.retreatCost = retreatCost;
		return *this;
	}

	Chain& weakness(EnergyType energyType) {
		card.weakness = energyType;
		return *this;
	}

	Chain& resistance(EnergyType energyType) {
		card.resistance = energyType;
		return *this;
	}

	Chain& evolvesFrom(const char8_t* evolvesFrom) {
		card.evolvesFrom = evolvesFrom;
		return *this;
	}


	Chain& basicEnergy(EnergyType energyType) {
		card.energyType = energyType;
		card.energyCount = 1;
		return *this;
	}

	Chain& specialEnergy(EnergyType energyType, int energyCount) {
		card.energyType = energyType;
		card.energyCount = energyCount;
		return *this;
	}


	Chain& tera() {
		card.tera = true;
		return *this;
	}

	Chain& trashMyTurnEnd() {
		card.trashMyTurnEnd = true;
		return *this;
	}

	Chain& cannotToHandOrDeckInTrash() {
		card.cannotToHandOrDeckInTrash = true;
		return *this;
	}

	Chain& canPlayFirstTurn() {
		card.canPlayFirstTurn = true;
		return *this;
	}

	Chain& transformOnly() {
		card.transformOnly = true;
		return *this;
	}

	Chain& ancient() {
		card.ancient = true;
		return *this;
	}

	Chain& future() {
		card.future = true;
		return *this;
	}

	Chain& hop() {
		card.hop = true;
		return *this;
	}

	Chain& lillie() {
		card.lillie = true;
		return *this;
	}

	Chain& iono() {
		card.iono = true;
		return *this;
	}

	Chain& n() {
		card.n = true;
		return *this;
	}

	Chain& ethan() {
		card.ethan = true;
		return *this;
	}

	Chain& cynthia() {
		card.cynthia = true;
		return *this;
	}

	Chain& misty() {
		card.misty = true;
		return *this;
	}

	Chain& arven() {
		card.arven = true;
		return *this;
	}

	Chain& steven() {
		card.steven = true;
		return *this;
	}

	Chain& marnie() {
		card.marnie = true;
		return *this;
	}

	Chain& erika() {
		card.erika = true;
		return *this;
	}

	Chain& larry() {
		card.larry = true;
		return *this;
	}

	Chain& teamRocket() {
		card.teamRocket = true;
		return *this;
	}

	Chain& aceSpec() {
		card.aceSpec = true;
		return *this;
	}



	void initSkill(int skillId, const char8_t* name, const char8_t* text) {
		assert(!SkillTable.contains(skillId));
		at = nullptr;
		sk = &SkillTable[skillId];
		*sk = {};
		sk->skillId = skillId;
		sk->cardId = card.cardId;
		sk->name = name;
		sk->text = text;
		ef = nullptr;
		tr = nullptr;
		tgt = nullptr;
		tgt2 = nullptr;
	}

	// 特性追加
	Chain& ability(int skillId, const char8_t* name, const char8_t* text, AreaType areaType = AreaType::All, AreaType areaType2 = AreaType::All) {
		assert(card.ability == nullptr);
		initSkill(skillId, name, text);

		if (areaType == AreaType::All) {
			throw std::runtime_error("area type not set");
		}
		sk->areas.push_back(areaType);
		if (areaType2 != AreaType::All) {
			sk->areas.push_back(areaType2);
		}
		card.ability = sk;
		return *this;
	}

	Chain& abilityBattleField(int skillId, const char8_t* name, const char8_t* text) {
		return ability(skillId, name, text, AreaType::Active, AreaType::Bench);
	}

	Chain& abilityActive(int skillId, const char8_t* name, const char8_t* text) {
		return ability(skillId, name, text, AreaType::Active);
	}

	Chain& abilityBench(int skillId, const char8_t* name, const char8_t* text) {
		return ability(skillId, name, text, AreaType::Bench);
	}

	Chain& abilityHand(int skillId, const char8_t* name, const char8_t* text) {
		return ability(skillId, name, text, AreaType::Hand);
	}

	Chain& abilityTrash(int skillId, const char8_t* name, const char8_t* text) {
		return ability(skillId, name, text, AreaType::Trash);
	}

	// 特殊な特性
	Chain& specialAbility(int skillId, const char8_t* name, const char8_t* text) {
		initSkill(skillId, name, text);
		card.ability = sk;
		return *this;
	}

	// 自分の番に、このカードを手札からベンチに出したとき、1回使える
	Chain& abilityPlay(int skillId, const char8_t* name, const char8_t* text) {
		ability(skillId, name, text, AreaType::Bench);
		sk->canSelectActivate = true;
		triggerMe(TriggerType::HandToBench);
		return conditionMyTurn();
	}

	// 自分の番に、このカードを手札から出して進化させたとき、1回使える
	Chain& abilityEvolve(int skillId, const char8_t* name, const char8_t* text) {
		abilityBattleField(skillId, name, text);
		sk->canSelectActivate = true;
		triggerMe(TriggerType::EvolveFromHand);
		return conditionMyTurn();
	}

	// 自分の番に、このポケモンがベンチからバトル場に出たとき、1回使える
	Chain& abilityBenchToActive(int skillId, const char8_t* name, const char8_t* text) {
		abilityActive(skillId, name, text);
		sk->canSelectActivate = true;
		sk->onceTurn = true;
		triggerMe(TriggerType::BenchToActive);
		return conditionMyTurn();
	}

	// 自分の番に、このポケモンがバトル場からベンチにもどったとき、1回使える
	Chain& abilityActiveToBench(int skillId, const char8_t* name, const char8_t* text) {
		abilityBench(skillId, name, text);
		sk->canSelectActivate = true;
		sk->onceTurn = true;
		triggerMe(TriggerType::ActiveToBench);
		return conditionMyTurn();
	}

	Chain& activateSkill(int skillId, const char8_t* name, const char8_t* text) {
		abilityBattleField(skillId, name, text);
		sk->mainAbility = true;
		return *this;
	}

	Chain& activateSkillOnceTurn(int skillId, const char8_t* name, const char8_t* text) {
		abilityBattleField(skillId, name, text);
		sk->mainAbility = true;
		sk->onceTurn = true;
		return *this;
	}

	Chain& activateSkillOnceTurnActive(int skillId, const char8_t* name, const char8_t* text) {
		abilityActive(skillId, name, text);
		sk->mainAbility = true;
		sk->onceTurn = true;
		return *this;
	}

	Chain& activateSkillOnceTurnBench(int skillId, const char8_t* name, const char8_t* text) {
		abilityBench(skillId, name, text);
		sk->mainAbility = true;
		sk->onceTurn = true;
		return *this;
	}


	Chain& activateSkillFirstTurn(int skillId, const char8_t* name, const char8_t* text) {
		activateSkillOnceTurn(skillId, name, text);
		return conditionFirstTurn();
	}

	// 使用時スキル追加
	Chain& playSkill(int skillId, const char8_t* text) {
		assert(card.play == nullptr);
		initSkill(skillId, card.name.c_str(), text);

		card.play = sk;
		return *this;
	}

	// 付けたときのスキル追加
	Chain& attachSkill(int skillId, const char8_t* text) {
		return playSkill(skillId, text);
	}

	// ベンチポケモンにつけたときのスキル追加
	Chain& attachSkillBench(int skillId, const char8_t* text) {
		return playSkill(skillId, text).attachBench();
	}

	// エネルギースキル追加
	Chain& energySkill(int skillId, const char8_t* text) {
		ability(skillId, card.name.c_str(), text, AreaType::Energy);
		return *this;
	}

	// どうぐスキル追加
	Chain& toolSkill(int skillId, const char8_t* text) {
		ability(skillId, card.name.c_str(), text, AreaType::Tool);
		return *this;
	}

	// スタジアムスキル追加
	Chain& stadiumSkill(int skillId, const char8_t* text) {
		ability(skillId, card.name.c_str(), text, AreaType::Stadium);
		return *this;
	}

	// 1ターンに1回使えるスタジアム起動スキル追加
	Chain& stadiumActivateSkillOnceTurn(int skillId, const char8_t* text) {
		stadiumSkill(skillId, text);
		sk->mainAbility = true;
		sk->onceTurn = true;
		return *this;
	}

	// 遅延発動スキル追加
	Chain& delaySkill(int skillId, const char8_t* name, const char8_t* text) {
		assert(card.play == nullptr);
		initSkill(skillId, name, text);
		sk->canActivateTrash = true;

		card.delay = sk;
		return *this;
	}

	// ワザ追加
	Chain& attack(int attackId, const char8_t* name, const char8_t* text, const char* damage, std::initializer_list<EnergyType> energies) {
		assert(!AttackTable.contains(attackId));
		attackEffectType = 0;
		sk = nullptr;
		at = &AttackTable[attackId];
		at->attackId = attackId;
		at->cardId = card.cardId;
		at->damageText = damage;
		at->name = name;
		at->text = text;
		for (EnergyType e : energies) {
			at->energies.push_back(e);
		}
		card.attacks.push_back(at);

		int val = 0;
		const std::string& s = at->damageText;
		if (s.size() > 0) {
			if (!s.ends_with("×")) {
				val = std::stoi(s);
			}
		}
		at->damage = val;

		return *this;
	}

	Chain& attack(int attackId, int damage, std::initializer_list<EnergyType> energies) {
		std::string s = std::to_string(damage);
		return attack(attackId, u8"", u8"", s.c_str(), energies);
	}

	Chain& setPreEffect() {
		assert(at != nullptr);
		attackEffectType = 1;
		return *this;
	}

	Chain& preEffect(EffectType type, TargetPlayer targetPlayer) {
		setPreEffect();
		effectInit(type, targetPlayer);
		return *this;
	}

	// このカード対象のEffect追加
	Chain& preEffectMe(EffectType type) {
		return preEffect(type, TargetPlayer::Me).target(AreaType::Me);
	}

	// 途中でconditionが失敗せず最後まで実行された場合にワザは成功する
	Chain& preEffectFailAttack() {
		at->lastCancelFailAttack = true;
		return preEffect(EffectType::FailAttack, TargetPlayer::None);
	}

	Chain& preEffectFailAttackCoinTail() {
		preEffectFailAttack();
		return effectBreakIfCoinTail();
	}

	Chain& preEffectSelectActivate() {
		return preEffect(EffectType::SelectActivate, TargetPlayer::None);
	}

	Chain& preEffectPostEffectActivate() {
		return preEffect(EffectType::PostEffectActivate, TargetPlayer::None);
	}

	Chain& preEffectAttackDamageChange(int damageChange) {
		return preEffect(EffectType::AttackDamageChange, TargetPlayer::None).eVal(damageChange);
	}

	Chain& preEffectAttackDamageChangeCoin(int coinCount, int damageChange) {
		return preEffect(EffectType::AttackDamageChangeCoin, TargetPlayer::None).eVal(coinCount, damageChange);
	}

	Chain& preEffectUnitedWings() {
		return preEffect(EffectType::AttackDamageChangeTargetCount, TargetPlayer::Me).eVal(20).targetTrash().targetNameCondition(TargetType::HasAttackName, u8"だんけつのつばさ");
	}

	Chain& preEffectRound(int damage) {
		return preEffect(EffectType::AttackDamageChangeTargetCount, TargetPlayer::Me).eVal(damage).targetPokemon().targetNameCondition(TargetType::HasAttackName, u8"りんしょう");
	}

	// 相手のバトルポケモンが～なら
	Chain& preExistEnemyActive() {
		setPreEffect();
		return exist(AreaType::Active, TargetPlayer::Enemy);
	}

	Chain& setPostEffect() {
		assert(at != nullptr);
		attackEffectType = 2;
		return *this;
	}

	Chain& postEffect(EffectType type, TargetPlayer targetPlayer) {
		setPostEffect();
		effectInit(type, targetPlayer);
		return *this;
	}

	// このカード対象のEffect追加
	Chain& postEffectMe(EffectType type) {
		return postEffect(type, TargetPlayer::Me).target(AreaType::Me);
	}

	Chain& postEffectActiveEnemy(EffectType type) {
		postEffect(type, TargetPlayer::Enemy).target(AreaType::Active);
		return *this;
	}


	// 相手のポケモンにダメージ
	Chain& postEffectDamagePokemon(int damage, int pokemonCount = 1) {
		setPostEffect();
		existPokemon(TargetPlayer::Enemy);
		effect(EffectType::AttackDamage, TargetPlayer::Enemy).eVal(damage).targetPokemon().multiSelect(pokemonCount);
		return *this;
	}

	// 相手のベンチポケモンにダメージ
	Chain& postEffectDamageBench(int damage, int pokemonCount = 1) {
		setPostEffect();
		existEnemyBench();
		int effectIndex = currentEffectIndex();
		effect(EffectType::AttackDamage, TargetPlayer::Enemy).eVal(damage).targetBench().multiSelect(pokemonCount);
		return setSecondTarget(effectIndex);
	}

	// 山札を上からcount枚トラッシュする
	Chain& postEffectDeckToTrash(int count) {
		setPostEffect();
		exist(AreaType::Deck, TargetPlayer::Enemy);
		return effect(EffectType::DeckToTrash, TargetPlayer::Enemy).eVal(count);
	}

	Chain& postEffectSelectActivate() {
		return postEffect(EffectType::SelectActivate, TargetPlayer::None);
	}

	Chain& postEffectBreakIfNotPostEffectActivated() {
		return postEffect(EffectType::BreakIfNotPostEffectActivated, TargetPlayer::None);
	}

	Chain& postEffectTrashEnergyMe(int count) {
		return postEffect(EffectType::ToTrash, TargetPlayer::Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).selectEnergy(count);
	}

	Chain& postEffectTrashEnergyMeAll() {
		return postEffect(EffectType::ToTrash, TargetPlayer::Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe);
	}

	// このポケモンにもダメージ
	Chain& postEffectDamageMe(int damage) {
		return postEffectMe(EffectType::AttackDamage).eVal(damage);
	}

	// コインを1回投げオモテなら、相手のバトルポケモンについているエネルギーを1個選び、トラッシュする
	Chain& postEffectActiveEnemyEnergyTrashIfCoinHead() {
		return setPostEffect()
			.effectBreakIfCoinTail()
			.postEffect(EffectType::ToTrash, TargetPlayer::Enemy).targetAttachedEnergy().targetCondition(TargetType::AttachedActivePokemon).selectEnergy(1);
	}

	// コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする
	Chain& postEffectParalyzeIfCoinHead() {
		return setPostEffect()
			.effectBreakIfCoinTail()
			.effect(EffectType::Paralyze, TargetPlayer::Enemy);
	}

	// コインを1回投げオモテなら、次の相手の番、このポケモンはワザのダメージや効果を受けない
	Chain& postEffectNoDamageAndEffectIfCoinHead() {
		return setPostEffect()
			.effectBreakIfCoinTail()
			.effectMe(EffectType::NoDamageAndEffectAttackNextEnemyTurn);
	}



	Chain& effectInit(EffectType type, TargetPlayer targetPlayer) {
		if (costHandTrashCount >= 1) {
			int count = costHandTrashCount;
			costHandTrashCount = 0;
			effect(EffectType::ToTrash, TargetPlayer::Me);
			targetHand();
			notMe();
			multiSelect(count);
		}
		ef = &currentEffects().emplace_back();
		*ef = {};
		if (sk == nullptr) {
			ef->attack = at;
		} else {
			ef->skill = sk;
			ef->skillId = sk->skillId;
		}

		ef->effectType = type;
		ef->target.targetPlayer = targetPlayer;

		if (type == EffectType::Ko) {
			ef->selectContext = SelectContext::Discard;
		} else if (type == EffectType::ToHand || type == EffectType::PrizeToHand || type == EffectType::ToHandReverse || type == EffectType::ToHandWithAttach) {
			ef->selectContext = SelectContext::ToHand;
		} else if(type == EffectType::ToTrash || type == EffectType::TransformTrash) {
			ef->selectContext = SelectContext::Discard;
		} else if (type == EffectType::ToDeckAndShuffle || type == EffectType::ToDeck || type == EffectType::ToDeckReverse || type == EffectType::LookToDeckReverse || type == EffectType::ToDeckReverseAndShuffle || type == EffectType::ToDeckWithAttach || type == EffectType::SwitchDeck) {
			ef->selectContext = SelectContext::ToDeck;
		} else if (type == EffectType::ToDeckBottom || type == EffectType::ToDeckBottomReverse || type == EffectType::ToDeckBottomClose) {
			ef->selectContext = SelectContext::ToDeckBottom;
		} else if (type == EffectType::ToActiveAndTrashActive) {
			ef->selectContext = SelectContext::ToActive;
		} else if (type == EffectType::ToBench) {
			ef->selectContext = SelectContext::ToBench;
		} else if (type == EffectType::ToPrize) {
			ef->selectContext = SelectContext::ToPrize;
		} else if (type == EffectType::ToLooking) {
			ef->selectContext = SelectContext::Look;
		} else if (type == EffectType::Switch) {
			ef->selectContext = SelectContext::Switch;
		} else if (type == EffectType::NotMove) {
			ef->selectContext = SelectContext::NotMove;
		} else if (type == EffectType::DamageCounter || type == EffectType::DamageCounterRemoved) {
			ef->selectContext = SelectContext::DamageCounter;
		} else if (type == EffectType::DamageCounterAny) {
			ef->selectContext = SelectContext::DamageCounterAny;
		} else if (type == EffectType::AttackDamage) {
			ef->selectContext = SelectContext::Damage;
		} else if (type == EffectType::RemoveDamageCounter || type == EffectType::RemoveDamageCounterAll) {
			ef->selectContext = SelectContext::RemoveDamageCounter;
		} else if (type == EffectType::Heal || type == EffectType::HealAll) {
			ef->selectContext = SelectContext::Heal;
		} else if (type == EffectType::SelectEvolvesFrom || type == EffectType::EvolvesFromEach) {
			ef->selectContext = SelectContext::EvolvesFrom;
		} else if (type == EffectType::EvolvesToEach || type == EffectType::SelectEvolvesTo) {
			ef->selectContext = SelectContext::EvolvesTo;
		} else if (type == EffectType::AttachFromEach || type == EffectType::SelectAttachFrom || type == EffectType::AttachSelectedCard || type == EffectType::SwitchSelectedCard || type == EffectType::EnergySwitchEach) {
			ef->selectContext = SelectContext::AttachFrom;
		} else if (type == EffectType::AttachToEach || type == EffectType::SelectAttachTo || type == EffectType::AttachEnergyMe) {
			ef->selectContext = SelectContext::AttachTo;
		} else if (type == EffectType::SelectSwitchEnergy) {
			ef->selectContext = SelectContext::SwitchEnergy;
		} else if (type == EffectType::SelectSwitchEnergyCard) {
			ef->selectContext = SelectContext::SwitchEnergyCard;
		} else if (type == EffectType::LookAndReturn) {
			ef->selectContext = SelectContext::Look;
		} else if (type == EffectType::Devolve || type == EffectType::DevolveAny) {
			ef->selectContext = SelectContext::Devolve;
		} else {
			ef->selectContext = SelectContext::None;
		}

		tr = nullptr;
		tgt = &ef->target;
		tgt2 = nullptr;
		return *this;
	}

	// Effect追加
	// これ以降のtarget系関数はこのEffectのtargetとなる
	Chain& effect(EffectType type, TargetPlayer targetPlayer) {
		assert(sk != nullptr || attackEffectType != 0); // ワザの最初でこの関数を呼んではいけない
		return effectInit(type, targetPlayer);
	}

	// 効果の切れ目
	Chain& separator() {
		setEffect((int)currentEffects().size() - 1);
		ef->separator = true;
		return *this;
	}

	// このカード対象のEffect追加
	Chain& effectMe(EffectType type) {
		return effect(type, TargetPlayer::Me).target(AreaType::Me);
	}

	// 前のEffectでの効果対象を対象とするEffect追加
	Chain& effectEffectedCard(EffectType type) {
		effect(type, TargetPlayer::Both).target(AreaType::Effected);
		return *this;
	}

	// 付けたポケモンを対象とするEffect追加
	Chain& effectAttachedPokemon(EffectType type) {
		effect(type, TargetPlayer::Both).target(AreaType::TriggerSubject);
		return *this;
	}

	// トリガー主体対象のEffect追加
	Chain& effectTriggerSubject(EffectType type) {
		effect(type, TargetPlayer::Both).target(AreaType::TriggerSubject);
		return *this;
	}

	// トリガー客体対象のEffect追加
	Chain& effectTriggerObject(EffectType type) {
		effect(type, TargetPlayer::Both).target(AreaType::TriggerObject);
		return *this;
	}

	Chain& effectEnergyContinual(EffectType type) {
		effect(type, TargetPlayer::None).target(AreaType::Attach);
		return *this;
	}

	Chain& effectToolContinual(EffectType type) {
		effect(type, TargetPlayer::None).target(AreaType::Attach);
		return *this;
	}


	// 効果の値。effectValue
	Chain& eVal(int val) {
		ef->values[0] = val;
		return *this;
	}

	Chain& eVal(int val, int val2) {
		ef->values[0] = val;
		ef->values[1] = val2;
		return *this;
	}

	Chain& setContext(SelectContext context) {
		ef->selectContext = context;
		return *this;
	}


	Chain& condition(ConditionType type, int value = 0, ComparatorType comparatorType = ComparatorType::Equal) {
		if (isSkill()) {
			if (sk->firstConditionCount == sk->effects.size()) {
				sk->firstConditionCount++;
			}
		}

		ef = &currentEffects().emplace_back();
		*ef = {};

		ef->skill = sk;
		ef->attack = at;
		if (sk != nullptr) {
			ef->skillId = sk->skillId;
		}

		ef->isCondition = true;
		ef->values[0] = value;
		ef->comparatorType = comparatorType;
		ef->conditionType = type;

		tr = nullptr;
		tgt = &ef->target;
		tgt2 = nullptr;
		return *this;
	}

	// 常に満たされない
	Chain& conditionAlwaysFail() {
		return condition(ConditionType::Always, 0, ComparatorType::NotEqual);
	}

	Chain& conditionCountTarget2(int count0, int count1, TargetPlayer player) {
		condition(ConditionType::CountTarget2, count0, ComparatorType::Equal);
		ef->values[1] = count1;
		return targetPlayer(player);
	}

	Chain& conditionCountTargetMeOrEnemy(int count0, int count1, ComparatorType comparatorType) {
		condition(ConditionType::CountTargetMeOrEnemy, count0, comparatorType);
		ef->values[1] = count1;
		return targetPlayer(TargetPlayer::None);
	}

	Chain& conditionGreaterEqual(int count, TargetPlayer player = TargetPlayer::Me) {
		condition(ConditionType::CountTarget, count, ComparatorType::GreaterEqual);
		return targetPlayer(player);
	}

	Chain& conditionLessEqual(int count, TargetPlayer player = TargetPlayer::Me) {
		condition(ConditionType::CountTarget, count, ComparatorType::LessEqual);
		return targetPlayer(player);
	}

	Chain& conditionLess(int count, TargetPlayer player = TargetPlayer::Me) {
		condition(ConditionType::CountTarget, count, ComparatorType::Less);
		return targetPlayer(player);
	}

	// 「手札をすべて山札にもどして切る」の使用条件
	Chain& canHandToDeckAndShuffle(TargetPlayer player = TargetPlayer::Me) {
		return exist(AreaType::Hand, player).target(AreaType::Deck).notMe();
	}

	Chain& exist(AreaType area, TargetPlayer player = TargetPlayer::Me) {
		conditionGreaterEqual(1, player);
		return target(area);
	}

	// Skill中のこれより後のEffectのTargetのどれかが1つ以上の対象カードを持つか
	Chain& existAnyTargetAfterEffect() {
		return condition(ConditionType::AnyTargetAfterEffect);
	}

	Chain& existEffected() {
		return exist(AreaType::Effected, TargetPlayer::Both);
	}

	Chain& existPokemon(TargetPlayer player = TargetPlayer::Me) {
		conditionGreaterEqual(1);
		targetPlayer(player);
		return targetPokemon();
	}

	Chain& existDamaged(TargetPlayer player) {
		return existPokemon(player).targetDamaged();
	}

	Chain& existStadium() {
		return exist(AreaType::Stadium, TargetPlayer::Both);
	}

	Chain& existMyDeck() {
		return exist(AreaType::Deck);
	}

	Chain& existEnemyDeck() {
		return exist(AreaType::Deck, TargetPlayer::Enemy);
	}

	Chain& existMyTrash() {
		return exist(AreaType::Trash);
	}

	Chain& existMyHand() {
		return exist(AreaType::Hand);
	}

	Chain& existMyBench() {
		return exist(AreaType::Bench);
	}

	Chain& existEnemyBench() {
		return exist(AreaType::Bench, TargetPlayer::Enemy);
	}

	Chain& existHandBothNotMe() {
		return exist(AreaType::Hand, TargetPlayer::Both).notMe();
	}

	Chain& notOpen4Card(const char8_t* name) {
		conditionLess(4);
		target(AreaType::Trash);
		target(AreaType::Active);
		target(AreaType::Bench);
		return targetName(name);
	}

	Chain& notFullBench(TargetPlayer player = TargetPlayer::Me) {
		condition(ConditionType::NotFullBench);
		return targetPlayer(player);
	}

	// このポケモンがバトル場にいるなら
	Chain& conditionActiveMe() {
		return exist(AreaType::Me).targetCondition(TargetType::Area, (int)AreaType::Active);
	}

	// このポケモンに指定タイプのエネルギーがついているなら
	Chain& conditionAttachEnergyMe(EnergyType type) {
		return exist(AreaType::Me).targetCondition(TargetType::IsAttachedEnergyType, (int)type);
	}

	// このポケモンに指定タイプのエネルギーが2個以上ついているなら
	Chain& conditionAttachEnergy2Me(EnergyType type) {
		return exist(AreaType::Me).targetCondition(TargetType::IsAttachedEnergy2Type, (int)type);
	}

	Chain& conditionNoSameNameSkillThisTurn() {
		return condition(ConditionType::NoSameNameSkillThisTurn);
	}

	Chain& conditionMyTurn() {
		return condition(ConditionType::MyTurn);
	}

	Chain& conditionEnemyTurn() {
		return condition(ConditionType::MyTurn, 0, ComparatorType::NotEqual);
	}

	Chain& conditionFirstTurn() {
		return condition(ConditionType::Turn, 2, ComparatorType::LessEqual);
	}

	Chain& conditionNotFirstTurn() {
		return condition(ConditionType::Turn, 3, ComparatorType::GreaterEqual);
	}

	Chain& conditionFirstTurnFirstPlayer() {
		return condition(ConditionType::Turn, 1);
	}

	Chain& conditionFirstTurnSecondPlayer() {
		return condition(ConditionType::Turn, 2);
	}

	Chain& conditionPrizeCountGreater() {
		return condition(ConditionType::CompareCountTargetMeEnemy, 0, ComparatorType::Greater)
			.targetPlayer(TargetPlayer::Both)
			.targetPrize();
	}



	// 発動タイミング追加
	// これ以降のtarget系関数はこのtriggerのsubjectのtargetとなる
	// この後targetPlayerを指定する場合は、エリアも指定すること
	Chain& trigger(TriggerType type) {
		tr = &sk->triggers.emplace_back();
		*tr = {};
		tr->triggerType = type;
		tgt = &tr->subject;
		return *this;
	}

	Chain& trigger(TriggerType type, AreaType area, TargetPlayer player) {
		trigger(type);
		tr->subject.areas.push_back(area);
		tr->subject.targetPlayer = player;
		return *this;
	}

	// このカード
	Chain& triggerMe(TriggerType type) {
		trigger(type);
		tr->subject.areas.push_back(AreaType::Me);
		tr->subject.targetPlayer = TargetPlayer::Me;
		return *this;
	}

	// 付いているポケモン
	Chain& triggerAttachedCard(TriggerType type) {
		trigger(type);
		tr->subject.areas.push_back(AreaType::Attach);
		tr->subject.targetPlayer = TargetPlayer::Me;
		return *this;
	}


	Chain& setSecondTarget(int index) {
		tgt2 = &currentEffects()[index].target;
		return *this;
	}

	// 対象エリア追加
	Chain& target(AreaType area) {
		if (tgt->targetPlayer == TargetPlayer::None) {
			tgt->targetPlayer = TargetPlayer::Both;
		}
		for (AreaType a : tgt->areas) {
			assert(a != area); // 同じエリア重複
		}
		tgt->areas.push_back(area);
		if (tgt2 != nullptr) {
			tgt2->areas.push_back(area);
		}
		return *this;
	}

	Chain& targetHand() {
		return target(AreaType::Hand);
	}

	Chain& targetTrash() {
		return target(AreaType::Trash);
	}

	Chain& targetDeck() {
		return target(AreaType::Deck);
	}

	// バトル場かベンチ
	Chain& targetPokemon() {
		target(AreaType::Active);
		return target(AreaType::Bench);
	}

	Chain& targetActive() {
		return target(AreaType::Active);
	}

	Chain& targetBench() {
		return target(AreaType::Bench);
	}

	Chain& targetPrize() {
		return target(AreaType::Prize);
	}

	Chain& targetAttachedEnergy() {
		return target(AreaType::Energy);
	}

	Chain& targetAttachedTool() {
		return target(AreaType::Tool);
	}

	Chain& targetLooking() {
		return target(AreaType::Looking);
	}



	Chain& targetCondition(TargetType type, int val = 0, ComparatorType comparatorType = ComparatorType::Equal) {
		TargetCondition condition = {};
		condition.targetType = type;
		condition.comparatorType = comparatorType;
		condition.val = val;
		tgt->conditions.push_back(condition);
		if (tgt2 != nullptr) {
			tgt2->conditions.push_back(condition);
		}
		return *this;
	}

	Chain& targetNameCondition(TargetType type, const char8_t* name, ComparatorType comparatorType = ComparatorType::Equal) {
		TargetCondition condition = {};
		condition.targetType = type;
		condition.comparatorType = comparatorType;
		condition.name = name;
		tgt->conditions.push_back(condition);
		if (tgt2 != nullptr) {
			tgt2->conditions.push_back(condition);
		}
		return *this;
	}

	Chain& targetCondition2(TargetType type, int val, int val2, ComparatorType comparatorType = ComparatorType::Equal) {
		TargetCondition condition = {};
		condition.targetType = type;
		condition.comparatorType = comparatorType;
		condition.val = val;
		condition.val2 = val2;
		tgt->conditions.push_back(condition);
		if (tgt2 != nullptr) {
			tgt2->conditions.push_back(condition);
		}
		return *this;
	}

	Chain& targetPlayer(TargetPlayer targetPlayer) {
		tgt->targetPlayer = targetPlayer;
		if (tgt2 != nullptr) {
			tgt2->targetPlayer = targetPlayer;
		}
		return *this;
	}

	Chain& notMe() {
		tgt->notMe = true;
		if (tgt2 != nullptr) {
			tgt2->notMe = true;
		}
		return *this;
	}

	Chain& skipEnemyTarget() {
		tgt->skipEnemyTarget = true;
		if (tgt2 != nullptr) {
			tgt2->skipEnemyTarget = true;
		}
		return *this;
	}

	// たねポケモン
	Chain& targetBasicPokemon() {
		return targetCondition(TargetType::BasicPokemon);
	}

	// 進化ポケモン
	Chain& targetEvolvedPokemon() {
		return targetCondition(TargetType::EvolvedPokemon);
	}

	Chain& targetPokemonCard() {
		return targetCondition(TargetType::PokemonCard);
	}

	Chain& targetBasicEnergy() {
		return targetCondition(TargetType::BasicEnergy);
	}

	Chain& targetSpecialEnergy() {
		return targetCondition(TargetType::SpecialEnergy);
	}

	Chain& targetEnergyCard() {
		return targetCondition(TargetType::EnergyCard);
	}

	Chain& targetItem() {
		return targetCondition(TargetType::Item);
	}

	Chain& targetToolCard() {
		return targetCondition(TargetType::Tool);
	}

	Chain& targetSupporter() {
		return targetCondition(TargetType::Supporter);
	}

	Chain& targetStadiumCard() {
		return targetCondition(TargetType::Stadium);
	}

	Chain& targetTrainer() {
		return targetCondition(TargetType::Trainer);
	}

	Chain& targetRulePokemon() {
		return targetCondition(TargetType::RulePokemon);
	}

	Chain& targetNotRulePokemon() {
		return targetCondition(TargetType::NotRulePokemon);
	}

	Chain& targetDamaged() {
		return targetCondition(TargetType::DamageCounter, 0, ComparatorType::Greater);
	}

	Chain& targetNotDamaged() {
		return targetCondition(TargetType::DamageCounter, 0, ComparatorType::Equal);
	}

	Chain& targetHpLessEqual(int hp) {
		return targetCondition(TargetType::Hp, hp, ComparatorType::LessEqual);
	}

	Chain& targetEnergyType(EnergyType type) {
		return targetCondition(TargetType::EnergyType, (int)type);
	}

	// 【type】エネルギーとして扱われているエネルギー
	Chain& targetEnergyTypeAttached(EnergyType type) {
		return targetCondition(TargetType::EnergyTypeAttached, (int)type);
	}

	Chain& targetCardId(int cardId) {
		return targetCondition(TargetType::CardId, cardId);
	}

	Chain& targetNotMyCardId() {
		return targetCondition(TargetType::CardId, card.cardId, ComparatorType::NotEqual);
	}

	Chain& targetName(const char8_t* name) {
		return targetNameCondition(TargetType::Name, name);
	}

	Chain& targetMyName() {
		return targetNameCondition(TargetType::Name, card.name.c_str());
	}

	Chain& targetNotMyName() {
		return targetNameCondition(TargetType::Name, card.name.c_str(), ComparatorType::NotEqual);
	}

	Chain& targetNameContains(const char8_t* name) {
		return targetNameCondition(TargetType::NameContains, name);
	}

	// このポケモンについているエネルギー
	Chain& targetEnergyCardMe() {
		return targetAttachedEnergy().targetCondition(TargetType::AttachedMe);
	}


	Chain& select(EffectSelectType type, int selectNumber) {
		ef->effectSelectType = type;
		ef->selectCount = selectNumber;
		return *this;
	}

	// 1枚選ぶ
	Chain& singleSelect() {
		return multiSelect(1);
	}

	// count枚選ぶ
	Chain& multiSelect(int count) {
		return select(EffectSelectType::CardCount, count);
	}

	// 順番を選ぶ
	Chain& allSelect() {
		return select(EffectSelectType::CardCount, 99);
	}

	// 最大count枚選ぶ
	Chain& maxSelect(int count) {
		return select(EffectSelectType::MaxCardCount, count);
	}

	// 前の対象カード枚数まで選ぶ
	Chain& maxSelectTargetCount() {
		ef->selectTargetCount = true;
		return select(EffectSelectType::MaxCardCount, 0);
	}

	// コインのオモテの数ちょうど選ぶ
	Chain& selectCoinHeadCount() {
		ef->selectCoinHeadCount = true;
		return select(EffectSelectType::CardCount, 0);
	}

	// コインのオモテの数まで選ぶ
	Chain& maxSelectCoinHeadCount() {
		ef->selectCoinHeadCount = true;
		return select(EffectSelectType::MaxCardCount, 0);
	}

	// コインのオモテの数×2まで選ぶ
	Chain& maxSelectCoinHeadCount2() {
		ef->selectCoinHeadCount2 = true;
		return select(EffectSelectType::MaxCardCount, 0);
	}

	// 相手のポケモン全員についているエネルギーの数ぶんまで選ぶ
	Chain& maxSelectEnemyEnergyCount() {
		ef->selectEnemyEnergyCount = true;
		return select(EffectSelectType::MaxCardCount, 0);
	}

	// 好きなだけ選ぶ
	// 基本的に0枚選択可能
	Chain& selectAny() {
		return select(EffectSelectType::MaxCardCount, 99).canNoSelect();
	}

	Chain& selectAnyCannotNoSelect() {
		return select(EffectSelectType::MaxCardCount, 99).cannotNoSelect();
	}

	// count個選ぶ
	Chain& selectEnergy(int count) {
		return select(EffectSelectType::Energy, count);
	}

	// コインのオモテの数まで選ぶ
	Chain& selectEnergyCoinHeadCount() {
		ef->selectCoinHeadCount = true;
		return select(EffectSelectType::Energy, 0);
	}

	Chain& maxSelectEnergy(int count) {
		return select(EffectSelectType::MaxEnergyCard, count);
	}

	// 好きなだけ選ぶ
	Chain& selectEnergyAny() {
		return select(EffectSelectType::MaxEnergyCard, 99).canNoSelect();
	}

	// count枚選ぶ
	Chain& selectTool(int count) {
		return select(EffectSelectType::ToolCard, count);
	}

	// ランダムに1枚選ぶ
	Chain& randomSelect() {
		ef->randomSelect = true;
		return singleSelect();
	}

	// ランダムにコインのオモテの数まで選ぶ
	Chain& randomSelectCoinHeadCount() {
		ef->randomSelect = true;
		ef->selectCoinHeadCount = true;
		return multiSelect(0);
	}

	// ランダムにN枚になるように選ぶ
	Chain& randomSelectUntil(int count) {
		ef->randomSelect = true;
		return select(EffectSelectType::CardUntil, count);
	}

	// 前のeffectにtargetを付けたいときに使用
	Chain& setEffect(int effectIndex) {
		ef = &currentEffects().at(effectIndex);
		tgt = &ef->target;
		return *this;
	}

	Chain& noEffect() {
		return effect(EffectType::NoEffect, TargetPlayer::Both);
	}

	Chain& effectDraw(int drawCount) {
		existMyDeck();
		return effect(EffectType::Draw, TargetPlayer::Me).eVal(drawCount);
	}

	Chain& effectDrawUntil(int count, TargetPlayer player = TargetPlayer::Me) {
		existMyDeck();
		condition(ConditionType::CountTarget, count, ComparatorType::Less)
			.targetPlayer(player)
			.targetHand().notMe();
		return effect(EffectType::DrawUntil, player).eVal(count);
	}

	Chain& effectLookAndToHandReverseRestDeckBottom(int lookCount) {
		existMyDeck();
		effect(EffectType::LookDeck, TargetPlayer::Me).eVal(lookCount);
		effect(EffectType::ToHandReverse, TargetPlayer::Me).targetLooking().singleSelect();
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToDeckBottomReverse, TargetPlayer::Me).targetLooking();
		return setEffect(effectIndex);
	}

	Chain& effectLookAndToHandRestDeckAndShuffle(int lookCount, int toHandCount = 1) {
		existMyDeck();
		effect(EffectType::LookDeck, TargetPlayer::Me).eVal(lookCount);
		effect(EffectType::ToHand, TargetPlayer::Me).targetLooking().maxSelect(toHandCount);
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToDeckReverse, TargetPlayer::Me).targetLooking();
		effectShuffle();
		return setEffect(effectIndex);
	}

	Chain& effectLookAndToHandReverseRestDeckAndShuffle(int lookCount, int toHandCount = 1) {
		existMyDeck();
		effect(EffectType::LookDeck, TargetPlayer::Me).eVal(lookCount);
		effect(EffectType::ToHandReverse, TargetPlayer::Me).targetLooking().maxSelect(toHandCount);
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToDeckReverse, TargetPlayer::Me).targetLooking();
		effectShuffle();
		return setEffect(effectIndex);
	}

	Chain& effectLookDeckBottomAndToHandRestDeckAndShuffle(int lookCount, int toHandCount = 1) {
		existMyDeck();
		effect(EffectType::LookDeckBottom, TargetPlayer::Me).eVal(lookCount);
		effect(EffectType::ToHand, TargetPlayer::Me).targetLooking().maxSelect(toHandCount);
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToDeckReverse, TargetPlayer::Me).targetLooking();
		effectShuffle();
		return setEffect(effectIndex);
	}

	Chain& effectLookAndToBenchRestDeckAndShuffle(int lookCount, TargetPlayer player) {
		exist(AreaType::Deck, player);
		notFullBench(player);
		effect(EffectType::LookDeck, player).eVal(lookCount);
		if (player == TargetPlayer::Enemy) {
			open();
		}
		effect(EffectType::ToBench, player).targetLooking().targetPokemonCard().maxSelect(lookCount);
		int effectIndex = currentEffectIndex();
		if (player == TargetPlayer::Me) {
			effect(EffectType::ToDeckReverse, player);
		} else {
			effect(EffectType::ToDeck, player);
		}
		targetLooking();
		effectShuffle(player);
		return setEffect(effectIndex);
	}

	// 見て戻す
	Chain& effectLookEnemyhand() {
		exist(AreaType::Hand, TargetPlayer::Enemy);
		effect(EffectType::ToLooking, TargetPlayer::Enemy).targetHand();
		return effect(EffectType::ToHand, TargetPlayer::Enemy).targetLooking();
	}

	Chain& effectDeckToHandNoShuffle(int count = 1) {
		existMyDeck();
		effect(EffectType::ToHand, TargetPlayer::Me).targetDeck();
		return maxSelect(count);
	}

	Chain& effectDeckToHandCardId(int cardId) {
		existMyDeck();
		effect(EffectType::ToHand, TargetPlayer::Me).targetDeck().targetCondition(TargetType::CardId, cardId);
		return maxSelect(1);
	}

	Chain& effectDeckToHandAndShuffle(int count = 1) {
		existMyDeck();
		effect(EffectType::ToHand, TargetPlayer::Me).targetDeck();
		maxSelect(count);
		int effectIndex = currentEffectIndex();
		effectShuffle();
		return setEffect(effectIndex);
	}

	// 好きなカードを選ぶとき
	Chain& effectDeckToHandReverseAndShuffle(int count = 1) {
		existMyDeck();
		effect(EffectType::ToHandReverse, TargetPlayer::Me).targetDeck();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		int effectIndex = currentEffectIndex();
		effectShuffle();
		return setEffect(effectIndex);
	}

	Chain& effectDeckToTrashAndShuffle(int count = 1) {
		existMyDeck();
		effect(EffectType::ToTrash, TargetPlayer::Me).targetDeck();
		maxSelect(count);
		int effectIndex = currentEffectIndex();
		effectShuffle();
		return setEffect(effectIndex);
	}

	Chain& effectDeckToBenchAndShuffle(int count = 1) {
		notFullBench();
		existMyDeck();
		effect(EffectType::ToBench, TargetPlayer::Me).targetDeck().targetPokemonCard();
		maxSelect(count);
		int effectIndex = currentEffectIndex();
		effectShuffle();
		return setEffect(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectTrashToHand(int count = 1) {
		exist(AreaType::Trash);
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToHand, TargetPlayer::Me).targetTrash();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectTrashToBench(int count = 1) {
		notFullBench();
		exist(AreaType::Trash);
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToBench, TargetPlayer::Me).targetTrash();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectTrashToDeckAndShuffle(int count = 1) {
		existMyTrash();
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToDeckAndShuffle, TargetPlayer::Me).targetTrash();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		return setSecondTarget(effectIndex);
	}

	Chain& effectTrashStadium() {
		exist(AreaType::Stadium, TargetPlayer::Both);
		return effect(EffectType::ToTrash, TargetPlayer::Both).target(AreaType::Stadium);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectSelectAttachBasicEnergyTrash(int count = 1) {
		existMyTrash().targetBasicEnergy();
		int effectIndex = currentEffectIndex();
		effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetTrash().targetBasicEnergy();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	Chain& effectSelectAttachEnergyDeck(int count = 1) {
		existMyDeck();
		return effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetDeck().targetEnergyCard().maxSelect(count);
	}

	// count枚まで選ぶ
	Chain& effectSelectAttachBasicEnergyDeck(int count = 1) {
		existMyDeck();
		return effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetDeck().targetBasicEnergy().maxSelect(count);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectSelectAttachEnergyHand(int count = 1) {
		existMyHand().targetEnergyCard();
		int effectIndex = currentEffectIndex();
		effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetHand().targetEnergyCard().maxSelect(count);
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectSelectAttachBasicEnergyHand(int count = 1) {
		existMyHand().targetBasicEnergy();
		int effectIndex = currentEffectIndex();
		effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetHand().targetBasicEnergy().maxSelect(count);
		return setSecondTarget(effectIndex);
	}

	Chain& effectAttachFromEach() {
		return effect(EffectType::AttachFromEach, TargetPlayer::Me).eachSelectedList().singleSelect();
	}

	// 自分の山札からエネルギーを選び、自分のポケモンに好きなようにつける
	Chain& effectDeckAttachEnergyAndShuffle(int count) {
		existMyDeck();
		effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetDeck().targetEnergyCard().maxSelect(count);
		int effectIndex = currentEffectIndex();
		effectAttachFromEach().targetPokemon().seeingDeck();
		effectShuffle();
		return setEffect(effectIndex);
	}

	// 自分の山札からエネルギーを選び、ベンチポケモンに好きなようにつける
	Chain& effectDeckAttachEnergyBenchAndShuffle(int count) {
		existMyDeck();
		effect(EffectType::SelectAttachTo, TargetPlayer::Me).targetDeck().targetEnergyCard().maxSelect(count);
		int effectIndex = currentEffectIndex();
		effectAttachFromEach().targetBench().seeingDeck();
		effectShuffle();
		return setEffect(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectHandAttachEnergyMe(int count = 1) {
		exist(AreaType::Hand);
		int effectIndex = currentEffectIndex();
		effect(EffectType::AttachEnergyMe, TargetPlayer::Me).targetHand().targetEnergyCard();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectTrashAttachEnergyMe(int count = 1) {
		exist(AreaType::Trash);
		int effectIndex = currentEffectIndex();
		effect(EffectType::AttachEnergyMe, TargetPlayer::Me).targetTrash().targetEnergyCard();
		if (count == 1) {
			singleSelect();
		} else {
			maxSelect(count);
		}
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	Chain& effectDeckAttachEnergyMeAndShuffle(int count = 1) {
		exist(AreaType::Deck);
		effect(EffectType::AttachEnergyMe, TargetPlayer::Me).targetDeck().targetEnergyCard().maxSelect(count);
		int effectIndex = currentEffectIndex();
		effectShuffle();
		return setEffect(effectIndex);
	}

	Chain& effectSwitch(TargetPlayer player) {
		exist(AreaType::Bench, player);
		int effectIndex = currentEffectIndex();
		effect(EffectType::Switch, player)
			.targetBench()
			.singleSelect();
		return setSecondTarget(effectIndex);
	}

	// 相手のベンチポケモンを1匹選び、バトルポケモンと入れ替える
	Chain& effectSwitchEnemyBench() {
		effectSwitch(TargetPlayer::Enemy);
		ef->effectTargetBench = true;
		return *this;
	}

	Chain& effectTrashEnergy(int count, TargetPlayer player) {
		exist(AreaType::Energy, player);
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToTrash, player).targetAttachedEnergy().selectEnergy(count);
		return setSecondTarget(effectIndex);
	}

	Chain& effectEnergySwitchEach(TargetPlayer player) {
		return effect(EffectType::EnergySwitchEach, player).eachSelectedList().targetCondition(TargetType::NotContextCardAttachedPokemon).singleSelect();
	}

	Chain& effectBasicEnergySwitch(TargetPlayer player) {
		exist(AreaType::Energy).targetBasicEnergy();
		conditionGreaterEqual(2).targetPokemon();
		effect(EffectType::SelectSwitchEnergyCard, player).targetAttachedEnergy().targetBasicEnergy().maxSelectEnergy(1);
		return effectEnergySwitchEach(player).targetPokemon();
	}

	Chain& effectHealSingle(int heal) {
		existDamaged(TargetPlayer::Me);
		return effect(EffectType::Heal, TargetPlayer::Me).eVal(heal).targetPokemon().targetDamaged().singleSelect();
	}

	Chain& effectShuffle(TargetPlayer player = TargetPlayer::Me) {
		effect(EffectType::Shuffle, player);
		ef->notUpdateTarget = true;
		return *this;
	}
	
	// バトル場ベンチ両方
	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectSelectAttachFrom(int count) {
		existPokemon();
		int effectIndex = currentEffectIndex();
		effect(EffectType::SelectAttachFrom, TargetPlayer::Me).targetPokemon().maxSelect(count);
		return setSecondTarget(effectIndex);
	}

	// count枚まで選ぶ
	// これ以降のtargetは使用条件にも適用される
	Chain& effectSelectAttachFromBench(int count) {
		existMyBench();
		int effectIndex = currentEffectIndex();
		effect(EffectType::SelectAttachFrom, TargetPlayer::Me).targetBench().maxSelect(count);
		return setSecondTarget(effectIndex);
	}

	// シャッフルしないので最後のシャッフルを忘れない
	Chain& effectDeckSelectBasicEnergyNotSameType(int count, SelectContext context) {
		effect(EffectType::SelectCard, TargetPlayer::Me).targetDeck().singleSelect();
		targetBasicEnergy().targetCondition(TargetType::NotChecked);
		return setContext(context).addCheckList().loopCount(count);
	}

	Chain& effectAttachToEach() {
		return effect(EffectType::AttachToEach, TargetPlayer::Me).eachSelectedList();
	}

	Chain& effectEvolvesToEach() {
		return effect(EffectType::EvolvesToEach, TargetPlayer::Me).targetCondition(TargetType::CanEvolveContextCard).eachSelectedList().maxSelect(1);
	}

	Chain& effectEvolvesFromEach() {
		return effect(EffectType::EvolvesFromEach, TargetPlayer::Me).targetCondition(TargetType::CanEvolvesToContextCard).eachSelectedList().singleSelect();
	}

	Chain& effectNoRetreatCost() {
		return effect(EffectType::NoRetreatCost, TargetPlayer::Me);
	}

	Chain& effectNoAbility(TargetPlayer player) {
		return effect(EffectType::NoAbility, player).priority();
	}

	Chain& effectNoKoMeAbility(TargetPlayer player) {
		return effect(EffectType::NoKoMeAbility, player).priority(1);
	}

	Chain& effectBreakIfCoinTail() {
		return effect(EffectType::BreakIfCoinTail, TargetPlayer::None);
	}

	Chain& effectBreakIfCoinTailMulti(int count) {
		return effect(EffectType::BreakIfCoinTailMulti, TargetPlayer::None).eVal(count);
	}

	Chain& effectTurnEnd() {
		return effect(EffectType::TurnEnd, TargetPlayer::None);
	}

	Chain& effectKoMe() {
		sk->koMeAbility = true;
		return effect(EffectType::Ko, TargetPlayer::Me).target(AreaType::Me);
	}

	// おまつりおんど
	Chain& effectFestivalLead() {
		exist(AreaType::Stadium, TargetPlayer::Both).targetName(u8"お祭り会場");
		return effectMe(EffectType::DoubleAttack);
	}

	// 制限無し手札コスト
	// 手札からプレイする場合、手札はこのカードを含めてcount + 1枚必要
	Chain& costHandTrash(int count) {
		costHandTrashCount = count;
		condition(ConditionType::CountTarget, count, ComparatorType::GreaterEqual);
		targetPlayer(TargetPlayer::Me);
		return targetHand().notMe();
	}

	// 制限付き手札コスト
	// もし他にも条件があるならこの関数を呼ぶ前に付ける
	Chain& costHandTrashLimited(int count) {
		condition(ConditionType::CountTarget, count, ComparatorType::GreaterEqual);
		targetPlayer(TargetPlayer::Me).targetHand().notMe();
		int effectIndex = currentEffectIndex();
		effect(EffectType::ToTrash, TargetPlayer::Me).targetHand().multiSelect(count);
		return setSecondTarget(effectIndex);
	}


	Chain& fossil() {
		card.hp = 60;
		card.pokemonType = PokemonType::PokemonItem;
		card.evolutionType = EvolutionType::Basic;
		card.energyType = EnergyType::Colorless;
		card.canTrash = true;
		card.toBench = true;
		notFullBench();
		return noEffect();
	}

	Chain& doll() {
		card.hp = 120;
		card.pokemonType = PokemonType::PokemonItem;
		card.evolutionType = EvolutionType::Basic;
		card.energyType = EnergyType::Colorless;
		card.canTrash = true;
		card.toBattleFieldOnlySetup = true;
		card.noPrize = true;
		return conditionAlwaysFail();
	}

	Chain& toActiveOnlySetup() {
		card.toActiveOnlySetup = true;
		return conditionAlwaysFail().noEffect();
	}

	Chain& eachSelectedList() {
		ef->eachSelectedList = true;
		return *this;
	}

	Chain& eachList() {
		ef->eachList = true;
		return *this;
	}

	Chain& addCheckList() {
		ef->addCheckList = true;
		return *this;
	}

	Chain& notClearSelectedList() {
		ef->notClearSelectedList = true;
		return *this;
	}

	Chain& notPreTarget() {
		ef->notPreTarget = true;
		return *this;
	}

	Chain& notUpdateTarget() {
		ef->notUpdateTarget = true;
		return *this;
	}

	Chain& multiplyEffectValuePreTargetCount() {
		ef->multiplyEffectValuePreTargetCount = true;
		return *this;
	}

	Chain& multiplyEffectValueCoinHeadCount() {
		ef->multiplyEffectValueCoinHeadCount = true;
		return *this;
	}

	Chain& canNoSelect() {
		ef->canNoSelect = true;
		return *this;
	}

	Chain& canNoSelectIfExistPreTarget() {
		ef->canNoSelectIfExistPreTarget = true;
		return *this;
	}

	Chain& cannotNoSelect() {
		ef->cannotNoSelect = true;
		return *this;
	}

	Chain& energyMaxSelect() {
		ef->energyMaxSelect = true;
		return *this;
	}

	Chain& skipNoTarget() {
		ef->skipNoTarget = true;
		return *this;
	}

	Chain& open() {
		ef->open = true;
		return *this;
	}

	Chain& setTargetSwitchBench() {
		ef->setTargetSwitchBench = true;
		return *this;
	}

	Chain& effectTargetActive() {
		ef->effectTargetActive = true;
		return *this;
	}

	Chain& removeEffectedIfNoEffect() {
		ef->removeEffectedIfNoEffect = true;
		return *this;
	}

	Chain& seeingDeck() {
		ef->seeingDeck = true;
		return *this;
	}

	Chain& enemySelect() {
		ef->enemySelect = true;
		return *this;
	}

	Chain& failSkip(int skipCount) {
		if (sk != nullptr && sk->firstConditionCount == sk->effects.size()) {
			sk->firstConditionCount--;
		}
		ef->failSkip = skipCount;
		return *this;
	}

	Chain& loopCount(int count) {
		ef->loopCount = count;
		return *this;
	}

	Chain& priority(int val = 2) {
		sk->priority = val;
		return *this;
	}

	Chain& canSelectActivate() {
		sk->canSelectActivate = true;
		return *this;
	}

	Chain& onceTurn() {
		sk->onceTurn = true;
		return *this;
	}

	Chain& notStack() {
		sk->notStack = true;
		return *this;
	}

	Chain& canActivateTrash() {
		sk->canActivateTrash = true;
		return *this;
	}

	Chain& attachBench() {
		sk->attachBench = true;
		return *this;
	}

	Chain& luckyBonus() {
		sk->luckyBonus = true;
		return *this;
	}

	Chain& secondEffectStartIndex(int index) {
		sk->secondEffectStartIndex = index;
		return *this;
	}

	Chain& secondEffectStartIndexEnemy(int index) {
		sk->secondEffectStartIndexEnemy = index;
		return *this;
	}

	Chain& triggerStartIndex(int index) {
		sk->triggerStartIndex = index;
		return *this;
	}

	Chain& asActiveEnemyPokemonAttack() {
		at->asActiveEnemyPokemonAttack = true;
		return *this;
	}

	Chain& asActiveEnemyTerastalPokemonAttack() {
		at->asActiveEnemyTerastalPokemonAttack = true;
		return *this;
	}

	Chain& asActiveEnemyPokemonAttackIfCoinHead() {
		at->asActiveEnemyPokemonAttackIfCoinHead = true;
		return *this;
	}

	Chain& asMyBenchNPokemonAttack() {
		at->asMyBenchNPokemonAttack = true;
		return *this;
	}

	Chain& asEnemyDeckTop10Attack() {
		at->asEnemyDeckTop10Attack = true;
		return *this;
	}

	Chain& noTargetEffect() {
		at->noTargetEffect = true;
		return *this;
	}

	Chain& noTargetWeaknessOnly() {
		at->noTargetWeakness = true;
		return *this;
	}

	Chain& noTargetResistance() {
		at->noTargetResistance = true;
		return *this;
	}

	Chain& noTargetWeaknessResistance() {
		at->noTargetWeakness = true;
		at->noTargetResistance = true;
		return *this;
	}

	Chain& noTargetEffectAndWeakness() {
		at->noTargetEffect = true;
		at->noTargetWeakness = true;
		at->noTargetResistance = true;
		return *this;
	}

	Chain& cannotUseFirstTurn() {
		at->cannotUseFirstTurn = true;
		return *this;
	}

	Chain& cannotUseSameNameAttackPreMyTurn() {
		at->cannotUseSameNameAttackPreMyTurn = true;
		return *this;
	}

	Chain& canOnlyUseEnemyPrizeOne() {
		at->canOnlyUseEnemyPrizeOne = true;
		return *this;
	}

	Chain& deckTopAttack() {
		at->deckTopAttack = true;
		return *this;
	}

	Chain& deckTopSupporter() {
		at->deckTopSupporter = true;
		return *this;
	}

	Chain& noEnergyIfSpecialCondition() {
		at->noEnergyIfSpecialCondition = true;
		return *this;
	}

	Chain& darkness1IfDamaged() {
		at->darkness1IfDamaged = true;
		return *this;
	}

	Chain& prizePlus1() {
		at->prizePlus1 = true;
		return *this;
	}

	Chain& koNoDamageAndEffectAttackNextEnemyTurn() {
		at->koNoDamageAndEffectAttackNextEnemyTurn = true;
		return *this;
	}

	Chain& canUseBench() {
		at->canUseBench = true;
		return *this;
	}

	Chain& canUseFirst() {
		at->canUseFirst = true;
		return *this;
	}

	Chain& noCheckCondition() {
		at->noCheckCondition = true;
		return *this;
	}

private:

	CardMaster& card;

	Attack* at = nullptr;
	Skill* sk = nullptr;
	Effect* ef = nullptr;
	Trigger* tr = nullptr;
	Target* tgt = nullptr;
	Target* tgt2 = nullptr;
	int attackEffectType = 0;
	int costHandTrashCount = 0;

	bool isSkill() {
		return sk != nullptr;
	}

	int currentEffectIndex() {
		return (int)currentEffects().size() - 1;
	}

	std::vector<Effect>& currentEffects() {
		if (sk == nullptr) {
			assert(at != nullptr);
			if (attackEffectType == 1) {
				return at->preEffects;
			} else if(attackEffectType == 2) {
				return at->postEffects;
			} else {
				Exception("attackEffectType not set");
			}
		} else {
			assert(at == nullptr);
			return sk->effects;
		}
	}
};

inline CardMaster DummyCardMaster;
inline Chain ChainInstance{ DummyCardMaster };

inline Chain& CreateCard(int cardId, const char8_t* name, CardType cardType, int no) {
	CardMaster& card = CardTable[cardId];
	card = {};
	card.cardId = cardId;
	card.cardType = cardType;
	card.name = name;
	card.no = no;

	NameTable[card.name] = cardId;

	Chain* c = new(&ChainInstance) Chain(card);
	return *c;
}
