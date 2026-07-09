// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Types.h"
#include "FixedList.h"

using CardId = int;

struct TargetCondition {
	TargetType targetType;
	ComparatorType comparatorType;
	int val;
	int val2;
	std::u8string name;
};

struct Target {
	TargetPlayer targetPlayer;
	bool notMe;
	bool skipEnemyTarget; // trueなら相手カードの場合conditionsを無視する
	ByteFixedList<AreaType, 4> areas;
	std::vector<TargetCondition> conditions;
};

struct Effect {
	bool isCondition;
	EffectType effectType;
	EffectSelectType effectSelectType;
	signed char selectCount;
	SelectContext selectContext;
	bool enemySelect; // 相手が選択
	bool randomSelect; // ランダム選択
	bool eachSelectedList; // selectedListに含まれている対象それぞれに効果発動
	bool eachList; // eachListに含まれている対象それぞれに効果発動
	bool addCheckList; // targetをselectedListに追加する
	bool notClearSelectedList; // selectedListを消去しない
	bool notPreTarget; // preTargetに含まれている対象を除く
	bool notUpdateTarget; // targetListをそのまま保持する
	bool multiplyEffectValuePreTargetCount; // preTargetの数をvaluesにかける
	bool multiplyEffectValueCoinHeadCount; // コインでオモテが出たの数をvaluesにかける
	bool canNoSelect; // 0枚選択可能
	bool canNoSelectIfExistPreTarget; // preTargetが存在すれば0枚選択可能
	bool cannotNoSelect; // 0枚選択不可能
	bool energyMaxSelect; // SelectType::Energyで～個まで選択
	bool selectTargetCount; // 前の対象カード数だけ選択できる
	bool selectCoinHeadCount; // コインがオモテの数だけ選択できる
	bool selectCoinHeadCount2; // コインがオモテの数×2だけ選択できる
	bool selectEnemyEnergyCount; // 相手のポケモン全員についているエネルギーの数ぶんまで選択できる
	bool skipNoTarget; // 対象がいなければ選択スキップ
	bool open; // 公開する
	bool setTargetSwitchBench; // 入れ替えでベンチに行ったポケモンをtargetListに入れる
	bool effectTargetActive; // バトル場対象の効果とみなす
	bool effectTargetBench; // ベンチ対象の効果とみなす
	bool removeEffectedIfNoEffect; // 効果なしの場合にtargetListから除外
	bool seeingDeck; // クライアントにデッキ情報を送る
	bool separator; // 効果の切れ目
	signed char loopCount; // この回数だけ実行される
	short priority;
	std::array<int, 2> values;
	Target target;
	ConditionType conditionType;
	ComparatorType comparatorType;
	signed char failSkip; // 設定されていたら、conditionが満たされなかったときに、この数だけ飛ばす
	int skillId;

	struct Skill* skill;
	struct Attack* attack;

	bool isInstantEffect() const {
		return effectType > EffectType::ContinualEffectSeparator;
	}

	bool isNotOpenSelect() const {
		if (target.areas.size() == 1) {
			AreaType area = target.areas[0];
			if (area == AreaType::Deck || area == AreaType::Looking) {
				return true;
			}
		}
		return false;
	}

	bool isNotOpenSelectNoCondition() const {
		if (isNotOpenSelect() && target.conditions.size() > 0) {
			return true;
		}
		return false;
	}
};

struct Trigger {
	TriggerType triggerType;
	Target subject;
};

struct Skill {
	int skillId;
	CardId cardId;
	SkillType skillType;
	bool mainAbility; // Main選択タイミングで使用可能ならtrue
	bool onceTurn; // ターン1回
	bool canSelectActivate; // trueなら発動するか選べる
	bool notStack; // 同じスキルが重ならない
	bool canActivateTrash; // トラッシュで発動できる
	bool attachBench; // ベンチポケモンにつけたときの効果
	bool koMeAbility; // このポケモンを【きぜつ】させる
	bool luckyBonus;
	signed char priority;
	signed char firstConditionCount;
	signed char secondEffectStartIndex; // 2つから効果を選べる場合の、2つ目の開始インデックス
	signed char secondEffectStartIndexEnemy; // 相手が2つから効果を選べる場合の、2つ目の開始インデックス
	signed char triggerStartIndex; // トリガー能力が始まるインデックス
	ByteFixedList<AreaType, 2> areas;
	std::vector<Trigger> triggers;
	std::vector<Effect> effects;
	std::u8string name;
	std::u8string nameEn;
	std::u8string text;
	std::u8string textEn;

	bool hasTrigger() const {
		return !triggers.empty();
	}

	bool hasContinual() const {
		return (triggers.empty() || triggerStartIndex > 0) && !mainAbility;
	}

	bool isAreaMatch(AreaType area) const {
		for (AreaType areaType : areas) {
			if (areaType == area) {
				return true;
			}
		}
		return false;
	}
};

using AttackEnergies = ByteFixedList<EnergyType, 7>;

// ワザ
struct Attack {
	int attackId;
	CardId cardId;
	int damage;
	union {
		unsigned attackFlags;
		struct {
			bool asActiveEnemyPokemonAttack : 1;
			bool asActiveEnemyTerastalPokemonAttack : 1;
			bool asActiveEnemyPokemonAttackIfCoinHead : 1;
			bool asMyBenchNPokemonAttack : 1;
			bool asEnemyDeckTop10Attack : 1; // 相手の山札を上から10枚オモテにする。のぞむなら、その中にあるポケモンが持つワザを1つ選び、このワザとして使う
			bool noTargetEffect : 1; // このワザのダメージは、対象のポケモンにかかっている効果を計算しない
			bool noTargetWeakness : 1; // このワザのダメージは、弱点を計算しない
			bool noTargetResistance : 1; // このワザのダメージは、抵抗力を計算しない
			bool cannotUseFirstTurn : 1; // 後攻プレイヤーの最初の番には使えない
			bool cannotUseSameNameAttackPreMyTurn : 1; // 前の自分の番に、自分のポケモンが同名のワザを使っていたなら、このワザは使えない
			bool canOnlyUseEnemyPrizeOne : 1; // このワザは、相手のサイドの残り枚数が1枚のときにしか使えない
			bool deckTopAttack : 1; // 自分の山札を上から1枚トラッシュし、そのカードがポケモン（「ルールを持つポケモン」をのぞく）なら、そのポケモンが持っているワザを1つ選び、このワザとして使う
			bool deckTopSupporter : 1; // 自分の山札を上から1枚トラッシュし、そのカードがサポートなら、その効果を、このワザの効果として使う
			bool noEnergyIfSpecialCondition : 1; // このワザは、このポケモンが特殊状態なら、このワザを使うためのエネルギーがこのポケモンについていなくても、使える。
			bool darkness1IfDamaged : 1; // このワザは、このポケモンにダメカンがのっているなら、【悪】エネルギー1個で使える
			bool prizePlus1 : 1; // このワザのダメージで、相手のポケモンが【きぜつ】したなら、サイドを1枚多くとる
			bool koNoDamageAndEffectAttackNextEnemyTurn : 1; // このワザのダメージで、相手のポケモンが【きぜつ】したなら、次の相手の番、このポケモンはワザのダメージや効果を受けない
			bool canUseBench : 1; // このワザは、このポケモンがベンチにいても使える
			bool canUseFirst : 1; // このワザは、先攻プレイヤーの最初の番でも使える
			bool noCheckCondition : 1; // Conditionによる使用可否チェックをしない
		};
	};

	AttackEnergies energies;
	std::vector<Effect> preEffects; // ワザの成否やダメージに影響がある効果や「ダメージを与える前に」と書かれた効果
	std::vector<Effect> postEffects; // 上記以外
	std::string damageText;
	std::u8string name;
	std::u8string nameEn;
	std::u8string text;
	std::u8string textEn;
	bool lastCancelFailAttack;
};
