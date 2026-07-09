// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Skill.h"

struct CardRef {
	unsigned char cardIndex;

	CardRef() = default;
	explicit CardRef(int cardIndex) : cardIndex((unsigned char)cardIndex) {}

	bool isNull() const {
		return cardIndex == 0;
	}
};

inline bool operator==(CardRef left, CardRef right) {
	return left.cardIndex == right.cardIndex;
}
inline bool operator!=(CardRef left, CardRef right) {
	return left.cardIndex != right.cardIndex;
}

struct AreaRef {
	CardRef card;
	int moveCounter;
};

inline bool operator==(AreaRef left, AreaRef right) {
	return left.card == right.card && left.moveCounter == right.moveCounter;
}
inline bool operator!=(AreaRef left, AreaRef right) {
	return !(left == right);
}

struct CardMaster {
	CardId cardId;
	CardType cardType;

	PokemonType pokemonType;
	EvolutionType evolutionType;
	signed char retreatCost;
	int hp;
	EnergyType weakness;
	EnergyType resistance;
	EnergyType energyType; // ポケモンのタイプかエネルギーのタイプ
	bool tera; // tera pokemon

	signed char energyCount; // エネルギー個数

	bool trashMyTurnEnd; // 自分の番の終わりにトラッシュする
	bool cannotToHandOrDeckInTrash; // トラッシュにあるかぎり、手札に加えられず、山札にもどせない
	bool canPlayFirstTurn; // 最初のターンに使用可能
	bool transformOnly; // 変身でしか場に出せない
	bool canTrash; // 自分の番の中でなら、場に出ているこのカードをトラッシュできる
	bool toBench; // 【たね】ポケモンとして、場に出る
	bool toBattleFieldOnlySetup; // 対戦準備でのみ【たね】ポケモンとして、場に出る
	bool toActiveOnlySetup; // 対戦準備でポケモンをバトル場に出すとき、このカードが手札にあるなら、ウラにしてバトル場に出してよい
	bool noPrize; // このカードが【きぜつ】しても、相手はサイドをとれない
	bool onlyTeamRocket; // 「ロケット団のポケモン」にしかつけられず、「ロケット団のポケモン」以外についているなら、トラッシュする

	bool ancient;
	bool future;

	bool hop; // ホップ
	bool lillie; // リーリエ
	bool iono; // ナンジャモ
	bool n; // N
	bool ethan; // ヒビキ
	bool cynthia; // シロナ
	bool misty; // カスミ
	bool arven; // ペパー
	bool steven; // ダイゴ
	bool marnie; // マリィ
	bool erika; // エリカ
	bool larry; // アオキ
	bool teamRocket; // ロケット団

	bool aceSpec; // ACE SPEC
	bool canUse;
	int no;
	std::u8string code;
	Skill* ability;
	Skill* play;
	Skill* delay; // 遅延発動スキル用
	std::vector<Attack*> attacks;
	std::u8string evolvesFrom;
	std::u8string evolvesFrom2; // stage2 -> basic
	std::u8string name;
	std::u8string nameEn;

	// 化石や人形は含まない
	bool isPokemonCard() const {
		return cardType == CardType::Pokemon;
	}

	// 対戦準備で出せるカード
	bool canSetup() const {
		return (cardType == CardType::Pokemon && evolutionType == EvolutionType::Basic) || toBattleFieldOnlySetup;
	}

	bool canSetupActive() const {
		return canSetup() || toActiveOnlySetup;
	}

	// 化石や人形を含む
	bool isPokemon() const {
		return pokemonType != PokemonType::NotPokemon;
	}

	// ルールを持つポケモン
	bool isRulePokemon() const {
		return pokemonType == PokemonType::Ex || pokemonType == PokemonType::MegaEx;
	}

	bool isEx() const {
		return pokemonType == PokemonType::Ex || pokemonType == PokemonType::MegaEx;
	}

	// ルールを持たないポケモン
	// 化石や人形を含む
	bool isNotRulePokemon() const {
		return !isRulePokemon() && pokemonType != PokemonType::NotPokemon;
	}

	// ルールを持たないポケモン
	// 化石や人形は含まない
	bool isNotRulePokemonCard() const {
		return pokemonType == PokemonType::Normal;
	}

	bool hasAttack(const Attack* attack) const {
		for (const Attack* at : attacks) {
			if (at == attack) {
				return true;
			}
		}
		return false;
	}

	FixedList<const Skill*, 2> getSkills() const {
		FixedList<const Skill*, 2> skills;
		if (ability) {
			skills.push_back(ability);
		}
		if (play) {
			skills.push_back(play);
		}
		return skills;
	}
};

inline std::unordered_map<CardId, CardMaster> CardTable; // Both card master
inline std::unordered_map<int, Skill> SkillTable;
inline std::unordered_map<int, Attack> AttackTable;
inline std::unordered_map<std::u8string, int> NameTable;

// 次の自分の番、自分は～/次の相手の番、相手は～
union CardNextTurnState {
	std::array<unsigned, 4> value;
	struct {
		short cannotUseAttackId; // ワザの選択自体ができない。ワザが使われることがあっても失敗する
		short cannotUseAttackId2;
		short damageChange;
		short damageChangeActive;
		short damageChangeMyAttack;
		signed char attackCostChange;
		signed char retreatCostChange;
		bool cannotRetreat : 1;
		bool cannotHandAttachEnergy : 1;
		bool cannotAttack : 1;
		bool cannotAttackLessEqualEnergy2 : 1;
		bool attackCoin : 1;
		bool attackCoin2 : 1;
	};
};

// 次の自分の番、相手は～
// この効果を受けるのは相手
union CardNextMyTurnEnemyState {
	std::array<unsigned, 1> value;
	struct {
		short takeDamageChange;
	};
};

struct Card {
	CardId cardId;
	int moveCounter;
	int attachMoveCounter;
	int skillOrder;
	int damage;

	CardNextTurnState thisTurn;
	CardNextTurnState nextTurn;

	CardNextMyTurnEnemyState thisTurnEnemy;
	CardNextMyTurnEnemyState nextTurnEnemy;

	int takeAttackDamageThisTurn;
	int takeAttackDamagePreTurn;

	signed char playerIndex;
	AreaType area;
	AreaType preArea;
	bool reverse;

	short cannotUseAttackIdNonActive; // バトル場をはなれるまで使えない
	FixedList<short, 8> abilityUsed; // onceTurnの特性を使ったときにセット

	union {
		unsigned short nextEnemyTurnEndStateBattleField; // ベンチにもどっても消えない
		struct {
			bool noDamageAndEffectEnemyExAttackNextEnemyTurn : 1;
		};
	};
	union {
		unsigned nextEnemyTurnEndState; // 次の相手の番の終わりまで(自分にかける効果)
		struct {
			short takeDamageChangeNextEnemyTurn;
			unsigned char noDamageLessEqualAttackNextEnemyTurn;
			bool noDamageAndEffectAttackNextEnemyTurn : 1;
			bool noDamageAndEffectEnemyAttackNextEnemyTurn : 1;
			bool noDamageAttackNextEnemyTurn : 1;
			bool noDamageBasicAttackNextEnemyTurn : 1;
			bool noDamageBasicColorAttackNextEnemyTurn : 1;
			bool noDamageAbilityAttackNextEnemyTurn : 1;
			bool noWeaknessNextEnemyTurn : 1;
		};
	};
	union {
		std::array<unsigned, 3> turnState; // このターン中状態
		struct {
			short damageChangeThisTurn;
			short damageChangeExThisTurn;
			unsigned char koCauseRef;
			signed char koPrizeChangeAlways;
			signed char koPrizeChange;
			bool appear : 1;
			bool evolved : 1;
			bool benchToActive : 1; // この番にベンチからバトル場に出ている
			bool ko : 1; // きぜつする
			bool koAttackDamage : 1;
			bool koEnemyAttackDamage : 1;
			bool koEnemyAttackDamageActive : 1;
			bool koEnemyExAttackDamage : 1;
			bool koEnemyTerastalAttackDamage : 1;
			bool koEnemyNAttackDamage : 1;
			bool koFull : 1; // HPがまんたんの状態で、ワザのダメージを受けてきぜつ
			bool koPrizePlus1 : 1; // サイドをとられる数が+1
			bool koPrizeDecreaseOnce : 1;
			bool koPrizeZero : 1;
			bool koNoDamageAndEffectAttackNextEnemyTurn : 1;
		};
	};
	union {
		std::array<unsigned long long, 5> continualState;
		struct {
			short hpChange;
			short damageChange;
			short damageChangeActive;
			short damageChangeEx;
			short damageChangeAbility;
			short damageChangeEvolved;
			short damageChangeEnemyTakenPrize;
			short takeDamageChange;
			short takeEnemyAttackDamageChange;
			short takeEnemyAbilityPokemonAttackDamageChange;
			short takeEnemyFireOrWaterPokemonAttackDamageChange;
			short takeEnemy4TypePokemonAttackDamageChange;
			short noDamageGreaterEqual;
			signed char retreatCostChange;
			signed char attackCostChangeColorless;
			signed char attackCostDown;
			signed char attackCostDownColorlessOwnAttack;
			signed char typeIndex;
			signed char weaknessIndex;
			bool noAbility : 1;
			bool noKoMeAbility : 1;
			bool noDamageEnemyAbilityPokemonAttack : 1;
			bool noDamageEnemyExAttack : 1;
			bool noDamageEnemyBasicExAttack : 1;
			bool noDamageAndEffectEnemyTerastalAttack : 1;
			bool noDamageAndEffectEnemySpecialEnergyAttack : 1;
			bool noDamageEnemyAttack : 1;
			bool noEffectEnemyAttack : 1;
			bool noEffectEnemyItem : 1;
			bool noEffectEnemySupporter : 1;
			bool noDamageCounterEnemyAttackAbility : 1;
			bool noEnemyAbility : 1;
			bool noSpecialCondition : 1;
			bool noSleepParalyzeConfuse : 1;
			bool noSleep : 1;
			bool noRetreatCost : 1;
			bool noPrizeEx : 1;
			bool notRecoverConfuseEvolve : 1;
			bool canUsePreEvolutionAttack : 1;
			bool canEvolveAppearTurn : 1;
			bool canEvolveGrassAppearTurn : 1;
			bool canAttackFirst : 1;
			bool cannotRetreat : 1;
			bool cannotAttack : 1;
			bool cannotToHand : 1;
			bool cannotMoveDamageCounter : 1;
			bool attackEnergyColoressOne : 1;
			bool attackEnergyPsychicOne : 1;
			bool doubleGrassEnergy : 1;
			bool noDamageCoin : 1;
			bool koByDamageToHand : 1;
			bool basicPrizePlus1 : 1;
			bool doubleAttack : 1;
			bool tool2 : 1;
			bool tool4 : 1;
			bool technicalMachine : 1;
			bool specialFlagTool : 1;
			bool rainbowDna : 1;
			bool canPlay : 1;
		};
	};



	void init(CardId cardId, int moveCounter, int playerIndex) {
		this->cardId = cardId;
		this->moveCounter = moveCounter;
		this->playerIndex = playerIndex;
		area = AreaType::Deck;
	}

	void clearState() {
		nextEnemyTurnEndStateBattleField = {};
		turnState = {};
		abilityUsed.clear();
		continualState = {};
		damage = 0;
		skillOrder = 0;
		takeAttackDamageThisTurn = 0;
		takeAttackDamagePreTurn = 0;
	}

	void clearNextTurnState() {
		thisTurn.value = {};
		nextTurn.value = {};
		thisTurnEnemy.value = {};
		nextTurnEnemy.value = {};
		nextEnemyTurnEndState = {};
	}

	void copyState(const Card& src) {
		CardId preId = cardId;
		*this = src;
		cardId = preId;
	}

	void turnStart(int activePlayerIndex) {
		takeAttackDamagePreTurn = takeAttackDamageThisTurn;
		takeAttackDamageThisTurn = 0;
		if (playerIndex == activePlayerIndex) {
			thisTurn = nextTurn;
			nextTurn.value = {};
		} else {
			thisTurnEnemy = nextTurnEnemy;
			nextTurnEnemy.value = {};
		}
	}

	void turnEnd(int activePlayerIndex) {
		turnState = {};
		abilityUsed.clear();
		thisTurn.value = {};
		thisTurnEnemy.value = {};
		if (playerIndex != activePlayerIndex) {
			nextEnemyTurnEndState = {};
			nextEnemyTurnEndStateBattleField = {};
		}
	}

	const CardMaster& getMaster() const {
		return CardTable.at(cardId);
	}

	ByteFixedList<EnergyType, 2> getEnergyType() const{
		ByteFixedList<EnergyType, 2> result;
		result.push_back(getMaster().energyType);
		if (typeIndex > 0) {
			result.push_back(EnergyTypes.at(typeIndex));
		}
		return result;
	}

	bool canAttack() const {
		return !cannotAttack && !thisTurn.cannotAttack;
	}
};


struct Deck {
	std::array<CardId, DECK_SIZE> cards;
};
