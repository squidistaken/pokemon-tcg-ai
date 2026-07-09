// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Core.h"
#include "ApiType.h"

enum class EffectType : unsigned char {
	NoEffect,

	SelectCard, // 選択結果をselectedListに格納
	ForEach, // 選択結果をeachListに格納
	Ko,
	ToHand,
	PrizeToHand, // サイドをとる
	ToHandReverse,
	ToHandWithAttach,
	ToTrash,
	ToDeck,
	ToDeckWithAttach,
	ToDeckReverse,
	LookToDeckReverse,
	ToDeckAndShuffle,
	ToDeckReverseAndShuffle,
	ToDeckBottom, // デッキの下に送る
	ToDeckBottomReverse, // デッキの下に送る。相手には見えない
	ToDeckBottomClose, // 両者から見えないままデッキの下に送る。複数枚の場合は順番ランダム
	ToActiveAndTrashActive,
	ToBench,
	ToPrize,
	ToLooking,
	ToPlayingFirst, // 最初の1枚を送る
	Switch,
	SwitchDeck, // 山札の上のカードと入れ替える
	NotMove,
	LookDeck,
	LookDeckReverse,
	LookDeckBottom,
	LookAndReturn, // オモテを見て、もとにもどす
	DamageCounter,
	DamageCounterRemoved,
	DamageCounterDamaged, // 受けたダメージぶん
	DamageCounterAny, // 好きなようにのせる
	DamageCounterDouble,
	DamageCounterHp, // 残りHPを指定してダメカンをのせる
	DamageCounterSwitchAny, // 相手の場のポケモンにのっているダメカンを好きなだけ選び、相手の場のポケモンに好きなようにのせ替える
	DamageCounterTypeEnergyCountMe, // 付いている特定タイプのエネルギーの数×effectValue2だけダメカンをのせる
	AttackDamage,
	AttackDamageMulti, // ポケモンをeffectValue回選び、選んだポケモン全員に、選んだ回数×effectValue2ダメージ
	AttackDamageCoin, // 対象に対してそれぞれ1回ずつコインを投げ、オモテが出たポケモン全員にダメージ
	RemoveDamageCounter,
	RemoveDamageCounterAll,
	Heal,
	HealAll, // HP全回復
	HealSand, // ペパーのサンドウィッチ
	ResetHp, // HPリセット
	Drain, // 相手のバトルポケモンに与えたダメージぶん、HPを回復する
	Devolve, // 退化。effectValueは退化したカードの移動先エリア
	DevolveAny, // 好きなだけ退化
	TransformDeck, // ポケモンを入れ替える。ついているカード・ダメカン・特殊状態・効果などは、すべて引きつぐ。元のカードは山札に戻す
	TransformTrash, // ポケモンを入れ替える。ついているカード・ダメカン・特殊状態・効果などは、すべて引きつぐ。元のカードはトラッシュする
	ExchangeSelected, // カードを入れ替え
	KoPrizeChangeAlways, // このポケモンが【きぜつ】したときのとられるサイド枚数変化
	KoPrizeChange, // このポケモンが相手のポケモンからワザのダメージを受けて【きぜつ】したときのとられるサイド枚数変化
	KoPrizeDecreaseOnce, // このポケモンが相手のポケモンからワザのダメージを受けて【きぜつ】したとき、とられるサイドは1枚少なくなる。対戦中1回のみはたらく
	SelectEvolvesFrom, // 進化元を選択
	EvolvesToEach, // 進化元それぞれの進化先を選択。このターンに進化しているかは見ない
	SelectEvolvesTo, // 進化先を選択
	EvolvesFromEach, // 進化先それぞれの進化元を選択。このターンに進化しているかは見ない
	SelectAttachFrom, // 付ける先を選択
	AttachToEach, // ポケモンそれぞれに大して付けるカードを選択
	SelectAttachTo, // ポケモンに付けるカードを選択
	AttachEnergyMe,
	AttachSelectedCard, // selectedListのエネルギーを付ける
	SwitchSelectedCard, // selectedListのエネルギーを付ける
	AttachFromEach, // ポケモンに付けるカードそれぞれに対してつけるポケモンを選択
	SelectSwitchEnergy, // 付け替えるエネルギーを選択
	SelectSwitchEnergyCard, // 付け替えるエネルギーカードを選択
	EnergySwitchEach, // 付けかえるエネルギーカードそれぞれに対してつけるポケモンを選択
	DelayEffect, // 効果発動予約

	Coin, // effectValue回コインを投げる
	CoinUntilTail, // ウラが出るまでコインを投げる。effectValueを1にすると、1回目でウラが出たときに効果終了する

	AttackDamageChange,
	AttackDamageChangeTargetCount, // ターゲットの数×effectValueだけダメージ変化
	EffectDamageChangeTargetCount, // ターゲットの数×effectValueだけダメージ変化
	AttackDamageChangeEnergyCount, // 付いているエネルギーの数（カード枚数ではない）×effectValueだけダメージ変化
	EffectDamageChangeEnergyCount, // 付いているエネルギーの数（カード枚数ではない）×effectValueだけダメージ変化
	AttackDamageChangeTypeEnergyCount, // 付いている特定タイプのエネルギーの数×effectValue2だけダメージ変化
	EffectDamageChangeTypeEnergyCount, // 付いている特定タイプのエネルギーの数×effectValue2だけダメージ変化
	AttackDamageChangeEnergyCountCoin, // 付いているエネルギーの数ぶんコインを投げる。オモテの数×effectValueダメージ
	AttackDamageChangeTypeEnergyCountCoin, // 付いている特定タイプのエネルギーの数だけコインを投げ、オモテの数×effectValue2だけダメージ変化
	AttackDamageChangeCoin, // コインをeffectValue回投げ、オモテの数×effectValue2ダメージ
	AttackDamageChangeCoinUntilTail, // ウラが出るまでコインを投げ、オモテの数×effectValueダメージ
	AttackDamageChangeTargetCountCoin, // 対象の数ぶんコインを投げる。オモテの数×effectValueダメージ
	AttackDamageChangeTargetCountEnemyCoin, // 相手は対象の数ぶんコインを投げる。ウラの数×effectValueダメージ
	AttackDamageChangeTakenPrize, // すでにとったサイドの枚数×effectValueだけダメージ変化
	EffectDamageChangeTakenPrize, // すでにとったサイドの枚数×effectValueだけダメージ変化
	AttackDamageChangeDamageCounter, // ターゲットのダメカン合計数×effectValueだけダメージ変化
	EffectDamageChangeDamageCounter, // ターゲットのダメカン合計数×effectValueだけダメージ変化
	AttackDamageChangePutDamageCounter, // このポケモンにダメカンをeffectValue個までのせ、のせた数×effectValue2ダメージ変化
	AttackDamageChangeRetreatCost, // ターゲットの【にげる】ためのエネルギーの合計数×effectValueだけダメージ変化
	AttackDamageChangeTypeCount, // ターゲットのタイプの数×effectValueだけダメージ変化
	AttackDamageChangeSpecialConditionCount, // バトルポケモンの特殊状態の数×effectValueだけダメージ変化
	AttackDamageChangeTakeAttackDamagePreTurn, // 前の相手の番に受けたワザのダメージと同じダメージ追加
	AttackDamageChangePreTurnTakePrizeCount, // 前の相手の番に、相手がとったサイドの枚数×effectValueダメージ変化

	Burn,
	Poison,
	Poison8,
	Poison16,
	Sleep,
	Confuse,
	Paralyze,
	RecoverSpecialCondition, // 特殊状態を、すべて回復する
	RecoverSpecialConditionSingle, // 特殊状態を1つ回復する
	Draw,
	DrawTargetCount, // ターゲットの数だけドロー
	DrawPrizeCount, // 自身のサイド残り枚数ぶんドロー
	DrawUntil, // 手札がeffectValue枚になるようにドロー
	DrawUntilPsychic, // 自分の手札の枚数が、自分の場の【超】ポケモンの数と同じ枚数になるようにドロー
	DrawMirror, // 自分の手札が、相手の手札と同じ枚数になるように、山札を引く
	DeckToTrash,
	DeckToTrashCoinUntilTail, // ウラが出るまでコインを投げ、オモテの数ぶん、山札を上からトラッシュする。
	DeckBottomToTrash,
	DeckToPrize,
	Shuffle,
	EffectWin, // 対戦に勝利する
	FailRetreat, // 【にげる】ためのエネルギーはトラッシュせず、入れ替えをしない

	TurnEnd, // 番が終わる

	// この番

	DamageChangeThisTurn, // この番、このポケモンが使うワザの、相手のバトルポケモンへのダメージ増減
	DamageChangeExThisTurn,

	// プレイヤー
	PlayerDamageChange, // この番、自分のポケモンが使うワザの、相手のバトルポケモンへのダメージ増減
	PlayerDamageChangeEx, // この番、自分のポケモンが使うワザの、相手のバトル場の「ポケモン【ex】」へのダメージ増減
	PlayerDamageChangeMyFighting, // この番、自分の【闘】ポケモンが使うワザの、相手のバトルポケモンへのダメージ増減
	TakePrizeCountChangeTerastalAttackKoActive,  // 自分の「テラスタル」のポケモンが使うワザのダメージで、相手のバトルポケモンが【きぜつ】したときのサイドを取る枚数変化
	TakePrizeCountChangeNAttackKoActive,  // この番、自分の「Nのポケモン」が使うワザのダメージで、相手のバトルポケモンが【きぜつ】したときのサイドを取る枚数変化

	TakeDamageChangeNextEnemyTurn,
	NoDamageLessEqualAttackNextEnemyTurn, // 次の相手の番、「effectValue」以下のワザのダメージを受けない
	NoDamageAndEffectAttackNextEnemyTurn, // 次の相手の番、ワザのダメージや効果を受けない
	NoDamageAndEffectEnemyAttackNextEnemyTurn, // 次の相手の番の終わりまで、相手のポケモンからワザのダメージや効果を受けない
	NoDamageAndEffectEnemyExAttackNextEnemyTurn, // 次の相手の番の終わりまで、相手のポケモンからワザのダメージや効果を受けない
	NoDamageAttackNextEnemyTurn, // 次の相手の番、ワザのダメージを受けない
	NoDamageBasicAttackNextEnemyTurn, // 次の相手の番、【たね】ポケモンからワザのダメージを受けない
	NoDamageBasicColorAttackNextEnemyTurn, // 次の相手の番、【たね】ポケモン（【無】ポケモンをのぞく）からワザのダメージを受けない
	NoDamageAbilityAttackNextEnemyTurn, // 次の相手の番、特性を持つポケモンからワザのダメージを受けない
	NoWeaknessNextEnemyTurn, // 次の相手の番、ポケモンの弱点は、すべてなくなる

	CannotUseThisAttackNextTurn, // 次の自分の番、このポケモンは「このワザ」が使えない
	DamageChangeMyAttackNextTurn, // 次の自分の番、このポケモンの持っているダメージワザのダメージ変化
	DamageChangeActiveNextTurn, // 次の自分の番、このポケモンが使うワザの、相手のバトルポケモンへのダメージ変化
	DamageChangeNextTurn, // 次の相手の番、ポケモンが使うワザのダメージ変化(相手限定ではなく、バトルポケモン限定でもない)
	AttackCostChangeNextTurn, // 次の相手の番、ワザを使うための【無】エネルギー変化
	RetreatCostChangeNextTurn, // 次の相手の番、【にげる】ための【無】エネルギー変化

	CannotRetreatNextTurn, // 次の相手の番、にげられない
	CannotHandAttachEnergyNextTurn, // 次の相手の番、手札から出すエネルギーをつけられない
	CannotAttackNextTurn, // 次の相手の番、ワザが使えない
	CannotAttackLessEqualEnergy2NextTurn, // 次の相手の番、ついているエネルギーが2個以下なら、ワザが使えない
	AttackCoinNextTurn, // 次の相手の番、このワザを受けたポケモンがワザを使うとき、相手はコインを1回投げる。ウラならそのワザは失敗
	AttackCoin2NextTurn, // 次の相手の番、このワザを受けたポケモンがワザを使うとき、相手はコインを2回投げる。1回でもウラなら、そのワザは失敗
	CannotUseSelectedAttack, // 相手のバトルポケモンが持つワザを1つ選ぶ。次の相手の番、このワザを受けたポケモンは、選ばれたワザが使えない

	// プレイヤー
	MetalDamageChangeNextTurn, // "次の相手の番、自分の【鋼】ポケモン全員が、相手のポケモンから受けるワザのダメージ変化（新しく場に出したポケモンもふくむ。）
	CannotAttackLessEqualEnergy2NextTurnPlayer, // 次の相手の番、ついているエネルギーが2個以下のポケモン全員は、ワザが使えない。（新しく場に出したポケモンもふくむ。）
	CannotPlayItemNextTurn, // 次の相手の番、手札からグッズを出して使えない
	CannotPlaySupporterNextTurn, // 次の相手の番、手札からサポートを出して使えない
	CannotPlayStadiumNextTurn, // 次の相手の番、手札からスタジアムを出せない
	CannotPlaySpecialEnergyNextTurn, // 次の相手の番、手札から特殊エネルギーを出してつけられない
	CannotEvolveNextTurn, // 次の相手の番、手札からポケモンを出して進化させられない
	CannotRetreatPoison, // 次の相手の番、相手の【どく】のポケモンは、にげられない。（新しく【どく】にしたポケモンもふくむ。）

	// 次の自分の番、相手は～
	// この効果を受けるのは相手
	TakeDamageChangeNextMyTurnEnemy,

	// 永続
	CannotUseThisAttackNonActive, // バトル場をはなれるまで「このワザ」が使えない

	// 効果選択
	SelectActivate, // Noを選んだら効果終了
	SelectEffect, // 効果を選ぶ

	// ワザ系
	FailAttack, // ワザ失敗
	CancelFailAttack, // ワザ失敗取り消し

	BreakIfCoinHead, // コインを1回投げオモテなら効果終了
	BreakIfCoinTail, // コインを1回投げウラなら効果終了
	BreakIfCoinTailMulti, // コインをeffectValue回投げ1回でもウラなら効果終了
	SkipIfCoinTail, // コインを1回投げウラならeffectValueだけEffectをスキップ。effectValue2が1なら相手が投げる
	PostEffectActivate, // postEffectを有効にする
	BreakIfNotPostEffectActivated, // postEffectが有効でなければ効果終了
	SelectPoisonBurnConfuse, // 【どく】・【やけど】・【こんらん】の中から1つ選ぶ
	SelectSpecialCondition, // 特殊状態の中から1つ選び、相手のバトルポケモンをその状態にする

	ContinualEffectSeparator, // ここからContinual Effect

	MaxHpChange,
	MaxHpChangeFighting, // このポケモンについている【闘】エネルギー1個につき、このポケモンの最大HPはeffectValue変化する
	DamageChange, // ポケモンが使うワザのダメージ増減
	DamageChangeActive, // ポケモンが使うワザの、相手のバトルポケモンへのダメージ増減
	DamageChangeEx, // ポケモンが使うワザの相手のバトル場の「ポケモン【ex】」へのダメージ増減
	DamageChangeAbility, // ポケモンが使うワザの相手のバトル場の特性を持つポケモンへのダメージ増減
	DamageChangeEvolved, // ポケモンが使うワザの、相手のバトル場の進化ポケモンへのダメージ増減
	DamageChangeEnemyTakenPrize, // ポケモンが使うワザの、相手のバトルポケモンへのダメージが、相手がすでにとったサイド1枚につき増減
	TakeDamageChange, // 受けるワザのダメージ増減
	TakeEnemyAttackDamageChange, // 相手のポケモンから受けるワザのダメージ増減
	TakeEnemyAbilityPokemonAttackDamageChange, // 相手の特性を持つポケモンから受けるワザのダメージ増減
	TakeEnemyFireOrWaterPokemonAttackDamageChange, // 相手の【炎】【水】ポケモンから受けるワザのダメージ増減
	TakeEnemy4TypePokemonAttackDamageChange, // 相手の【草】【炎】【水】【雷】ポケモンから受けるワザのダメージ増減
	NoDamageGreaterEqual, // 相手のポケモンから「effectValue」以上のワザのダメージを受けない
	RetreatCostChange,
	AttackCostChangeColorless,
	AttackCostDown,
	AttackCostDownColorlessTargetCount, // このポケモンがワザを使うための【無】エネルギーがターゲットの数だけ少なくなる
	AttackCostDownColorlessOwnAttackEnemyTakenPrize,
	AddEnergyType, // 追加タイプ。effectValueはEnergyTypeIndex
	SetWeakness, // effectValueはEnergyTypeIndex

	NoAbility, // 特性はすべてなくなる
	NoKoMeAbility, // 自身を【きぜつ】させる効果の特性が、すべてなくなる
	NoDamageEnemyAttack, // 相手のワザのダメージを受けない
	NoDamageEnemyAbilityPokemonAttack, // 相手の特性を持つポケモンからワザのダメージを受けない
	NoDamageEnemyExAttack, // 相手の「ポケモン【ex】」からワザのダメージを受けない
	NoDamageEnemyBasicExAttack, // 相手の【たね】ポケモンの「ポケモン【ex】」からワザのダメージを受けない
	NoDamageAndEffectEnemyTerastalAttack, // 相手の「テラスタル」のポケモンからワザのダメージや効果を受けない
	NoDamageAndEffectEnemySpecialEnergyAttack, // 特殊エネルギーがついている相手のポケモンから、ワザのダメージや効果を受けない
	NoEffectEnemyAttack, // 相手のポケモンが使うワザの効果を受けない
	NoDamageAndEffectEnemyAttack, // 相手のポケモンが使うワザのダメージや効果を受けない
	NoEffectEnemyItem, // 相手が手札からグッズを出して使ったとき、その効果を受けない
	NoEffectEnemySupporter, // 相手が手札からサポートを出して使ったとき、その効果を受けない
	NoDamageCounterEnemyAttackAbility, // 相手のワザや特性の効果で、ダメカンがのらない
	NoEnemyAbility, // 相手のポケモンから特性の効果を受けない
	NoSpecialCondition, // 特殊状態にならず、受けている特殊状態は、すべて回復する
	NoSleepParalyzeConfuse, // 【ねむり】・【マヒ】・【こんらん】にならず、受けている【ねむり】・【マヒ】・【こんらん】は、すべて回復する
	NoSleep, // 【ねむり】にならない
	NoRetreatCost, // 【にげる】ためのエネルギーは、すべてなくなる
	NoPrizeEx, // 相手の「ポケモン【ex】」からワザのダメージを受けて【きぜつ】しても、相手はサイドをとれない
	NotRecoverConfuseEvolve, // 進化・退化しても【こんらん】が回復しない
	CanUsePreEvolutionAttack, // 進化前に持っていたワザを、すべて使える
	CanEvolveAppearTurn, // 出したばかりでも進化できる。最初の自分の番に進化できない制限を受けない
	CanEvolveGrassAppearTurn, // 出したばかりでも【草】ポケモンに進化できる
	CanAttackFirst, // 先攻プレイヤーの最初の番でも、ワザが使える
	CannotRetreat, // にげられない
	CannotAttack, // ワザが使えない
	CannotToHand, // 手札に戻せない
	CannotMoveDamageCounter, // ダメカンを、別のポケモンにのせ替えられない
	AttackEnergyColoressOne, // 元々持っているワザを使うためのエネルギーは、【無】エネルギー1個になる。他の増減効果よりも後に適用する
	AttackEnergyPsychicOne, // 元々持っているワザを使うためのエネルギーは、【超】エネルギー1個になる。他の増減効果よりも後に適用する
	DoubleGrassEnergy, // ついている「基本【草】エネルギー」は、それぞれ【草】エネルギー2個ぶんとしてはたらく
	NoDamageCoin, // このポケモンがワザのダメージを受けるとき、自分はコインを1回投げる。オモテなら、このポケモンはそのダメージを受けない
	KoByDamageToHand, // 相手のポケモンからワザのダメージを受けて【きぜつ】したとき、トラッシュせず、手札にもどす。（ポケモン以外のカードは、すべてトラッシュする。）
	BasicPrizePlus1, // このポケモンが使うワザのダメージで、相手の【たね】ポケモンが【きぜつ】したなら、サイドを1枚多くとる
	DoubleAttack, // 持っているワザを2回連続で使える
	Tool2, // 「ポケモンのどうぐ」を2枚までつけられる
	Tool4, // 「ポケモンのどうぐ」を4枚までつけられる
	TechnicalMachine, // わざマシン
	SpecialFlagTool, // どうぐ用個別対応フラグ
	RainbowDna, // にじいろDNA。「イーブイ」から進化する「ポケモン【ex】」を手札から出して、このポケモンにのせて進化できる。（最初の自分の番や、出したばかりの番には進化できない。）

	CanPlay, // 手札状態。使用可能

	// プレイヤー
	PoisonDamageChange, // どくのダメージ増減
	BurnDamageChange, // やけどのダメージ増減
	PoisonDamageChangeNotDarkness, // 悪タイプ以外のどくのダメージ増減
	BenchCapacity,

	CannotPlayItem, // 手札からグッズを出して使えない
	CannotPlayStadium, // 手札からスタジアムを出せない
	CannotPlayTool, // 手札から「ポケモンのどうぐ」を出してつけられない
	CannotPlayAceSpec, // 手札から「【ACE SPEC】」のカードを出して使えない
	CannotPlayAbilityPokemonNotRocket, // 手札から特性を持つポケモン（「ロケット団のポケモン」をのぞく）を場に出せない
	CannotTrashToHandAbilityOrTrainers, // トラッシュにあるカードは、自分の特性またはトレーナーズの効果で、手札に加えられない

	// 全体
	NoToolEffect,
};

// 対象タイプ
enum class TargetType : unsigned char {
	All,

	Hp,
	MaxHp,
	RetreatCost,
	EnergyType,
	EnergyType2, // 2つのどちらかのタイプ
	Resistance, // 抵抗力タイプ

	PokemonCard,
	BasicPokemon, // たねポケモン
	EvolvedPokemon, // 進化ポケモン
	Stage1, // 1進化ポケモン
	Stage2, // 2進化ポケモン
	BasicEnergy,
	SpecialEnergy,
	EnergyCard,
	Item,
	Tool,
	Supporter,
	Stadium,
	Trainer,
	CardId,

	PokemonOrBasicEnergy,
	BasicPokemonOrBasicEnergy,
	NotRulePokemonCardOrBasicEnergy,
	EnergyTypePokemonOrStadium,
	ItemOrTool,
	EnemyToolOrSpecialEnergyOrStadium,
	EthanPokemonOrBasicFireEnergy,

	HasAbility,
	HasAbilityName, // 特定のとくせいを持っているか
	HasAttackName, // 特定のワザを持っているか
	RulePokemon,
	NotRulePokemon, // 化石含む
	Ex,
	MegaEx,
	Terastal,
	Ancient,
	Future,
	Hop,
	Lillie,
	Iono,
	N,
	Ethan,
	Cynthia,
	Misty,
	Arven,
	Steven,
	Marnie,
	Erika,
	Larry,
	TeamRocket,

	SilcoonOrCascoon, // 「カラサリス」または「マユルド」
	KoffingOrWeezing, // 名前に「ドガース」または「マタドガス」とつく
	HonedgeOrDoubladeOrAegislash, // 「ヒトツキ」「ニダンギル」「ギルガルド」

	Name,
	NameContains, // 「○○」とつくカード

	CanEvolve, // 進化可能
	CanEvolve2, // can evolve Basic -> Stage2
	CanEvolveMe, // 自身から進化可能
	CanEvolveContextCard, // contextCardから進化可能
	CanEvolvesToContextCard, // contextCardに進化可能
	CanEvolveField, // 場のポケモンから進化可能
	CanEvolveFieldNotAppearThisTurn, // 出したばかりではない場のポケモンから進化可能
	Evolved, // 進化している（下に進化前カードが存在する）
	EvolvedThisTurnName, // この番「○○」から進化している
	NotAppearThisTurn, // この番に出したばかりのカードではない
	HealThisTurn, // この番にHPを回復している
	AttachedMe,
	AttachedEffected, // 前の効果対象に付いている
	AttachedTriggerSubject,
	AttachedTriggerObject,
	AttachedContextCard,
	AttachedActivePokemon, // バトル場のポケモンに付いている
	AttachedBenchPokemon, // ベンチのポケモンに付いている
	IsAttachedEnergy, // エネルギーが付いている
	AttachedEnergyCount, // ～個エネルギーが付いているポケモン
	IsAttachedSpecialEnergy, // 特殊エネルギーが付いている
	IsAttachedEnergyType, // 特定のタイプのエネルギーが付いている
	IsAttachedEnergy2Type, // 特定のタイプのエネルギーが2個以上付いている
	IsAttachedEnergyName, // 特定のエネルギーが付いている
	IsAttachedTool, // どうぐが付いている
	IsAttachedToolName, // 特定のどうぐが付いている
	IsAttachedToolOrSpecialEnergy, // どうぐか特殊エネルギーが付いている
	NotContextCardAttachedPokemon, // contextCardが付いているポケモンでない
	NotSelectedListAttachedPokemon, // selectedListのカードが付いているポケモンでない
	EnergyTypeAttached, // 特定タイプのエネルギーとして扱われている、付けられたエネルギー
	Reverse, // 裏向き
	Area,
	TriggerSubject,
	TriggerObject,
	DamageCounter, // のっているダメカンの数
	MinHp, // おたがいの場のポケモン（このポケモンをのぞく）の中から、残りHPが一番少ないポケモン
	SameTypeEnemy, // 相手の場のポケモンと同じタイプ
	SpecialCondition, // 特殊状態
	SpecialConditionOrDamaged, // 特殊状態またはダメカンがのっている
	Poison, // どく
	Burn, // やけど
	Confuse, // こんらん
	PoisonOrBurn, // 【どく】または【やけど】
	BenchToActiveThisTurn, // この番にベンチからバトル場に出ている
	SameNameEnemyField, // 相手の場のポケモンと同じ名前のポケモン
	NotChecked, // checkListに含まれていない(カードIDで判定)
};


// 事前条件タイプ
enum class ConditionType : unsigned char {
	Always,

	AnyTargetAfterEffect, // Skill中のこれより後のEffectのTargetのどれかが1つ以上の対象カードを持つか
	CountTarget, // 対象の数
	CountTarget2, // 対象の数2つ
	CountTargetMeOrEnemy, // 自分と相手のどちらかの条件が満たされていればOK
	CompareCountTargetMeEnemy, // 左辺が自分で右辺が相手。TargetPlayerはBothにする
	CountEnergy, // 場のエネルギー全ての個数
	CountEnergyType, // 指定タイプのエネルギーの個数
	CompareCountEnergyMeEnemy, // エネルギー個数。左辺が自分で右辺が相手。TargetPlayerはBothにする
	AttackEnergyExtra, // このワザを使うためのエネルギーより、N個多くエネルギーがついているなら
	NotFullBench,

	MyTurn, // Equalなら自分ターン、NotEqualなら相手ターン
	Turn, // ターン数
	KoPreEnemyTurn, // 前の相手の番に自分のポケモンがきぜつしたか
	KoPreEnemyTurnTeamRocket, // 前の相手の番に自分の「ロケット団のポケモン」がきぜつしたか
	KoAttackDamagePreEnemyTurn, // 前の相手の番にワザのダメージで自分のポケモンがきぜつしたか
	KoAttackDamageEthanPreEnemyTurn, // 前の相手の番にワザのダメージで自分の「ヒビキのポケモン」が【きぜつ】したか
	KoAttackDamageHopPreEnemyTurn, // 前の相手の番にワザのダメージで自分の「ホップのポケモン」が【きぜつ】したか
	NoSameNameSkillThisTurn, // 同名のスキルをこの番に使っていない
	SameAttackPreMyTurn, // 前の自分の番に、このポケモンがこのワザを使っているか
	CoinHeadCount, // コインでオモテが出た回数
	AttachActive, // バトルポケモンにつけたか

	MysteryGarden, // 自分の手札の枚数が、自分の場の【超】ポケモンの数以下
	LoveBall, // ラブラブボール
};

// ビット演算使用
enum class TargetPlayer : unsigned char {
	None = 0,
	Me = 1,
	Enemy = 2,
	Both = 3
};

inline bool IsTargetPlayer(int myPlayerIndex, int targetPlayerIndex, TargetPlayer targetPlayer) {
	return ((1 + (myPlayerIndex ^ targetPlayerIndex)) & (int)targetPlayer) != 0;
}
// 選択数タイプ

enum class EffectSelectType : unsigned char {
	All,

	// ～枚指定
	CardCount,
	MaxCardCount, // 1以上N枚まで
	CardUntil, // N枚になるように（選択肢数 - N）
	MaxCardUntil, // 0～N枚になるように。

	Energy, // N個選ぶ
	MaxEnergyCard, // 1以上N個まで
	ToolCard,

	CardOrAttachedCardCount,

	Evolve,
	Evolve2, // ふしぎなアメ
};

// 能力発動タイミング
enum class TriggerType : unsigned char {
	None,
	TurnEnd,
	PokemonCheckup,
	HandToBench,
	ActiveToBench,
	ToBenchMyTurn, // バトル場からベンチは含まない
	BenchToActive,
	DeckToTrashEnemyEffect, // 相手のワザ・特性・グッズ・サポートの効果で山札からトラッシュされたとき
	EvolveFromHand,
	EnergyAttachFromHand, // 主体はポケモン
	DamagedEnemyAttack, // 相手のポケモンからワザのダメージを受けたとき
	DamagedEnemyAttackActive, // バトル場で相手のポケモンからワザのダメージを受けたとき
	PreKo, // ワザのダメージを受けて【きぜつ】するとき
	PreKoFull, // HPがまんたんの状態で、ワザのダメージを受けて【きぜつ】するとき
	PreKoFullEnemy, // HPがまんたんの状態で、相手のポケモンからワザのダメージを受けて【きぜつ】するとき
	Ko,
	KoEnemyAttackDamage,
	KoEnemyExAttackDamage, // 相手の「ポケモン【ex】」からワザのダメージを受けて【きぜつ】したとき
	KoEnemyAttackDamageActive, // バトル場で相手のポケモンからワザのダメージを受けて【きぜつ】したとき。ワザのダメージをバトル場で受けてHP0になってさえいれば、きぜつのタイミングでベンチにいてもいい
	PreRetreat, // 【にげる】とき
	Attach, // PullTriggerを介さない
};

// Comparator types
enum class ComparatorType : unsigned char {
	Equal,
	GreaterEqual,
	LessEqual,
	NotEqual,
	Greater,
	Less,
};

inline bool Compare(int val, int effectVal, ComparatorType type) {
	switch (type) {
	case ComparatorType::Equal:
		return val == effectVal;
	case ComparatorType::GreaterEqual:
		return val >= effectVal;
	case ComparatorType::LessEqual:
		return val <= effectVal;
	case ComparatorType::NotEqual:
		return val != effectVal;
	case ComparatorType::Greater:
		return val > effectVal;
	case ComparatorType::Less:
		return val < effectVal;
	default:
		assert(false);
		return false;
	}
}

inline bool BoolCompare(bool flag, ComparatorType type) {
	if (type == ComparatorType::Equal) {
		return flag;
	} else {
		return !flag;
	}
}


enum class SkillType : unsigned char {
	Ability,
	Play,
	Attach,
};

enum class EnergyType : unsigned short {
	Colorless = 0,
	Grass = 1 << 0,
	Fire = 1 << 1,
	Water = 1 << 2,
	Lightning = 1 << 3,
	Psychic = 1 << 4,
	Fighting = 1 << 5,
	Darkness = 1 << 6,
	Metal = 1 << 7,
	Dragon = 1 << 8,
	All = (1 << 9) - 1,
};

constexpr EnergyType operator|(EnergyType t0, EnergyType t1) {
	return (EnergyType)((int)t0 | (int)t1);
}

constexpr std::array<EnergyType, 12> EnergyTypes = {
	EnergyType::Colorless,
	EnergyType::Grass,
	EnergyType::Fire,
	EnergyType::Water,
	EnergyType::Lightning,
	EnergyType::Psychic,
	EnergyType::Fighting,
	EnergyType::Darkness,
	EnergyType::Metal,
	EnergyType::Dragon,
	EnergyType::All,
	EnergyType::Psychic | EnergyType::Darkness,
};

inline bool ContainsEnergyType(EnergyType t0, EnergyType t1) {
	return ((int)t0 & (int)t1) != 0;
}

// 無色をタイプの一つとしてとらえる場合
inline bool MatchEnergyType(EnergyType t0, EnergyType t1) {
	if (t1 == EnergyType::Colorless) {
		return t0 == EnergyType::Colorless;
	} else {
		return ((int)t0 & (int)t1) != 0;
	}
}

inline int EnergyTypeIndex(EnergyType type) {
	switch (type)
	{
	case EnergyType::Colorless:
		return 0;
	case EnergyType::Grass:
		return 1;
	case EnergyType::Fire:
		return 2;
	case EnergyType::Water:
		return 3;
	case EnergyType::Lightning:
		return 4;
	case EnergyType::Psychic:
		return 5;
	case EnergyType::Fighting:
		return 6;
	case EnergyType::Darkness:
		return 7;
	case EnergyType::Metal:
		return 8;
	case EnergyType::Dragon:
		return 9;
	case EnergyType::All:
		return 10;
	case EnergyType::Psychic | EnergyType::Darkness:
		return 11;
	default:
		assert(false);
		return 0;
	}
}

inline const char* EnergyText(EnergyType type) {
	switch (type)
	{
	case EnergyType::Colorless:
		return "Colorless";
	case EnergyType::Grass:
		return "Grass";
	case EnergyType::Fire:
		return "Fire";
	case EnergyType::Water:
		return "Water";
	case EnergyType::Lightning:
		return "Lightning";
	case EnergyType::Psychic:
		return "Psychic";
	case EnergyType::Fighting:
		return "Fighting";
	case EnergyType::Darkness:
		return "Darkness";
	case EnergyType::Metal:
		return "Metal";
	case EnergyType::Dragon:
		return "Dragon";
	case EnergyType::All:
		return "Rainbow";
	case EnergyType::Psychic | EnergyType::Darkness:
		return "PsychicAndDarkness";
	default:
		assert(false);
		return "";
	}
}

enum class PokemonType : unsigned char {
	NotPokemon, // not pokemon
	Normal,
	PokemonItem, // 化石や人形。明示的に書かれていなければ対戦準備では出せない
	Ex, // Pokemon ex
	MegaEx, // メガシンカex
};

enum class AreaType : unsigned char {
	All,
	Deck,
	Hand,
	Trash,
	Active, // バトル場
	Bench,
	Prize,
	Stadium,
	Energy,    // 場のエネルギー
	Tool,      // 場のどうぐ
	PreEvolution, // 進化前
	Player,
	Looking,
	Playing, // Playing item and supporter
	DeckBottom, // デッキの下。移動元では使わない

	Me,
	Effected,
	EffectedPreTarget, // 前の前の対象
	SelectedList, // selectedList
	TriggerSubject,
	TriggerObject,
	Attach, // 付いているポケモン
	TurnPlay, // この番に使ったカード
	AttackPreMyTurn, // 前の自分のターンにワザを使ったポケモン
	Temporary, // 一時エリア
};

// 特殊状態
enum class BadStatusType : unsigned char {
	None, // no bad status
	Asleep, // ねむり
	Paralyzed, // マヒ
	Confused, // こんらん
};

// 特殊状態の選択
enum class SelectSpecialConditionType : unsigned char {
	Poison, // どく
	Burn, // やけど
	Sleep, // ねむり
	Paralyze, // マヒ
	Confuse, // こんらん
};

enum class CardType : unsigned char {
	Pokemon,
	Item, // グッズ
	Tool, // ポケモンのどうぐ
	Supporter,
	Stadium,
	BasicEnergy, // 基本エネルギー
	SpecialEnergy, // 特殊エネルギー
};

inline bool IsTrainer(CardType type) {
	return type == CardType::Item
		|| type == CardType::Tool
		|| type == CardType::Supporter
		|| type == CardType::Stadium;
}

inline bool IsEnergy(CardType type) {
	return type == CardType::BasicEnergy
		|| type == CardType::SpecialEnergy;
}


enum class EvolutionType : unsigned char {
	NoEvolutionType, // ポケモンでない
	Basic, // たね
	Stage1, // 1進化
	Stage2, // 2進化
};
