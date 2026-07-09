// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Game.h"
#include "PlayerState.h"
#include "ActivateInfo.h"
#include "GameFunction.h"
#include "Binary.h"

// 勝敗
enum class GameResult : unsigned char {
	None = 0, // 結果未確定
	Player0Win = 1,
	Player1Win = 2,
	Draw = 3,
};

// 終了理由
enum class FinishReason : unsigned char {
	None = 0, // 未終了
	Prize0 = 1,
	Deck0 = 2,
	NoActivePokemon = 3,
	Effect = 4, // その他能力による勝利
	Other = 9, // その他
};

enum class GamePhase : unsigned char {
	Setup = 0,
	Main = 1,
	PokemonCheckup = 2,
	PokemonCheckupEnd = 3,
};

struct CardPosition {
	AreaType area;
	int areaIndex;
	int playerIndex;

	CardPosition() = default;
	CardPosition(AreaType area, int areaIndex, int playerIndex) : area(area), areaIndex(areaIndex), playerIndex(playerIndex) {}
};

struct AttachedCard {
	CardRef ref;
	CardRef targetRef;
	int attachedIndex;
};

using AttachedCardList = FixedList<AttachedCard, DECK_SIZE>;

struct AttachedToolEnergy {
	CardRef ref;
	int listIndex;
	const Card* card;

	const std::u8string& name() const {
		return card->getMaster().name;
	}
};

struct SelectOption {
	SelectOptionType type;
	short param0;
	short param1;
	short param2;
	short param3;
	short param4;

	CardPosition getCardPosition() const {
		return { (AreaType)param0, param1, param2 };
	}
};

struct EnergyInfo {
	EnergyType type;
	int count;
};

struct alignas(4) EvolveInfo {
	CardRef preRef;
	CardRef ref;
};

struct TurnHistory {
	bool ko;
	bool koTeamRocket;
	bool koAttackDamage;
	bool koAttackDamageEthan;
	bool koAttackDamageHop;
	CardRef turnAttackCard;
	signed char takePrizeCountTurnPlayer;
	int turnAttackId;
};

struct EffectStackElement {
	EffectState effectState;
	TriggerInfo triggerInfo;
	unsigned char effectJump;
	CardList selectedList;
	std::vector<AreaRef> preTargetList;
	std::vector<AreaRef> targetList;
};

struct State {
	Game* game;

	// ターン数
	// 1なら先攻1ターン目
	// 2なら後攻1ターン目
	// 0は1ターン目より前
	int turn;
	int turnActionCount;
	int effectActionCount;
	int turnAttackCount;

	GamePhase phase;
	GameResult gameResult;
	FinishReason finishReason;

	std::array<bool, 2> setupDone;
	std::array<bool, 2> mulligan; // 引き直すならtrue
	std::array<int, 2> mulliganCount; // 引き直した回数

	signed char firstPlayer; // 先攻プレイヤーインデックス

	signed char lookingPlayer; // 見ているプレイヤーのインデックス。2なら双方、3と4は裏向きで見ている
	bool lookingReverse; // lookingが裏向きならtrue

	bool isBreak; // trueならbreakする
	bool effectLoopStop; // trueなら現在のEffectのループを止める
	bool changed; // 効果で何かが変わったらtrue
	bool stateChanged; // trueならもう一度Refresh
	bool updateOrder; // trueなら効果順の更新が必要

	union { // この番
		unsigned char turnState;
		struct {
			bool supporterPlayed : 1;
			bool stadiumPlayed : 1;
			bool energyPlayed : 1;
			bool retreated : 1;
			bool turnEnd : 1; // ターンエンド予約
		};
	};

	union {
		unsigned char continualState;
		struct {
			bool noToolEffect : 1; // どうぐの効果は無くなる
		};
	};

	union {
		unsigned reserved; // 予約領域
		struct {
			unsigned char reserved0;
		};
	};

	int currentCardEffectIndex;
	int coinHeadCount; // コインで表が出た回数
	signed char lastStadiumPlayer;
	bool failRetreat;

	// 選択関連
	SelectType selectType;
	SelectContext selectContext; // 厳密に設定されているわけではない。期待と違うcontextな可能性もあるので必ず確かめる
	CardRef contextCard; // selectで送るカード
	bool selectDeck; // デッキからの選択
	signed char selectPlayer; // プレイヤーインデックス
	int selectMin; // 最小選択数
	int selectMax; // 最大選択数
	int remainDamageCounter;
	int energyCost;
	int remainEnergyCost; // 必要な残りエネルギー選択個数。selectTypeがEnergyのときのみ利用
	int selectedEnergyCardCount;
	int removedDamageCounter;
	std::array<int, BENCH_SIZE_MAX + 1> selectCounts;
	CardRef selectingEnergyPokemonRef;

	// 能力ステート。パッシブ能力以外の処理中に設定される
	EffectState effectState; // 発動中能力
	TriggerInfo triggerInfo; // トリガー情報
	unsigned char effectJump; // この数だけ次のEffectをスキップ
	bool attachActive;

	int currentAttackId; // 発動中ワザ
	int srcAttackId; // 発動中の元のワザ
	int attackDamageChange; // ダメージ変化
	int lastAttackDamage;
	CardRef attacker;
	bool postAttackEffect;
	bool postEffectActivate;
	bool failAttack;
	bool secondAttack;

	int moveCounter; // カードが移動する度に増やす
	int currentSkillOrder;

	std::array<TurnHistory, 3> turnHistories; // インデックス0が今のターン

	ByteFixedList<CardRef, 1> stadium;
	CardList looking; // 能力で一時的に見ているカード
	CardList selectedList; // 特定のEffectで格納
	CardList eachList; // 特定のEffectで格納
	ByteFixedList<CardRef, 2> playing; // 使用中カード
	ByteFixedList<CardRef, 9> checkList;
	std::array<PlayerState, 2> players;

	std::array<int, 2> logIndex;

	std::array<Card, 128> allCard;

	std::vector<SelectOption> options;
	std::vector<int> selected; // optionsのインデックス

	std::vector<AreaRef> preTargetList;
	std::vector<AreaRef> targetList;
	std::vector<AreaRef> koList;

	std::vector<TriggeredAbility> delayTriggerStack;
	std::vector<TriggeredAbility> temporaryTriggerStack;
	std::vector<TriggeredAbility> triggerStack;

	std::vector<int> turnUsedSkill;
	std::vector<CardRef> turnPlay;
	std::vector<CardRef> turnHeal;
	std::vector<EvolveInfo> turnEvolve;
	std::vector<GameFunction> functionStack;
	std::vector<Log> logs;

	void clear() {
		int count = (int)((unsigned char*)&options - (unsigned char*)&turn);
		std::memset(&turn, 0, count);
	}

	void serialize(BinaryWriter& b) const {
		b.set(&turn, &options);

		b.set(options);
		b.set(selected);
		b.set(preTargetList);
		b.set(targetList);
		b.set(koList);
		b.set(delayTriggerStack);
		b.set(temporaryTriggerStack);
		b.set(triggerStack);
		b.set(turnUsedSkill);
		b.set(turnPlay);
		b.set(turnHeal);
		b.set(turnEvolve);
		b.set(functionStack);
		b.set(logs);
	}

	void deserialize(BinaryReader& b) {
		b.set(&turn, &options);

		b.set(options);
		b.set(selected);
		b.set(preTargetList);
		b.set(targetList);
		b.set(koList);
		b.set(delayTriggerStack);
		b.set(temporaryTriggerStack);
		b.set(triggerStack);
		b.set(turnUsedSkill);
		b.set(turnPlay);
		b.set(turnHeal);
		b.set(turnEvolve);
		b.set(functionStack);
		b.set(logs);
	}

	void eraseCardList(CardList& list) {
		for (CardRef& ref : list) {
			eraseCard(ref);
		}
		list.allClear();
	}

	void erasePlayerData(int myPlayerIndex) {
		{
			PlayerState& ps = players.at(myPlayerIndex);
			eraseCardList(ps.prize);
			if (!selectDeck) {
				eraseCardList(ps.deck);
			}
		}
		{
			PlayerState& ps = players.at(1 - myPlayerIndex);
			eraseCardList(ps.prize);
			eraseCardList(ps.hand);
			eraseCardList(ps.deck);
			for (CardRef& ref : ps.active) {
				Card& card = getCard(ref);
				if (card.reverse) {
					eraseCard(ref);
					card.reverse = true;
				}
			}
		}
		looking.outClear();
		if (phase == GamePhase::Setup) {
			targetList.clear();
		}
		logs.clear();
		logIndex = {};
	}

	void eraseCard(CardRef& ref) {
		getCard(ref) = {};
		ref = {};
	}

	std::array<int, 2> basicPlayerOrder() const {
		return { firstPlayer, 1 - firstPlayer };
	}

	std::array<int, 2> reversePlayerOrder() const {
		return { 1 - firstPlayer, firstPlayer };
	}

	std::array<int, 2> activePlayerOrder() const {
		int playerIndex = activePlayerIndex();
		return { playerIndex, 1 - playerIndex };
	}

	int winPlayer() const {
		if (gameResult == GameResult::Player0Win) {
			return 0;
		} else if (gameResult == GameResult::Player1Win) {
			return 1;
		} else {
			return 2;
		}
	}

	void setResult(int losePlayerIndex, FinishReason reason) {
		if (losePlayerIndex == 0) {
			gameResult = GameResult::Player1Win;
		} else if (losePlayerIndex == 1) {
			gameResult = GameResult::Player0Win;
		} else {
			gameResult = GameResult::Draw;
		}
		finishReason = reason;
	}

	int apiResult() const {
		if (gameResult == GameResult::None) {
			return -1;
		} else if (gameResult == GameResult::Player0Win) {
			return 0;
		} else if (gameResult == GameResult::Player1Win) {
			return 1;
		} else {
			return 2;
		}
	}

	bool isFinish() const {
		return gameResult != GameResult::None;
	}

	bool finishCheck() {
		if (isFinish()) {
			return true;
		}
		FinishReason reason = FinishReason::None;
		std::array<int, 2> score = {};
		for (int i : basicPlayerOrder()) {
			const PlayerState& ps = players[i];
			if (ps.prize.empty()) {
				score[i]++;
				reason = FinishReason::Prize0;
			}
			if (ps.active.empty() && ps.bench.empty()) {
				score[1 - i]++;
				reason = FinishReason::NoActivePokemon;
			}
		}
		if (reason != FinishReason::None) {
			int losePlayerIndex = 2;
			if (score[0] < score[1]) {
				losePlayerIndex = 0;
			} else if (score[0] > score[1]) {
				losePlayerIndex = 1;
			}
			setResult(losePlayerIndex, reason);
		}
		return isFinish();
	}

	bool isPlayerTurn() const {
		return phase == GamePhase::Main;
	}

	int lookingOpenType() const {
		if (lookingPlayer == 2) {
			return 0;
		} else {
			return 3 + lookingPlayer;
		}
	}

	TurnHistory& thisTurnHistory() {
		return turnHistories[0];
	}


	bool isAddLog() const {
		return game->isAddLog();
	}
	Log& addLog(LogType type) {
		return logs.emplace_back(type);
	}

	int nextLogStart() {
		int playerIndex = selectPlayer;
		int logStart = logIndex[playerIndex];
		logIndex.at(playerIndex) = (int)logs.size();
		return logStart;
	}

	void setSelect(SelectType selectType, SelectContext selectContext, int selectPlayer, int selectMin = 1, int selectMax = 1) {
		if (selectContext == SelectContext::None) {
			Exception("no context");
		}
		if (SelectContext::Main < selectContext && selectContext < SelectContext::DiscardEnergyCard) {
			if (selectType != SelectType::Card) {
				Exception("not Card");
			}
		}
		this->selectType = selectType;
		this->selectContext = selectContext;
		this->selectPlayer = selectPlayer;
		this->selectMin = selectMin;
		this->selectMax = selectMax;
	}

	// 選択後の処理で、選択情報が不要になったら呼ぶ
	void clearSelect() {
		selectType = SelectType::None;
		options.clear();
		selected.clear();
		contextCard = {};
		selectDeck = false;
	}

	void setSelectMinMaxAny() {
		selectMin = 0;
		selectMax = (int)options.size();
	}

	SelectOption& addOption(SelectOptionType type) {
		if (selectType == SelectType::Card) {
			if (type != SelectOptionType::Card) {
				Exception("not Card");
			}
		} else if (selectType == SelectType::AttachedCard) {
			if (type != SelectOptionType::EnergyCard && type != SelectOptionType::ToolCard) {
				Exception("not AttachedCard");
			}
		}
		SelectOption& option = options.emplace_back();
		option = {};
		option.type = type;
		return option;
	}

	// selectedのindexを指定
	SelectOption selectedOption(int index) const {
		if (selected.size() <= index) {
			Exception("invalid index");
		}
		int selectedIndex = selected.at(index);
		if (selectedIndex >= options.size()) {
			Exception("invalid select");
		}
		return options.at(selectedIndex);
	}

	SelectOption firstSelected() const {
		return selectedOption(0);
	}

	bool selectedYes() const {
		SelectOptionType type = firstSelected().type;
		return type == SelectOptionType::Yes;
	}

	int selectedNumber() const {
		return firstSelected().param0;
	}

	// 効果対象など選択後
	inline void setSelectedCardTarget() {
		targetList.clear();
		for (int index : selected) {
			const SelectOption& option = options.at(index);
			targetList.push_back(makeAreaRef(option));
		}

		clearSelect();
	}


	CardRef getCardRef(AreaType area, int areaIndex, int playerIndex) const {
		const PlayerState& ps = players.at(playerIndex);
		switch (area)
		{
		case AreaType::Deck:
			return ps.deck.at(areaIndex);
		case AreaType::Hand:
			return ps.hand.at(areaIndex);
		case AreaType::Trash:
			return ps.trash.at(areaIndex);
		case AreaType::Active:
			return ps.active.at(areaIndex);
		case AreaType::Bench:
			return ps.bench.at(areaIndex);
		case AreaType::Prize:
			return ps.prize.at(areaIndex);
		case AreaType::Stadium:
			return stadium.at(areaIndex);
		case AreaType::Energy:
			return ps.energy.at(areaIndex);
		case AreaType::Tool:
			return ps.tool.at(areaIndex);
		case AreaType::PreEvolution:
			return ps.preEvolution.at(areaIndex);
		case AreaType::Player:
			return playerCardRef(playerIndex);
		case AreaType::Looking:
			return looking.at(areaIndex);
		case AreaType::Playing:
			return playing.at(areaIndex);
		case AreaType::Temporary:
			return ps.temporary.at(areaIndex);
		default:
			Exception("getCardRef unexpected area " + std::to_string((int)area));
			return {};
		}
	}

	CardRef getCardRef(const CardPosition& pos) const {
		return getCardRef(pos.area, pos.areaIndex, pos.playerIndex);
	}

	CardRef removeCardRef(AreaType area, int areaIndex, int playerIndex) {
		PlayerState& ps = players.at(playerIndex);
		switch (area)
		{
		case AreaType::Deck:
			return ps.deck.take(areaIndex);
		case AreaType::Hand:
			return ps.hand.take(areaIndex);
		case AreaType::Trash:
			return ps.trash.take(areaIndex);
		case AreaType::Active:
			return ps.active.take(areaIndex);
		case AreaType::Bench:
			return ps.bench.take(areaIndex);
		case AreaType::Prize:
			return ps.prize.take(areaIndex);
		case AreaType::Stadium:
			return stadium.take(areaIndex);
		case AreaType::Energy:
			return ps.energy.take(areaIndex);
		case AreaType::Tool:
			return ps.tool.take(areaIndex);
		case AreaType::PreEvolution:
			return ps.preEvolution.take(areaIndex);
		case AreaType::Looking:
			return looking.take(areaIndex);
		case AreaType::Playing:
			return playing.take(areaIndex);
		case AreaType::Temporary:
			return ps.temporary.take(areaIndex);
		default:
			Exception("removeCardRef unexpected area " + std::to_string((int)area));
			return {};
		}
	}

	void pushCardRef(AreaType area, int playerIndex, CardRef ref) {
		PlayerState& ps = players.at(playerIndex);
		switch (area)
		{
		case AreaType::Deck:
			return ps.deck.push_back(ref);
		case AreaType::Hand:
			return ps.hand.push_back(ref);
		case AreaType::Trash:
			return ps.trash.push_back(ref);
		case AreaType::Active:
			return ps.active.push_back(ref);
		case AreaType::Bench:
			return ps.bench.push_back(ref);
		case AreaType::Prize:
			return ps.prize.push_back(ref);
		case AreaType::Stadium:
			return stadium.push_back(ref);
		case AreaType::Energy:
			return ps.energy.push_back(ref);
		case AreaType::Tool:
			return ps.tool.push_back(ref);
		case AreaType::PreEvolution:
			return ps.preEvolution.push_back(ref);
		case AreaType::Looking:
			return looking.push_back(ref);
		case AreaType::Playing:
			return playing.push_back(ref);
		case AreaType::DeckBottom:
			return ps.deck.push_front(ref);
		default:
			Exception("pushCardRef unexpected area " + std::to_string((int)area));
		}
	}

	CardRef getCardRef(const SelectOption& option) const {
		CardPosition p = option.getCardPosition();
		return getCardRef(p.area, p.areaIndex, p.playerIndex);
	}

	CardRef getAttachCardRef(const SelectOption& option, int playerIndex) const {
		AreaType area = (AreaType)option.param0;
		return getCardRef(area, option.param1, playerIndex);
	}

	CardRef getAttachPokemonCardRef(const SelectOption& option, int playerIndex) const {
		AreaType area = (AreaType)option.param2;
		return getCardRef(area, option.param3, playerIndex);
	}

	CardRef getAbilityCardRef(const SelectOption& option, int playerIndex) const {
		AreaType area = (AreaType)option.param0;
		return getCardRef(area, option.param1, playerIndex);
	}

	CardRef getPlayCardRef(const SelectOption& option, int playerIndex) const {
		return getCardRef(AreaType::Hand, option.param0, playerIndex);
	}

	// プレイヤー本体を表すCardRef
	CardRef playerCardRef(int playerIndex) const {
		return CardRef{ 1 + playerIndex };
	}

	AreaRef makeAreaRef(CardRef card) const {
		AreaRef ref;
		ref.card = card;
		ref.moveCounter = getCard(card).moveCounter;
		return ref;
	}
	AreaRef makeAreaRef(const SelectOption& option) const {
		CardPosition p = option.getCardPosition();
		CardRef ref = getCardRef(p.area, p.areaIndex, p.playerIndex);
		return makeAreaRef(ref);
	}

	Card& getCard(CardRef ref) {
		assert(ref.cardIndex > 0);
		return allCard.at(ref.cardIndex);
	}
	const Card& getCard(CardRef ref) const {
		assert(ref.cardIndex > 0);
		return allCard.at(ref.cardIndex);
	}

	Card* checkGetCard(AreaRef area) {
		if (area.card.isNull()) {
			return nullptr;
		}
		Card& card = getCard(area.card);
		if (card.moveCounter == area.moveCounter) {
			return &card;
		} else {
			return nullptr;
		}
	}

	Card* checkGetCardNotPreventEffect(AreaRef area) {
		if (area.card.isNull()) {
			return nullptr;
		}
		Card& card = getCard(area.card);
		if (card.moveCounter != area.moveCounter) {
			return nullptr;
		}
		if (isPreventEffect(card)) {
			return nullptr;
		}
		return &card;
	}

	CardId getCardId(CardRef ref) const {
		return getCard(ref).cardId;
	}

	const std::u8string& getCardName(CardRef ref) const {
		return getCard(ref).getMaster().name;
	}

	void cardMoved(CardRef ref, AreaType newArea, bool reverse = false) {
		Card& card = getCard(ref);
		if (card.area == newArea) {
			return;
		}
		AreaType preArea = card.area;
		if (preArea == AreaType::Active) {
			players[card.playerIndex].clearActiveState();
			for (int i = (int)delayTriggerStack.size() - 1; i >= 0; i--) {
				const TriggeredAbility& ta = delayTriggerStack[i];
				const Card& subject = getCard(ta.trigger.subject.card);
				if (subject.playerIndex == card.playerIndex) {
					delayTriggerStack.erase(delayTriggerStack.begin() + i);
				}
			}
		}

		if (preArea == AreaType::Active || preArea == AreaType::Bench) {
			if (newArea == AreaType::Active || newArea == AreaType::Bench) {
				// empty
			} else {
				card.clearState();
				card.moveCounter = moveCounter++;
			}
		} else {
			card.clearState();
			card.moveCounter = moveCounter++;
			if (newArea == AreaType::Active || newArea == AreaType::Bench) {
				card.appear = true;
			}
		}

		if (newArea != AreaType::Active) {
			if (newArea == AreaType::Bench && preArea == AreaType::Active) {
				bool cannotAttackLessEqualEnergy2ThisTurn = card.thisTurn.cannotAttackLessEqualEnergy2;
				bool cannotAttackLessEqualEnergy2NextTurn = card.nextTurn.cannotAttackLessEqualEnergy2;
				card.clearNextTurnState();
				card.thisTurn.cannotAttackLessEqualEnergy2 = cannotAttackLessEqualEnergy2ThisTurn;
				card.nextTurn.cannotAttackLessEqualEnergy2 = cannotAttackLessEqualEnergy2NextTurn;
			} else {
				card.clearNextTurnState();
			}
			card.cannotUseAttackIdNonActive = 0;
		}

		card.area = newArea;
		card.preArea = preArea;
		card.reverse = reverse;
		card.attachMoveCounter = 0;
	}

	// AreaRefが作られたときからエリア移動していなければtrue
	bool notMoved(AreaRef area) const {
		const Card& card = getCard(area.card);
		return card.moveCounter == area.moveCounter;
	}

	template<typename TList>
	static int currentAreaIndex(const TList& list, CardRef ref) {
		for (int i = 0; i < list.size(); i++) {
			if (list[i].cardIndex == ref.cardIndex) {
				return i;
			}
		}
		return -1;
	}

	int currentAreaIndex(int playerIndex, AreaType area, CardRef ref) const {
		const PlayerState& ps = players.at(playerIndex);
		switch (area)
		{
		case AreaType::Deck:
			return currentAreaIndex(ps.deck, ref);
		case AreaType::Hand:
			return currentAreaIndex(ps.hand, ref);
		case AreaType::Trash:
			return currentAreaIndex(ps.trash, ref);
		case AreaType::Active:
			return currentAreaIndex(ps.active, ref);
		case AreaType::Bench:
			return currentAreaIndex(ps.bench, ref);
		case AreaType::Prize:
			return currentAreaIndex(ps.prize, ref);
		case AreaType::Stadium:
			return currentAreaIndex(stadium, ref);
		case AreaType::Energy:
			return currentAreaIndex(ps.energy, ref);
		case AreaType::Tool:
			return currentAreaIndex(ps.tool, ref);
		case AreaType::PreEvolution:
			return currentAreaIndex(ps.preEvolution, ref);
		case AreaType::Player:
			return 0;
		case AreaType::Looking:
			return currentAreaIndex(looking, ref);
		case AreaType::Temporary:
			return currentAreaIndex(ps.temporary, ref);
		default:
			Exception("currentAreaIndex unexpected area " + std::to_string((int)area));
		}
		return -1;
	}

	int currentAreaIndex(CardRef ref, const Card& card) const {
		return currentAreaIndex(card.playerIndex, card.area, ref);
	}

	CardPosition currentCardPosition(CardRef ref) const {
		const Card& card = getCard(ref);
		int areaIndex = currentAreaIndex(card.playerIndex, card.area, ref);
		return CardPosition{ card.area, areaIndex, card.playerIndex };
	}

	// カードが移動済みの場合、返り値のareaIndexは-1となる
	CardPosition currentCardPosition(AreaRef ref) const {
		const Card& card = getCard(ref.card);
		if (card.moveCounter != ref.moveCounter) {
			return CardPosition{ card.area, -1, card.playerIndex };
		}
		int areaIndex = currentAreaIndex(card.playerIndex, card.area, ref.card);
		return CardPosition{ card.area, areaIndex, card.playerIndex };
	}

	// 付いているポケモンの場所とCardRefを返す
	RefPosition attachedCardPosition(const Card& card) const {
		int moveCounter = card.attachMoveCounter;
		int playerIndex = card.playerIndex;
		const PlayerState& ps = players[playerIndex];
		if (ps.active.size() > 0) {
			CardRef activeRef = ps.getActive();
			if (getCard(activeRef).moveCounter == moveCounter) {
				return { activeRef, AreaType::Active, 0 };
			}
		}
		for (int i : range(ps.bench)) {
			CardRef ref = ps.bench[i];
			if (getCard(ps.bench[i]).moveCounter == moveCounter) {
				return { ref, AreaType::Bench, i };
			}
		}
		return {};
	}

	int energyIndex(CardRef energyRef, CardRef pokemonRef) const {
		const Card& pokemonCard = getCard(pokemonRef);
		int index = 0;
		for (CardRef ref : players[pokemonCard.playerIndex].energy) {
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == pokemonCard.moveCounter) {
				if (ref == energyRef) {
					return index;
				} else {
					index++;
				}
			}
		}
		assert(false);
		return 0;
	}

	RefPosition attachedEnergyPositionFromOption(const SelectOption& option) const {
		assert(option.type == SelectOptionType::Energy || option.type == SelectOptionType::EnergyCard);
		CardPosition p = option.getCardPosition();
		CardRef targetRef = getCardRef(p.area, p.areaIndex, p.playerIndex);
		int moveCounter = getCard(targetRef).moveCounter;
		const PlayerState& ps = players[p.playerIndex];
		int index = 0;
		for (int i : range(ps.energy)) {
			CardRef ref = ps.energy[i];
			if (moveCounter == getCard(ref).attachMoveCounter) {
				if (index == option.param3) {
					return { ref, AreaType::Energy, i };
				} else {
					index++;
				}
			}
		}
		return {};
	}

	RefPosition attachedToolPositionFromOption(const SelectOption& option) const {
		assert(option.type == SelectOptionType::ToolCard);
		CardPosition p = option.getCardPosition();
		CardRef targetRef = getCardRef(p.area, p.areaIndex, p.playerIndex);
		int moveCounter = getCard(targetRef).moveCounter;
		const PlayerState& ps = players[p.playerIndex];
		int index = 0;
		for (int i : range(ps.tool)) {
			CardRef ref = ps.energy[i];
			if (moveCounter == getCard(ref).attachMoveCounter) {
				if (index == option.param3) {
					return { ref, AreaType::Tool, i };
				} else {
					index++;
				}
			}
		}
		return {};
	}

	CardRef attachedCardRefFromOption(const SelectOption& option) const {
		if (option.type == SelectOptionType::ToolCard) {
			RefPosition rp = attachedToolPositionFromOption(option);
			return rp.ref;
		} else {
			RefPosition rp = attachedEnergyPositionFromOption(option);
			return rp.ref;
		}
	}

	AttachedCardList getAttachedList(const PlayerState& ps, const CardList& list) {
		AttachedCardList attachedList;
		int activeIndex = 0;
		std::array<int, BENCH_SIZE_MAX> benchIndex = {};
		for (CardRef attachRef : list) {
			int moveCounter = getCard(attachRef).attachMoveCounter;
			bool found = false;
			for (CardRef ref : ps.active) {
				if (getCard(ref).moveCounter == moveCounter) {
					found = true;
					attachedList.push_back({ attachRef, ref, activeIndex++ });
					break;
				}
			}
			if (!found) {
				for (int j : range(ps.bench)) {
					CardRef ref = ps.bench[j];
					if (getCard(ref).moveCounter == moveCounter) {
						attachedList.push_back({ attachRef, ref, benchIndex[j]++ });
						break;
					}
				}
			}
		}
		return attachedList;
	}

	int getMaxHp(const Card& card) const {
		return card.getMaster().hp + card.hpChange;
	}

	int getHp(const Card& card) const {
		return getMaxHp(card) - card.damage;
	}

	EnergyType getEnergyType(const Card& card) const {
		auto types = card.getEnergyType();
		int type = {};
		for (EnergyType t : types) {
			type |= (int)t;
		}
		return (EnergyType)type;
	}

	bool isEvolved(CardRef ref) const {
		const Card& card = getCard(ref);
		bool found = false;
		for (CardRef r : players.at(card.playerIndex).preEvolution) {
			const Card& preCard = getCard(r);
			if (preCard.attachMoveCounter == card.moveCounter) {
				return true;
			}
		}
		return false;
	}

	bool isKo(const Card& card) const {
		return getMaxHp(card) <= card.damage;
	}

	bool cannotToHand(CardRef ref) const {
		const Card& card = getCard(ref);
		if (card.area == AreaType::Energy || card.area == AreaType::Tool || card.area == AreaType::PreEvolution) {
			RefPosition rp = attachedCardPosition(card);
			return getCard(rp.ref).cannotToHand;
		} else {
			return card.cannotToHand;
		}
	}

	int energyCount(int playerIndex, CardRef pokemonRef) const {
		getEnergies(playerIndex, pokemonRef, game->energyList);
		return (int)game->energyList.size();
	}

	int typeEnergyCount(int playerIndex, CardRef pokemonRef, EnergyType type) const {
		int count = 0;
		getEnergies(playerIndex, pokemonRef, game->energyList);
		for (EnergyType energyType : game->energyList) {
			if (MatchEnergyType(type, energyType)) {
				count++;
			}
		}
		return count;
	}

	void getEnergies(int playerIndex, CardRef pokemonRef, std::vector<EnergyType>& output) const {
		output.clear();
		const PlayerState& ps = players.at(playerIndex);
		int moveCounter = getCard(pokemonRef).moveCounter;
		for (CardRef ref : ps.energy) {
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == moveCounter) {
				EnergyInfo ei = getEnergyInfo(card, pokemonRef);
				for (int _ : range(ei.count)) {
					output.push_back(ei.type);
				}
			}
		}
	}

	int getEnergyCount(int playerIndex, CardRef pokemonRef) const {
		getEnergies(playerIndex, pokemonRef, game->energyList);
		return (int)game->energyList.size();
	}

	int targetListEnergyCount() {
		int count = 0;
		for (const AreaRef& ref : targetList) {
			Card* card = checkGetCard(ref);
			if (card != nullptr) {
				getEnergies(card->playerIndex, ref.card, game->energyList);
				count += (int)game->energyList.size();
			}
		}
		return count;
	}

	int targetListTypeEnergyCount(EnergyType type) {
		int count = 0;
		for (const AreaRef& ref : targetList) {
			Card* card = checkGetCard(ref);
			if (card != nullptr) {
				count += typeEnergyCount(card->playerIndex, ref.card, type);
			}
		}
		return count;
	}

	void getEnergyCards(CardRef pokemonRef, std::vector<CardRef>& output) const {
		output.clear();
		const Card& c = getCard(pokemonRef);
		const PlayerState& ps = players.at(c.playerIndex);
		for (CardRef ref : ps.energy) {
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == c.moveCounter) {
				output.push_back(ref);
			}
		}
	}

	bool isAttachedSpecialEnergy(CardRef pokemonRef) const {
		const Card& c = getCard(pokemonRef);
		const PlayerState& ps = players.at(c.playerIndex);
		for (CardRef ref : ps.energy) {
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == c.moveCounter) {
				if (card.getMaster().cardType == CardType::SpecialEnergy) {
					return true;
				}
			}
		}
		return false;
	}

	EnergyInfo getEnergyInfo(const Card& energyCard, CardRef pokemonRef) const {
		const CardMaster& master = energyCard.getMaster();
		if (master.cardType == CardType::SpecialEnergy) {
			const Card& pokemonCard = getCard(pokemonRef);
			if (master.cardId == NEO_UPPER_ENERGY) {
				const CardMaster& master = pokemonCard.getMaster();
				if (master.evolutionType == EvolutionType::Stage2) {
					return EnergyInfo(EnergyType::All, 2);
				} else {
					return EnergyInfo(EnergyType::Colorless, 1);
				}
			} else if (master.cardId == PRISM_ENERGY) {
				const CardMaster& master = pokemonCard.getMaster();
				if (master.evolutionType == EvolutionType::Basic) {
					return EnergyInfo(EnergyType::All, 1);
				} else {
					return EnergyInfo(EnergyType::Colorless, 1);
				}
			} else if (master.cardId == IGNITION_ENERGY) {
				const CardMaster& master = pokemonCard.getMaster();
				if (master.evolutionType == EvolutionType::Stage1 || master.evolutionType == EvolutionType::Stage2) {
					return EnergyInfo(EnergyType::Colorless, 3);
				} else {
					return EnergyInfo(EnergyType::Colorless, 1);
				}
			}
		} else if (master.cardId == GRASS_ENERGY) {
			const Card& pokemonCard = getCard(pokemonRef);
			if (pokemonCard.doubleGrassEnergy) {
				return EnergyInfo(EnergyType::Grass, 2);
			}
		}
		EnergyInfo ei(master.energyType, master.energyCount);

		return ei;
	}

	int retreatCost(const Card& card) const {
		const CardMaster& master = card.getMaster();
		int cost = master.retreatCost + card.retreatCostChange + card.thisTurn.retreatCostChange;
		if (card.noRetreatCost) {
			cost = 0;
		}
		if (master.pokemonType == PokemonType::PokemonItem) {
			cost = 0;
		}
		return std::max(0, cost);
	}

	int retreatCost(int playerIndex) const {
		CardRef ref = players.at(playerIndex).getActive();
		const Card& card = getCard(ref);
		return retreatCost(card);
	}

	EnergyType getWeakness(const Card& card, const CardMaster& master) const {
		if (card.noWeaknessNextEnemyTurn) {
			return EnergyType::Colorless;
		}
		if (card.weaknessIndex > 0) {
			return EnergyTypes.at(card.weaknessIndex);
		}
		return master.weakness;
	}

	int getPrizeCount(const Card& card) const {
		if (card.koPrizeZero) {
			return 0;
		}
		const CardMaster& master = card.getMaster();
		int count = 1;
		if (master.pokemonType == PokemonType::Ex) {
			count = 2;
		} else if (master.pokemonType == PokemonType::MegaEx) {
			count = 3;
		}
		count += card.koPrizeChangeAlways;
		if (card.koEnemyAttackDamage) {
			count += card.koPrizeChange;
			if (card.koPrizeDecreaseOnce) {
				count--;
			}
		}
		if (card.koEnemyTerastalAttackDamage && card.area == AreaType::Active) {
			count += players[1 - card.playerIndex].takePrizeCountChangeTerastalAttackKoActive;
		}
		if (card.koEnemyNAttackDamage && card.area == AreaType::Active) {
			count += players[1 - card.playerIndex].takePrizeCountChangeNAttackKoActive;
		}
		if (card.koPrizePlus1) {
			count++;
		}
		return std::max(count, 0);
	}

	int takenPrizeCount(int playerIndex) const {
		int count = PRIZE_SIZE - players[playerIndex].prize.size();
		return std::max(0, count);
	}

	// ワザの特殊使用条件を満たしているか
	bool satisfyAttackCondition(const Card& card, const Attack& attack, int srcAttackId) const {
		if (card.thisTurn.cannotUseAttackId == attack.attackId || card.thisTurn.cannotUseAttackId2 == attack.attackId || card.cannotUseAttackIdNonActive == attack.attackId) {
			if (srcAttackId == 0 || srcAttackId == attack.attackId) {
				return false;
			}
		}
		if (attack.cannotUseFirstTurn && turn <= 2) {
			return false;
		}
		if (attack.cannotUseSameNameAttackPreMyTurn) {
			int id = turnHistories[2].turnAttackId;
			if (id > 0) {
				const Attack& preAttack = AttackTable.at(id);
				if (attack.name == preAttack.name) {
					return false;
				}
			}
		}
		if (attack.canOnlyUseEnemyPrizeOne) {
			if (players[1 - card.playerIndex].prize.size() != 1) {
				return false;
			}
		}
		return true;
	}

	const Skill* getAbility(const Card& card, const CardMaster& master) const {
		if (master.ability == nullptr) {
			return nullptr;
		}
		if (card.noAbility) {
			return nullptr;
		}
		if (card.noKoMeAbility && master.ability->koMeAbility) {
			return nullptr;
		}
		return master.ability;
	}

	int activePlayerIndex() const {
		return ((turn + 1) ^ firstPlayer) & 1;
	}

	int benchCapacity(int playerIndex) const {
		int capacity = players.at(playerIndex).benchCapacity;
		return capacity == 0 ? BENCH_SIZE_DEFAULT : capacity;
	}

	int remainingBench(int playerIndex) const {
		return benchCapacity(playerIndex) - players.at(playerIndex).bench.size();
	}

	bool transformOnly(const Card& card) const {
		return card.getMaster().transformOnly;
	}

	bool canExchangeActive(int playerIndex) const {
		return players[playerIndex].bench.size() > 0;
	}

	// カード種類によらない共通処理
	bool canPlay(const PlayerState& ps, const Card& card, const CardMaster& master) const {
		if (ps.cannotPlayAceSpec) {
			if (card.area == AreaType::Hand && master.aceSpec) {
				return false;
			}
		}
		return true;
	}

	bool canAttack(CardRef ref, const Card& card) const {
		if (!card.canAttack()) {
			return false;
		}
		if (card.thisTurn.cannotAttackLessEqualEnergy2) {
			if (energyCount(card.playerIndex, ref) <= 2) {
				return false;
			}
		}
		return true;
	}

	bool canEvolveEffect(const State& state, const Card& handCard, const CardMaster& handMaster, CardRef pokemonRef) const {
		if (handMaster.cardType != CardType::Pokemon) {
			return false;
		}
		const Card& pokemonCard = getCard(pokemonRef);
		if (transformOnly(handCard)) {
			return false;
		}
		const CardMaster& pokemonMaster = pokemonCard.getMaster();
		if (pokemonCard.rainbowDna) {
			if (state.turn > 2
				&& !pokemonCard.appear
				&& handCard.area == AreaType::Hand
				&& handMaster.isEx()
				&& handMaster.evolvesFrom == u8"イーブイ") {
				return true;
			}
		}
		return handMaster.evolvesFrom == pokemonMaster.name;
	}

	bool canEvolve(const State& state, const Card& handCard, const CardMaster& handMaster, CardRef pokemonRef, bool cannotAppearThisTurn = true) const {
		if (handMaster.cardType != CardType::Pokemon) {
			return false;
		}
		const Card& pokemonCard = getCard(pokemonRef);
		bool cannotEvolveAppearTurn = !pokemonCard.canEvolveAppearTurn;
		if (state.turn <= 2 && cannotEvolveAppearTurn) {
			return false;
		}
		if (cannotAppearThisTurn && pokemonCard.appear) {
			if (cannotEvolveAppearTurn && (!pokemonCard.canEvolveGrassAppearTurn || !ContainsEnergyType(handMaster.energyType, EnergyType::Grass))) {
				return false;
			}
		}
		return canEvolveEffect(state, handCard, handMaster, pokemonRef);
	}

	bool canEvolve2(const Card& handCard, const CardMaster& handMaster, CardRef pokemonRef) const {
		if (handMaster.cardType != CardType::Pokemon) {
			return false;
		}
		const Card& pokemonCard = getCard(pokemonRef);
		if (pokemonCard.appear) {
			return false;
		}
		if (transformOnly(handCard)) {
			return false;
		}
		const CardMaster& pokemonMaster = pokemonCard.getMaster();
		return handMaster.evolvesFrom2 == pokemonMaster.name;
	}

	bool canAttachEnergy(const Card& energyCard, const CardMaster& energyMaster, CardRef pokemonRef) const {
		const Card& pokemonCard = getCard(pokemonRef);
		if (energyMaster.onlyTeamRocket && !pokemonCard.getMaster().teamRocket) {
			return false;
		}
		return true;
	}

	int remainingToolCapacity(const Card& pokemonCard) const {
		int attached = attachToolCount(pokemonCard);
		int canAttachCount = (pokemonCard.tool4 ? 4 : (pokemonCard.tool2 ? 2 : 1));
		return canAttachCount - attached;
	}

	bool canAttachTool(const CardMaster& toolMaster, CardRef pokemonRef) const {
		const Card& pokemonCard = getCard(pokemonRef);
		return remainingToolCapacity(pokemonCard) > 0;
	}

	int attachToolCount(const Card& pokemonCard) const {
		int count = 0;
		for (CardRef ref : players[pokemonCard.playerIndex].tool) {
			if (getCard(ref).attachMoveCounter == pokemonCard.moveCounter) {
				count++;
			}
		}
		return count;
	}

	FixedList<CardRef, 4> getAttachedToolRef(const Card& pokemonCard) const {
		FixedList<CardRef, 4> refs;
		for (CardRef ref : players.at(pokemonCard.playerIndex).tool) {
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == pokemonCard.moveCounter) {
				refs.push_back(ref);
			}
		}
		return refs;
	}

	FixedList<AttachedToolEnergy, 4> getAttachedTools(const Card& pokemonCard) const {
		FixedList<AttachedToolEnergy, 4> result;
		auto& tool = players.at(pokemonCard.playerIndex).tool;
		for (int i : range(tool)) {
			CardRef ref = tool[i];
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == pokemonCard.moveCounter) {
				result.push_back({ref, i, &card});
			}
		}
		return result;
	}

	CardId getAttachedToolId(const Card& pokemonCard) const {
		CardId id = 0;
		auto tools = getAttachedTools(pokemonCard);
		for (const AttachedToolEnergy& tool : tools) {
			id = tool.card->getMaster().cardId;
			break;
		}
		return id;
	}

	FixedList<CardRef, 8> getPreEvolutions(const Card& pokemonCard) const {
		FixedList<CardRef, 8> refs;
		for (CardRef ref : players.at(pokemonCard.playerIndex).preEvolution) {
			const Card& card = getCard(ref);
			if (card.attachMoveCounter == pokemonCard.moveCounter) {
				refs.push_back(ref);
			}
		}
		return refs;
	}

	// ワザの効果中
	bool onAttackEffect() const {
		if (!onAttack()) {
			return false;
		}
		if (onEffect()) {
			if (getAttacker() != getEffectCard().card) {
				return false;
			}
		}
		return true;
	}

	// 効果を受けないならtrue
	bool isPreventEffect(const Card& srcCard) const {
		const Card* targetPokemon = nullptr;
		if (srcCard.area == AreaType::Energy || srcCard.area == AreaType::Tool) {
			// 付いているカードに対する効果は、ポケモンに対する効果扱い
			RefPosition rp = attachedCardPosition(srcCard);
			targetPokemon = &getCard(rp.ref);
		}
		const Card& card = (targetPokemon == nullptr ? srcCard : *targetPokemon);
		if (onAttackEffect()) {
			const Card& attacker = getCard(getAttacker());
			if (attacker.playerIndex != card.playerIndex) {
				if (card.noDamageAndEffectEnemyTerastalAttack) {
					if (attacker.getMaster().tera) {
						return true;
					}
				}
				if (card.noDamageAndEffectEnemySpecialEnergyAttack) {
					if (isAttachedSpecialEnergy(getAttacker())) {
						return true;
					}
				}
				if (card.noEffectEnemyAttack || card.noDamageAndEffectEnemyAttackNextEnemyTurn) {
					return true;
				}
				if (card.noDamageAndEffectEnemyExAttackNextEnemyTurn) {
					if (attacker.getMaster().isEx()) {
						return true;
					}
				}
			}
			if (card.noDamageAndEffectAttackNextEnemyTurn) {
				return true;
			}
		} else if(onEffect()){
			const Card& effectCard = getCard(getEffectCard().card);
			if (effectCard.playerIndex != card.playerIndex) {
				if (card.noEffectEnemyItem) {
					if (effectCard.getMaster().cardType == CardType::Item) {
						return true;
					}
				}
				if (card.noEffectEnemySupporter) {
					if (effectCard.getMaster().cardType == CardType::Supporter) {
						return true;
					}
				}
				if (card.noEnemyAbility) {
					if (effectCard.area == AreaType::Active || effectCard.area == AreaType::Bench) {
						return true;
					}
				}
			}
		}

		return false;
	}

	// 効果を受けない系の判定はしない
	bool isPreventDamageCounter(const Card& targetPokemon) {
		const Card& effectCard = getCard(getEffectCard().card);
		if (targetPokemon.noDamageCounterEnemyAttackAbility) {
			if (effectCard.getMaster().isPokemon() && targetPokemon.playerIndex != effectCard.playerIndex) {
				return true;
			}
		}
		return false;
	}

	bool isPreventEffectActive(int playerIndex) const {
		const PlayerState& ps = players.at(playerIndex);
		if (ps.active.empty()) {
			return true;
		}
		const Card& card = getCard(ps.getActive());
		return isPreventEffect(card);
	}

	bool noSpecialCondition(const Card& card) const {
		return card.noSpecialCondition || card.getMaster().pokemonType == PokemonType::PokemonItem;
	}

	bool afterConfuse(const Card& card) {
		return players.at(card.playerIndex).badStatus == BadStatusType::Confused && card.notRecoverConfuseEvolve;
	}

	void setActivateAbility(const ActivateAbilityInfo& ability) {
		effectState.init();
		effectState.ability = ability;
		triggerInfo.type = TriggerType::None;
	}

	void setTriggeredAbility(const TriggeredAbility& ability) {
		effectState.init();
		effectState.ability = ability.activateInfo;
		triggerInfo = ability.trigger;
	}

	bool onSkill() const {
		return effectState.ability.skillId > 0;
	}

	bool onEffect() const {
		return effectState.onEffect;
	}

	bool onAttack() const {
		return currentAttackId > 0;
	}

	void clearAbility() {
		effectState = {};
		triggerInfo = {};
		contextCard = {};
		targetList.clear();
		preTargetList.clear();
		selectedList.clear();
		eachList.clear();
		checkList.clear();
		removedDamageCounter = 0;
		effectJump = 0;
		attachActive = false;
	}

	const Effect& getEffectByState(const EffectState& es) const {
		return SkillTable.at(es.ability.skillId).effects.at(es.effectIndex);
	}

	auto getValuesByEffect(const Effect& effect) const {
		auto values = effect.values;
		if (effect.multiplyEffectValuePreTargetCount) {
			for (int& v : values) {
				v *= (int)preTargetList.size();
			}
		}
		if (effect.multiplyEffectValueCoinHeadCount) {
			for (int& v : values) {
				v *= coinHeadCount;
			}
		}
		return values;
	}

	int getValueByEffect(const Effect& effect) const {
		return getValuesByEffect(effect)[0];
	}

	int getEffectValueByState(const EffectState& es) const {
		return getValueByEffect(getEffectByState(es));
	}

	// Instant Effect処理中のみ
	const Skill& getSkill() const {
		return SkillTable.at(effectState.ability.skillId);
	}

	// Instant Effect処理中のみ
	const std::vector<Effect>& getEffects() const {
		if (effectState.ability.skillId == 0) {
			const Attack& attack = getAttack();
			if (postAttackEffect) {
				return attack.postEffects;
			} else {
				return attack.preEffects;
			}
		} else {
			const Skill& skill = getSkill();
			return skill.effects;
		}
	}

	// Instant Effect処理中のみ
	const Effect& getEffect() const {
		return getEffects().at(effectState.effectIndex);
	}

	// Instant Effect処理中のみ
	AreaRef getEffectCard() const {
		return effectState.ability.effectCard;
	}

	// Instant Effect処理中のみ
	int getEffectValue() const {
		return getValueByEffect(getEffect());
	}

	// Instant Effect処理中のみ
	int getSecondEffectValue() const {
		return getValuesByEffect(getEffect())[1];
	}

	// Instant Effect処理中のみ
	int getEffectTargetPlayerIndex() const {
		TargetPlayer targetPlayer = getEffect().target.targetPlayer;
		if (targetPlayer == TargetPlayer::Me) {
			return effectPlayerIndex();
		} else if (targetPlayer == TargetPlayer::Enemy) {
			return 1 - effectPlayerIndex();
		} else {
			return 2;
		}
	}

	// Instant Effect処理中のみ
	int effectPlayerIndex() const {
		return effectState.ability.usePlayerIndex;
	}

	// Instant Effect処理中のみ
	FixedList<int, 2> effectTargetPlayer(const Effect& effect) const {
		return effectTargetPlayerContinual(effect, effectPlayerIndex());
	}

	// Instant Effect処理中のみ
	int effectLookingPlayerIndex() const {
		if (getEffect().open) {
			return 2;
		} else {
			return effectPlayerIndex();
		}
	}

	// Attack中のみ
	const Attack& getAttack() const {
		return AttackTable.at(currentAttackId);
	}

	// Attack中のみ
	CardRef getAttacker() const {
		return attacker;
	}

	void breakEffect() {
		effectJump = 99;
	}

	static FixedList<int, 2> effectTargetPlayerContinual(const Effect& effect, int effectPlayer) {
		FixedList<int, 2> target;
		TargetPlayer targetPlayer = effect.target.targetPlayer;
		if (targetPlayer == TargetPlayer::Both) {
			target.push_back(0);
			target.push_back(1);
		} else if (targetPlayer == TargetPlayer::Me) {
			target.push_back(effectPlayer);
		} else if (targetPlayer == TargetPlayer::Enemy) {
			target.push_back(1 - effectPlayer);
		}
		return target;
	}

	GameFunction& lastFunction() {
		return functionStack.back();
	}

	void setLastFunctionCallCount(int count) {
		if (count == 0) {
			functionStack.pop_back();
		} else {
			lastFunction().setCallCount(count);
		}
	}

	int calledCount() const {
		return functionStack.back().calledCount;
	}

	// 次に呼ぶ関数をスタックにプッシュ
	void pushFunction(GameFunctionNoArg func) {
		GameFunction gf{ (void*)func, ArgType::None };
		functionStack.push_back(gf);
	}
	void pushFunction(GameFunctionI func, int arg0) {
		GameFunction gf{ (void*)func, ArgType::I };
		gf.arg0 = arg0;
		functionStack.push_back(gf);
	}
	void pushFunction(GameFunctionB func, bool arg0) {
		GameFunction gf{ (void*)func, ArgType::B };
		gf.arg0 = arg0;
		functionStack.push_back(gf);
	}
	void pushFunction(GameFunctionII func, int arg0, int arg1) {
		GameFunction gf{ (void*)func, ArgType::II };
		gf.arg0 = arg0;
		gf.arg1 = arg1;
		functionStack.push_back(gf);
	}
	void pushFunction(GameFunctionIII func, int arg0, int arg1, int arg2) {
		GameFunction gf{ (void*)func, ArgType::III };
		gf.arg0 = arg0;
		gf.arg1 = arg1;
		gf.arg2 = arg2;
		functionStack.push_back(gf);
	}

	void callFunction() {
		if (functionStack.empty()) {
			Exception("no function");
		}
		int index = (int)functionStack.size() - 1;
		GameFunction gf = functionStack[index];
		void* fp = FunctionTable.at(gf.functionIndex);
		switch (gf.argType)
		{
		case ArgType::None:
			((GameFunctionNoArg)fp)(*this);
			break;
		case ArgType::I:
			((GameFunctionI)fp)(*this, gf.arg0);
			break;
		case ArgType::B:
			((GameFunctionB)fp)(*this, (bool)gf.arg0);
			break;
		case ArgType::II:
			((GameFunctionII)fp)(*this, gf.arg0, gf.arg1);
			break;
		case ArgType::III:
			((GameFunctionIII)fp)(*this, gf.arg0, gf.arg1, gf.arg2);
			break;
		default:
			Exception("unexpected arg type");
			break;
		}

		GameFunction& gameFunc = functionStack[index];
		gameFunc.calledCount += 1;
		if (isBreak || gameFunc.calledCount >= gameFunc.callCount) {
			// 指定回数呼び出し完了
			functionStack.erase(functionStack.begin() + index);
			isBreak = false;
		}
	}

	// 1手進める
	// 終了したらfalseを返す
	bool step() {
		callFunction();
		while (!isFinish()) {
			if (selectType == SelectType::None) {
				callFunction();
				continue;
			}
			int optionSize = (int)options.size();
			if (selectMax > optionSize) {
				selectMax = optionSize;
			}
			if (selectMin > selectMax) {
				selectMin = selectMax;
			}
			turnActionCount++;
			return true;
		}
		return false;
	}


	int checkPlayerSelect() const {
		if (selected.size() <= 60) {
			FixedList<int, 60> s;
			for (int i : selected) {
				s.push_back(i);
			}
			std::sort(s.begin(), s.end());
			for (int i = 1; i < s.size(); i++) {
				if (s[i - 1] == s[i]) {
					return 6;
				}
			}
		}

		for (int i : selected) {
			if (i < 0 || options.size() <= i) {
				return 5;
			}
		}

		if (selectMin > selected.size() || selectMax < selected.size()) {
			return 4;
		}
		return 0;
	}
};

