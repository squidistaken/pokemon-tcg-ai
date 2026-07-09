// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "SetupProc.h"

class BattleData {
public:

	Game game = {};
  State state = {};

	void setState(const GameConfig& config) {
		game.init(config);

		int cardIndex = 0;
		state.game = &game;
		state.moveCounter = 1;
		state.firstPlayer = -1;
		state.allCard[cardIndex++] = {}; // 最初を空ける
		for (int i : range(2)) {
			Card& card = state.allCard[cardIndex++]; // プレイヤー
			card.init(0, state.moveCounter++, i);
			card.area = AreaType::Player;
		}
	}

	void init(const GameConfig& config, bool checkDeck = true) {
		setState(config);
		int cardIndex = 3;
		for (int i : range(2)) {
			PlayerState& ps = state.players[i];
			ps.playerIndex = i;

			int basicPokemonCount = 0;
			const Deck& deck = config.decks[i];
			ps.deck.resize(DECK_SIZE);
			for (int j : range(DECK_SIZE)) {
				int index = cardIndex++;
				Card& card = state.allCard[index];
				card.init(deck.cards[j], state.moveCounter++, i);
				ps.deck[DECK_SIZE - j - 1] = CardRef(index);
				const CardMaster& master = card.getMaster();
				if (master.cardType == CardType::Pokemon && master.evolutionType == EvolutionType::Basic) {
					basicPokemonCount++;
				}
			}
			if (checkDeck) {
				if (basicPokemonCount == 0) {
					Exception("No basic pokemon");
				}
			}

      game.remainingTime[i] = config.timeLimit * 1000;
		}
	}

	void start() {
		state.pushFunction(SetupGame);
	}

	bool next() {
		bool isContinue = state.step();

		if (!state.game->selecting) {
			state.game->actionCount++;
			if (state.game->actionCount >= 3000) {
				state.setResult(2, FinishReason::Other);
				isContinue = false;
			}
		}

		if (!isContinue) {
			int reason = (int)state.finishReason;
			LogResult(state, (int)state.gameResult - 1, reason);
		}
		return isContinue;
	}
};

inline void ReservedFunction() {}
inline void ReservedFunction2() {}
inline void ReservedFunction3() {}

inline void InitializeBattleFunction() {
  FunctionTable.push_back((void*)ActivateEffect2);
  FunctionTable.push_back((void*)ActivateEffectEachSelected);
  FunctionTable.push_back((void*)ActivateEffectForEach);
  FunctionTable.push_back((void*)ActivateEffectMultiple);
  FunctionTable.push_back((void*)ActivateSkillEffect);
  FunctionTable.push_back((void*)AfterAbility);
  FunctionTable.push_back((void*)AfterAttack);
  FunctionTable.push_back((void*)AfterAttack2);
  FunctionTable.push_back((void*)AfterAttack3);
  FunctionTable.push_back((void*)AfterAttack4);
  FunctionTable.push_back((void*)AfterAttackDamageChangeCoin);
  FunctionTable.push_back((void*)AfterAttackDamageChangeCoinUntilTail);
  FunctionTable.push_back((void*)AfterAttackDamageChangeTargetCountEnemyCoin);
  FunctionTable.push_back((void*)AfterAttackDamageCoin);
  FunctionTable.push_back((void*)AfterAttackTrigger);
  FunctionTable.push_back((void*)AfterBreakIfCoinHead);
  FunctionTable.push_back((void*)AfterBreakIfCoinTail);
  FunctionTable.push_back((void*)AfterBreakIfCoinTailMulti);
  FunctionTable.push_back((void*)AfterBurnProc);
  FunctionTable.push_back((void*)AfterDeckToTrashCoinUntilTail);
  FunctionTable.push_back((void*)AfterEffect);
  FunctionTable.push_back((void*)AfterPlay);
  FunctionTable.push_back((void*)AfterRefresh);
  FunctionTable.push_back((void*)AfterResetupActivePokemon);
  FunctionTable.push_back((void*)AfterRetreat);
  FunctionTable.push_back((void*)AfterSetupActivePokemon);
  FunctionTable.push_back((void*)AfterSkipIfCoinTail);
  FunctionTable.push_back((void*)AfterSleepProc);
  FunctionTable.push_back((void*)AfterTriggerAbility);
  FunctionTable.push_back((void*)AttackCoinProc);
  FunctionTable.push_back((void*)AttackCoinProc2);
  FunctionTable.push_back((void*)AttackDamage);
  FunctionTable.push_back((void*)AttackDamageCoin);
  FunctionTable.push_back((void*)AttackEffect);
  FunctionTable.push_back((void*)AttackEffects);
  FunctionTable.push_back((void*)AttackNoDamageCoin);
  FunctionTable.push_back((void*)BothSetupActivePokemon);
  FunctionTable.push_back((void*)BurnProc);
  FunctionTable.push_back((void*)CoinLuckyBonus);
  FunctionTable.push_back((void*)CoinPrize);
  FunctionTable.push_back((void*)ConfuseProc);
  FunctionTable.push_back((void*)EffectAttackDamage);
  FunctionTable.push_back((void*)EffectAttackNoDamageCoin);
  FunctionTable.push_back((void*)EnemySelectedWhichEffect);
  FunctionTable.push_back((void*)KOProc);
  FunctionTable.push_back((void*)KOProc2);
  FunctionTable.push_back((void*)KOProc3);
  FunctionTable.push_back((void*)MainSelect);
  FunctionTable.push_back((void*)ParalyzeProc);
  FunctionTable.push_back((void*)PokemonCheckup);
  FunctionTable.push_back((void*)PokemonCheckupEnd);
  FunctionTable.push_back((void*)PreSetupActivePokemon);
  FunctionTable.push_back((void*)ResolveTriggerStack);
  FunctionTable.push_back((void*)SelectActivePokemon);
  FunctionTable.push_back((void*)SelectBenchMaxTrash);
  FunctionTable.push_back((void*)SelectCoinSingle);
  FunctionTable.push_back((void*)SelectDamageCounterAny);
  FunctionTable.push_back((void*)SelectDamageMulti);
  FunctionTable.push_back((void*)SelectedActivate);
  FunctionTable.push_back((void*)SelectedActivateAbility);
  FunctionTable.push_back((void*)SelectedAddDamageCounterSwitchAny);
  FunctionTable.push_back((void*)SelectedAttackDamageChangePutDamageCounter);
  FunctionTable.push_back((void*)SelectedAttackId);
  FunctionTable.push_back((void*)SelectedBenchMaxTrash);
  FunctionTable.push_back((void*)SelectedBurn);
  FunctionTable.push_back((void*)SelectedCardOrAttachedCard);
  FunctionTable.push_back((void*)SelectedCoinSingle);
  FunctionTable.push_back((void*)SelectedCoinUntilTail);
  FunctionTable.push_back((void*)SelectedDamageCounterAny);
  FunctionTable.push_back((void*)SelectedDamageMulti);
  FunctionTable.push_back((void*)SelectedDamageMultiAll);
  FunctionTable.push_back((void*)SelectedDisableAttack);
  FunctionTable.push_back((void*)SelectedDrawCount);
  FunctionTable.push_back((void*)SelectedEffectTarget);
  FunctionTable.push_back((void*)SelectedEnergyTarget);
  FunctionTable.push_back((void*)SelectedEvolveTarget);
  FunctionTable.push_back((void*)SelectedFirstEffect);
  FunctionTable.push_back((void*)SelectedIsFirst);
  FunctionTable.push_back((void*)SelectedLuckyBonus);
  FunctionTable.push_back((void*)SelectedMain);
  FunctionTable.push_back((void*)SelectedMoreDevolve);
  FunctionTable.push_back((void*)SelectedMulligan);
  FunctionTable.push_back((void*)SelectedPokemonEnergy);
  FunctionTable.push_back((void*)SelectedPokemonEnergyLoop);
  FunctionTable.push_back((void*)SelectedPrize);
  FunctionTable.push_back((void*)SelectedRecoverSpecialCondition);
  FunctionTable.push_back((void*)SelectedRemoveDamageCounter);
  FunctionTable.push_back((void*)SelectedRemoveDamageCounterSwitchAny);
  FunctionTable.push_back((void*)SelectedSecondAttack);
  FunctionTable.push_back((void*)SelectedSetupActivePokemon);
  FunctionTable.push_back((void*)SelectedSetupBenchPokemon);
  FunctionTable.push_back((void*)SelectedSkillOrder);
  FunctionTable.push_back((void*)SelectedSpecialCondition);
  FunctionTable.push_back((void*)SelectedSwitchPokemon);
  FunctionTable.push_back((void*)SelectedToolMaxTrash);
  FunctionTable.push_back((void*)SelectedToolTarget);
  FunctionTable.push_back((void*)SelectedWhichEffect);
  FunctionTable.push_back((void*)SelectLuckyBonus);
  FunctionTable.push_back((void*)SelectPokemonEnergyLoop);
  FunctionTable.push_back((void*)SelectPrize);
  FunctionTable.push_back((void*)SelectedRetreat2);
  FunctionTable.push_back((void*)SelectSwitchPokemon);
  FunctionTable.push_back((void*)SelectToolMaxTrash);
  FunctionTable.push_back((void*)SeparatorProc);
  FunctionTable.push_back((void*)SetupActivePokemon);
  FunctionTable.push_back((void*)SetupBenchPokemon);
  FunctionTable.push_back((void*)SetupGame);
  FunctionTable.push_back((void*)SleepProc);
  FunctionTable.push_back((void*)StartSetupBench);
  FunctionTable.push_back((void*)ToMain);
  FunctionTable.push_back((void*)ToolCountProc);
  FunctionTable.push_back((void*)TurnEnd2);
  FunctionTable.push_back((void*)TurnStart);
  FunctionTable.push_back((void*)ReservedFunction);
  FunctionTable.push_back((void*)ReservedFunction2);
  FunctionTable.push_back((void*)ReservedFunction3);

  for (int i : range(FunctionTable)) {
    FunctionIndexTable[(long long)FunctionTable[i]] = i;
  }
}

