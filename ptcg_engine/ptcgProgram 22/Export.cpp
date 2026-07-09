// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "All.h"

#ifdef _MSC_VER
#	define GAME_API __declspec(dllexport)
#else
#	define GAME_API __attribute__ ((visibility("default")))
#endif


static JsonBuilder AllCardJson;
static JsonBuilder AllAttackJson;

static const char8_t* JsonResult(ApiData* data, const SearchInfo& si) {
  SearchReturnJson(data->jsonBuilder, si);
  return data->jsonBuilder.buf.c_str();
}

static void CopyIdPtr(int* src, std::vector<int>& dest, int count, int& error) {
  if (error) {
    return;
  }
  dest.resize(count);
  for (int i : range(count)) {
    dest[i] = src[i];
    if (!CardTable.contains(dest[i])) {
      error = 1;
      break;
    }
  }
}

extern "C" {

  GAME_API void GameInitialize() {
    InitializeAll();
  }

  GAME_API StartData BattleStart(int* cards) {
    return ApiBattleStart(cards);
  }

  GAME_API ApiData* AgentStart() {
    return ApiAgentStart();
  }

  GAME_API void BattleFinish(ApiData* data) {
    return ApiBattleFinish(data);
  }

  GAME_API SerialData GetBattleData(ApiData* data) {
    if (data->apiDataType != 1) {
      return {};
    }

    if (data->preGetSelectCount != data->selectCount) {
      const State& state = data->state;
      int index = std::max(state.logIndex[0], state.logIndex[1]);
      const std::vector<int>* selected = nullptr;
      if (data->visData.size() > 0) {
        selected = &data->selected;
      }
      ToJsonVis(data->state, data->jsonBuilder, index, selected);
      data->visData.push_back(data->jsonBuilder.buf);
      data->preGetSelectCount = data->selectCount;
    }

    return ApiGetBattleData(data);
  }

  GAME_API int Select(ApiData* data, int* select, int selectCount) {
    if (data->apiDataType != 1) {
      return 30;
    }
    return ApiSelect(data, select, selectCount);
  }

  GAME_API const char8_t* VisualizeData(ApiData* data) {
    if (data->apiDataType != 1) {
      return nullptr;
    }

    ApiVisualizeData(data->jsonBuilder, data->visData);
    
    return data->jsonBuilder.buf.c_str();
  }

  GAME_API const char8_t* SearchBegin(ApiData* data, const char* serialized, int count, int* myDeck, int* myPrize, int* enemyDeck, int* enemyPrize, int* enemyHand, int* enemyActive, int manualCoin) {
    SearchInfo si;
    if (data->apiDataType != 2) {
      si = SearchInfo::error(30);
    } else {
      SetBattleData(data, serialized, count);

      try {
        const State& state = data->state;
        int myIndex = state.selectPlayer;

        int error = 0;
        SearchStartConfig config;
        if (!state.selectDeck) {
          CopyIdPtr(myDeck, config.myDeck, state.players[myIndex].deck.size(), error);
        }
        CopyIdPtr(myPrize, config.myPrize, state.players[myIndex].prize.size(), error);
        CopyIdPtr(enemyDeck, config.enemyDeck, state.players[1 - myIndex].deck.size(), error);
        CopyIdPtr(enemyPrize, config.enemyPrize, state.players[1 - myIndex].prize.size(), error);
        CopyIdPtr(enemyHand, config.enemyHand, state.players[1 - myIndex].hand.size(), error);
        if (IsActiveNull(state, 1 - myIndex)) {
          CopyIdPtr(enemyActive, config.enemyActive, state.players[1 - myIndex].active.size(), error);
        }
        config.manualCoin = (bool)manualCoin;

        if (error) {
          si = SearchInfo::error(error);
        } else {
          si = ApiSearchBegin(data, config);
        }

      } catch (...) {
        si = SearchInfo::error(99);
      }
    }
    return JsonResult(data, si);
  }

  GAME_API const char8_t* SearchStep(ApiData* data, long long searchId, int* select, int selectCount) {
    SearchInfo si;
    if (data->apiDataType != 2) {
      si = SearchInfo::error(30);
    } else {
      try {
        si = ApiSearchStep(data, searchId, select, selectCount);
      } catch (...) {
        si = SearchInfo::error(99);
      }
    }
    return JsonResult(data, si);
  }

  GAME_API void SearchEnd(ApiData* data) {
    if (data->apiDataType != 2) {
      return;
    }
    ApiSearchEnd(data);
  }

  GAME_API void SearchRelease(ApiData* data, long long searchId) {
    if (data->apiDataType != 2) {
      return;
    }
    ApiSearchRelease(data, searchId);
  }

  GAME_API const char8_t* AllCard() {
    if (AllCardJson.buf.empty()) {
      ApiAllCard(AllCardJson);
    }
    return AllCardJson.buf.c_str();
  }

  GAME_API const char8_t* AllAttack() {
    if (AllAttackJson.buf.empty()) {
      ApiAllAttack(AllAttackJson);
    }
    return AllAttackJson.buf.c_str();
  }
}
