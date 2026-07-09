// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "BattleData.h"
#include "Search.h"

class ApiData : public BattleData {
public:

	int apiDataType = 0; // 1:battle, 2:agent
	int selectCount = 0; // 選択回数
	int preGetSelectCount = -1; // 前にGetBattleDataが呼ばれたときのselectCount
	int initializeError = -1;
	Search search;
	std::vector<int> selected;
	JsonBuilder jsonBuilder;
	BinaryWriter writer;
	BinaryReader reader;
	std::vector<std::u8string> visData;
};
