// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Framework.h"

constexpr int DECK_SIZE = 60;
constexpr int BENCH_SIZE_DEFAULT = 5;
constexpr int BENCH_SIZE_MAX = 8;
constexpr int PRIZE_SIZE = 6;

constexpr int FIRST_HAND = 7;

// デッキに同じカードを入れられる枚数
constexpr int DECK_SAME_CARD_MAX = 4;


constexpr int GRASS_ENERGY = 1;
constexpr int FIRE_ENERGY = 2;
constexpr int WATER_ENERGY = 3;
constexpr int LIGHTNING_ENERGY = 4;
constexpr int PSYCHIC_ENERGY = 5;
constexpr int FIGHTING_ENERGY = 6;
constexpr int DARKNESS_ENERGY = 7;
constexpr int METAL_ENERGY = 8;

constexpr int BOOMERANG_ENERGY = 9;
constexpr int NEO_UPPER_ENERGY = 10;
constexpr int PRISM_ENERGY = 16;
constexpr int IGNITION_ENERGY = 17;
constexpr int NITRO_FIRE_ENERGY = 1268;

constexpr int ANGE_FLOETTE = 1429;

#ifdef NDEBUG
constexpr bool IS_DEBUG = false;
#else
constexpr bool IS_DEBUG = true;
#endif



[[noreturn]]
inline void Exception(const std::string& message) {
	std::cerr << message << std::endl;
	throw std::runtime_error(message);
}
