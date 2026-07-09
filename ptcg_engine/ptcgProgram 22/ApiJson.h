// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "State.h"

inline void SelectOptionJson(JsonBuilder& j, const SelectOption& option, bool web) {
	j.append('{');
	j.appendKey("type");
	if (!web) {
		j.append((int)option.type);
	}
	switch (option.type) {
	case SelectOptionType::Number:
		if (web) {
			j.appendDoubleQuote("Number");
		}
		j.appendCommaKeyValue("number", option.param0);
		break;
	case SelectOptionType::Yes:
		if (web) {
			j.appendDoubleQuote("Yes");
		}
		break;
	case SelectOptionType::No:
		if (web) {
			j.appendDoubleQuote("No");
		}
		break;
	case SelectOptionType::Card:
		if (web) {
			j.appendDoubleQuote("Card");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		j.appendCommaKeyValue("playerIndex", option.param2);
		break;
	case SelectOptionType::ToolCard:
		if (web) {
			j.appendDoubleQuote("ToolCard");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		j.appendCommaKeyValue("playerIndex", option.param2);
		j.appendCommaKeyValue("toolIndex", option.param3);
		break;
	case SelectOptionType::EnergyCard:
		if (web) {
			j.appendDoubleQuote("EnergyCard");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		j.appendCommaKeyValue("playerIndex", option.param2);
		j.appendCommaKeyValue("energyIndex", option.param3);
		break;
	case SelectOptionType::Energy:
		if (web) {
			j.appendDoubleQuote("Energy");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		j.appendCommaKeyValue("playerIndex", option.param2);
		j.appendCommaKeyValue("energyIndex", option.param3);
		j.appendCommaKeyValue("count", option.param4);
		break;
	case SelectOptionType::Play:
		if (web) {
			j.appendDoubleQuote("Play");
		}
		j.appendCommaKeyValue("index", option.param0);
		break;
	case SelectOptionType::Attach:
		if (web) {
			j.appendDoubleQuote("Attach");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		j.appendCommaKeyValue("inPlayArea", option.param2);
		j.appendCommaKeyValue("inPlayIndex", option.param3);
		break;
	case SelectOptionType::Evolve:
		if (web) {
			j.appendDoubleQuote("Evolve");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		j.appendCommaKeyValue("inPlayArea", option.param2);
		j.appendCommaKeyValue("inPlayIndex", option.param3);
		break;
	case SelectOptionType::Ability:
		if (web) {
			j.appendDoubleQuote("Ability");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		break;
	case SelectOptionType::Discard:
		if (web) {
			j.appendDoubleQuote("Discard");
		}
		j.appendCommaKeyValue("area", option.param0);
		j.appendCommaKeyValue("index", option.param1);
		break;
	case SelectOptionType::Retreat:
		if (web) {
			j.appendDoubleQuote("Retreat");
		}
		break;
	case SelectOptionType::Attack:
		if (web) {
			j.appendDoubleQuote("Attack");
		}
		j.appendCommaKeyValue("attackId", option.param0);
		break;
	case SelectOptionType::End:
		if (web) {
			j.appendDoubleQuote("End");
		}
		break;
	case SelectOptionType::Skill:
		if (web) {
			j.appendDoubleQuote("Skill");
		}
		j.appendCommaKeyValue("cardId", option.param0);
		j.appendCommaKeyValue("serial", option.param1);
		break;
	case SelectOptionType::SpecialCondition:
		if (web) {
			j.appendDoubleQuote("SpecialCondition");
		}
		j.appendCommaKeyValue("specialConditionType", option.param0);
		break;
	default:
		j.appendDoubleQuote("None");
		assert(false);
		break;
	}
	j.append('}');
}

inline void LogJson(JsonBuilder& j, const Log& log, int playerIndex, bool web) {
	j.append('{');
	j.appendKey("type");
	switch (log.logType) {
	case LogType::Shuffle:
		if (web) {
			j.appendDoubleQuote("Shuffle");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		break;
	case LogType::HasBasicPokemon:
		if (web) {
			j.appendDoubleQuote("HasBasicPokemon");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("hasBasicPokemon", (bool)log.param[1]);
		break;
	case LogType::TurnStart:
		if (web) {
			j.appendDoubleQuote("TurnStart");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		break;
	case LogType::TurnEnd:
		if (web) {
			j.appendDoubleQuote("TurnEnd");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		break;
	case LogType::Draw:
		if (log.param[0] == playerIndex || playerIndex == 2) {
			if (web) {
				j.appendDoubleQuote("Draw");
			} else {
				j.append((int)log.logType);
			}
			j.appendCommaKeyValue("playerIndex", log.param[0]);
			j.appendCommaKeyValue("cardId", log.param[1]);
			j.appendCommaKeyValue("serial", log.param[2]);
		} else {
			if (web) {
				j.appendDoubleQuote("DrawReverse");
			} else {
				j.append((int)LogType::DrawReverse);
			}
			j.appendCommaKeyValue("playerIndex", log.param[0]);
		}
		break;
	case LogType::DrawReverse:
		if (web) {
			j.appendDoubleQuote("DrawReverse");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		break;
	case LogType::MoveCard:
		if (log.param[5] == 0 || (log.param[5] == 1 && log.param[0] == playerIndex) || (log.param[5] == 3 && playerIndex == 0) || (log.param[5] == 4 && playerIndex == 1) || playerIndex == 2) {
			if (web) {
				j.appendDoubleQuote("MoveCard");
			} else {
				j.append((int)log.logType);
			}
			j.appendCommaKeyValue("playerIndex", log.param[0]);
			j.appendCommaKeyValue("cardId", log.param[1]);
			j.appendCommaKeyValue("serial", log.param[2]);
			j.appendCommaKeyValue("fromArea", log.param[3]);
			j.appendCommaKeyValue("toArea", log.param[4]);
		} else {
			if (web) {
				j.appendDoubleQuote("MoveCardReverse");
			} else {
				j.append((int)LogType::MoveCardReverse);
			}
			j.appendCommaKeyValue("playerIndex", log.param[0]);
			j.appendCommaKeyValue("fromArea", log.param[3]);
			j.appendCommaKeyValue("toArea", log.param[4]);
		}
		break;
	case LogType::MoveCardReverse:
		if (web) {
			j.appendDoubleQuote("MoveCardReverse");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("fromArea", log.param[1]);
		j.appendCommaKeyValue("toArea", log.param[2]);
		break;
	case LogType::Switch:
		if (web) {
			j.appendDoubleQuote("Switch");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardIdActive", log.param[1]);
		j.appendCommaKeyValue("serialActive", log.param[2]);
		j.appendCommaKeyValue("cardIdBench", log.param[3]);
		j.appendCommaKeyValue("serialBench", log.param[4]);
		break;
	case LogType::Change:
		if (web) {
			j.appendDoubleQuote("Change");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardIdBefore", log.param[1]);
		j.appendCommaKeyValue("serialBefore", log.param[2]);
		j.appendCommaKeyValue("cardIdAfter", log.param[3]);
		j.appendCommaKeyValue("serialAfter", log.param[4]);
		break;
	case LogType::Play:
		if (web) {
			j.appendDoubleQuote("Play");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		break;
	case LogType::Attach:
		if (web) {
			j.appendDoubleQuote("Attach");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		j.appendCommaKeyValue("cardIdTarget", log.param[3]);
		j.appendCommaKeyValue("serialTarget", log.param[4]);
		break;
	case LogType::Evolve:
		if (web) {
			j.appendDoubleQuote("Evolve");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		j.appendCommaKeyValue("cardIdTarget", log.param[3]);
		j.appendCommaKeyValue("serialTarget", log.param[4]);
		break;
	case LogType::Devolve:
		if (web) {
			j.appendDoubleQuote("Devolve");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		j.appendCommaKeyValue("cardIdTarget", log.param[3]);
		j.appendCommaKeyValue("serialTarget", log.param[4]);
		break;
	case LogType::MoveAttached:
		if (web) {
			j.appendDoubleQuote("MoveAttached");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		j.appendCommaKeyValue("cardIdBefore", log.param[3]);
		j.appendCommaKeyValue("serialBefore", log.param[4]);
		j.appendCommaKeyValue("cardIdAfter", log.param[5]);
		j.appendCommaKeyValue("serialAfter", log.param[6]);
		break;
	case LogType::Attack:
		if (web) {
			j.appendDoubleQuote("Attack");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		j.appendCommaKeyValue("attackId", log.param[3]);
		break;
	case LogType::HpChange:
		if (web) {
			j.appendDoubleQuote("HpChange");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("cardId", log.param[1]);
		j.appendCommaKeyValue("serial", log.param[2]);
		j.appendCommaKeyValue("value", log.param[3]);
		j.appendCommaKeyValue("putDamageCounter", (bool)log.param[4]);
		break;
	case LogType::Poisoned:
		if (web) {
			j.appendDoubleQuote("Poisoned");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("isRecover", (bool)log.param[1]);
		j.appendCommaKeyValue("cardId", log.param[2]);
		j.appendCommaKeyValue("serial", log.param[3]);
		break;
	case LogType::Burned:
		if (web) {
			j.appendDoubleQuote("Burned");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("isRecover", (bool)log.param[1]);
		j.appendCommaKeyValue("cardId", log.param[2]);
		j.appendCommaKeyValue("serial", log.param[3]);
		break;
	case LogType::Asleep:
		if (web) {
			j.appendDoubleQuote("Asleep");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("isRecover", (bool)log.param[1]);
		j.appendCommaKeyValue("cardId", log.param[2]);
		j.appendCommaKeyValue("serial", log.param[3]);
		break;
	case LogType::Paralyzed:
		if (web) {
			j.appendDoubleQuote("Paralyzed");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("isRecover", (bool)log.param[1]);
		j.appendCommaKeyValue("cardId", log.param[2]);
		j.appendCommaKeyValue("serial", log.param[3]);
		break;
	case LogType::Confused:
		if (web) {
			j.appendDoubleQuote("Confused");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("isRecover", (bool)log.param[1]);
		j.appendCommaKeyValue("cardId", log.param[2]);
		j.appendCommaKeyValue("serial", log.param[3]);
		break;
	case LogType::Coin:
		if (web) {
			j.appendDoubleQuote("Coin");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("playerIndex", log.param[0]);
		j.appendCommaKeyValue("head", (bool)log.param[1]);
		break;
	case LogType::Result:
		if (web) {
			j.appendDoubleQuote("Result");
		} else {
			j.append((int)log.logType);
		}
		j.appendCommaKeyValue("result", log.param[0]);
		j.appendCommaKeyValue("reason", log.param[1]);
		break;
	default:
		j.appendDoubleQuote("None");
		assert(false);
		break;
	}
	j.append('}');
}

