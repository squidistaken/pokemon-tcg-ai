// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "State.h"

inline bool IsTargetSingle(const State& state, CardRef ref, const Card& card, const CardMaster& master, const TargetCondition& target, const AreaRef& effectCard) {
	switch (target.targetType)
	{
	case TargetType::All:
		return true;
	case TargetType::Hp:
		return Compare(state.getHp(card), target.val, target.comparatorType);
	case TargetType::MaxHp:
		return Compare(master.hp, target.val, target.comparatorType);
	case TargetType::RetreatCost:
		return Compare(state.retreatCost(card), target.val, target.comparatorType);
	case TargetType::EnergyType:
		return BoolCompare(MatchEnergyType(state.getEnergyType(card), (EnergyType)target.val), target.comparatorType);
	case TargetType::EnergyType2:
		return BoolCompare(MatchEnergyType(state.getEnergyType(card), (EnergyType)target.val) || MatchEnergyType(state.getEnergyType(card), (EnergyType)target.val2), target.comparatorType);
	case TargetType::Resistance:
		return BoolCompare(MatchEnergyType(master.resistance, (EnergyType)target.val), target.comparatorType);
	case TargetType::PokemonCard:
		return BoolCompare(master.isPokemonCard(), target.comparatorType);
	case TargetType::BasicPokemon:
		if (card.area == AreaType::Active || card.area == AreaType::Bench) {
			return BoolCompare(master.evolutionType == EvolutionType::Basic, target.comparatorType);
		} else {
			return BoolCompare(master.evolutionType == EvolutionType::Basic && master.cardType == CardType::Pokemon, target.comparatorType);
		}
	case TargetType::EvolvedPokemon:
		return BoolCompare(master.evolutionType == EvolutionType::Stage1 || master.evolutionType == EvolutionType::Stage2, target.comparatorType);
	case TargetType::Stage1:
		return BoolCompare(master.evolutionType == EvolutionType::Stage1, target.comparatorType);
	case TargetType::Stage2:
		return BoolCompare(master.evolutionType == EvolutionType::Stage2, target.comparatorType);
	case TargetType::BasicEnergy:
		return BoolCompare(master.cardType == CardType::BasicEnergy, target.comparatorType);
	case TargetType::SpecialEnergy:
		return BoolCompare(master.cardType == CardType::SpecialEnergy, target.comparatorType);
	case TargetType::EnergyCard:
		return BoolCompare(IsEnergy(master.cardType), target.comparatorType);
	case TargetType::Item:
		return BoolCompare(master.cardType == CardType::Item, target.comparatorType);
	case TargetType::Tool:
		return BoolCompare(master.cardType == CardType::Tool, target.comparatorType);
	case TargetType::Supporter:
		return BoolCompare(master.cardType == CardType::Supporter, target.comparatorType);
	case TargetType::Stadium:
		return BoolCompare(master.cardType == CardType::Stadium, target.comparatorType);
	case TargetType::Trainer:
		return BoolCompare(IsTrainer(master.cardType), target.comparatorType);
	case TargetType::CardId:
		return BoolCompare(master.cardId == target.val, target.comparatorType);
	case TargetType::PokemonOrBasicEnergy:
		return BoolCompare(master.cardType == CardType::Pokemon || master.cardType == CardType::BasicEnergy, target.comparatorType);
	case TargetType::BasicPokemonOrBasicEnergy:
		if (master.cardType == CardType::Pokemon) {
			return BoolCompare(master.evolutionType == EvolutionType::Basic, target.comparatorType);
		} else {
			return BoolCompare(master.cardType == CardType::BasicEnergy, target.comparatorType);
		}
	case TargetType::NotRulePokemonCardOrBasicEnergy:
		return BoolCompare(master.isNotRulePokemonCard() || master.cardType == CardType::BasicEnergy, target.comparatorType);
	case TargetType::EnergyTypePokemonOrStadium: {
		bool match = false;
		if (master.cardType == CardType::Pokemon) {
			match = MatchEnergyType(state.getEnergyType(card), (EnergyType)target.val);
		} else if (master.cardType == CardType::Stadium) {
			match = true;
		}
		return BoolCompare(match, target.comparatorType);
	}
	case TargetType::ItemOrTool:
		return BoolCompare(master.cardType == CardType::Item || master.cardType == CardType::Tool, target.comparatorType);
	case TargetType::EnemyToolOrSpecialEnergyOrStadium: {
		const Card& c = state.getCard(effectCard.card);
		if (card.area == AreaType::Tool) {
			return c.playerIndex != card.playerIndex;
		} else if (card.area == AreaType::Energy) {
			return c.playerIndex != card.playerIndex && master.cardType == CardType::SpecialEnergy;
		} else {
			return true;
		}
	}
	case TargetType::EthanPokemonOrBasicFireEnergy:
		return BoolCompare(master.ethan || master.cardId == FIRE_ENERGY, target.comparatorType);
	case TargetType::HasAbility:
		return BoolCompare(state.getAbility(card, master) != nullptr, target.comparatorType);
	case TargetType::HasAbilityName:
	{
		const Skill* ability = state.getAbility(card, master);
		return BoolCompare(ability != nullptr && ability->name == target.name, target.comparatorType);
	}
	case TargetType::HasAttackName:
	{
		bool found = false;
		for (const Attack* attack : master.attacks) {
			if (attack->name == target.name) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::RulePokemon:
		return BoolCompare(master.isRulePokemon(), target.comparatorType);
	case TargetType::NotRulePokemon:
		return BoolCompare(master.isNotRulePokemon(), target.comparatorType);
	case TargetType::Ex:
		return BoolCompare(master.isEx(), target.comparatorType);
	case TargetType::MegaEx:
		return BoolCompare(master.pokemonType == PokemonType::MegaEx, target.comparatorType);
	case TargetType::Terastal:
		return BoolCompare(master.tera, target.comparatorType);
	case TargetType::Ancient:
		return BoolCompare(master.ancient, target.comparatorType);
	case TargetType::Future:
		return BoolCompare(master.future, target.comparatorType);
	case TargetType::Hop:
		return BoolCompare(master.hop, target.comparatorType);
	case TargetType::Lillie:
		return BoolCompare(master.lillie, target.comparatorType);
	case TargetType::Iono:
		return BoolCompare(master.iono, target.comparatorType);
	case TargetType::N:
		return BoolCompare(master.n, target.comparatorType);
	case TargetType::Ethan:
		return BoolCompare(master.ethan, target.comparatorType);
	case TargetType::Cynthia:
		return BoolCompare(master.cynthia, target.comparatorType);
	case TargetType::Misty:
		return BoolCompare(master.misty, target.comparatorType);
	case TargetType::Arven:
		return BoolCompare(master.arven, target.comparatorType);
	case TargetType::Steven:
		return BoolCompare(master.steven, target.comparatorType);
	case TargetType::Marnie:
		return BoolCompare(master.marnie, target.comparatorType);
	case TargetType::Erika:
		return BoolCompare(master.erika, target.comparatorType);
	case TargetType::Larry:
		return BoolCompare(master.larry, target.comparatorType);
	case TargetType::TeamRocket:
		return BoolCompare(master.teamRocket, target.comparatorType);
	case TargetType::SilcoonOrCascoon:
		return BoolCompare(master.name == u8"カラサリス" || master.name == u8"マユルド", target.comparatorType);
	case TargetType::KoffingOrWeezing:
		return BoolCompare(master.name.find(u8"ドガース") != std::u8string::npos || master.name.find(u8"マタドガス") != std::u8string::npos, target.comparatorType);
	case TargetType::HonedgeOrDoubladeOrAegislash:
		return BoolCompare(master.name == u8"ヒトツキ" || master.name == u8"ニダンギル" || master.name == u8"ギルガルド", target.comparatorType);
	case TargetType::Name:
		return BoolCompare(master.name == target.name, target.comparatorType);
	case TargetType::NameContains:
		return BoolCompare(master.name.find(target.name) != std::u8string::npos, target.comparatorType);
	case TargetType::CanEvolve:
		if (master.evolutionType != EvolutionType::Stage1 && master.evolutionType != EvolutionType::Stage2) {
			return false;
		}
		for (RefPosition rp : state.players.at(card.playerIndex).getInPlayPokemon()) {
			if (state.canEvolveEffect(state, card, master, rp.ref)) {
				return true;
			}
		}
		return false;
	case TargetType::CanEvolve2:
		if (master.evolutionType != EvolutionType::Stage2) {
			return false;
		}
		for (RefPosition rp : state.players.at(card.playerIndex).getInPlayPokemon()) {
			if (state.canEvolve2(card, master, rp.ref)) {
				return true;
			}
		}
		return false;
	case TargetType::CanEvolveMe:
	{
		return BoolCompare(state.canEvolveEffect(state, card, master, effectCard.card), target.comparatorType);
	}
	case TargetType::CanEvolveContextCard:
	{
		if (state.contextCard.isNull()) {
			return false;
		}
		return BoolCompare(state.canEvolveEffect(state, card, master, state.contextCard), target.comparatorType);
	}
	case TargetType::CanEvolvesToContextCard:
	{
		if (state.contextCard.isNull()) {
			return false;
		}
		const Card& contextCard = state.getCard(state.contextCard);
		return BoolCompare(state.canEvolveEffect(state, contextCard, contextCard.getMaster(), ref), target.comparatorType);
	}
	case TargetType::CanEvolveField:
	{
		bool found = false;
		for (RefPosition rp : state.players.at(card.playerIndex).getInPlayPokemon()) {
			if (state.canEvolveEffect(state, card, master, rp.ref)) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::CanEvolveFieldNotAppearThisTurn:
	{
		bool found = false;
		for (RefPosition rp : state.players.at(card.playerIndex).getInPlayPokemon()) {
			if (state.getCard(rp.ref).appear) {
				continue;
			}
			if (state.canEvolveEffect(state, card, master, rp.ref)) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::Evolved:
		return BoolCompare(state.isEvolved(ref), target.comparatorType);
	case TargetType::EvolvedThisTurnName:
		for (auto info : state.turnEvolve) {
			if (info.ref == ref) {
				if (state.getCard(info.preRef).getMaster().name == target.name) {
					return true;
				}
			}
		}
		return false;
	case TargetType::NotAppearThisTurn:
		return BoolCompare(!card.appear, target.comparatorType);
	case TargetType::HealThisTurn:
		for (CardRef r : state.turnHeal) {
			if (r == ref) {
				return true;
			}
		}
		return false;
	case TargetType::AttachedMe:
		return BoolCompare(card.attachMoveCounter == state.getCard(effectCard.card).moveCounter, target.comparatorType);
	case TargetType::AttachedEffected:
	{
		bool found = false;
		for (const AreaRef& r : state.preTargetList) {
			const Card& c = state.getCard(r.card);
			if (card.attachMoveCounter == c.moveCounter) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::AttachedTriggerSubject:
	{
		if (!state.onSkill()) {
			return false;
		}
		CardRef r = state.triggerInfo.subject.card;
		if (r.isNull()) {
			return false;
		}
		return BoolCompare(card.attachMoveCounter == state.getCard(r).moveCounter, target.comparatorType);
	}
	case TargetType::AttachedTriggerObject:
	{
		if (!state.onSkill()) {
			return false;
		}
		CardRef r = state.triggerInfo.object.card;
		if (r.isNull()) {
			return false;
		}
		return BoolCompare(card.attachMoveCounter == state.getCard(r).moveCounter, target.comparatorType);
	}
	case TargetType::AttachedContextCard:
	{
		CardRef r = state.contextCard;
		if (r.isNull()) {
			return false;
		}
		return BoolCompare(card.attachMoveCounter == state.getCard(r).moveCounter, target.comparatorType);
	}
	case TargetType::AttachedActivePokemon:
	{
		const PlayerState& ps = state.players.at(card.playerIndex);
		if (ps.active.empty()) {
			return false;
		}
		return BoolCompare(card.attachMoveCounter == state.getCard(ps.getActive()).moveCounter, target.comparatorType);
	}
	case TargetType::AttachedBenchPokemon:
	{
		const PlayerState& ps = state.players.at(card.playerIndex);
		bool found = false;
		for (CardRef r : ps.bench) {
			const Card& c = state.getCard(r);
			if (card.attachMoveCounter == c.moveCounter) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::IsAttachedEnergy:
		state.getEnergies(card.playerIndex, ref, state.game->energyList);
		return BoolCompare(state.game->energyList.size() > 0, target.comparatorType);
	case TargetType::AttachedEnergyCount:
		state.getEnergies(card.playerIndex, ref, state.game->energyList);
		return Compare((int)state.game->energyList.size(), target.val, target.comparatorType);
	case TargetType::IsAttachedSpecialEnergy:
	{
		state.getEnergyCards(ref, state.game->cardList);
		bool found = false;
		for (CardRef r : state.game->cardList) {
			if (state.getCard(r).getMaster().cardType == CardType::SpecialEnergy) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::IsAttachedEnergyType:
	{
		state.getEnergies(card.playerIndex, ref, state.game->energyList);
		EnergyType targetType = (EnergyType)target.val;
		bool found = false;
		for (EnergyType type : state.game->energyList) {
			if (ContainsEnergyType(type, targetType)) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::IsAttachedEnergy2Type:
	{
		state.getEnergies(card.playerIndex, ref, state.game->energyList);
		EnergyType targetType = (EnergyType)target.val;
		int count = 0;
		for (EnergyType type : state.game->energyList) {
			if (ContainsEnergyType(type, targetType)) {
				count++;
			}
		}
		return BoolCompare(count >= 2, target.comparatorType);
	}
	case TargetType::IsAttachedEnergyName:
	{
		auto& cards = state.game->cardList;
		state.getEnergyCards(ref, cards);
		bool found = false;
		for (CardRef r : cards) {
			if (target.name == state.getCard(r).getMaster().name) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::IsAttachedTool:
	{
		auto tools = state.getAttachedToolRef(card);
		return BoolCompare(tools.size() > 0, target.comparatorType);
	}
	case TargetType::IsAttachedToolName:
	{
		auto tools = state.getAttachedToolRef(card);
		bool found = false;
		for (CardRef r : tools) {
			if (target.name == state.getCard(r).getMaster().name) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::IsAttachedToolOrSpecialEnergy:
	{
		auto tools = state.getAttachedToolRef(card);
		bool found = (tools.size() > 0);
		if (!found) {
			state.getEnergyCards(ref, state.game->cardList);
			for (CardRef ref : state.game->cardList) {
				if (state.getCard(ref).getMaster().cardType == CardType::SpecialEnergy) {
					found = true;
					break;
				}
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::NotContextCardAttachedPokemon:
	{
		if (state.contextCard.isNull()) {
			return true;
		}
		const Card& contextCard = state.getCard(state.contextCard);
		return BoolCompare(contextCard.attachMoveCounter != card.moveCounter, target.comparatorType);
	}
	case TargetType::NotSelectedListAttachedPokemon:
	{
		bool found = false;
		for (CardRef r : state.selectedList) {
			const Card& c = state.getCard(r);
			if (card.moveCounter == c.attachMoveCounter) {
				found = true;
				break;
			}
		}
		return BoolCompare(!found, target.comparatorType);
	}
	case TargetType::EnergyTypeAttached:
	{
		CardRef pokemonRef = state.attachedCardPosition(card).ref;
		EnergyInfo ei = state.getEnergyInfo(card, pokemonRef);
		return BoolCompare(MatchEnergyType(ei.type, (EnergyType)target.val), target.comparatorType);
	}
	case TargetType::Reverse:
		if (card.area == AreaType::Hand) {
			return true;
		}
		return BoolCompare(card.reverse, target.comparatorType);
	case TargetType::Area:
		return BoolCompare((int)card.area == target.val, target.comparatorType);
	case TargetType::TriggerSubject:
	{
		bool found = false;
		if (state.onSkill()) {
			if (ref == state.triggerInfo.subject.card) {
				found = true;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::TriggerObject:
	{
		bool found = false;
		if (state.onSkill()) {
			if (ref == state.triggerInfo.object.card) {
				found = true;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::DamageCounter:
		return Compare(card.damage / 10, target.val, target.comparatorType);
	case TargetType::MinHp:
	{
		int minHp = 10000000;
		for (int pi : range(2)) {
			for (const RefPosition& rp : state.players[pi].getInPlayPokemon()) {
				if (rp.ref == effectCard.card) {
					continue;
				}
				int hp = state.getHp(state.getCard(rp.ref));
				minHp = std::min(minHp, hp);
			}
		}
		return BoolCompare(state.getHp(card) == minHp, target.comparatorType);
	}
	case TargetType::SameTypeEnemy:
	{
		int type = 0;
		bool hasColorless = false;

		auto inPlayEnemy = state.players.at(1 - card.playerIndex).getInPlayPokemon();
		for (RefPosition& rp : inPlayEnemy) {
			auto types = state.getCard(rp.ref).getEnergyType();
			for (EnergyType t : types) {
				type |= (int)t;
				if (t == EnergyType::Colorless) {
					hasColorless = true;
				}
			}
		}

		bool found = false;
		auto inPlay = state.players.at(card.playerIndex).getInPlayPokemon();
		for (RefPosition& rp : inPlay) {
			auto types = state.getCard(rp.ref).getEnergyType();
			for (EnergyType t : types) {
				if (t == EnergyType::Colorless) {
					if (hasColorless) {
						found = true;
						break;
					}
				} else {
					if (type & (int)t) {
						found = true;
						break;
					}
				}
			}
		}

		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::SpecialCondition:
		if (card.area != AreaType::Active) {
			return false;
		}
		return BoolCompare(state.players.at(card.playerIndex).isSpecialCondition(), target.comparatorType);
	case TargetType::SpecialConditionOrDamaged:
		if (card.area != AreaType::Active) {
			return false;
		}
		return BoolCompare(state.players.at(card.playerIndex).isSpecialCondition() || card.damage > 0, target.comparatorType);
	case TargetType::Poison:
		if (card.area != AreaType::Active) {
			return false;
		}
		return BoolCompare(state.players.at(card.playerIndex).isPoisoned(), target.comparatorType);
	case TargetType::Burn:
		if (card.area != AreaType::Active) {
			return false;
		}
		return BoolCompare(state.players.at(card.playerIndex).isBurned(), target.comparatorType);
	case TargetType::Confuse:
		if (card.area != AreaType::Active) {
			return false;
		}
		return BoolCompare(state.players.at(card.playerIndex).badStatus == BadStatusType::Confused, target.comparatorType);
	case TargetType::PoisonOrBurn:
		if (card.area != AreaType::Active) {
			return false;
		}
		return BoolCompare(state.players.at(card.playerIndex).isPoisoned() || state.players.at(card.playerIndex).isBurned(), target.comparatorType);
	case TargetType::BenchToActiveThisTurn:
		return BoolCompare(card.benchToActive, target.comparatorType);
	case TargetType::SameNameEnemyField:
	{
		bool found = false;
		for (RefPosition rp : state.players[1 - state.getCard(effectCard.card).playerIndex].getInPlayPokemon()) {
			if (card.getMaster().name == state.getCard(rp.ref).getMaster().name) {
				found = true;
				break;
			}
		}
		return BoolCompare(found, target.comparatorType);
	}
	case TargetType::NotChecked:
	{
		bool found = false;
		for (CardRef r : state.checkList) {
			if (card.cardId == state.getCardId(r)) {
				found = true;
				break;
			}
		}
		return BoolCompare(!found, target.comparatorType);
	}
	default:
		Exception("invalid target type");
		return false;
	}
}

inline bool IsTarget(const State& state, CardRef ref, const Target& target, const AreaRef& effectCard) {
	if (target.conditions.empty()) {
		return true;
	}
	const Card& card = state.getCard(ref);
	const CardMaster& master = card.getMaster();
	for (const TargetCondition& tc : target.conditions) {
		if (!IsTargetSingle(state, ref, card, master, tc, effectCard)) {
			return false;
		}
	}
	return true;
}

inline void AddIfTarget(const State& state, CardRef ref, const Target& target, std::vector<AreaRef>& output, const AreaRef& effectCard) {
	if (IsTarget(state, ref, target, effectCard)) {
		output.push_back(state.makeAreaRef(ref));
	}
}

// targetに合致する対象をoutputに入れる
inline void TargetList(const State& state, const Target& target, std::vector<AreaRef>& output, AreaRef effectCard, int effectOwner) {
	if (target.areas.size() == 0) {
		output.clear();
		return;
	}
	AreaType firstArea = target.areas[0];
	switch (firstArea) {
	case AreaType::Me:
		output.clear();
		if (state.notMoved(effectCard)) {
			AddIfTarget(state, effectCard.card, target, output, effectCard);
		}
		return;
	case AreaType::Effected:
		if (&output == &state.targetList) {
			for (int i = (int)output.size() - 1; i >= 0; i--) {
				if (IsTarget(state, output[i].card, target, effectCard)) {
					output[i] = state.makeAreaRef(output[i].card);
				} else {
					output.erase(output.begin() + i);
				}
			}
		} else {
			output.clear();
			for (const AreaRef& ref : state.targetList) {
				AddIfTarget(state, ref.card, target, output, effectCard);
			}
		}
		break;
	case AreaType::EffectedPreTarget:
		output.clear();
		for (const AreaRef& ref : state.preTargetList) {
			AddIfTarget(state, ref.card, target, output, effectCard);
		}
		break;
	case AreaType::SelectedList:
		output.clear();
		for (CardRef ref : state.selectedList) {
			AddIfTarget(state, ref, target, output, effectCard);
		}
		break;
	case AreaType::TriggerSubject:
		output.clear();
		if (state.notMoved(state.triggerInfo.subject)) {
			AddIfTarget(state, state.triggerInfo.subject.card, target, output, effectCard);
		}
		return;
	case AreaType::TriggerObject:
		output.clear();
		if (state.notMoved(state.triggerInfo.object)) {
			AddIfTarget(state, state.triggerInfo.object.card, target, output, effectCard);
		}
		return;
	case AreaType::Attach: {
		output.clear();
		const Card& card = state.getCard(effectCard.card);
		RefPosition targetPos = state.attachedCardPosition(card);
		if (targetPos.area != AreaType::All) {
			AddIfTarget(state, targetPos.ref, target, output, effectCard);
		}
		return;
	}
	case AreaType::TurnPlay:
		output.clear();
		for (CardRef ref : state.turnPlay) {
			AddIfTarget(state, ref, target, output, effectCard);
		}
		break;
	case AreaType::AttackPreMyTurn: {
		output.clear();
		const Card& card = state.getCard(effectCard.card);
		int turnIndex = 2;
		if (state.activePlayerIndex() != card.playerIndex) {
			turnIndex = 1;
		}
		CardRef ref = state.turnHistories[turnIndex].turnAttackCard;
		if (!ref.isNull()) {
			AddIfTarget(state, ref, target, output, effectCard);
		}
		break;
	}
	default:
		output.clear();
		for (int playerIndex : state.basicPlayerOrder()) {
			if (!IsTargetPlayer(effectOwner, playerIndex, target.targetPlayer)) {
				continue;
			}
			const PlayerState& ps = state.players[playerIndex];
			for (AreaType areaType : target.areas) {
				switch (areaType)
				{
				case AreaType::Deck:
					for (CardRef ref : ps.deck) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Hand:
					if (target.skipEnemyTarget && playerIndex != state.getCard(effectCard.card).playerIndex) {
						for (CardRef ref : ps.hand) {
							output.push_back(state.makeAreaRef(ref));
						}
					} else {
						for (CardRef ref : ps.hand) {
							AddIfTarget(state, ref, target, output, effectCard);
						}
					}
					break;
				case AreaType::Trash:
					for (CardRef ref : ps.trash) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Active:
					for (CardRef ref : ps.active) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Bench:
					for (CardRef ref : ps.bench) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Prize:
					for (CardRef ref : ps.prize) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Stadium:
					for (CardRef ref : state.stadium) {
						if (state.getCard(ref).playerIndex == playerIndex) {
							AddIfTarget(state, ref, target, output, effectCard);
						}
					}
					break;
				case AreaType::Player:
					output.push_back(state.makeAreaRef(state.playerCardRef(playerIndex)));
					break;
				case AreaType::Energy:
					for (CardRef ref : ps.energy) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Tool:
					for (CardRef ref : ps.tool) {
						AddIfTarget(state, ref, target, output, effectCard);
					}
					break;
				case AreaType::Looking:
					for (CardRef ref : state.looking) {
						if (state.getCard(ref).playerIndex == playerIndex) {
							AddIfTarget(state, ref, target, output, effectCard);
						}
					}
					break;
				default:
					Exception("unexpected area target " + std::to_string((int)areaType));
					break;
				}
			}
		}
		break;
	}

	if (target.notMe) {
		for (int i = 0; i < output.size(); i++) {
			if (output[i].card == effectCard.card) {
				output.erase(output.begin() + i);
				break;
			}
		}
	}
}

