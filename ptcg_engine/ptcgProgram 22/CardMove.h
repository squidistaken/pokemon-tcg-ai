// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "SetProperty.h"
#include "PullTrigger.h"

inline void RefreshEffect(State& state, int depth = 0);
inline CardRef MoveCardAttach(State& state, int playerIndex, AreaType fromArea, int fromIndex, AreaType toArea, bool ko);

// 移動先のmaxチェックは呼び出し元で
template<typename FromList, typename ToList>
inline CardRef MoveListCard(FromList& from, ToList& to, int areaIndex) {
	CardRef card = from.at(areaIndex);
	from.remove(areaIndex);
	to.push_back(card);
	return card;
}

template<typename FromList, typename ToList>
inline CardRef MoveListCardFront(FromList& from, ToList& to, int areaIndex) {
	CardRef card = from.at(areaIndex);
	from.remove(areaIndex);
	to.push_front(card);
	return card;
}

template<typename FromList, typename ToList>
inline std::pair<CardRef, int> MoveCardFromLast(FromList& from, ToList& to) {
	int index = (int)from.size() - 1;
	CardRef ref = MoveListCard(from, to, index);
	return { ref, index };
}

inline void AppearProc(PlayerState& ps, Card& card) {
	if (!card.noEffectEnemyAttack) {
		if (ps.nextTurn.cannotAttackLessEqualEnergy2) {
			card.nextTurn.cannotAttackLessEqualEnergy2 = true;
		} else if (ps.thisTurn.cannotAttackLessEqualEnergy2) {
			card.thisTurn.cannotAttackLessEqualEnergy2 = true;
		}
	}
}

// openTypeが1なら相手には見えない、2なら両方見えない、3ならプレイヤー0だけ見える、4ならプレイヤー1だけ見える
inline CardRef MoveCard(State& state, int playerIndex, AreaType fromArea, int fromIndex, AreaType toArea, int openType = 0, bool noLog = false, bool withAttach = false, bool mustMove = false, bool ko = false) {
	assert(fromArea != toArea);

	PlayerState& ps = state.players.at(playerIndex);
	CardRef ref = state.getCardRef(fromArea, fromIndex, playerIndex);
	if (fromArea == AreaType::Trash) {
		if (toArea == AreaType::Hand || toArea == AreaType::Deck || toArea == AreaType::DeckBottom) {
			const Card& card = state.getCard(ref);
			if (card.getMaster().cannotToHandOrDeckInTrash) {
				return ref;
			}
		}
		if (toArea == AreaType::Hand) {
			if (ps.cannotTrashToHandAbilityOrTrainers) {
				if (!state.onAttack()) {
					return ref;
				}
			}
		}
	}

	if (!mustMove) {
		if (toArea == AreaType::Hand) {
			if (state.cannotToHand(ref)) {
				if (ko) {
					toArea = AreaType::Trash;
				} else {
					return ref;
				}
			}
		}
		if (toArea == AreaType::Bench) {
			const Card& card = state.getCard(ref);
			if (fromArea == AreaType::Hand || (fromArea == AreaType::Looking && card.preArea == AreaType::Hand)) {
				if (ps.cannotPlayAbilityPokemonNotRocket) {
					const CardMaster& master = card.getMaster();
					if (state.getAbility(card, master) != nullptr && !master.teamRocket) {
						return ref;
					}
				}
			}
		}
	}

	if (toArea == AreaType::Stadium) {
		if (state.stadium.size() > 0) {
			if (fromArea != AreaType::Stadium) {
				const Card& card = state.getCard(state.stadium[0]);
				MoveCard(state, card.playerIndex, AreaType::Stadium, 0, AreaType::Trash);
			}
		}
	}

	AreaType logArea = fromArea;
	if (logArea == AreaType::Temporary) {
		const Card& card = state.getCard(ref);
		logArea = card.area;
	}

	ref = state.removeCardRef(fromArea, fromIndex, playerIndex);
	state.pushCardRef(toArea, playerIndex, ref);

	int moveCounter = state.getCard(ref).moveCounter;
	AreaType area = toArea;
	if (area == AreaType::DeckBottom) {
		area = AreaType::Deck;
	}
	state.cardMoved(ref, area);
	if (!noLog) {
		if (toArea != AreaType::PreEvolution) {
			LogMoveCard(state, playerIndex, ref, logArea, toArea, openType);
		}
	}

	if (fromArea == AreaType::Stadium) {
		state.lastStadiumPlayer = playerIndex;
	} else if (fromArea == AreaType::Prize) {
		if (toArea == AreaType::Hand || toArea == AreaType::Bench) {
			const Card& card = state.getCard(ref);
			if (state.isPlayerTurn() && state.activePlayerIndex() == card.playerIndex) {
				state.thisTurnHistory().takePrizeCountTurnPlayer += 1;
			}
		}
	}

	if ((fromArea == AreaType::Active && toArea != AreaType::Bench)
		|| (fromArea == AreaType::Bench && toArea != AreaType::Active)) {
		for (int i = ps.preEvolution.size() - 1; i >= 0; i--) {
			CardRef ref = ps.preEvolution[i];
			if (state.getCard(ref).attachMoveCounter == moveCounter) {
				if (withAttach) {
					MoveCardAttach(state, playerIndex, AreaType::PreEvolution, i, toArea, ko);
				} else {
					if (toArea == AreaType::Hand) {
						MoveCardAttach(state, playerIndex, AreaType::PreEvolution, i, AreaType::Hand, ko);
					} else {
						MoveCardAttach(state, playerIndex, AreaType::PreEvolution, i, AreaType::Trash, ko);
					}
				}
			}
		}
		for (int i = ps.energy.size() - 1; i >= 0; i--) {
			CardRef ref = ps.energy[i];
			if (state.getCard(ref).attachMoveCounter == moveCounter) {
				if (withAttach) {
					MoveCardAttach(state, playerIndex, AreaType::Energy, i, toArea, ko);
				} else {
					MoveCardAttach(state, playerIndex, AreaType::Energy, i, AreaType::Trash, ko);
				}
			}
		}
		for (int i = ps.tool.size() - 1; i >= 0; i--) {
			CardRef ref = ps.tool[i];
			if (state.getCard(ref).attachMoveCounter == moveCounter) {
				if (withAttach) {
					MoveCardAttach(state, playerIndex, AreaType::Tool, i, toArea, ko);
				} else {
					MoveCardAttach(state, playerIndex, AreaType::Tool, i, AreaType::Trash, ko);
				}
			}
		}
	}

	if (toArea == AreaType::Active || toArea == AreaType::Bench) {
		RefreshEffect(state);
		if (fromArea != AreaType::Active && fromArea != AreaType::Bench) {
			Card& card = state.getCard(ref);
			AppearProc(ps, card);
		}
	}

	if (toArea == AreaType::Bench) {
		const Card& card = state.getCard(ref);
		if (state.activePlayerIndex() == card.playerIndex && fromArea != AreaType::Active) {
			PullTrigger(state, TriggerType::ToBenchMyTurn, ref);
		}
		if (fromArea == AreaType::Hand) {
			PullTrigger(state, TriggerType::HandToBench, ref);
		} else if (fromArea == AreaType::Active) {
			PullTrigger(state, TriggerType::ActiveToBench, ref);
		}
	} else if (toArea == AreaType::Active) {
		if (fromArea == AreaType::Bench) {
			if (state.phase == GamePhase::Main) {
				Card& card = state.getCard(ref);
				card.benchToActive = true;
			}
			PullTrigger(state, TriggerType::BenchToActive, ref);
		}
	} else if (toArea == AreaType::Trash) {
		if (fromArea == AreaType::Deck) {
			if (state.onEffect()) {
				const Card& effectCard = state.getCard(state.getEffectCard().card);
				CardType type = effectCard.getMaster().cardType;
				if (type == CardType::Pokemon || type == CardType::Item || type == CardType::Supporter) {
					const Card& card = state.getCard(ref);
					if (effectCard.playerIndex != card.playerIndex) {
						PullTrigger(state, TriggerType::DeckToTrashEnemyEffect, ref);
					}
				}
			}
		}
	}

	return ref;
}

inline CardRef MoveCardAttach(State& state, int playerIndex, AreaType fromArea, int fromIndex, AreaType toArea, bool ko) {
	return MoveCard(state, playerIndex, fromArea, fromIndex, toArea, 0, false, false, true, ko);
}

inline CardRef MoveCardBase(State& state, int playerIndex, AreaType fromArea, int fromIndex, AreaType toArea) {
	return MoveCard(state, playerIndex, fromArea, fromIndex, toArea);
}

inline void MoveRefCard(State& state, CardRef ref, const Card& card, AreaType toArea, int openType = 0, bool noLog = false, bool withAttach = false) {
	int index = state.currentAreaIndex(ref, card);
	MoveCard(state, card.playerIndex, card.area, index, toArea, openType, noLog, withAttach);
}

inline void SwitchPokemon(State& state, int playerIndex, int benchIndex) {
	PlayerState& ps = state.players.at(playerIndex);
	CardRef activeRef;
	if (ps.active.empty()) {
		activeRef = MoveCard(state, playerIndex, AreaType::Bench, benchIndex, AreaType::Active);
		RefreshEffect(state);
	} else {
		CardRef& ref1 = ps.active.at(0);
		CardRef& ref2 = ps.bench.at(benchIndex);
		LogSwitch(state, playerIndex, ref1, ref2);
		ClearSpecialCondition(state, playerIndex);
		std::swap(ref1, ref2);
		state.cardMoved(ref1, AreaType::Active);
		state.cardMoved(ref2, AreaType::Bench);
		activeRef = ref1;

		if (state.phase == GamePhase::Main) {
			Card& card = state.getCard(activeRef);
			card.benchToActive = true;
		}

		RefreshEffect(state);
		PullTrigger(state, TriggerType::ActiveToBench, ref2);
	}
	PullTrigger(state, TriggerType::BenchToActive, activeRef);
}

// デッキシャッフル
inline void ShuffleDeck(State& state, int playerIndex, bool noLog = false) {
	PlayerState& ps = state.players.at(playerIndex);
	if (ps.deck.size() > 0) {
		state.changed = true;
		if (state.game->config.deviceRand) {
			std::shuffle(ps.deck.begin(), ps.deck.end(), std::random_device());
		} else {
			std::shuffle(ps.deck.begin(), ps.deck.end(), state.game->rng);
		}
		if (!noLog) {
			LogShuffle(state, playerIndex);
		}
	}
}

// 手札を公開しながら全て戻してシャッフル
inline void OpenReturnAndShuffle(State& state, int playerIndex) {
	PlayerState& ps = state.players.at(playerIndex);
	while (ps.hand.size() > 0) {
		auto [ref, index] = MoveCardFromLast(ps.hand, ps.deck);
		LogMoveCard(state, playerIndex, ref, AreaType::Hand, AreaType::Deck, 0);
		state.cardMoved(ref, AreaType::Deck);
	}
	ShuffleDeck(state, playerIndex);
}


// カードを引く
inline void Draw(State& state, int playerIndex, int count) {
	PlayerState& ps = state.players.at(playerIndex);
	for (int i = 0; i < count; i++) {
		if (ps.deck.size() > 0) {
			state.changed = true;
			auto [ref, _] = MoveCardFromLast(ps.deck, ps.hand);
			LogDraw(state, playerIndex, ref);
			state.cardMoved(ref, AreaType::Hand);
		}
	}
}

inline void DeckToPrize(State& state, int playerIndex, int count) {
	PlayerState& ps = state.players[playerIndex];
	for (int _ : range(count)) {
		if (ps.deck.empty()) {
			break;
		}
		auto [ref, index] = MoveCardFromLast(ps.deck, ps.prize);
		state.cardMoved(ref, AreaType::Prize, true);
		LogMoveCard(state, playerIndex, ref, AreaType::Deck, AreaType::Prize, 2);
	}
}

inline void EvolveProc(State& state, AreaType area, int index, AreaType inPlayArea, int inPlayIndex, int playerIndex) {
	PlayerState& ps = state.players[playerIndex];
	CardRef& inPlayRef = (inPlayArea == AreaType::Active ? ps.active.at(0) : ps.bench.at(inPlayIndex));
	Card& preEvolveCard = state.getCard(inPlayRef);
	if (preEvolveCard.area != AreaType::Active && preEvolveCard.area != AreaType::Bench) {
		return;
	}
	if (ps.thisTurn.cannotEvolve && area == AreaType::Hand) {
		return;
	}
	if (ps.cannotPlayAbilityPokemonNotRocket && area == AreaType::Hand) {
		CardRef r = state.getCardRef(area, index, playerIndex);
		const CardMaster& master = state.getCard(r).getMaster();
		if (master.ability != nullptr && !master.teamRocket) {
			return;
		}
	}
	bool afterConfuse = state.afterConfuse(preEvolveCard);

	CardRef ref = state.removeCardRef(area, index, playerIndex);
	Card& card = state.getCard(ref);
	LogEvolve(state, playerIndex, ref, inPlayRef);

	EvolveInfo ei = {};
	ei.preRef = inPlayRef;
	ei.ref = ref;
	state.turnEvolve.push_back(ei);

	int moveCounter = preEvolveCard.moveCounter;
	int damage = preEvolveCard.damage;
	ps.preEvolution.push_back(inPlayRef);
	state.cardMoved(inPlayRef, AreaType::PreEvolution);
	preEvolveCard.attachMoveCounter = moveCounter;
	inPlayRef = ref;
	state.cardMoved(ref, inPlayArea);
	card.moveCounter = moveCounter;
	card.damage = damage;


	if (inPlayArea == AreaType::Active) {
		ClearSpecialCondition(state, playerIndex);
		ps.activeState = 0;
		if (afterConfuse) {
			ps.badStatus = BadStatusType::Confused;
		}
	}

	RefreshEffect(state);
	if (area == AreaType::Hand) {
		PullTrigger(state, TriggerType::EvolveFromHand, ref);
	}
	AppearProc(ps, card);
}

inline void DevolveProc(State& state, CardRef ref, Card& card, AreaType toArea) {
	PlayerState& ps = state.players.at(card.playerIndex);
	for (int i = ps.preEvolution.size() - 1; i >= 0; i--) {
		CardRef preRef = ps.preEvolution[i];
		Card& preCard = state.getCard(preRef);
		if (preCard.attachMoveCounter == card.moveCounter) {
			bool afterConfuse = state.afterConfuse(card);
			AreaType area = card.area;
			int areaIndex = state.currentAreaIndex(ref, card);
			int moveCounter = card.moveCounter;
			CardRef& currentRef = (area == AreaType::Active ? ps.active.at(0) : ps.bench.at(areaIndex));
			int damage = card.damage;
			if (toArea == AreaType::Hand) {
				if (state.cannotToHand(currentRef)) {
					continue;
				}
			}

			LogDevolve(state, preCard.playerIndex, preRef, ref);

			ps.preEvolution.remove(i);
			if (toArea == AreaType::Hand) {
				ps.hand.push_back(currentRef);
				state.cardMoved(currentRef, AreaType::Hand);
			} else {
				ps.deck.push_back(currentRef);
				state.cardMoved(currentRef, AreaType::Deck);
			}
			currentRef = preRef;
			state.cardMoved(preRef, area);
			preCard.moveCounter = moveCounter;
			preCard.damage = damage;

			state.changed = true;
			if (area == AreaType::Active) {
				ClearSpecialCondition(state, card.playerIndex);
				ps.activeState = 0;
				if (afterConfuse) {
					ps.badStatus = BadStatusType::Confused;
				}
			}

			RefreshEffect(state);
			AppearProc(ps, preCard);
			break;
		}
	}
}

inline void AttachProc(State& state, AreaType area, int index, int playerIndex, CardRef targetRef, bool isEffect) {
	CardRef ref = state.getCardRef(area, index, playerIndex);
	Card& card = state.getCard(ref);
	const CardMaster& master = card.getMaster();

	const Card& targetCard = state.getCard(targetRef);
	if (targetCard.area != AreaType::Active && targetCard.area != AreaType::Bench) {
		return;
	}
	if (state.players[playerIndex].thisTurn.cannotPlaySpecialEnergy) {
		if (card.area == AreaType::Hand && master.cardType == CardType::SpecialEnergy) {
			return;
		}
	}
	if (targetCard.thisTurn.cannotHandAttachEnergy) {
		if (card.area == AreaType::Hand && IsEnergy(master.cardType)) {
			MoveCard(state, playerIndex, area, index, AreaType::Trash, 0, false);
			return;
		}
	}
	if (IsEnergy(master.cardType) && !state.canAttachEnergy(card, master, targetRef)) {
		return;
	}
	LogAttach(state, playerIndex, ref, targetRef);

	if (master.cardType == CardType::Tool) {
		MoveCard(state, playerIndex, area, index, AreaType::Tool, 0, true);
	} else {
		if (!isEffect) {
			state.energyPlayed = true;
		}
		MoveCard(state, playerIndex, area, index, AreaType::Energy, 0, true);
	}
	card.attachMoveCounter = targetCard.moveCounter;

	if (area == AreaType::Hand) {
		if (IsEnergy(master.cardType)) {
			PullTrigger(state, TriggerType::EnergyAttachFromHand, targetRef);
			for (int i = (int)state.delayTriggerStack.size() - 1; i >= 0; i--) {
				TriggeredAbility ta = state.delayTriggerStack[i];
				if (ta.trigger.type == TriggerType::EnergyAttachFromHand && ta.trigger.subject.card == targetRef) {
					state.temporaryTriggerStack.push_back(ta);
				}
			}
		}
		if (master.play != nullptr) {
			if (master.cardType == CardType::Tool && state.noToolEffect) {
				return;
			}
			if (master.play->attachBench && targetCard.area != AreaType::Bench) {
				return;
			}
			TriggeredAbility ta = {};
			ta.trigger.type = TriggerType::Attach;
			ta.trigger.subject = state.makeAreaRef(targetRef);
			ta.activateInfo = ActivateAbilityInfo(master.play->skillId, state.makeAreaRef(ref), card.playerIndex);
			state.temporaryTriggerStack.push_back(ta);
		}
	}
}

inline void SwitchEnergyProc(State& state, CardRef energyRef, CardRef pokemonRef) {
	Card& energyCard = state.getCard(energyRef);
	const Card& pokemonCard = state.getCard(pokemonRef);

	RefPosition rp = state.attachedCardPosition(energyCard);
	LogMoveAttached(state, pokemonCard.playerIndex, energyRef, rp.ref, pokemonRef);

	energyCard.attachMoveCounter = pokemonCard.moveCounter;
}
