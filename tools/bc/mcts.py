"""
PUCT search over the engine's determinized simulator, in the AlphaZero shape.

The earlier one-ply attempts discarded the policy and ranked successors by a
single noisy number, which cannot beat the policy it replaced. Here the policy
is the prior inside the selection rule, so with few simulations the search
reproduces the policy's own ranking and only departs from it where the backed
up values agree. Values are always from the searching seat, so our nodes
maximize and the opponent's nodes minimize.
"""
import argparse
import math
import random
import sys

import torch
from tensordict import TensorDict

from cg import api
from cg.api import Observation
from src.env.battle_handle import BattleHandle
from src.env.decks.deck import load_deck
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent

sys.path.insert(0, "/tmp/claude-1000/-home-matthijs-programming-pokemon-tcg-ai/1907c66a-6c97-40f1-9fd2-7a220e78b811/scratchpad")
from lookahead import build_network, determinize  # noqa: E402

MAX_OPTIONS = 128


class Node:
    """
    One search node: a forked engine state plus its visit statistics.
    """

    __slots__ = ("search_id", "observation", "seat_to_move", "prior", "children",
                 "visits", "value_sum", "terminal_value", "n_options")

    def __init__(self, search_id: int, observation: Observation, our_seat: int):
        """
        :param search_id: Engine search-state ID this node wraps.
        :param observation: Observation at this node.
        :param our_seat: The seat the search is run for.
        """
        self.search_id = search_id
        self.observation = observation
        state = observation.current
        self.seat_to_move = state.yourIndex if state else our_seat
        self.prior: list[float] = []
        self.children: dict[int, Node | None] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.terminal_value: float | None = None
        select = observation.select
        self.n_options = len(select.option) if select else 0

    @property
    def mean_value(self) -> float:
        """
        :return: Mean backed-up value, or 0 for an unvisited node.
        """
        return self.value_sum / self.visits if self.visits else 0.0


class MCTSPolicy:
    """
    Choose an action by PUCT search, falling back to the raw policy on error.

    :param network: Trained actor-critic supplying both prior and value.
    :param encoder: Observation encoder matching training.
    :param own_list: Our decklist, for determinization.
    :param opponent_list: Guessed opponent decklist.
    :param simulations: PUCT simulations per decision.
    :param c_puct: Exploration constant.
    :param seed: Seed for determinization.
    """

    def __init__(self, network, encoder, own_list, opponent_list,
                 simulations: int = 64, c_puct: float = 1.5, seed: int = 0):
        self._network = network
        self._encoder = encoder
        self._own = own_list
        self._opponent = opponent_list
        self._simulations = simulations
        self._c_puct = c_puct
        self._rng = random.Random(seed)
        self._fallback = GreedyPolicyOpponent(network, encoder)
        self.searched = 0
        self.fell_back = 0

    @torch.inference_mode()
    def _evaluate(self, observation: Observation, our_seat: int, n_options: int):
        """
        Run the network once for a node's prior and value.

        The option table belongs to whoever is about to act, so the node is
        encoded from *that* seat: scoring the opponent's options through our
        view of the board models them playing a game they cannot see. The
        value that comes back is therefore in the mover's frame, and is
        negated into ours when the mover is the opponent.

        :param observation: State at the node.
        :param our_seat: Seat the search is run for.
        :param n_options: Legal option count at this node.
        :return: ``(prior over options, value from our seat)``.
        """
        state = observation.current
        mover = state.yourIndex if state else our_seat
        encoded = TensorDict(
            {"observation": self._encoder.encode(observation, mover, 0)},
            batch_size=torch.Size(()),
        )
        self._network(encoded)
        logits = encoded.get("logits").reshape(-1)[:n_options]
        value = float(encoded.get("state_value").reshape(-1)[0])
        prior = torch.softmax(logits.float(), dim=-1).tolist()
        return prior, value if mover == our_seat else -value

    def _terminal_value(self, observation: Observation, our_seat: int):
        """
        :param observation: State to check.
        :param our_seat: Seat the search is run for.
        :return: 1/0/0.5 if the battle ended, else None.
        """
        state = observation.current
        if state is None or state.result == -1:
            return None
        if state.result == our_seat:
            return 1.0
        if state.result == 1 - our_seat:
            return -1.0
        return 0.0

    def _select_child(self, node: Node) -> int:
        """
        PUCT selection, minimizing our value at the opponent's nodes.

        :param node: Node to select from.
        :return: Option index to descend into.
        """
        total = math.sqrt(max(node.visits, 1))
        best, best_score = 0, -float("inf")
        maximizing = node.seat_to_move == self._our_seat
        for option in range(node.n_options):
            child = node.children.get(option)
            visits = child.visits if child else 0
            q = child.mean_value if child and child.visits else 0.0
            if not maximizing:
                q = -q
            prior = node.prior[option] if option < len(node.prior) else 1e-6
            score = q + self._c_puct * prior * total / (1 + visits)
            if score > best_score:
                best, best_score = option, score
        return best

    def _simulate(self, root: Node) -> None:
        """
        Run one PUCT simulation from the root and back the value up.

        :param root: Root node of the tree.
        """
        path = [root]
        node = root
        while True:
            if node.terminal_value is not None or node.n_options == 0:
                break
            option = self._select_child(node)
            child = node.children.get(option)
            if child is None:
                try:
                    stepped = api.search_step(node.search_id, [option])
                except Exception:
                    node.children[option] = None
                    break
                child = Node(stepped.searchId, stepped.observation, self._our_seat)
                child.terminal_value = self._terminal_value(
                    stepped.observation, self._our_seat
                )
                if child.terminal_value is None and child.n_options:
                    child.prior, value = self._evaluate(
                        stepped.observation, self._our_seat, child.n_options
                    )
                else:
                    value = child.terminal_value if child.terminal_value is not None else 0.0
                node.children[option] = child
                path.append(child)
                self._backup(path, value)
                return
            node = child
            path.append(node)
        value = node.terminal_value if node.terminal_value is not None else node.mean_value
        self._backup(path, value)

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        """
        Add one simulation's value to every node on the path.

        :param path: Root-to-leaf node path.
        :param value: Value from the searching seat.
        """
        for node in path:
            node.visits += 1
            node.value_sum += value

    def __call__(self, observation: Observation) -> list[int]:
        """
        Pick a selection by PUCT search, or fall back to the raw policy.

        :param observation: Live agent observation.
        :return: Option indices to submit.
        """
        select = observation.select
        state = observation.current
        if (select is None or state is None
                or observation.search_begin_input is None
                or len(select.option) < 2
                or select.minCount != 1 or select.maxCount != 1):
            self.fell_back += 1
            return self._fallback(observation)
        self._our_seat = state.yourIndex
        try:
            begin = api.search_begin(
                observation,
                **determinize(observation, self._our_seat, self._own,
                              self._opponent, self._rng),
            )
        except Exception:
            self.fell_back += 1
            return self._fallback(observation)
        try:
            root = Node(begin.searchId, begin.observation, self._our_seat)
            root.prior, _ = self._evaluate(
                begin.observation, self._our_seat, root.n_options
            )
            for _ in range(self._simulations):
                self._simulate(root)
            visited = {k: v.visits for k, v in root.children.items() if v}
        finally:
            try:
                api.search_end()
            except Exception:
                pass
        if not visited:
            self.fell_back += 1
            return self._fallback(observation)
        self.searched += 1
        return [max(visited, key=lambda k: visited[k])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/deck-pinned-150m-local/tf-ptr-pinned-selfplay-10m-s42/"
        "checkpoints/snapshot_000175702016.pt")
    parser.add_argument(
        "--deck", default="decks/top20/alakazam-dudunsparce/alakazam-dudunsparce-4.csv")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    args = parser.parse_args()

    network, encoder = build_network(args.checkpoint)
    deck = load_deck(args.deck)
    searcher = MCTSPolicy(network, encoder, deck, deck,
                          simulations=args.simulations, seed=5)
    raw = GreedyPolicyOpponent(network, encoder)

    wins = losses = draws = 0
    rng = random.Random(11)
    for game in range(args.games):
        our_seat = rng.randint(0, 1)
        handle = BattleHandle()
        try:
            observation = handle.start(deck, deck)
            for _ in range(2000):
                state = observation.current
                if state is None or state.result != -1 or observation.select is None:
                    break
                actor = searcher if state.yourIndex == our_seat else raw
                observation = handle.select(actor(observation))
            result = observation.current.result if observation.current else -1
        finally:
            handle.finish()
        wins += result == our_seat
        losses += result == 1 - our_seat
        draws += result not in (our_seat, 1 - our_seat)
        print(f"  game {game + 1}: W{wins} L{losses} D{draws} "
              f"(searched {searcher.searched}, fell back {searcher.fell_back})",
              flush=True)

    scored = wins + losses + draws
    print(f"\nPUCT search ({args.simulations} sims) vs raw argmax, same weights")
    print(f"  games : {scored}")
    print(f"  score : {(wins + 0.5 * draws) / max(scored, 1):.3f} (0.50 = no gain)")
    print(f"  W/L/D : {wins}/{losses}/{draws}")


if __name__ == "__main__":
    main()
