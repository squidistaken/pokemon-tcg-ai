import torch
from tensordict import TensorDict

from cg.api import all_attack, all_card_data


class CardDatabase:
    """
    Static, card-ID-indexed lookup tables built from the engine's card list.

    Complements the per-state observation (``docs/torchrl_environment.md``):
    observations carry raw card/attack IDs, and these tables map an ID to
    the card's static properties. They are meant for model-side use, e.g.
    initializing card embeddings from real card features instead of learning
    identity from scratch, or concatenating ``card_features[ids]`` onto
    embedded IDs.

    All tables have ``max_id + 1`` rows so a card/attack ID indexes its own
    row directly; row 0 is all zeros and corresponds to the observation
    convention that ID 0 means "none / padding / unknown". Categorical
    columns store ``enum value + 1`` with 0 for "absent", matching the
    observation encoding.
    """

    CARD_FEATURE_COUNT = 12
    CARD_CATEGORICAL_COUNT = 4
    ATTACK_FEATURE_COUNT = 14
    ENERGY_TYPE_COUNT = 12
    MAX_ATTACKS_PER_CARD = 2

    def __init__(self) -> None:
        """
        Load the card and attack lists from the engine and build the tables.
        """
        cards = all_card_data()
        attacks = all_attack()
        self._card_names: dict[int, str] = {card.cardId: card.name for card in cards}
        self._attack_names: dict[int, str] = {
            attack.attackId: attack.name for attack in attacks
        }

        n_card_rows = max(card.cardId for card in cards) + 1
        self._card_features = torch.zeros(
            n_card_rows, self.CARD_FEATURE_COUNT, dtype=torch.float32
        )
        self._card_cats = torch.zeros(
            n_card_rows, self.CARD_CATEGORICAL_COUNT, dtype=torch.int64
        )
        self._card_attack_ids = torch.zeros(
            n_card_rows, self.MAX_ATTACKS_PER_CARD, dtype=torch.int64
        )
        for card in cards:
            row = card.cardId
            self._card_features[row] = torch.tensor(
                [
                    float(card.hp),
                    float(card.retreatCost),
                    float(card.basic),
                    float(card.stage1),
                    float(card.stage2),
                    float(card.ex),
                    float(card.megaEx),
                    float(card.tera),
                    float(card.aceSpec),
                    float(len(card.skills)),
                    float(len(card.attacks)),
                    1.0,
                ],
                dtype=torch.float32,
            )
            self._card_cats[row] = torch.tensor(
                [
                    int(card.cardType) + 1,
                    int(card.energyType) + 1,
                    int(card.weakness) + 1 if card.weakness is not None else 0,
                    int(card.resistance) + 1 if card.resistance is not None else 0,
                ],
                dtype=torch.int64,
            )
            for column, attack_id in enumerate(
                card.attacks[: self.MAX_ATTACKS_PER_CARD]
            ):
                self._card_attack_ids[row, column] = attack_id

        n_attack_rows = max(attack.attackId for attack in attacks) + 1
        self._attack_features = torch.zeros(
            n_attack_rows, self.ATTACK_FEATURE_COUNT, dtype=torch.float32
        )
        for attack in attacks:
            row = attack.attackId
            cost_counts = [0.0] * self.ENERGY_TYPE_COUNT
            for energy in attack.energies:
                energy_index = int(energy)
                if 0 <= energy_index < self.ENERGY_TYPE_COUNT:
                    cost_counts[energy_index] += 1.0
            self._attack_features[row] = torch.tensor(
                [float(attack.damage), float(len(attack.energies))] + cost_counts,
                dtype=torch.float32,
            )

    @property
    def card_features(self) -> torch.Tensor:
        """
        Float features per card ID.

        Columns: hp, retreat cost, basic, stage1, stage2, ex, mega ex, tera,
        ace spec, skill count, attack count, exists flag (0 on unused rows).

        :return: Float32 tensor of shape ``(max_card_id + 1, CARD_FEATURE_COUNT)``.
        """
        return self._card_features

    @property
    def card_cats(self) -> torch.Tensor:
        """
        Categorical features per card ID, shifted so 0 means "absent".

        Columns: card type, energy type, weakness, resistance.

        :return: Int64 tensor of shape ``(max_card_id + 1, CARD_CATEGORICAL_COUNT)``.
        """
        return self._card_cats

    @property
    def card_attack_ids(self) -> torch.Tensor:
        """
        Attack IDs per card ID, zero-padded.

        :return: Int64 tensor of shape ``(max_card_id + 1, MAX_ATTACKS_PER_CARD)``.
        """
        return self._card_attack_ids

    @property
    def attack_features(self) -> torch.Tensor:
        """
        Float features per attack ID.

        Columns: damage, total energy cost, then the cost count per
        ``EnergyType`` (12 columns).

        :return: Float32 tensor of shape ``(max_attack_id + 1, ATTACK_FEATURE_COUNT)``.
        """
        return self._attack_features

    def card_name(self, card_id: int) -> str:
        """
        Human-readable card name, for debugging and inspection.

        :param card_id: Engine card ID.
        :return: The card's name, or "<none>" for ID 0 / unknown IDs.
        """
        return self._card_names.get(card_id, "<none>")

    def attack_name(self, attack_id: int) -> str:
        """
        Human-readable attack name, for debugging and inspection.

        :param attack_id: Engine attack ID.
        :return: The attack's name, or "<none>" for ID 0 / unknown IDs.
        """
        return self._attack_names.get(attack_id, "<none>")

    def as_tensordict(self) -> TensorDict:
        """
        Bundle all tables into a single TensorDict (e.g. for saving to disk).

        :return: TensorDict with the four lookup tables, batch size ``()``.
        """
        return TensorDict(
            {
                "card_features": self._card_features,
                "card_cats": self._card_cats,
                "card_attack_ids": self._card_attack_ids,
                "attack_features": self._attack_features,
            },
            batch_size=torch.Size(()),
        )
