def load_deck(path: str) -> list[int]:
    """
    Load a 60-card deck from a CSV file with one card ID per line.

    :param path: Path to the deck CSV file.
    :return: List of 60 card IDs.
    """
    with open(path, "r") as file:
        lines = [line for line in file.read().split("\n") if line.strip()]

    num_valid_cards = 60
    if len(lines) != num_valid_cards:
        raise ValueError(
            f"Deck must contain exactly 60 cards, found {len(lines)}"
        )

    return [int(line) for line in lines]
