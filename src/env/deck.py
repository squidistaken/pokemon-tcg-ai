def load_deck(path: str) -> list[int]:
    """
    Load a 60-card deck from a CSV file with one card ID per line.

    :param path: Path to the deck CSV file.
    :return: List of 60 card IDs.
    """
    with open(path, "r") as file:
        lines = file.read().split("\n")
    return [int(lines[i]) for i in range(60)]
