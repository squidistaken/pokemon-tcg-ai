"""Tests for Bulbapedia decklist identity parsing."""

from scraper.sources.bulbapedia import BulbapediaSource


def test_card_links_preserve_expansion_and_collection_number():
    html = """
    <table>
      <tr><th>Quantity</th><th>Card</th><th>Type</th></tr>
      <tr>
        <td>2×</td>
        <td><a href="/wiki/Charmeleon_(Obsidian_Flames_27)"
               title="Charmeleon (Obsidian Flames 27)">Charmeleon</a></td>
        <td>Pokémon</td>
      </tr>
      <tr>
        <td>4×</td>
        <td><a href="/wiki/Professor_Oak_(TCG)"
               title="Professor Oak (TCG)">Professor Oak</a></td>
        <td>Trainer</td>
      </tr>
    </table>
    """

    decks = BulbapediaSource._parse_decklist_tables(html)  # noqa: SLF001

    assert len(decks) == 1
    assert (decks[0][0].name, decks[0][0].set_code, decks[0][0].number) == (
        "Charmeleon",
        "Obsidian Flames",
        "27",
    )
    assert (decks[0][1].set_code, decks[0][1].number) == (None, None)
