from pathlib import Path

# Alle csv's en png's staan in de map "data" naast de scripts. Het pad wordt berekend vanaf de
# locatie van dit bestand, dus het werkt op elke laptop en vanuit elke werkmap (geen vaste paden).
# Wil je een andere mapnaam, pas dan enkel DATA_MAP aan.
DATA_MAP = "data"

BASIS = Path(__file__).resolve().parent
DATA = BASIS / DATA_MAP
DATA.mkdir(exist_ok=True)

def pad(bestandsnaam):
    """Geeft het volledige pad van een bestand in de data-map terug (als string)."""
    return str(DATA / bestandsnaam)