# pipeline/bird_names.py
"""Assign cute unique names to identified birds."""

# Name pools themed by species
JUNCO_NAMES = [
    "Pepper", "Slate", "Storm", "Ember", "Ash", "Flint", "Shadow", "Dusty",
    "Pebble", "Smoky", "Charcoal", "Granite", "Onyx", "Graphite", "Cinder",
    "Obsidian", "Basalt", "Cobalt", "Dusk", "Twilight", "Mercury", "Nimbus",
    "Thunder", "Fossil", "Pumice", "Shale", "Marble", "Jet", "Raven", "Gunmetal",
]

CHICKADEE_NAMES = [
    "Pip", "Chickpea", "Bean", "Twig", "Acorn", "Hazel", "Clover", "Nutmeg",
    "Thistle", "Wren", "Cricket", "Sparky", "Biscuit", "Mochi", "Poppy",
    "Maple", "Birch", "Almond", "Cashew", "Walnut", "Pecan", "Pistachio",
    "Juniper", "Sage", "Rosemary", "Basil", "Thyme", "Ginger", "Cinnamon",
    "Clove",
]

FINCH_NAMES = [
    "Ruby", "Scarlet", "Blaze", "Coral", "Rusty", "Copper", "Garnet",
    "Crimson", "Sienna", "Auburn", "Brick", "Cayenne", "Paprika", "Salsa",
    "Cherry",
]

SPARROW_NAMES = [
    "Goldie", "Sunny", "Marigold", "Honey", "Buttercup", "Dandelion",
    "Saffron", "Amber", "Topaz", "Citrine", "Canary", "Lemon", "Bumblebee",
    "Sunflower", "Dijon",
]

# Fallback names for any species
FALLBACK_NAMES = [
    "Feathers", "Chirpy", "Tweety", "Birdie", "Scout", "Skipper", "Patches",
    "Ziggy", "Doodle", "Noodle", "Pickle", "Sprout", "Button", "Widget",
    "Pixel",
]

SPECIES_NAME_MAP = {
    "Dark-Eyed Junco": JUNCO_NAMES,
    "Black-Capped Chickadee": CHICKADEE_NAMES,
    "Chickadee": CHICKADEE_NAMES,
    "House Finch": FINCH_NAMES,
    "Golden-Crowned Sparrow": SPARROW_NAMES,
    "House Sparrow": SPARROW_NAMES,
    "Song Sparrow": SPARROW_NAMES,
    "Chestnut-backed Chickadee": CHICKADEE_NAMES,
}


def assign_names(bird_ids: list[str], species_map: dict[str, str]) -> dict[str, str]:
    """Assign cute names to bird IDs.

    Args:
        bird_ids: list of bird_id strings like "Dark-Eyed Junco_bird_3"
        species_map: dict mapping bird_id -> species

    Returns:
        dict mapping bird_id -> cute name
    """
    name_assignments = {}
    species_counters = {}

    for bird_id in sorted(bird_ids):
        if "noise" in bird_id:
            name_assignments[bird_id] = "Unknown Visitor"
            continue

        species = species_map.get(bird_id, "")
        pool = SPECIES_NAME_MAP.get(species, FALLBACK_NAMES)

        idx = species_counters.get(species, 0)
        species_counters[species] = idx + 1

        if idx < len(pool):
            name = pool[idx]
        else:
            name = f"{pool[idx % len(pool)]}-{idx // len(pool) + 1}"

        name_assignments[bird_id] = name

    return name_assignments
