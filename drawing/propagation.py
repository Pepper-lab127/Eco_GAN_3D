"""How the riparian plants of the Lockport list spread: how far and when.

REACH: each way a plant moves, as a distance range (m) from the parent per
event or per year, with `est` True where the number is an estimate rather
than a published figure for the species.

YEAR: each stage of the plant's year as (stage, start, end) in months from
January 1 (0) to December 31 (12); a range that wraps the new year has
start > end. Stages: flower, ripen (fruit or seed on the plant), disperse,
cold (dormant seed that needs winter cold), germinate, spread (vegetative
growth: rhizomes, stolons, suckers, layering). Dates are typical for
southern Manitoba, rounded to the half month: indicative, not observed.

Sources (see the reading guide): UW propagation notes, Nature Manitoba,
Alberta species profile (red-osier dogwood); NRCS plant guides (prairie
cordgrass, swamp milkweed); USFS Tree Atlas and FEIS (boxelder); Iowa State
extension and FEIS (American elm); EOPugetSound, UAlberta and Tree Planters'
Notes (trembling aspen); Ecology and Evolution 2020 (trillium seeds and ants).
"""

PLANTS = [
    dict(key="trillium", name="NODDING TRILLIUM", latin="Trillium cernuum", draw=("flora", "trillium", 0.4),
         reach=[("ants", 0.5, 2.0, "per seed", True), ("falls", 0.05, 0.3, "per seed", True)],
         year=[("flower", 4.5, 6.0), ("ripen", 6.5, 8.0), ("disperse", 7.0, 8.0), ("cold", 10.5, 4.0),
               ("germinate", 4.5, 5.5)],
         notes="ANTS CARRY THE SEED FOR ITS OILY TIP · TWO WINTERS BEFORE A SHOOT"),
    dict(key="cordgrass", name="PRAIRIE CORD GRASS", latin="Spartina pectinata", draw=("flora", "cordgrass", 1.8),
         reach=[("rhizome", 0.3, 0.6, "per year", False)],
         year=[("spread", 4.5, 9.0), ("flower", 6.5, 8.0), ("ripen", 8.0, 9.5), ("cold", 10.5, 4.0),
               ("germinate", 5.0, 6.0)],
         notes="RHIZOMES 0.3–0.6 M A YEAR, SHOOTS RUNNING DOWN TOWARD THE WATER"),
    dict(key="dogwood", name="RED-OSIER DOGWOOD", latin="Cornus sericea", draw=("flora", "dogwood", 2.2),
         reach=[("stolon", 0.45, 1.1, "per year", True), ("layer", 1.0, 2.5, "per stem", True),
                ("birds", 10.0, 300.0, "per seed", True)],
         year=[("spread", 5.0, 9.0), ("flower", 5.0, 7.0), ("ripen", 6.5, 9.5), ("disperse", 7.0, 10.0),
               ("cold", 10.5, 4.0), ("germinate", 4.5, 6.0)],
         notes="STOLONS AND LAYERED STEMS ROOT AT THE NODES · BIRDS CARRY THE WHITE FRUIT"),
    dict(key="aspen", name="TREMBLING ASPEN", latin="Populus tremuloides", draw=("tree", "aspen", 14.0),
         reach=[("suckers", 1.0, 30.0, "per clone", True), ("wind", 5.0, 3000.0, "per seed", True)],
         year=[("flower", 3.5, 4.5), ("ripen", 4.5, 5.0), ("disperse", 5.0, 6.0), ("germinate", 5.0, 6.0),
               ("spread", 5.5, 8.0)],
         notes="ONE CLONE OF ROOT SUCKERS · COTTONY SEED, ALIVE ONLY 2–4 WEEKS"),
    dict(key="oak", name="BUR OAK", latin="Quercus macrocarpa", draw=("tree", "oak", 13.0),
         reach=[("falls", 0.0, 15.0, "per acorn", True), ("cached", 5.0, 60.0, "per acorn", True)],
         year=[("flower", 4.5, 5.5), ("ripen", 7.5, 9.0), ("disperse", 8.5, 10.5), ("germinate", 9.0, 10.5),
               ("cold", 10.5, 4.5), ("germinate", 4.5, 5.5)],
         notes="ACORNS FALL UNDER THE CROWN; SQUIRRELS AND MICE BURY AND EAT THEM"),
    dict(key="maple", name="MANITOBA MAPLE", latin="Acer negundo", draw=("tree", "maple", 12.0),
         reach=[("wind", 1.0, 200.0, "per seed", False)],
         year=[("flower", 4.0, 5.0), ("ripen", 7.5, 10.0), ("disperse", 9.5, 4.0), ("cold", 10.5, 4.0),
               ("germinate", 4.5, 5.5)],
         notes="WINGED SEED, UP TO 200 M ON THE WIND, ACROSS THE SNOW ALL WINTER"),
    dict(key="elm", name="AMERICAN ELM", latin="Ulmus americana", draw=("tree", "elm", 16.0),
         reach=[("wind", 1.0, 400.0, "per seed", True)],
         year=[("flower", 3.5, 4.5), ("ripen", 4.5, 5.0), ("disperse", 5.0, 6.0), ("germinate", 5.0, 6.5)],
         notes="SEED FALLS IN SPRING AND SPROUTS AT ONCE ON MOIST GROUND"),
    dict(key="milkweed", name="SWAMP MILKWEED", latin="Asclepias incarnata", draw=("flora", "milkweed", 1.2),
         reach=[("wind", 2.0, 500.0, "per seed", True)],
         year=[("flower", 6.5, 8.0), ("ripen", 7.5, 9.5), ("disperse", 9.0, 11.0), ("cold", 11.0, 4.0),
               ("germinate", 5.0, 6.0)],
         notes="SILK-TUFTED SEED ON THE WIND, OCTOBER–NOVEMBER · ABOUT 270 SEEDS A PLANT"),
]

# How each way of moving is drawn.
KIND = {
    "falls": "drop", "ants": "ants", "rhizome": "under", "stolon": "under", "suckers": "under",
    "layer": "arch", "birds": "flight", "cached": "flight", "wind": "wind",
}
