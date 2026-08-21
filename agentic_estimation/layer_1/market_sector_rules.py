"""Hand-authored market-name -> EXIOBASE sector rules.

Deterministic and auditable: no LLM, no embeddings. Ordered list of
(regex, EXIOBASE sector) -- FIRST match wins, so put specific patterns
before general ones. Anything unmatched stays NULL rather than being
forced into a wrong bucket (fail closed, same principle as
market_climate_trace_mapper).

Rationale for ordering: a "cancer drug delivery device" is a device, not a
drug, so device/instrument patterns must precede drug patterns; "packaging"
beats the product being packaged; explicit service words beat manufacturing.
"""

RULES: list[tuple[str, str]] = [
    # ---- PRIORITY OVERRIDES ----
    # Found by hand-checking a random sample of the mapping output: these were
    # being captured by a later, broader rule that fires on a substring
    # ("circulator" -> chemical stem /circ/, "response" -> /res/, "laser" ->
    # optical instruments even in "laser welding machine"). They must be
    # matched FIRST because first-match wins.
    (r"\b(welding|cutting|drilling|milling|grinding|machining)\b",
     "Manufacture of machinery and equipment n.e.c. (29)"),
    (r"\b(rf\b|radio ?frequency|isolator|circulator|amplifier|oscillator|waveguide|antenna)\b",
     "Manufacture of radio, television and communication equipment and apparatus (32)"),
    (r"\b(emergency response|alert system|alarm|intercom|siren)\b",
     "Manufacture of electrical machinery and apparatus n.e.c. (31)"),
    (r"\b(shipping container|freight container|intermodal container|steel drum)\b",
     "Manufacture of fabricated metal products, except machinery and equipment (28)"),

    # ---- services that would otherwise be swallowed by product words ----
    (r"\b(consult|advisory|outsourc|bpo|staffing|recruit|market research|due diligence)\b",
     "Other business activities (74)"),
    (r"\b(insurance|reinsur|annuit|underwrit)\b",
     "Insurance and pension funding, except compulsory social security (66)"),
    (r"\b(bank|lending|loan|mortgage|payment|fintech|credit card|wealth manage|asset manage|trading platform)\b",
     "Financial intermediation, except insurance and pension funding (65)"),
    (r"\b(e-?commerce|retail|supermarket|store|shop|vending)\b",
     "Retail trade, except of motor vehicles and motorcycles; repair of personal and household goods (52)"),
    (r"\b(wholesale|distribution|distributor|supply chain|logistics|freight forward|warehous)\b",
     "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)"),
    (r"\b(hospital|clinic|physiotherap|nursing|home care|patient|telemedicine|telehealth|dental care|healthcare service|medical claim|health insurance|diagnos(is|tic service))\b",
     "Health and social work (85)"),
    (r"\b(education|e-?learning|training|tutor|school|university|courseware)\b",
     "Education (80)"),
    (r"\b(hotel|restaurant|catering|hospitality|food service|travel|tourism)\b",
     "Hotels and restaurants (55)"),
    (r"\b(real estate|property manage|facility manage|construction|building material|cement|concrete|infrastructure|hvac install|roofing|flooring)\b",
     "Construction (45)"),
    (r"\b(research and development|clinical trial|contract research|cro\b|preclinical|drug discovery|biotech research)\b",
     "Research and development (73)"),

    # ---- ICT ----
    (r"\b(software|saas|platform|app\b|apps\b|cloud|cyber ?security|firewall|analytics|artificial intelligence|machine learning|blockchain|smart contract|data (center|centre|management|analytics)|iot platform|digital twin|erp|crm|devops|api\b)\b",
     "Computer and related activities (72)"),
    (r"\b(telecom|5g|broadband|satellite communication|network operator|mobile network|wireless carrier)\b",
     "Post and telecommunications (64)"),
    (r"\b(computer|laptop|server|data storage|semiconductor|microprocessor|integrated circuit|pcb|printed circuit)\b",
     "Manufacture of office machinery and computers (30)"),
    (r"\b(smartphone|smartwatch|wearable|television|radio|antenna|broadcast|communication equipment|set-top|headphone|earbud)\b",
     "Manufacture of radio, television and communication equipment and apparatus (32)"),

    # ---- instruments / devices (BEFORE drugs: a delivery device is a device) ----
    (r"\b(sensor|spectromet|spectral|microscop|laser|optic|lens|imaging|scanner|monitor(ing)? (device|system)|catheter|guidewire|stent|implant|prosthe|orthopedic|surgical|endoscop|ultrasound|mri\b|ct scan|x-?ray|diagnostic (device|equipment|kit)|biosensor|biodetection|wearable medical|medical device|dental (equipment|implant)|hearing aid|pacemaker|infusion pump|drug delivery (device|platform|system))\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),

    # ---- pharma / chemicals ----
    (r"\b(drug|pharmaceutic|therapeutic|vaccine|antibod|biologic|biosimilar|api\b|active pharmaceutical|treatment|therapy|oncolog|cancer|disease|syndrome|disorder|infection|diabet|cardio|neuro|dermatolog|ophthalmo|analgesic|antibiotic|inhibitor|monoclonal|peptide|enzyme|probiotic|nutraceutical|supplement)\b",
     "Chemicals nec"),
    (r"\b(chemical|polymer|resin|adhesive|coating|paint|dye|pigment|solvent|surfactant|catalyst|acid|oxide|chloride|sulfate|benzoate|reagent|electrolyte|lubricant|specialty gas|industrial gas)\b",
     "Chemicals nec"),
    (r"\b(fertiliz|fertilis|urea|ammonia|nitrogen fertil)\b", "N-fertiliser"),
    (r"\b(pesticide|herbicide|insecticide|agrochemical|crop protection)\b", "Chemicals nec"),

    # ---- materials ----
    (r"\b(plastic|rubber|foam|elastomer|silicone|pvc|polyethylene|polypropylene|packaging film)\b",
     "Manufacture of rubber and plastic products (25)"),
    (r"\b(packaging|container|bottle|carton|pouch|label)\b",
     "Manufacture of rubber and plastic products (25)"),
    (r"\b(paper|pulp|tissue|cardboard|corrugated)\b", "Paper"),
    (r"\b(glass|glazing)\b", "Manufacture of glass and glass products"),
    (r"\b(ceramic|porcelain)\b", "Manufacture of ceramic goods"),
    (r"\b(steel|iron|ferro|alloy steel)\b",
     "Manufacture of basic iron and steel and of ferro-alloys and first products thereof"),
    (r"\b(aluminium|aluminum)\b", "Aluminium production"),
    (r"\b(copper)\b", "Copper production"),
    (r"\b(textile|fabric|yarn|fiber|fibre|muslin|cotton|woven|nonwoven)\b",
     "Manufacture of textiles (17)"),
    (r"\b(apparel|clothing|garment|footwear|shoe|fashion wear)\b",
     "Manufacture of wearing apparel; dressing and dyeing of fur (18)"),
    (r"\b(leather|handbag|luggage)\b",
     "Tanning and dressing of leather; manufacture of luggage, handbags, saddlery, harness and footwear (19)"),
    (r"\b(furniture|mattress)\b", "Manufacture of furniture; manufacturing n.e.c. (36)"),
    (r"\b(wood|timber|lumber|plywood)\b",
     "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials (20)"),

    # ---- machinery / vehicles / energy ----
    (r"\b(motor control|electrical machinery|transformer|switchgear|circuit breaker|cable|battery|capacitor|electric motor|generator set|power supply|inverter)\b",
     "Manufacture of electrical machinery and apparatus n.e.c. (31)"),
    (r"\b(automotive|vehicle|car\b|truck|bus\b|motorcycle|infotainment|tire|tyre|autonomous driving|ev charging)\b",
     "Manufacture of motor vehicles, trailers and semi-trailers (34)"),
    (r"\b(aircraft|aerospace|drone|uav|satellite|spacecraft|shipbuild|railcar|locomotive)\b",
     "Manufacture of other transport equipment (35)"),
    (r"\b(pump|valve|compressor|turbine|bearing|robot|cnc|3d print|additive manufactur|industrial machinery|conveyor|actuator|hydraulic|pneumatic)\b",
     "Manufacture of machinery and equipment n.e.c. (29)"),
    (r"\b(solar|photovoltaic|pv\b)\b", "Production of electricity by solar photovoltaic"),
    (r"\b(wind (power|turbine|energy))\b", "Production of electricity by wind"),
    (r"\b(hydro ?power|hydroelectric)\b", "Production of electricity by hydro"),
    (r"\b(nuclear power|nuclear energy)\b", "Production of electricity by nuclear"),
    (r"\b(oil ?field|crude oil|petroleum|refinery|refining)\b", "Petroleum Refinery"),
    (r"\b(natural gas|lng\b)\b",
     "Extraction of natural gas and services related to natural gas extraction, excluding surveying"),
    (r"\b(coal)\b", "Mining of coal and lignite; extraction of peat (10)"),
    (r"\b(mining|ore\b|quarry)\b",
     "Mining of chemical and fertilizer minerals, production of salt, other mining and quarrying n.e.c."),
    (r"\b(electricity|power grid|smart grid|energy storage)\b", "Distribution and trade of electricity"),

    # ---- food & agri ----
    (r"\b(dairy|milk|cheese|yogurt|yoghurt)\b", "Processing of dairy products"),
    (r"\b(meat|poultry|beef|pork)\b", "Production of meat products nec"),
    (r"\b(beverage|drink|juice|coffee|tea\b|water bottle|soda|alcohol|beer|wine|spirits)\b",
     "Manufacture of beverages"),
    (r"\b(food|snack|bakery|confection|nutrition|infant formula|flavor|flavour|ingredient)\b",
     "Processing of Food products nec"),
    (r"\b(cannabis|hemp|crop|seed|agricultur|farming|horticultur|greenhouse)\b",
     "Cultivation of crops nec"),
    (r"\b(fish|aquacultur|seafood)\b",
     "Fishing, operating of fish hatcheries and fish farms; service activities incidental to fishing (05)"),
    (r"\b(tobacco|cigarette|vape|e-?cigarette)\b", "Manufacture of tobacco products (16)"),

    # ---- waste / water / misc ----
    (r"\b(recycl|waste manage|scrap)\b", "Recycling of waste and scrap"),
    (r"\b(water treatment|waste ?water|desalination|sewage)\b", "Waste water treatment, other"),
    (r"\b(publishing|printing|media content|broadcasting content)\b",
     "Publishing, printing and reproduction of recorded media (22)"),
    (r"\b(entertainment|gaming|sport|fitness|leisure|casino|music)\b",
     "Recreational, cultural and sporting activities (92)"),
    (r"\b(defence|defense|military|weapon|ammunition)\b",
     "Public administration and defence; compulsory social security (75)"),
    (r"\b(rail(way)? transport)\b", "Transport via railways"),
    (r"\b(air(line| transport|port))\b", "Air transport (62)"),
    (r"\b(shipping|maritime|port terminal)\b", "Sea and coastal water transport"),
    (r"\b(pipeline)\b", "Transport via pipelines"),
    (r"\b(metal (product|fabricat)|fastener|casting|forging|welding)\b",
     "Manufacture of fabricated metal products, except machinery and equipment (28)"),
]

# ---- second pass: broadened stems + chemical-nomenclature shapes ----
# Written after measuring the first pass (33.9% coverage): the top unmatched
# tokens were words the pass-1 rules *should* have caught but missed because of
# over-tight word boundaries ("drug" missing "drugs", "chemical" missing
# "chemicals"), plus IUPAC-style names no keyword list can enumerate.
RULES += [
    # chemical nomenclature by SHAPE, not by name
    (r"^[\d,\-]+[a-z]?[\s\-]", "Chemicals nec"),                      # "1,3 Propanediol", "2,4-d"
    (r"\b\w+(ol|ane|ene|yne|ate|ide|ite|amine|amide|acid|oxide|phenol|ester|ketone)\b", "Chemicals nec"),
    (r"\b(sodium|potassium|calcium|magnesium|zinc|lithium|ammonium|chlor|fluor|silic|sulph|sulf|nitr|phosph)\w*", "Chemicals nec"),

    # broadened stems the first pass was too tight for
    (r"\b(drugs?|medicine|medicat|pharma\w*|vaccin\w*|antigen|antivir\w*|steroid|hormone|insulin|serum|plasma|reagent)\b", "Chemicals nec"),
    (r"\b(chemical\w*|coating\w*|polymer\w*|additive\w*|pigment\w*|resin\w*|solvent\w*)\b", "Chemicals nec"),
    (r"\b(supplement\w*|vitamin\w*|protein\w*|amino acid|collagen|omega)\b", "Processing of Food products nec"),
    (r"\b(medical|dental|surgical|clinical|diagnostic\w*|implant\w*|orthotic\w*|prosthetic\w*|biopsy|catheter\w*)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(healthcare|health care|patient care|elder ?care|home ?health|wellness|nursing)\b", "Health and social work (85)"),
    (r"\b(cell\w*|stem cell|tissue|genom\w*|proteom\w*|crispr|dna|rna|biomarker|microbiome|bioprint\w*)\b", "Research and development (73)"),
    (r"\b(test\w*|assay|screening|metrolog\w*|calibration|inspection|quality control)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(sensor\w*|detector\w*|transducer|probe|meter\b|gauge|instrument\w*)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(smart|digital|ai\b|artificial|automation|iot|virtual|augmented|metaverse|saas|cyber)\b",
     "Computer and related activities (72)"),
    (r"\b(software|comput\w*|semiconductor\w*|chip\w*|processor|memory|display|screen|panel|wafer|electronic\w*)\b",
     "Manufacture of office machinery and computers (30)"),
    (r"\b(management|monitoring|tracking|planning|scheduling|optimization|analytics|intelligence)\b",
     "Computer and related activities (74)".replace("(74)", "(72)")),
    (r"\b(power|energy|battery|batteries|fuel cell|charging|voltage|electric\w*|grid)\b",
     "Manufacture of electrical machinery and apparatus n.e.c. (31)"),
    (r"\b(water|hydro|irrigation|filtration|purification)\b", "Collection, purification and distribution of water (41)"),
    (r"\b(material\w*|composite\w*|alloy\w*|ceramic\w*|graphene|nanomaterial|nanoparticle)\b",
     "Manufacture of other non-metallic mineral products n.e.c."),
    (r"\b(machine|machinery|industrial|manufactur\w*|robot\w*|tooling|press\b|mill\b)\b",
     "Manufacture of machinery and equipment n.e.c. (29)"),
    (r"\b(network\w*|telecom\w*|wireless|bluetooth|router|modem|gateway)\b", "Post and telecommunications (64)"),
    (r"\b(security|surveillance|authentication|encryption|biometric)\b", "Computer and related activities (72)"),
    (r"\b(skin|cosmetic\w*|beauty|hair|fragrance|perfume|soap|detergent|cleaning)\b", "Chemicals nec"),
    (r"\b(injection|infusion|syringe|needle|vial|ampoule)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(blood|cardiac|respirator\w*|oncolog\w*|neurolog\w*|ophthalm\w*|orthoped\w*|urolog\w*|gastro\w*)\b",
     "Health and social work (85)"),
    (r"\b(service\w*|platform|solution\w*|consulting|outsourcing)\b", "Other business activities (74)"),
]

# ---- third pass: remaining long tail ----
# The residual after pass 2 was dominated by specific compound names
# (Abamectin, Acetaminophen, Acetone, Acetonitrile...) plus assorted devices.
# Chemical/drug morphology covers most of it; the rest are explicit nouns.
RULES += [
    (r"\b\w*(mycin|cillin|statin|sartan|prazole|dipine|ciclib|tinib|zumab|ximab|mab\b|vir\b|pril\b|olol\b|azole|caine|profen)\b", "Chemicals nec"),
    (r"\b(acet|benz|ethyl|methyl|propyl|butyl|phenyl|glyc|carb|hydro|per|poly|iso|tri|di|mono)\w{3,}", "Chemicals nec"),
    (r"\b(extract|berry|herbal|botanical|essential oil|flavour|flavor)\b", "Processing of Food products nec"),
    (r"\b(abrasive|adhesive|sealant|lubricant|grease|wax)\b", "Chemicals nec"),
    (r"\b(pad|dressing|bandage|gauze|suture|hemostat|glove|mask|ppe)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(chiller|hvac|refrigerat|air condition|heat exchanger|boiler|furnace|compressor)\b",
     "Manufacture of machinery and equipment n.e.c. (29)"),
    (r"\b(cart|utv|atv|scooter|bicycle|trailer|forklift)\b", "Manufacture of motor vehicles, trailers and semi-trailers (34)"),
    (r"\b(sms|messaging|voip|call center|contact center)\b", "Post and telecommunications (64)"),
    (r"\b(folding carton|corrugat|box\b|crate|drum\b|pallet)\b", "Paper"),
    (r"\b(aneurysm|ablation|catheter|valve repair|graft|stent)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
]

# ---- fourth pass: disease names, brand drugs, remaining compounds ----
# A "<disease> Market" is a market for its TREATMENT, i.e. pharmaceuticals.
RULES += [
    (r"\b(itis|osis|emia|pathy|plasia|trophy|algia|ectomy|opsy|oma)\b", "Chemicals nec"),
    (r"\b(acute|chronic|syndrome|keratosis|stroke|seizure|pancreatitis|arthritis|asthma|copd|hiv|hepatitis|malaria|tuberculosis|alzheimer|parkinson|epilep|sclerosis|fibrosis|anemia|leukemia|lymphoma|melanoma|carcinoma|tumor|tumour)\b", "Chemicals nec"),
    (r"\b(nitrile|acrylo|amine|aldehyde|anhydride|glycol|urethane|siloxane|carbon black|activated carbon)\b", "Chemicals nec"),
    (r"\b(regulator|stabilizer|stabiliser|emulsifier|preservative|thickener|sweetener|colorant)\b", "Processing of Food products nec"),
    (r"\b(wheelchair|stroller|walker|crutch|hearing|mobility aid|acupuncture|orthosis)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(radar|sonar|avionics|missile|armor|armour|protection system)\b", "Manufacture of other transport equipment (35)"),
    (r"\b(wear|clothing|footwear)\b", "Manufacture of wearing apparel; dressing and dyeing of fur (18)"),
    (r"\b(chair|table|desk|sofa|cabinet|shelving)\b", "Manufacture of furniture; manufacturing n.e.c. (36)"),
    (r"\b(matrix|matrices|scaffold|graft|dermal|collagen)\b", "Research and development (73)"),
]

# ---- fifth pass: generic head-nouns (LAST -- lowest priority) ----
# The residual tail is dominated by generic nouns (devices 101, products 53,
# equipment 48, systems 41) that carry no domain of their own. These fire only
# after every specific rule above has failed, so they act as typed catch-alls
# rather than overriding real signal. Ordered medical -> electronic -> generic
# machinery, matching the observed composition of the tail.
RULES += [
    (r"\b(cannula|catheter|syringe|ecg|ekg|electrocardiogram|spectroscop|analyz|analys|laborator|surgery|surgical|rehabilitation|diagnostic|respirator|breathing apparatus|nebuliz|oximet|endoscop|defibrillat)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(heparin|mimetic|laxative|remed|antacid|lozenge|ointment|topical|syrup|tablet|capsule|dose|dosage)\b",
     "Chemicals nec"),
    (r"\b(insulation|thermal barrier|acoustic panel|drywall|plaster)\b", "Construction (45)"),
    (r"\b(home decor|decor|furnishing|upholster|curtain|rug\b|carpet)\b",
     "Manufacture of furniture; manufacturing n.e.c. (36)"),
    (r"\b(rental|renting|leasing|hire)\b",
     "Renting of machinery and equipment without operator and of personal and household goods (71)"),
    (r"\b(online|e-?tail|home shopping|marketplace|creator economy|subscription)\b",
     "Retail trade, except of motor vehicles and motorcycles; repair of personal and household goods (52)"),
    (r"\b(laundry|dry clean|cleaning service|sanitation)\b", "Other service activities (93)"),
    (r"\b(bag|film|sheet|foil|wrap|tape)\b", "Manufacture of rubber and plastic products (25)"),
    (r"\b(storage|retrieval|warehouse automation|conveyor|palletiz)\b",
     "Manufacture of machinery and equipment n.e.c. (29)"),
    (r"\b(agricultur|farm|tractor|harvest|irrigation)\b", "Cultivation of crops nec"),
    (r"\b(accessor|apparel|tuxedo|garment|tattoo)\b",
     "Manufacture of wearing apparel; dressing and dyeing of fur (18)"),
    # typed generic catch-alls -- deliberately last
    (r"\b(device|instrument|apparatus)\b",
     "Manufacture of medical, precision and optical instruments, watches and clocks (33)"),
    (r"\b(equipment|machinery|machine)\b", "Manufacture of machinery and equipment n.e.c. (29)"),
    (r"\b(system|systems|automation|control)\b", "Computer and related activities (72)"),
    (r"\b(product|goods|supplies|consumable)\b", "Manufacture of furniture; manufacturing n.e.c. (36)"),
]
