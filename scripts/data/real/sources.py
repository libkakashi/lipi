"""
Source registry for real-data ingestion (v1: HuggingFace-hosted crops).

Each spec describes one hub dataset of line/word crops with transcriptions.
Column names auto-detect when unset (first Image feature / first plausible
string feature), so specs stay robust to card churn; override per-spec when
detection would pick wrong.

license_tier: 1 = explicit commercial permission, 2 = unstated/gray (email
pending — usable for experiments, filter with --max-tier before shipping),
3 = blocked (never train; kept only so eval builds can reference them).
Full license notes per source live in the research report
(~/Documents/OCR_RealData_Research_20260713/).
"""

from dataclasses import dataclass, field


@dataclass
class SourceSpec:
    name: str            # registry key + chunk-dir suffix
    hub: str             # HF dataset id
    script: str | None   # fixed script, or None → use lang_col + lang_map
    license_tier: int
    splits: tuple = ("train",)   # hub splits to ingest (never test splits)
    config: str | None = None
    # Multi-config hubs (one config per language): each config name maps
    # to a script via lang_map; unmapped configs are skipped.
    configs: tuple = ()
    image_col: str | None = None
    text_col: str | None = None
    lang_col: str | None = None
    lang_map: dict = field(default_factory=dict)
    # "hub" = load_dataset streaming (parquet/imagefolder repos).
    # "url_zip" = download zips from `urls`, parse split txt annotations.
    # "hub_url_image" = hub rows whose image is an http URL to fetch.
    # Anything else is a placeholder for a loader not written yet.
    loader: str = "hub"
    urls: list = field(default_factory=list)   # [(zip_url, script), ...]
    notes: str = ""


# ISO/dataset language tags → taxonomy script names (Mozhi-style Indic sets).
_INDIC_LANG_TO_SCRIPT = {
    "hindi": "devanagari", "marathi": "devanagari", "nepali": "devanagari",
    "sanskrit": "devanagari", "bengali": "bengali", "bangla": "bengali",
    "assamese": "bengali", "gujarati": "gujarati", "punjabi": "gurmukhi",
    "gurumukhi": "gurmukhi", "odia": "odia", "oriya": "odia",
    "kannada": "kannada", "telugu": "telugu", "malayalam": "malayalam",
    "tamil": "tamil", "urdu": "arabic", "manipuri": None,  # not in taxonomy
}

REGISTRY: dict[str, SourceSpec] = {s.name: s for s in [
    SourceSpec("norhand_v3", "Teklia/NorHand-v3-line", "latin", 1,
               notes="247K Norwegian HW lines, CC-BY-4.0"),
    SourceSpec("catmus_modern", "CATMuS/modern", "latin", 1,
               notes="118K modern-era lines FR/ES/DE/EN/IT, CC-BY-4.0"),
    SourceSpec("casia_hwdb2", "Teklia/CASIA-HWDB2-line", "han", 2,
               notes="52K zh HW lines; mirror MIT tag vs CASIA "
                     "research-only upstream — gray"),
    SourceSpec("hhd_ethiopic", "zenodo.org/records/7978722", "ethiopic", 1,
               loader="url_zip", urls=[
                   ("https://zenodo.org/api/records/7978722/files/hhd-Ethiopic.zip/content", "ethiopic"),
               ],
               notes="80K HW lines CC-BY-4.0, single Zenodo zip; pair "
                     "layout (img ↔ .gt.txt); test paths quarantined. "
                     "(HF repo's tain-val entry is an empty marker; its "
                     "train data is numpy-packed — Zenodo is the clean "
                     "source. Images are small → upscale to 64px.)"),
    SourceSpec("thai_hw", "iapp/thai_handwriting_dataset", "thai", 2,
               notes="BEST2019+Wang, Apache tag, NECTEC provenance caveat"),
    SourceSpec("burmese_bl", "chuuhtetnaing/burmese_ocr_dataset_hf",
               "burmese", 1, configs=("cleaned",),
               notes="Burma Library lines, public domain (mirror of "
                     "alexbeatson/burmese_ocr_data whose raw zips aren't "
                     "hub-loadable); 'uncleaned' config = Document-AI "
                     "pseudo-labels, ingest separately as pseudo class"),
    SourceSpec("khatt", "benhachem/KHATT", "arabic", 2,
               notes="modern Arabic HW lines; KFUPM research terms"),
    SourceSpec("mozhi", "darknight054/indic-mozhi-ocr", None, 2,
               lang_map=_INDIC_LANG_TO_SCRIPT,
               configs=("assamese", "bengali", "gujarati", "hindi",
                        "kannada", "malayalam", "manipuri", "marathi",
                        "oriya", "punjabi", "tamil", "telugu", "urdu"),
               notes="1.2M printed Indic words + lines, 12 langs; "
                     "CVIT license email pending"),
    SourceSpec("tibetan_gbooks", "openpecha/OCR-Google_Books", "tibetan", 1,
               notes="751K real lines, ODC-BY; images embedded in parquet "
                     "(schema-verified 2026-07-13)"),
    SourceSpec("tibetan_cursive", "openpecha/OCR-Handwritten_Tibetan_Cursive",
               "tibetan", 2, loader="hub_url_image",
               notes="70K real HW cursive lines; images on monlam.ai S3 "
                     "(parquet columns misaligned — URL detected per-row); "
                     "license unstated, BDRC email pending"),
    SourceSpec("iiit_indic_hw_words", "cvit.iiit.ac.in", None, 2,
               loader="url_zip", urls=[
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/bengal.zip", "bengali"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/gu.zip", "gujarati"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/pn.zip", "gurmukhi"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/kn.zip", "kannada"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/od.zip", "odia"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/ma.zip", "malayalam"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/ta.zip", "tamil"),
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/ur.zip", "arabic"),
               ],
               notes="872K HW words, 8 scripts; no license stated — "
                     "tier-2 until CVIT permission lands"),
    SourceSpec("iiit_uc", "cvit.iiit.ac.in", None, 2, loader="uc_zip",
               urls=[
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/hindi.zip", "devanagari"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/marathi.zip", "devanagari"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/bengali.zip", "bengali"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/assamese.zip", "bengali"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/gujarati.zip", "gujarati"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/punjabi.zip", "gurmukhi"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/oriya.zip", "odia"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/kannada.zip", "kannada"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/telugu.zip", "telugu"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/malayalam.zip", "malayalam"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/tamil.zip", "tamil"),
                   ("https://cvit.iiit.ac.in/images/datasets/indic_handwritten/urdu.zip", "arabic"),
               ],
               notes="IIIT-Indic-HW-UC (2024): 2.6M camera-captured HW "
                     "words, 13 langs; manipuri (Meitei) skipped — not in "
                     "taxonomy. gt.txt + Google text/ sidecars. tier-2, CVIT"),
    SourceSpec("iiit_ma_fix", "cvit.iiit.ac.in", "malayalam", 2,
               loader="url_zip", urls=[
                   ("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/ma.zip", "malayalam"),
               ], notes="malayalam re-ingest under its own chunk prefix — "
                        "the preemption-retry of iiit_indic_hw_words lost "
                        "ma.zip to CDN truncation"),
    SourceSpec("iiit_hw_dev", "cvit.iiit.ac.in", "devanagari", 2,
               loader="url_zip", urls=[
                   ("https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Dev.zip", "devanagari"),
               ], notes="95K HW Devanagari words; tier-2, CVIT"),
    SourceSpec("iiit_hw_telugu", "cvit.iiit.ac.in", "telugu", 2,
               loader="url_zip", urls=[
                   ("https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/IIIT-HW-Telugu.zip", "telugu"),
               ], notes="120K HW Telugu words; tier-2, CVIT"),
    SourceSpec("digital_peter", "ai-forever/digital_peter_aij2020", "cyrillic",
               1, notes="Petrine cursive lines, MIT (GitHub-hosted; HF "
                        "mirror availability varies)"),
    # --- wave 3: scanned/camera printed (real degradation + labels) ---
    SourceSpec("heidata_fid4sa", "heidata.uni-heidelberg.de", None, 1,
               loader="alto_zip", urls=[
                   ("https://heidata.uni-heidelberg.de/api/access/dataset/:persistentId?persistentId=doi:10.11588/data/EGOKEI", "devanagari"),
                   ("https://heidata.uni-heidelberg.de/api/access/dataset/:persistentId?persistentId=doi:10.11588/data/L2KRZO", "malayalam"),
                   ("https://heidata.uni-heidelberg.de/api/access/dataset/:persistentId?persistentId=doi:10.11588/data/AIQSXL", "bengali"),
               ],
               notes="FID4SA-GT: scanned 19-20thC printed books, ALTO-v4 "
                     "line boxes, CC-BY-4.0. REAL scan degradation + human "
                     "GT. Devanagari/Malayalam/Bengali. ~78% keep "
                     "(mixed-script lines dropped under v1 policy)."),
    SourceSpec("tibetan_khyentse", "BDRC/KhyentseWangpo", "tibetan", 1,
               image_col="line", text_col="transcription",
               notes="13.5K real typeset-PRINT Tibetan line crops, ODC-BY "
                     "(vs our woodblock sets — adds clean-print domain). "
                     "Tibetan already over-represented → low mix weight."),
    SourceSpec("gt4histocr", "zenodo.org/records/1344132", "latin", 1,
               loader="url_zip", urls=[
                   ("https://zenodo.org/api/records/1344132/files/GT4HistOCR.tar/content", "latin"),
               ], notes="313K scanned Fraktur/early-Latin line pairs, "
                        "CC-BY-4.0; real scan degradation, precise labels. "
                        "Per-subcorpus transcription conventions vary."),
    SourceSpec("bmod", "zenodo.org/records/15310982", "latin", 1,
               loader="bmod_zip", urls=[
                   ("https://zenodo.org/api/records/15310982/files/b-mod_lines.zip/content", "latin"),
               ], notes="~500K real PHONE-CAMERA line photos of printed "
                        "pages, CC-BY-4.0, precise labels — best "
                        "camera-degradation source. lines/*.jpg + "
                        "train.{easy,medium,hard} annotation files."),

    # --- wave 2 (schema-probed 2026-07-13) ---
    SourceSpec("burmese_bl_pseudo", "chuuhtetnaing/burmese_ocr_dataset_hf",
               "burmese", 1, configs=("uncleaned",),
               notes="163K Document-AI pseudo-labeled Burma Library lines "
                     "(PD); pseudo-class — cap share at train time"),
    SourceSpec("riks_gbg_polis",
               "Riksarkivet/goteborgs_poliskammare_fore_1900_lines",
               "latin", 2, notes="357K Swedish police-record HW lines; "
                                 "license metadata absent (RA default open)"),
    SourceSpec("riks_svea", "Riksarkivet/svea_hovratt_lines", "latin", 2,
               notes="Svea Court of Appeal HW lines"),
    SourceSpec("riks_krigshovratt",
               "Riksarkivet/krigshovrattens_dombocker_lines", "latin", 2,
               notes="military court records"),
    SourceSpec("riks_bergskollegium",
               "Riksarkivet/bergskollegium_relationer_och_skrivelser_lines",
               "latin", 2, notes="Bergskollegium HW lines"),
    SourceSpec("riks_trolldom", "Riksarkivet/trolldomskommissionen_lines",
               "latin", 2, notes="witch-trial commission records"),
    SourceSpec("riks_frihetstiden",
               "Riksarkivet/frihetstidens_utskottshandlingar_lines",
               "latin", 2, notes="parliamentary committee records"),
    SourceSpec("riks_alvsborg", "Riksarkivet/alvsborgs_losen_lines",
               "latin", 2, notes="Älvsborg ransom tax records"),
    SourceSpec("riks_fraktur", "Riksarkivet/swedish_fraktur", "latin", 1,
               notes="printed Swedish Fraktur, Apache-2.0"),
    SourceSpec("teklia_belfort", "Teklia/Belfort-line", "latin", 1,
               notes="32.7K French cursive lines"),
    SourceSpec("teklia_popp", "Teklia/POPP-line", "latin", 1,
               notes="Paris census table lines"),
    SourceSpec("teklia_himanis", "Teklia/Himanis-line", "latin", 1,
               notes="23K medieval French/Latin royal chancery lines"),
    SourceSpec("rimes", "Teklia/RIMES-2011-line", "latin", 1,
               notes="12K French HW lines; 2024 permissive re-release"),
    SourceSpec("tibetan_lhasakanjur", "openpecha/OCR-Lhasakanjur",
               "tibetan", 2, notes="161K woodblock lines, embedded images; "
                                   "license unstated, BDRC email pending"),
    SourceSpec("tibetan_drutsa", "openpecha/OCR-Drutsa", "tibetan", 2,
               notes="32K drutsa-style cursive lines"),
    SourceSpec("tibetan_betsug", "openpecha/OCR-Betsug", "tibetan", 2,
               notes="28K betsug-style lines"),
    SourceSpec("tibetan_norbuketaka", "openpecha/OCR-Norbuketaka",
               "tibetan", 2, loader="hub_url_image",
               notes="2.24M modern typeset print lines; images on "
                     "monlam.ai S3 — mirror while ingesting"),
    SourceSpec("tibetan_uchan", "openpecha/OCR_Uchan", "tibetan", 2,
               loader="hub_url_image", notes="~1.8M uchen lines, S3 URLs"),
    SourceSpec("tibetan_derge", "openpecha/OCR-Dergetenjur", "tibetan", 2,
               loader="hub_url_image", notes="~845K Derge Tenjur woodblock "
                                             "lines, S3 URLs"),
]}


_PREFERRED_TEXT = ("text", "label", "transcription", "transcript",
                   "sentence", "ground_truth", "gt", "words", "word")


def detect_columns(first_row: dict,
                   spec: SourceSpec) -> tuple[str, str, str | None]:
    """Resolve (image_col, text_col, lang_col) from a sample row.

    Streaming datasets resolve features lazily (ds.features is often None
    before iteration), so detection works on the first decoded row: the
    image column holds a PIL image, the text column a plausible string.
    """
    from PIL import Image as PILImage

    image_col, text_col, lang_col = spec.image_col, spec.text_col, spec.lang_col
    if image_col is None:
        for name, val in first_row.items():
            if isinstance(val, PILImage.Image):
                image_col = name
                break
    if text_col is None:
        str_cols = [n for n, v in first_row.items() if isinstance(v, str)]
        for p in _PREFERRED_TEXT:
            if p in str_cols:
                text_col = p
                break
        if text_col is None and str_cols:
            text_col = str_cols[0]
    if lang_col is None and spec.script is None:
        for cand in ("language", "lang", "lang_code"):
            if cand in first_row:
                lang_col = cand
                break
    if image_col is None or text_col is None:
        raise RuntimeError(
            f"{spec.name}: could not resolve columns from "
            f"{list(first_row)}; set image_col/text_col in the SourceSpec")
    return image_col, text_col, lang_col
