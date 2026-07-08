#!/usr/bin/env python3
"""
Build Hindi legal word list for adapter training.

Sources:
  - Common Hindi words (top frequency)
  - Indian legal terminology in Hindi
  - IPC/BNS section references
  - Court names, procedural terms
  - Numbers and mixed Hindi-English terms

Output: training_data/word_lists/hindi_legal.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Common Hindi words (high frequency in legal documents)
COMMON_HINDI = [
    # Pronouns and basic words
    "यह", "वह", "इस", "उस", "जो", "कि", "और", "या", "से", "के",
    "का", "की", "को", "में", "पर", "ने", "है", "हैं", "था", "थी",
    "थे", "हो", "हुआ", "हुई", "होता", "होती", "होते", "गया", "गई", "गये",
    "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
    "सौ", "हजार", "लाख", "करोड़",

    # Common verbs
    "करना", "होना", "जाना", "आना", "देना", "लेना", "कहना", "बताना",
    "मिलना", "रखना", "चलना", "पाना", "मानना", "सोचना", "समझना",
    "लिखना", "पढ़ना", "सुनना", "देखना", "बोलना",
]

# Legal terminology
LEGAL_HINDI = [
    # Court system
    "न्यायालय", "उच्चतम", "उच्च", "जिला", "सत्र", "दीवानी", "फौजदारी",
    "न्यायाधीश", "न्यायमूर्ति", "पीठासीन", "अधिकारी", "मजिस्ट्रेट",
    "अभियोजन", "बचाव", "अभियुक्त", "अभियोक्ता", "प्रतिवादी", "याचिकाकर्ता",
    "वादी", "प्रतिवादी", "गवाह", "साक्षी",

    # Legal procedures
    "याचिका", "अपील", "पुनर्विचार", "समीक्षा", "रिट", "आवेदन",
    "शिकायत", "प्राथमिकी", "चार्जशीट", "आरोपपत्र", "दोषसिद्धि",
    "बरी", "दण्ड", "सजा", "जमानत", "हिरासत", "गिरफ्तारी",
    "तलाशी", "जब्ती", "कुर्की", "नीलामी",

    # Legal concepts
    "अधिकार", "कर्तव्य", "अनुबंध", "करार", "संविदा", "विवाद",
    "मुकदमा", "वाद", "कार्यवाही", "सुनवाई", "बहस", "तर्क",
    "निर्णय", "आदेश", "फैसला", "डिक्री", "रिपोर्ट", "प्रमाण",
    "साक्ष्य", "गवाही", "शपथ", "हलफनामा",

    # Indian legal codes
    "भारतीय", "दण्ड", "संहिता", "न्याय", "सुरक्षा", "नागरिक",
    "दंड", "प्रक्रिया", "अपराध", "धारा", "उपधारा", "अनुच्छेद",
    "खंड", "अध्याय", "भाग", "अनुसूची", "परिशिष्ट", "संशोधन",

    # Property and civil
    "सम्पत्ति", "भूमि", "मकान", "किराया", "पट्टा", "बिक्री",
    "खरीद", "हस्तांतरण", "विरासत", "उत्तराधिकार", "वसीयत",
    "गिरवी", "बंधक", "ऋण", "कर्ज",

    # Family law
    "विवाह", "तलाक", "भरण-पोषण", "अभिरक्षा", "दत्तक", "ग्रहण",
    "पति", "पत्नी", "संतान", "नाबालिग", "अभिभावक",

    # Administrative
    "सरकार", "प्रशासन", "कार्यालय", "विभाग", "मंत्रालय",
    "अधिसूचना", "परिपत्र", "आदेशपत्र", "राजपत्र",
    "पंजीकरण", "प्रमाणपत्र", "लाइसेंस", "अनुमति",

    # Common legal phrases (as individual words)
    "माननीय", "श्रीमान", "अदालत", "पेशी", "तारीख",
    "हाजिर", "गैरहाजिर", "स्थगित", "खारिज", "मंजूर",
    "स्वीकार", "अस्वीकार", "लागू", "प्रभावी", "तत्काल",
]

# Section numbers and references
SECTION_REFS = [
    # IPC/BNS sections commonly cited
    "302", "304", "306", "307", "323", "324", "325", "326",
    "354", "376", "377", "379", "380", "384", "392", "395",
    "397", "406", "409", "411", "413", "415", "417", "418",
    "420", "427", "452", "456", "467", "468", "471", "489",
    "498", "498A", "500", "506", "509",
    # CrPC sections
    "125", "144", "154", "155", "156", "161", "164", "167",
    "173", "190", "197", "200", "202", "204", "227", "228",
    "239", "240", "241", "245", "246", "250", "311", "313",
    "354", "357", "374", "378", "389", "397", "399", "401",
    "438", "439", "482",
    # Constitution articles
    "14", "19", "21", "25", "32", "44", "51A", "136", "141",
    "142", "143", "226", "227", "243", "300A", "311", "352",
]

# Mixed Hindi-English terms common in legal docs
MIXED_TERMS = [
    "FIR", "IPC", "CrPC", "BNS", "BNSS", "BSA",
    "PIL", "SLP", "CPC", "CBI", "NIA",
    "HC", "SC", "DC", "ADJ", "CMM", "ACMM",
]


def main():
    output_path = Path("training_data/word_lists/hindi_legal.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_words = set()
    all_words.update(COMMON_HINDI)
    all_words.update(LEGAL_HINDI)
    all_words.update(SECTION_REFS)
    all_words.update(MIXED_TERMS)

    # Remove empty strings
    all_words.discard("")

    # Sort for reproducibility
    words = sorted(all_words)

    with open(output_path, "w", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

    print(f"Written {len(words)} Hindi legal words to {output_path}")
    print(f"  Hindi words: {len(COMMON_HINDI) + len(LEGAL_HINDI)}")
    print(f"  Section refs: {len(SECTION_REFS)}")
    print(f"  Mixed terms: {len(MIXED_TERMS)}")

    # Also generate an expanded English legal word list
    eng_path = Path("training_data/word_lists/english_legal.txt")

    ENGLISH_LEGAL = [
        # Court system
        "Court", "Judge", "Justice", "Magistrate", "Tribunal", "Bench",
        "Petitioner", "Respondent", "Appellant", "Plaintiff", "Defendant",
        "Accused", "Prosecution", "Defence", "Witness", "Advocate",

        # Procedures
        "Petition", "Appeal", "Review", "Revision", "Application", "Complaint",
        "FIR", "Chargesheet", "Conviction", "Acquittal", "Bail", "Custody",
        "Arrest", "Search", "Seizure", "Attachment", "Auction",

        # Legal concepts
        "Section", "Article", "Chapter", "Part", "Schedule", "Amendment",
        "Statute", "Act", "Rule", "Regulation", "Notification", "Order",
        "Judgment", "Decree", "Verdict", "Sentence", "Evidence", "Proof",

        # Common
        "Honourable", "Dated", "Filed", "Heard", "Reserved", "Pronounced",
        "Dismissed", "Allowed", "Granted", "Rejected", "Disposed", "Adjourned",
        "Versus", "State", "Union", "India", "Government", "Ministry",
        "District", "Division", "National", "Supreme", "High", "Sessions",

        # Property
        "Property", "Land", "House", "Rent", "Lease", "Sale", "Purchase",
        "Transfer", "Inheritance", "Succession", "Will", "Mortgage",

        # Numbers and references
        "No.", "W.P.", "Crl.", "M.C.", "C.A.", "S.L.P.", "Writ",
    ]

    eng_words = set(ENGLISH_LEGAL + SECTION_REFS + MIXED_TERMS)
    eng_words = sorted(eng_words)

    with open(eng_path, "w", encoding="utf-8") as f:
        for word in eng_words:
            f.write(word + "\n")

    print(f"Written {len(eng_words)} English legal words to {eng_path}")


if __name__ == "__main__":
    main()
