# landmark_id : (name, tissue_depth_row_name, count)

LANDMARKS = {
    1:  ("Supraglabella", "Supraglabella", 1),
    2:  ("Glabella", "Glabella", 1),
    3:  ("Nasion", "Nasion", 1),
    4:  ("End of Nasals", "End of Nasals", 1),
    5:  ("Mid Philtrum", "Mid Philtrum", 1),
    6:  ("Upper Lip Margin", "Upper Lip Margin", 1),
    7:  ("Lower Lip Margin", "Lower Lip Margin", 1),
    8:  ("Chin Lip Fold", "Chin Lip Fold", 1),
    9:  ("Mental Eminence", "Mental Eminence", 1),
    10: ("Beneath Chin", "Beneath Chin", 1),

    11: ("Frontal Eminence", "Frontal Eminence", 2),
    12: ("Supraorbital", "Supraorbital", 2),
    13: ("Suborbital", "Suborbital", 2),
    14: ("Inferior Malar", "Inferior Malar", 2),
    15: ("Lateral Orbit", "Lateral Orbit", 2),
    16: ("Zygomatic Arch", "Zygomatic Arch Midway", 2),

    # 17 skipped

    18: ("Gonion", "Gonion", 2),
    19: ("Supra M2", "Supra M2", 2),
    20: ("Occlusal Line", "Occlusal Line", 2),
    21: ("Infra M2", "Infra M2", 2),
}

SKIP_LANDMARKS = {17}
