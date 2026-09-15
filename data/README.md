# Data Directory

## Primary dataset (NOT included)
`response.xlsx` - KNUST Student Cybersecurity Behaviour Survey (n=1,250).

This file is **not distributed** because the KNUST ethics approval
(HURRRESEC/AP/[PENDING]/26) restricts sharing of the raw responses.
Access may be requested from the corresponding author subject to
institutional approval.

### Expected structure
- Column 0: timestamp/consent
- Columns 1-5 (Q1-Q5): demographic + causal predictors (features)
- Columns 6-13 (Q6-Q13): feature-half behaviour items (features)
- Columns 14-21 (Q14-Q21): label-half behaviour items (build the risk label)

Five-point items encoded Never=0, Rarely=1, Sometimes=2, Often=3, Always=4.
Risk label from sum of Q14-Q21: Low>=22, Medium 12-21, High<12.

## Public transfer datasets (download separately)
1. **Moroccan HTTPS Awareness** - Mendeley Data, DOI 10.17632/g3fbm5pwm6
   File: `Encoded_collected_dataset.csv`
2. **WUSTL-EHMS-2020** - https://www.cse.wustl.edu/~jain/ehms/index.html
   File: `wustl-ehms-2020_with_attacks_categories.csv`

Place any dataset in this directory before running the corresponding script.
