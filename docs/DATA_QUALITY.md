# Data quality findings and treatments

The supplied files contain mixed timestamp styles (ISO, day-first, written month, and Unix time), duplicate records, inconsistent casing, unit-bearing numbers, multiple Boolean encodings, currency symbols, and missing values.

Key treatments:

- Drop exact duplicates and retain the last record for duplicate primary identifiers.
- Parse heterogeneous timestamps with explicit Unix handling.
- Convert grams/milligrams to kilograms and strip distance/currency units.
- Normalize categorical case and collapse equivalent fleet statuses.
- Convert Yes/No, Y/N, true/false, and 1/0 into binary values.
- Convert Fahrenheit weather readings to Celsius.
- Infer missing review sentiment only from rating: 1-2 negative, 3 neutral, 4-5 positive.
- Preserve informative missing fields and impute inside model pipelines rather than globally.
- Flatten GPS waypoints into a separate relational table.

The original raw files remain unchanged.

