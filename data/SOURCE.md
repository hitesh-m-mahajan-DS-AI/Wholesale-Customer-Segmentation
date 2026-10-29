# Source, attribution and transformations

- **Dataset:** Wholesale customers, UCI dataset 292.
- **Source page:** https://archive.ics.uci.edu/dataset/292/wholesale+customers
- **Direct no-login download:** https://archive.ics.uci.edu/static/public/292/wholesale%2Bcustomers.zip
- **Attribution:** Cardoso, M. (2013). Wholesale customers [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X
- **License:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- **Original archive member:** Wholesale customers data.csv. Saved locally as wholesale_customers.csv with unchanged bytes.
- **SHA-256:** `c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef`.

The source describes 440 wholesale distributor customers with annual category spending in unspecified monetary units. It provides six spending fields plus channel and region codes. Exact observation dates, order frequency, product quantities, prices, costs and profit are unavailable.

Channel codes: 1=Horeca (hotel/restaurant/cafe), 2=retail. Region codes: 1=Lisbon, 2=Oporto, 3=Other. These fields are retained for descriptive profiles only and excluded from clustering.

The original has no missing values or exact duplicate rows; all six spending columns are positive in this version. No rows are removed or imputed. The raw spelling `Delicassen` is preserved; charts display the category as Delicatessen.

The project adds a one-based row ID (`ROW-001` to `ROW-440`), total recorded category spending for reporting, cluster labels and silhouette diagnostics. Row IDs are not source account identifiers. Clustering applies log1p and standard scaling to the six raw category amounts only, so both spending magnitude and category pattern affect distance. It is not a pure proportional-mix analysis.

This is an exploratory fit to the whole cohort. No target labels, customer purchases, dates or synthetic observations are invented, and no held-out or future-customer performance is claimed.
