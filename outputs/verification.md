# Verification record - Project 9

- Executed the complete source audit -> six-feature log/scaling -> fixed K-means -> silhouette/profile exports -> chart workflow successfully.
- Environment: Windows, Python 3.12, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 and matplotlib 3.11.2.
- Public original SHA-256 verified: `c3d018c643565b85cee733c4a2ac76dd76e080e857cb23f0ccfcc2e15a6c17ef`. Bundled source bytes remain unchanged under a local filename; attribution and newline-preservation rules are included.
- All **440** customers were retained, with no missing values or duplicate source rows. Only **six** spending columns enter the model; source channel/region, added IDs and derived spending totals are excluded.
- Fixed model: **three clusters**, `random_state=42`, `n_init=20`, Lloyd algorithm. Cluster sizes **145, 212 and 83** reconcile to 440.
- Measured internal silhouette: **0.2582784274194209**. Inertia in standardized log space: **1553.4127220608457**. **22** customers have negative silhouettes; labels are arbitrary and the overall score does not establish predictive accuracy or business value.
- Six unittest checks passed in **0.453 seconds**: source/checksum integrity; invalid spending rejection; repeatable assignment/preprocessing; exclusion of metadata/derived totals; profile reconciliation in original units; independent silhouette distance arithmetic.
- Independently calculated each customer's within-cluster mean distance and minimum other-cluster mean distance from a full Euclidean pairwise matrix. Both individual silhouettes and their mean match scikit-learn's output.
- Changing channel, region and derived total values while preserving spending inputs left assignments unchanged. Counts, shares and median-ratio formulas reconcile to raw grouped records.
- Re-ran main.py from the portfolio root; all **11** generated CSV checksums and the silhouette score were unchanged. Final application timer: **0.914 seconds**, excluding interpreter startup/top-level imports; runtimes vary by laptop.
- Both figures were visually checked for readable labels, consistent cluster colors, raw-customer counts and correctly identified feature/profile transformations. The scatter is explicitly a two-feature view, not the clustering input space in full.
- No alternate cluster-count search, held-out/future-customer evaluation, temporal demand inference, measured business action, paid service, GPU or other project import was used.

The result is an exploratory full-cohort segmentation with limited separation, not validated commercial customer classes or an operational forecasting system.
