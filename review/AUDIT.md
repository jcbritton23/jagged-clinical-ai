# Accuracy and completeness review

## Source of record

The historical master CSV contains 1,470 distinct rows. All 1,320 benchmark rows reconcile with the long-form table by benchmark, raw entity name, dimension, printed score and provenance. The other 150 rows are supplemental. The 26 distinct benchmark labels include adjacent general benchmarks, and several labels refer to variants of one instrument. The 26 is a count of labels.

The separately supplied workbook and master CSV matched their corresponding archive members byte for byte. The archive’s 143 manifest entries passed size and SHA-256 verification. The workbook is an earlier snapshot. All 1,059 of its benchmark rows match the later CSV by benchmark, raw name, size, dimension, source label and printed score. The 261 additional CSV rows are the EmoBench-Reddit extraction. The CSV is the release source of record. For the 324 EmoBench-M rows the release pins the provenance link to paper version 3. The original benchmark values and labels are retained.

## Source audit results

Every benchmark and supplemental row has a source check and disposition. The ledgers identify the original CSV row, recorded value, reference value when available, and source locator. Row numbers count the header as row 1. [Benchmark ledger](primary-source-checks.csv) · [Supplemental ledger](supplemental-source-checks.csv) · [Coverage by benchmark](coverage.csv)

| Status | Benchmark records | Supplemental records |
| --- | ---: | ---: |
| Match | 1,262 | 138 |
| Qualified | 41 | 9 |
| Differs | 4 | 3 |
| Invalid | 4 | 0 |
| Unresolved | 9 | 0 |
| Total | 1,320 | 150 |

**Match** means the reported value or qualitative claim reconciles with the named source. **Qualified** means the value matches and needs a specific interpretation, checkpoint or protocol note. **Differs** means the cited condition has a different value. **Invalid** means the archived field cannot serve as a score. **Unresolved** means an exact source for the archived value could not be established. These are counts of archive records, which can include multiple representations of one result.

| Finding | Disposition |
| --- | --- |
| CounselingBench | The paper reports 78.0% for GPT-4o zero-shot and 76.5% for GPT-4o zero-shot with self-consistency. The archive has 59.9% and 62.5%. Exact-value searches, the available preprint and proceedings tables, and the authors’ public repository did not establish the nine other model-setting combinations in the archive. Exclude all 11 archived rows from a source-verified analysis unless their exact evaluation becomes identifiable. Document the paper's two GPT-4o results under their original settings if incorporating them. [Paper, Table 2](https://aclanthology.org/2025.findings-naacl.418.pdf) |
| MoralBench | Two MFV totals differ from Table 1. Llama 2 is 51.4 rather than 52.6, and Gemma 1.1 is 51.8 rather than 44.4. Use the paper totals. MFV and MFQ scores measure responses to moral items on the paper's scoring scheme. [Paper](https://arxiv.org/pdf/2406.04428) |
| MentalChat16K | Four rows contain 10 because the import interpreted the denominator of “score / 10” as a second judge rating. Exclude those rows. Table 3 also distinguishes fine-tuned variants and the Mixtral base and instruction models, as recorded in the ledger. [Paper, Table 3](https://arxiv.org/html/2503.13509#S4.T3) |
| SIRI-2 supplemental averages | Three approximate readings differ from the arithmetic means across the six Appendix A3a configurations. The archived Gemini 2.5 Pro value of 38.5 is 40.3667, Claude 3.5 Sonnet 43.5 is 41.2167, and GPT-4o 46.5 is 45.55. The article's headline scores already use the exact appendix aggregation. All 54 configuration rows and 27 reasoning rows were checked against the publicly posted appendix, which matches the copy in the original archive byte for byte. [Preprint and appendix](https://www.medrxiv.org/content/10.64898/2026.03.20.26348900v1.supplementary-material) |
| EmoBench-Reddit | All 252 component scores match Table 1 and the nine review-derived means reproduce. GPT-4o’s mean is 0.75464, which rounds to 0.75 in the case table. [Paper](https://arxiv.org/abs/2509.11101) |
| EmoBench-M | All 324 scores match version 3 Table 3. Version 4 uses the distinct Video-LLaMA2.1-16F checkpoint in place of the archived Video-LLaMA2-16F configuration. [Version 3, Table 3](https://arxiv.org/html/2502.04424v3#S3.T3) · [Version 4, Table 3](https://arxiv.org/html/2502.04424v4#S3.T3) |
| MMLU-Pro | All 66 numbers match the official leaderboard's overall and psychology columns for the 33 model labels. This confirms Gemini 2.5 Pro Experimental 03-25's 0.8726 psychology result. [TIGER-Lab leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) |
| EmoBench, MoralChoice and theory of mind | The imported results match the runs in *Evaluating Large Language Models with Psychometrics*. The paper identifies GPT-4 Turbo and gpt-3.5-turbo-0125 for the imported generic model labels. MoralChoice high-ambiguity values are average choice scores and have no accuracy interpretation. [Psychometrics paper, Tables 2, 18–19 and 25](https://openreview.net/pdf?id=31UkFGMy8t) |

Three raw benchmark records have text with an empty numeric field. Two correctly preserve paired classification metrics. The third is PaLM's 54.6% five-shot RACE-h score, which remains available as text and in the audit. Some prompt-table rows repeat workbook scores. Across the master data, historical provenance fields and recorded results are preserved for reproducing the article. Use the audit ledger for source URLs and the disposition before reusing each score.

## Published claims

| Population | AA matches | Pearson r with psych score | Pearson r with family score range |
| --- | ---: | ---: | ---: |
| At least four benchmark variants | 13 of 15 | .4843 | −.0597 |
| Also observed in at least four capability families | 9 of 10 | .6141 | −.2227 |

The article prose uses the first population. The comparison figure uses the second. All nine displayed psych scores and ranges agree with the original data at published precision. The open-weight parameter-size correlation also reproduces at .7705 across five records using the archived metadata. Model variants and parameter definitions affect this comparison, especially the generic Mixtral label. Release timing gives .289 using exact release dates across 14 available records.

The published GPT-4o profile spans seven families and 11 distinct benchmark variants. GPT-4 spans 12 variants across four families. Gemini 2.5 Pro ties GPT-4o at seven families. The GPT-4o case table omits MMLU-Pro Psychology, which contributes two dated snapshot rows to the knowledge family.

The SIRI-2 values 45.55 and 40.37 are mean raw discrepancies from expert ratings, with lower scores indicating greater agreement. The review transforms them to 54.45 and 59.63 for plotting. Reading 45.55 as a crisis-safety percentage reverses the interpretation. The source project used the 2026 preprint appendix, Tables A3a and A4a. Repeated-rating consistency is distinct from following a patient through a multi-turn exchange. The benchmark map’s MoralChoice link points to an unrelated economics paper. Cite Scherrer et al. for the benchmark and *Evaluating Large Language Models with Psychometrics* for the imported results. [MoralChoice paper](https://arxiv.org/abs/2307.14324) · [Psychometrics paper](https://openreview.net/pdf?id=31UkFGMy8t) · [SIRI-2 source](https://www.medrxiv.org/content/10.64898/2026.03.20.26348900v1)

## Impact on the published results

In a temporary copy, I removed the 11 CounselingBench rows and four invalid MentalChat16K rows, substituted the two primary-source MoralBench totals, and rebuilt the article analysis. The matched cohorts stayed at 13 and nine models. The correlations between the AA index and the overall psychology score moved from .4843 to .4827 and from .6141 to .6091, respectively. Both round to the published two-decimal values. The correlations with the family ranges were unchanged. GPT-4o moved from 74.1 to 74.5, with 10 benchmark variants across seven families. Gemini 2.5 Pro remained at 72.4. This check isolates the score defects and missing CounselingBench evidence. The qualified rows and the historical model aliasing and aggregation rules still apply.

## Rebuild differences

The original percentile sort treated zero as missing. The packaged builder sorts zero as a numeric score. This changes five historical Theory of Mind Timeline percentile values in the derived tables. The article figures use transformed headline scores, so the published values and both correlations are unaffected. The archived comparison SVG is retained. On the tested Python version the rebuilt GPT-4 bar rounds to a width 0.1 SVG units larger, with the same printed score.

## Scoring and coverage limitations

The observed analysis includes 325 headline rows and 254 entity–family means across nine families. Within each observed family the script averages headline rows, then gives each available family equal weight. Jaggedness in the article is the difference between highest and lowest observed family means. Missing families stay missing. Different coverage can change either result.

The plot clips its transformed scores to 0–100. CBT knowledge uses the reported human baseline of 90.7 as its upper anchor. HEART maps 11.2 to zero and 53.0 to 100. MindEval maps the endpoints of the 1–6 rubric to 0–100. These conventions do not establish equivalence between instruments. For example, GPT-4o’s HEART win rate of 48.5 becomes 89.23 under the article’s anchor rule.

There are 18 repeated model–benchmark-variant keys beyond their first row in the headline data after alias merging. The historical family means include all of these rows while benchmark coverage counts distinct labels. Some represent distinct checkpoints, source overlaps or framework criteria. The heatmap also pools Llama 3 and 3.1 8B visually. Several reference entities are tagged as models upstream, and some parameter counts for closed models are visual-order estimates. Missing cells and repeated variants limit comparisons across models. Keep raw names, reference status and version information when reusing these data. The current literature and current model capabilities extend beyond this historical review.
