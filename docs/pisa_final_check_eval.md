# PISA Final Check acceptance cases

Source snapshot: indexed page 4, **PISA 2025 FT - Questionnaires Final Check**,
revision `291833327c434fefbbfd3698c77d29ab` (checked 2026-09-23).
These are evaluation labels, **not** special-case rules for production code.
Re-check the source and update this file if the page revision changes.

All questions below are asked while page 4 is open. For the three detail
questions, `search_current_page` must be used; a partial whole-page summary is
not sufficient evidence. The final trace must contain only page-4 chunks and
the answer must not claim that information absent from a sampled overview is
absent from the full page.

1. **How should a verifier implement a correction in a trend segment in OmegaT
   so the changes are saved?** Expected: use **Create Alternative
   Translation**, enter the updated translation, press **Ctrl+S**, then **F5**
   and verify the change. Do not substitute committing target files and
   closing the project for these steps. Relevant indexed chunks: section 2,
   children 20–21.
2. **If a selected erratum was missing and the verifier corrects it in OmegaT,
   what exact text must be entered in the QAS column AD?** Expected: enter
   **NOT OK** and **“Erratum REFXXX implemented by VER”**, replacing `REFXXX`
   with the erratum's reference ID. Do not say that no exact wording is given.
   Relevant indexed chunks: section 2, children 23–24.
3. **Can the verifier use the “Sort by” function in Excel to group the pale-blue
   cells in column AD? Why or why not?** Expected: **No**; the page warns that
   “Sort by” is not reversible. Filtering column AD by cell color is allowed,
   and the filter can be cleared. Do not claim that the page is silent on this.
   Relevant indexed chunks: section 2, children 8 and 10.
4. **Summarize this page.** An overview may be explicitly partial and cite only
   page 4. It is not a substitute for any of the three detail questions.

Release gate for this set: all three detail answers must contain their expected
facts and none of the forbidden claims. Inspect both the displayed answer and
`evidence_json.final_chunk_ids`. Record the provider, model, input/output
tokens, and latency for the run. This set alone is too small to establish
general production quality; combine it with the broader holdout set in
`evidence_contract.md`.
