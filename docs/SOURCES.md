# Sources and provenance

[Source identity](#identity-checks-and-limits) identifies the audited files; it
does not establish eligibility. The authors' [code licence](../LICENSE) and
[weights declaration](../WEIGHTS_LICENSE.md) are separate from dataset rights.
[Submission compliance](#submission-compliance) records the unresolved rules
questions and other remaining checks.

## Official contract

The reader, frame contract and unmodified validator are fixed to
[Metrica Sports commit f8f2629](https://github.com/metrica-sports/more-than-a-hack-2027/tree/f8f2629ff1e0985f81d7e59d8413678514ee1530).
The organisation's 3 November 2026 clarification, ORG-2026-11-03, takes precedence
over older documentation and individual answers. It prohibits comparison of
outputs with supplied video, manual data about that match, and training or tuning
on supplied assets. Supplied data must not be uploaded to AI services.

The selected card records the original components, training commands, input
fingerprints and software versions. It is preserved byte for byte; historical
training commits are provenance records, not commits contained in this delivery.

## External training data

| Source | Matches | Fixed version | Licence | Role |
|---|---:|---|---|---|
| Sportec Solutions / DFL, Bassek et al. (2025) | 7 | [figshare v1](https://doi.org/10.6084/m9.figshare.28196177.v1) | CC BY 4.0 | All components and DFL leave-one-match-out evaluation |
| SkillCorner Open Data | 20 | [4340d274572876239c154c90bc507a9b3250a656](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656) | MIT, copyright SkillCorner | All components, using phases of play |
| SkillCorner 2019/20 Open Data | 9 | [1ed50b775f93075c2a6d00d0d795f13b7b89709e](https://github.com/SkillCorner/opendata/tree/1ed50b775f93075c2a6d00d0d795f13b7b89709e) | MIT, copyright 2020 SkillCorner | Control head of component 3 only; no D supervision |

Cite Bassek et al., *An integrated dataset of spatiotemporal and event data in
elite soccer*, [Scientific Data](https://www.nature.com/articles/s41597-025-04505-y).
Retain the [SkillCorner licence](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/LICENSE)
and attribution when reusing its material. Dataset licences do not license the
team's code or supplied challenge assets. The data themselves are not bundled.

The older SkillCorner reader is required by the selected component's training;
it is not a discarded model or compatibility layer. The `skillcorner_legacy`
format name remains the historical manifest identifier.

For current SkillCorner, H/A follows the team in each provider possession phase.
The reader assigns D to the complement of those phases within each period and
X outside the periods. This is our construction of the dead-ball reference,
rather than an explicit D label supplied with the phases.

## Identity checks and limits

The recorded audit of the base component matched the seven DFL metadata files
to figshare v1 and its seven position files by the custodian's MD5/SHA256 receipt.
For the 20 SkillCorner matches, metadata matched the fixed commit; phase CSVs
matched after accounting for LF/CRLF, and tracking identity matched Git LFS
pointers. These checks identify files; they do not certify every training step.
The component records in the selected card retain their own input fingerprints.

All further training and evaluation must audit source versions, rights, file
hashes, match identity and partition separation. Synthetic tests do not establish
real data quality, and development scores have selection bias.

Apache 2.0 applies to original code. The 24 selected weights have a separate
[Apache 2.0 declaration](../WEIGHTS_LICENSE.md) by the three authors, with
attribution in [NOTICE](../NOTICE) and the complete [SkillCorner MIT notice](../LICENSES/SkillCorner-MIT.md).
Dataset and challenge-asset rights remain separate.

## Submission compliance

This section records how the package addresses the rules; it does not certify
eligibility. The governing
sources are the official [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/RULES.md),
[submission contract](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/SUBMISSION.md),
the organisers' 3 November clarification and the terms accepted at enrolment.
The clarification takes precedence where older guidance differs. The historical
`combined_dead` scenario and three-second reference were selected partly because
the external baseline approximated published full-match scores. The decision to
test finer references was also partly motivated by the published shift-sensitivity
table, computed from challenge ground truth. The selected two-second recipe
retains `combined_dead`. The organisers' reply of 16 November repeated the
external-only training and tuning rules without resolving the specific use of
those published figures. The admissibility of both selection decisions remains
unresolved. The selected candidate applies no fixed D interval.
The [event page](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
also confirms registration dates and the 3–5-member team requirement.

| Requirement | Evidence and limits in this package |
|---|---|
| Algorithmic output; no manual match annotations or output/video comparison | Tracking-only inference; no video input, manual corrections or labels in the delivered pipeline. Software checks cannot prove historical conduct. |
| Training and tuning on external data with usage rights | Three fixed sources, provider-label provenance and training-only transformations; held-out match identity checked across copies and SkillCorner formats. Both historical selection questions described above remain unresolved. |
| No prohibited data about the supplied match | No private ground truth or video used by inference; official public labels restricted to pipeline checks. The candidate applies no fixed D interval. |
| Exact regeneration of 199,319 `frame,label` rows | CPU inference, repeat check, independent card/weight pins, expected CSV SHA-256 and unmodified official validator. These check bytes and format, not sports accuracy. |
| Accessible and inspectable method | Source, dependencies, 24 selected weights with the authors' Apache 2.0 declaration, training/evaluation/inference commands and technical document. Third-party rights remain separate. |
| No redistribution of restricted assets or reconstructive exports | Data, CSVs and generated artifacts ignored; tracked export checker rejects forbidden paths and missing required files. Only the selected weights are bundled. |
| Technical explanation of assets, labels, possession rules and limitations | Report source `docs/report/main.tex`, with disclosed development selection bias. |

A format PASS cannot settle historical admissibility. If the organisers reject the historical protocol choice, agree an external-only protocol, retrain and reevaluate every affected component, then regenerate the whole delivery. If weight distribution is not permitted, resolve rights or replace the affected training sources and retrain; the code licence alone is insufficient.

Eligibility also depends on team registration and the terms accepted at enrolment.
The software checks do not establish either of them or certify compliance with
additional obligations in those terms.
