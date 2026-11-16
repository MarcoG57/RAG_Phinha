# Third-party notices

Original RAG_Phinha code is licensed under [Apache 2.0](LICENSE). Copyright and
credit for Marco, Casti and Ariel are retained in [NOTICE](NOTICE).

## Metrica Sports official starter

The pinned external checkout supplies the official input contract, reader
metadata and validator. Its source is [commit f8f2629](https://github.com/metrica-sports/more-than-a-hack-2027/tree/f8f2629ff1e0985f81d7e59d8413678514ee1530).
The complete [Metrica Sports licence](LICENSES/Metrica-Sports-MIT.md) is retained
unchanged. The checkout remains outside this repository under `.official/` and
keeps its original notices. No alternate baseline model is bundled.

Its code licence grants no rights to competition tracking, video, tags or other
supplied assets. External dataset versions, attribution and provenance limits
are listed in [SOURCES](docs/SOURCES.md). DFL and SkillCorner data are not bundled.

Dependencies are installed separately from the pinned requirements. This source
package does not redistribute their binaries or replace their notices. The
team's code licence does not automatically license trained weights or datasets.
The selected B4T weights have their own [licence declaration](WEIGHTS_LICENSE.md).
The complete [SkillCorner MIT notice](LICENSES/SkillCorner-MIT.md) is retained
for the two pinned training sources; attribution to DFL and SkillCorner is in
[NOTICE](NOTICE). No training dataset is redistributed in this package.
