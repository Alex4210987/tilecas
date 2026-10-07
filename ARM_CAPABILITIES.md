# Comparison configurations

| Configuration | Representation and context |
|---|---|
| High | Editing remains in the high-level representation |
| Low | Editing remains in native source |
| Fixed | Independent High agent, validated export, then a fresh Low agent without the High conversation or history |
| Unrestricted | AKO's existing representation choices within the platform contract |
| Random | Frequency-matched random action choice; after Rollback is sampled, the same agent selects a retained correct High source and records its rationale. Lower uses the current correct High |
| AKO+TileCas | One agent chooses source candidates, levels, diagnosis, and continued search using retained history |
| KDA / KDA+TileCas | KDA draft/plan/implementation/evidence workflow, with TileCas policy added in the combined configuration |
| w/o Diagnosis | Removal of Cross-Level Diagnosis |
| w/o Scheduling | Fixed High-to-Low progression |
| w/o Tree | Removal of editable alternative historical branches while retaining the incumbent |

The definitions follow PDF Sections 4.1 and 4.4. The source includes platform-specific execution, correctness, timing, conversion, and restoration support. Launch records specify the case-level execution settings.

KDA rows describe the comparison arms in the result tables. Local KDA integration is included; see `KDA_REFERENCE.md` for the external upstream version and invocation commands.


`RANDOM_ROUTING.md` describes Random action selection and retained-source restoration.
