# Review: TN-Check: CSL Model Checking on Tensor-Train CME

**Reviewer:** Prof. Marcus Chen (Machine Learning & Statistical Learning Theory, Stanford)
**Score: 5/10**
**Expertise Weight: 0.2** (limited alignment; this is primarily a verification/numerical methods project)

---

This project is largely outside my area of expertise. My comments are limited to the optional ML component (adaptive rank controller) and the general evaluation methodology.

**Strengths.** The evaluation plan is thorough and fully automated. The pre-build validation gate with explicit go/no-go criteria is exemplary scientific methodology. The comparison against established baselines (PRISM, Storm, STAMINA) on the same hardware and models provides a fair evaluation framework.

**Weaknesses.** The mention of an MLP rank controller (deferred to follow-up work) is appropriately scoped as optional, but the description provides no detail on what the MLP would predict, what features it would use, or how it would be trained. The greedy-doubling heuristic for adaptive rank selection is a reasonable baseline but is unlikely to be optimal — a learned controller could potentially reduce computational cost significantly. However, this is a minor point for what is fundamentally a verification/numerical methods contribution. The evaluation could benefit from a more systematic characterization of which biological model features predict TT compressibility — this would be amenable to ML-based analysis (e.g., predicting TT rank from network topology features).

**Verdict.** A specialized verification contribution that I am not well-positioned to evaluate. The evaluation methodology is sound. The ML component is appropriately deferred. I defer to the formal methods and computational mathematics reviewers.
