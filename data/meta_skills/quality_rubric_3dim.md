The following principles determine whether an extracted skill will actually improve agent performance. These three dimensions were identified through per-dimension validation as having the strongest correlation with downstream utility (65-66% predictive accuracy each).

**Priority: domain-specific failure knowledge over general advice.**

### Three Quality Principles

1. **Failure Mechanism Encoding**: The skill must identify SPECIFIC reasons why agents fail in this domain — not generic warnings. Name the concrete failure condition and the causal chain that leads to task failure. Example: "The API returns paginated results capped at 100 — the agent assumes all data is in the first response and silently drops rows beyond page 1." Anti-example: "handle errors carefully."

2. **Actionable Specificity**: Provide step-by-step procedures referencing domain objects, tools, or APIs. The procedure must be concrete enough that an agent can execute it without further interpretation. Example: "After mutation, re-query the object to get the server-assigned ID before referencing it in subsequent calls." Anti-example: "decompose into smaller steps."

3. **High-Risk Action Blacklist**: Explicitly forbid specific common-but-harmful action patterns with concrete reasons why they fail. These are patterns that LOOK correct but reliably cause task failure. Example: "NEVER pass user-supplied strings directly as regex patterns — escape special characters first, or the match silently fails on inputs containing '.', '*', etc." Anti-example: "be careful with dangerous operations."

### Anti-patterns to AVOID

- Generic process advice that could apply to any domain ("verify results", "be systematic")
- Polished, comprehensive-sounding guidance that lacks concrete failure knowledge
- Abstract principles without executable procedures
- Trying to cover every possible edge case — focus on the TOP failure modes

**Key insight**: A rough, narrow skill that encodes one domain-specific failure mechanism with an executable fix is MORE valuable than an elegant, well-structured skill full of generic best practices.
