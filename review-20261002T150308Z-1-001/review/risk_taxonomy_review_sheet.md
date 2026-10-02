# Risk taxonomy: face-validity review

**What this is.** The 41 clause categories of the CUAD dataset, each assigned a risk level of
High, Medium or Low **from the weaker party's point of view** (the side with less bargaining
power -- typically the smaller company or the individual signer), together with the one-line
explanation our application shows the user.

**What we are asking.** For each row, please say whether you agree with the level. If you do not,
write the level you would give and, if you can, one line on why. You do not need to comment on the
wording of the reason unless it is misleading. There are no right answers we are fishing for --
disagreements are the useful part, and we will report how many there were either way.

**Time needed.** About 20-30 minutes.

**How many reviewers.** One is enough for a face-validity check. **Two or three is much better**,
because it lets us report chance-corrected agreement (Cohen's or Fleiss' kappa) rather than a raw
percentage -- and with 22 of 41 categories sitting at Medium, a raw percentage flatters any reviewer
who defaults to the middle. Give each reviewer their own blank copy and do not let them confer.

**When the sheets come back**, run:

    python review/agreement.py reviewer1.csv reviewer2.csv

which prints per-reviewer agreement, Cohen's kappa against our taxonomy, reviewer-vs-reviewer kappa,
and Fleiss' kappa across everyone, plus the list of disputed categories.

**Note.** The level is assigned per *category*, not per clause: it says 'a clause of this type is
typically worth attention', not 'this particular wording is dangerous'. Please judge it on that basis.

| # | Category | Our level | Our reason | Agree? | Your level | Comment |
|---|---|---|---|---|---|---|
| 1 | Non-Compete | High | Limits your future income and freedom to work. |  |  |  |
| 2 | Uncapped Liability | High | Unlimited damages -- no ceiling on what you can owe. |  |  |  |
| 3 | IP Ownership Assignment | High | You lose ownership of a core asset for good. |  |  |  |
| 4 | Exclusivity | High | Locks you to a single source or buyer. |  |  |  |
| 5 | Most Favored Nation | High | You must always give the other side your best terms. |  |  |  |
| 6 | Liquidated Damages | High | Fixed penalties trigger automatically on breach. |  |  |  |
| 7 | Irrevocable or Perpetual License | High | A grant you can never take back. |  |  |  |
| 8 | Minimum Commitment | High | You are forced to buy or deliver a minimum amount. |  |  |  |
| 9 | Renewal Term | Medium | Auto-renewal can extend the contract unexpectedly. |  |  |  |
| 10 | Notice Period to Terminate Renewal | Medium | Miss the window and you are locked in for another term. |  |  |  |
| 11 | Competitive Restriction Exception | Medium | Carve-outs to competitive limits -- read carefully. |  |  |  |
| 12 | No-Solicit of Customers | Medium | Restricts approaching the other side's customers. |  |  |  |
| 13 | No-Solicit of Employees | Medium | Restricts hiring the other side's staff. |  |  |  |
| 14 | Non-Disparagement | Medium | Limits what you can publicly say. |  |  |  |
| 15 | Termination for Convenience | Medium | The other side can walk away with notice. |  |  |  |
| 16 | ROFR / ROFO / ROFN | Medium | First-refusal rights can constrain your options. |  |  |  |
| 17 | Change of Control | Medium | A sale or merger may trigger termination. |  |  |  |
| 18 | Anti-Assignment | Medium | You cannot transfer the contract without consent. |  |  |  |
| 19 | Revenue/Profit Sharing | Medium | You must share a slice of revenue or profit. |  |  |  |
| 20 | Price Restrictions | Medium | Limits what prices you may set or change. |  |  |  |
| 21 | Volume Restriction | Medium | Caps how much you can sell or distribute. |  |  |  |
| 22 | Joint IP Ownership | Medium | IP is co-owned -- control is shared, not sole. |  |  |  |
| 23 | Non-Transferable License | Medium | The licence cannot be passed on. |  |  |  |
| 24 | Affiliate License (Licensor) | Medium | Licensor's affiliates are pulled into the grant. |  |  |  |
| 25 | Unlimited / All-You-Can-Eat License | Medium | Broad, unlimited licence scope -- check who benefits. |  |  |  |
| 26 | Source Code Escrow | Medium | Source is held by a third party under conditions. |  |  |  |
| 27 | Post-Termination Services | Medium | Obligations continue after the contract ends. |  |  |  |
| 28 | Audit Rights | Medium | The other side can inspect your records. |  |  |  |
| 29 | Cap on Liability | Medium | Caps recovery -- may under-compensate real losses. |  |  |  |
| 30 | Covenant Not to Sue | Medium | You waive the right to sue over listed matters. |  |  |  |
| 31 | Document Name | Low | Identifies the agreement -- informational. |  |  |  |
| 32 | Parties | Low | Names who is signing -- informational. |  |  |  |
| 33 | Agreement Date | Low | The signing date -- informational. |  |  |  |
| 34 | Effective Date | Low | When terms start -- informational. |  |  |  |
| 35 | Expiration Date | Low | When the contract ends -- informational. |  |  |  |
| 36 | Governing Law | Low | Which jurisdiction's law applies -- neutral. |  |  |  |
| 37 | License Grant | Low | You receive usage rights -- generally favorable. |  |  |  |
| 38 | Warranty Duration | Low | States how long the warranty protects you. |  |  |  |
| 39 | Insurance | Low | Required coverage -- a protection. |  |  |  |
| 40 | Affiliate License (Licensee) | Low | Extends rights to your affiliates -- favorable. |  |  |  |
| 41 | Third Party Beneficiary | Low | Names an outside beneficiary -- informational. |  |  |  |

Reviewer name / role: ______________________  Date: ____________