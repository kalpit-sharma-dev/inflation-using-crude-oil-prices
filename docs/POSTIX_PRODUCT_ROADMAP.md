# Postix Product Roadmap

**Inspired by a full feature audit of [Makoro](https://makoro.co/) — Manufacturing ERP for SME factories**  
**Target product:** [Postix](https://postix.in/) — The AI Marketing Operating System  
**Date:** 13 August 2026  
**Status:** Strategy document for product, engineering, and GTM

---

## 1. Executive summary

Makoro and Postix sit in different industries. Makoro is a factory ERP (sales order → BOM shortfall → work order → inventory → dispatch → receivable). Postix is an AI Marketing OS (create → publish → advertise → attribute → optimize). Copying Makoro’s shop-floor modules into Postix would dilute the product and confuse buyers.

What *is* transferable is Makoro’s product architecture: **one connected operating system, one intelligence layer, one confirm-before-write copilot, and one action that updates three modules.** That is the pattern Postix should steal.

This roadmap answers three questions:

1. Which Makoro features can Postix build as-is, adapt, or should refuse?
2. What does the resulting Postix look like when those patterns are applied to marketing?
3. In what order should we ship, and how do we know it worked?

**Recommendation:** Do not become a factory ERP. Become the marketing equivalent of what Makoro is for factories — a system of record *and* a system of decision — so an agency owner can ask “will this week hit target?” and get an intervention, not a dashboard.

---

## 2. Product snapshots

### 2.1 Makoro today

| Layer | What it is |
| --- | --- |
| Positioning | Manufacturing ERP for small and mid-sized Indian factories |
| Core loop | Add partners → define products & BOM → take sales order → stock → produce → dispatch → get paid |
| Modules | Fulfillment, Production, Inventory, Dispatch, Finance, Workforce, Factory Floor, Intelligence, Copilot |
| Cross-cutting | Workorder printouts, itemized labels, WhatsApp notifications, multi-unit, RBAC, CSV everywhere, Tally + Zoho Books, Android app |
| AI | Copilot: type/speak English or Hinglish → draft entry → user confirms → logged write |
| Intelligence | Real-time synthesis: production velocity, inventory health, financial clarity, machine load |
| Pricing | Free forever (10 entries/section) · Pro from ~₹5,417/month billed yearly · dedicated migration person on every paid plan |
| GTM | Custom implementations, partner program, WhatsApp-first sales |

### 2.2 Postix today

| Layer | What it is |
| --- | --- |
| Positioning | AI Marketing Operating System for agencies and SMBs |
| Core loop | Create → Publish → Advertise → Measure → Optimize |
| Modules | Social scheduling (30+ networks), Calendar, Campaigns, Inbox, Approvals, AI Agent, Comet Attribution, Comet Ads Manager, Analytics |
| Cross-cutting | MCP + REST API + CLI, webhooks, n8n/Make, white-label, audit trail, self-host Go binary, Razorpay billing |
| AI | In-app Agent for posts/images/UGC video; BYOK; human-in-the-loop approvals; MCP for Cursor/Claude |
| Intelligence | Comet: first-party pixel, 8 MTA models, CAPI (Meta/Google/TikTok/LinkedIn), attributed ROAS, audience sync |
| Pricing | Social plans + optional Comet add-ons (USD $29–$99 band; INR available); 25% Comet discount for Social subscribers |
| GTM | Cloud + self-host, public docs, compare/alternatives pages, affiliate |

### 2.3 The honest overlap

Almost none of Makoro’s *domain objects* (BOM, GRN, e-way bill, wage rate, spare parts, machine registry) belong in Postix.

Almost all of Makoro’s *operating-system patterns* do:

- Shared event bus: one write fans out to inventory, finance, and notifications
- Confirm-before-write copilot that drafts, never silently saves
- Status machines that operators can read from across the room
- Aging, alerts, and “what to do in the next 15 minutes”
- Public no-login links for the people outside the system (clients)
- Spreadsheet-speed data entry for people who live in Excel
- Physical/operational artifacts (printouts, WhatsApp, labels)
- Free-to-explore with a hard but honest cap

---

## 3. Feature transferability matrix

Legend:

- **Adopt** — build the same capability, marketing-native
- **Adapt** — keep the pattern, change the objects
- **Already have** — exists in Postix; deepen rather than invent
- **Do not build** — factory-specific; would make Postix a worse marketing product

### 3.1 Operations / Fulfillment

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Sales orders with line items, rates, delivery dates | **Adapt** | **Client briefs / retainers / campaign orders** — a signed demand object the rest of the OS plans against |
| BOM shortfall (“exactly what to buy or make”) | **Adapt** | **Campaign BOM** — required assets (copy, image, video, landing page, UTM, pixel, CAPI event) vs what exists |
| Purchase orders to vendors | **Adapt** | **Vendor / freelancer POs** — designer, editor, UGC creator, media buyer |
| Order status draft → confirmed → closed | **Adapt** | Brief status: Draft → Scoped → In production → Live → Closed |
| Value-by-client visibility | **Adopt** | Revenue and pipeline by client / brand / workspace |
| Flows into work orders with no re-entry | **Adopt** | Confirmed brief auto-creates campaign + calendar slots + asset tasks |

### 3.2 Production

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Digital work orders (outputs + consumption) | **Adapt** | **Creative work orders** — outputs (posts, ads, landing pages) + consumption (brand kit, footage, offers) |
| Spreadsheet-style batch form | **Adopt** | Bulk calendar composer: one row per post, fill-down brand/campaign/UTM |
| Batch add (one batch, many items) | **Adapt** | One content batch (theme/offer) → many network-native variants |
| Quick-add catalog inline | **Already have / deepen** | Inline create offer, product, or brand asset without leaving the composer |
| FIFO deduction + stock check before complete | **Adapt** | Block publish if required assets, approval, or pixel are missing |
| A4 workorder printouts | **Adapt** | Printable / PDF **campaign brief** and **daily run sheet** for the studio or client workshop |
| Unit economics per batch | **Adapt** | Cost and attributed revenue per campaign / content batch |

### 3.3 Inventory

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Live register (qty, value, batch, movement) | **Adapt** | **Creative inventory** — every asset with usage count, last used, expiry, brand, campaign |
| Low-stock / zero-stock alerts | **Adapt** | Pipeline health: “LinkedIn has 2 approved posts in the next 14 days” |
| Party-linked batches | **Adapt** | Every asset tagged to a client / brand |
| Deletion audit with reason | **Already have / deepen** | Require a reason on destructive deletes; surface in movement history |
| CSV import/export | **Adopt** | CSV export on every table; import for posts, assets, clients |
| Itemized labels | **Adapt** | Asset IDs + QR on creative files; “scan to see where this ran and what it earned” |
| Movement history | **Adapt** | Asset ledger: created → used in post → used in ad → retired |

### 3.4 Dispatch

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Full dispatch record (client, items, vehicle, LR, e-way) | **Do not build** (logistics fields) / **Adapt** (the record) | **Publish packet**: client, assets, networks, UTMs, CAPI event, scheduled window |
| Status Draft → Packed → Dispatched → In Transit → Delivered | **Adapt** | Draft → Approved → Scheduled → Publishing → Live → Attributed |
| WhatsApp to client on every dispatch | **Adopt** | WhatsApp/email to client when a campaign goes live or a report is ready |
| Inventory auto-deduct on dispatch | **Adapt** | Publishing consumes the creative slot and marks the asset “used” |
| Bulk actions + CSV | **Adopt** | Bulk reschedule, pause, export |

### 3.5 Finance

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Receivables auto-raised on dispatch | **Adapt** | Invoice / retainer line auto-raised when a campaign goes live or a monthly package delivers |
| Payables (vendor bills) | **Adapt** | Freelancer and ad-platform bills against the same campaign |
| Client aging analysis | **Adopt** | Who owes what, for how long — retainers, overages, ad spend recoup |
| WhatsApp/SMS overdue reminders + cooldown | **Adopt** | Same, for agency collections |
| PDF invoice in one click | **Adopt** | Professional client invoice + campaign delivery note |
| Discounts, write-offs, ledger per receivable | **Adapt** | Agency ledger; do not become a full accounting suite |
| Two-way Zoho Books / Tally | **Adapt** | Zoho Books + Tally + QuickBooks + Stripe/Razorpay sync — push invoices, pull payments |
| Net position (cash + AR − AP − committed cost) | **Adapt** | Agency net: cash + retainers due − freelancer AP − committed ad spend |

### 3.6 Partners (clients, vendors, workforce)

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Clients & vendors directory | **Adopt** | First-class **Clients** and **Vendors** (not just “organizations”) |
| Employee profiles, shifts, wage rates | **Do not build** as HR | Optional later: contractor rate cards only, not attendance/payroll |
| Daily attendance / overtime / wages | **Do not build** | Out of category; integrate with existing HR if ever needed |
| Absenteeism patterns | **Do not build** | — |
| Skill-based operator routing | **Adapt (light)** | Route creative work orders by skill (copy, design, video, media buy) |

### 3.7 Factory floor → Marketing floor

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Quality issues by severity, linked to work orders | **Adapt** | **Quality / brand-safety issues** linked to posts, ads, and clients |
| Public complaint links (no login) | **Adopt** | Public client feedback / revision link; public “report a brand issue” |
| Machine registry + downtime | **Adapt** | **Channel & ad-account registry** — token health, API errors, spend pauses |
| Preventive maintenance schedule | **Adapt** | Preventive: OAuth refresh, pixel health, CAPI failure rate, posting quota |
| Spare parts + reorder | **Do not build** | No physical parts analog worth the complexity |
| Downtime reason codes | **Adapt** | Failed-publish reason codes; ad-account disable reasons |

### 3.8 Intelligence

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Every module feeds one intelligence layer | **Adopt** | Comet becomes the intelligence layer for *all* modules, not only ads |
| Production velocity / cycle time / yield | **Adapt** | Brief → live cycle time; approval yield; publish success rate |
| Inventory health / slow-moving capital | **Adapt** | Stale creatives, unused brand assets, dark landing pages |
| Financial clarity / cash cycle | **Adapt** | Client cash cycle; CAC payback; attributed net |
| Machine load | **Adapt** | Channel load (slots used vs safe cadence) and ad-account budget load |
| “Will today hit target? If not, which intervention?” | **Adopt** | The North-star test for Postix Intelligence |
| Exportable datasets | **Adopt** | CSV / warehouse later (already sketched in Attribution Enterprise) |

### 3.9 Copilot

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Type or speak, English or Hinglish | **Adopt** | Voice + Hinglish on the existing Agent |
| Drafts an entry; user confirms; nothing saved until Confirm | **Adopt** | Make this the *only* write path for Agent/MCP |
| Do / Ask / Learn / Guide modes | **Adopt** | Do (create post), Ask (ROAS this week), Learn (what is CAPI), Guide (how do I connect Instagram) + deep-link button |
| Writes across every module | **Adopt** | Agent must write calendar, campaigns, ads pause/budget, invoices, clients — not only posts |
| Every action logged | **Already have / deepen** | Show the confirm card in the audit trail |
| “I don’t know” before guessing | **Adopt** | Grounded product answers with citations to `/docs` |

### 3.10 Cross-cutting platform

| Makoro feature | Verdict | Postix equivalent |
| --- | --- | --- |
| Multi-unit (switch factory in one click) | **Adapt** | Multi-brand / multi-client workspace switcher that agencies actually use |
| Role-based access | **Deepen** | Roles: Owner, Strategist, Creator, Media buyer, Client (read/approve), Accountant |
| CSV export everywhere | **Adopt** | Non-negotiable on every list view |
| WhatsApp notifications (operational, not just a channel) | **Adopt** | Approvals, go-lives, low pipeline, overdue invoices, failed publishes |
| Android app (iOS later) | **Adopt** | Approve, reply inbox, pause ads, confirm Copilot drafts from the phone |
| Free forever, 10 entries/section | **Adapt** | Free cloud: 10 posts + 1 channel + 1,000 Comet events — no card, no expiry |
| Dedicated migration person on paid plans | **Adopt** | Named onboarding for Pro/Agency; import Buffer/Later/Hootsuite/Meta |
| Custom implementations (4–8 week go-live) | **Adapt** | Agency/enterprise: custom approval graphs, SSO, warehouse, client portals |
| Partner / reseller program | **Adapt** | Certified implementation partners for agencies that resell Postix |
| Encrypted cloud + automated backups | **Already have / communicate** | Make this a first-class trust page, not only a footer line |

---

## 4. What we will not build

These Makoro capabilities are real and well-designed. They still do not belong in Postix.

- Physical inventory, FIFO of raw materials, GRNs, batch codes for steel/chemicals
- E-way bills, LR numbers, vehicle/driver dispatch
- Shop-floor attendance, overtime, wage summaries, payroll export
- Machine registry, spare parts, preventive maintenance of equipment
- Quality issues about physical defects
- Tally as a *factory* ledger of production costing
- Industry packs for textiles, plastics, pharma, fabrication

If a customer asks for those, the answer is “that is a different product.” Postix stays a marketing OS.

---

## 5. Target architecture: the Marketing Operating System

Makoro’s sentence: *Sales orders to work orders, GRNs to dispatch, log book to ledger — every module feeds a shared intelligence layer.*

Postix’s equivalent sentence:

> **Briefs to work orders, assets to calendar, publish to ledger — every module feeds Comet, so creative decisions reflect real pipeline and every go-live can raise a receivable and a CAPI event.**

```
                    ┌─────────────────────────┐
                    │     Postix Copilot      │
                    │  Do · Ask · Learn · Guide│
                    │   confirm before write  │
                    └────────────┬────────────┘
                                 │
     ┌──────────┬──────────┬─────┴──────┬──────────┬──────────┐
     ▼          ▼          ▼            ▼          ▼          ▼
  Partners   Fulfillment  Studio     Publish    Finance    Floor
  clients    briefs &     work       calendar   AR/AP      quality
  vendors    campaign     orders     ads        invoices   channel
             BOM          assets     inbox      aging      health
     └──────────┴──────────┴────────────┴──────────┴──────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Comet Intelligence    │
                    │  will this week hit?    │
                    │  ROAS · pipeline · cash │
                    └─────────────────────────┘
```

### 5.1 The connected loop (Makoro’s 7 steps, rewritten)

| Step | Makoro | Postix |
| --- | --- | --- |
| 01 | Add partners | Add clients, brands, vendors |
| 02 | Define products & BOM | Define offers, brand kit, campaign BOM |
| 03 | Take sales order | Take brief / retainer / campaign order |
| 04 | Stock inventory | Stage assets in creative inventory |
| 05 | Create work order | Produce content batch (posts + ads) |
| 06 | Dispatch | Publish / go live (WhatsApp the client) |
| 07 | Raise receivable | Invoice + CAPI + Comet attribution |

One write. No re-entry. Data flows automatically.

---

## 6. Phased roadmap

Phases are sequenced by dependency and buyer value, not by calendar guesses. Each phase has an exit test. Do not start the next phase until the exit test is green.

---

### Phase 0 — Foundations (make the OS possible)

**Goal:** The primitives later phases hang on. Without these, Copilot and Intelligence are demos.

| ID | Work | Why it is first |
| --- | --- | --- |
| P0.1 | **Canonical event bus** | Every create/update/delete emits a typed event (`brief.confirmed`, `post.published`, `ad.paused`, `invoice.raised`). This is Makoro’s “one action, three modules.” |
| P0.2 | **Clients & brands as first-class objects** | Today Postix is org/channel-centric. Agencies think client → brand → channels. |
| P0.3 | **RBAC v2** | Owner, Admin, Strategist, Creator, Media buyer, Client, Accountant. Field-level where it matters (billing, pause ads). |
| P0.4 | **CSV export on every list** | Tablets and owners still live in Excel. This is table stakes and unblocks migration. |
| P0.5 | **Confirm-before-write contract** | Agent, MCP, CLI, and API writes that mutate money or publishing must return a draft + `confirm` token. |
| P0.6 | **Audit reasons** | Destructive actions require a reason; visible in History. |

**Exit test:** A confirmed brief can be created via API, appears under a client, is visible only to the right roles, and can be exported as CSV. An Agent draft to publish does not persist until Confirm.

**Already in Postix that we reuse:** audit trail, approvals, orgs/invites, public API, MCP, webhooks.

---

### Phase 1 — Copilot that actually runs the OS

**Goal:** Match Makoro Copilot’s four modes on marketing objects. Postix already has an Agent; this phase makes it operational, not generative.

| ID | Work | User-facing example |
| --- | --- | --- |
| P1.1 | **Do** across modules | “Schedule this Diwali offer on Instagram and LinkedIn next Tuesday 11am, UTM `diwali26`.” Draft card → Confirm. |
| P1.2 | **Ask** against live data | “Which client’s ROAS fell this week?” “How many approved posts sit in the next 14 days?” |
| P1.3 | **Learn** with citations | “Do you support TikTok CAPI?” Answers from `/docs` or says “I don’t know.” |
| P1.4 | **Guide** + deep link | “How do I connect a LinkedIn Page?” Steps + one-tap button to the right screen. |
| P1.5 | **Voice + Hinglish** | “Instagram pe yeh post kal subah 10 baje daal do.” India-first, same as Makoro. |
| P1.6 | **Write coverage** | Posts, campaigns, ad pause/budget, client notes, invoice recorded-paid. Same confirm card everywhere. |

**Exit test:** Ten real operator utterances (mix of English/Hinglish) complete without leaving chat, and every write shows in Audit. Hallucinated product answers are zero in a 50-question eval.

**Do not do in this phase:** autonomous publishing without Confirm. Makoro is explicit: *Nothing is saved until you tap Confirm.* Keep that religion.

---

### Phase 2 — Fulfillment + Studio (the demand and production chain)

**Goal:** Replace “a pile of posts” with a demand-driven studio, the way Makoro replaced whiteboards with work orders.

#### 2A. Fulfillment

| ID | Capability | Acceptance |
| --- | --- | --- |
| P2.1 | **Client briefs** | Line items (deliverables, networks, rates, go-live date). Status Draft → Confirmed → Closed. |
| P2.2 | **Campaign BOM** | Template per campaign type (always-on social, product launch, performance). Checklist: copy, creative, landing, pixel, CAPI, UTM, audience. |
| P2.3 | **Shortfall view** | “To fulfil 4 open briefs you are missing 11 assets and 2 pixel installs.” |
| P2.4 | **Vendor POs** | Assign a freelancer or studio; mark received when assets land in inventory. |

#### 2B. Studio (work orders)

| ID | Capability | Acceptance |
| --- | --- | --- |
| P2.5 | **Creative work orders** | Outputs + consumed brand assets; link to brief. |
| P2.6 | **Spreadsheet batch composer** | One batch (offer) → N posts. Fill-down campaign, UTM, brand. Inline quick-add offer. |
| P2.7 | **Gate on complete** | Cannot mark ready-to-publish if BOM items or approval are missing. |
| P2.8 | **Run sheet PDF** | A4/PDF: what is going out today, who approved, which UTMs. The workorder printout analog. |

**Exit test:** A confirmed brief auto-creates a work order and a shortfall list. Completing the work order fills the calendar without re-typing captions. A PDF run sheet is generated in one click.

---

### Phase 3 — Creative inventory + Publish packets

**Goal:** Makoro’s inventory + dispatch, for assets and go-lives.

#### 3A. Creative inventory

| ID | Capability | Acceptance |
| --- | --- | --- |
| P3.1 | Asset register | Every file/copy block has id, client, brand, campaign, usage count, last used, expiry, value (production cost). |
| P3.2 | Movement history | Created, used in post, used in ad, revised, retired — with reasons. |
| P3.3 | Pipeline alerts | Per channel: “fewer than N approved items in the next 14 days.” Analog of low-stock. |
| P3.4 | Stale-capital report | Assets unused for 60 days; dark landing pages still in live ads. |
| P3.5 | CSV import | Bring a media library from Drive/Dropbox/Frame.io with metadata. |

#### 3B. Publish packets (dispatch)

| ID | Capability | Acceptance |
| --- | --- | --- |
| P3.6 | Publish packet object | Client, items, networks, UTMs, CAPI event, window. Status Draft → Approved → Scheduled → Publishing → Live → Attributed. |
| P3.7 | One action, three modules | Go-live deducts inventory usage, notifies the client, and can raise a receivable (flag). |
| P3.8 | Operational WhatsApp | Client: “Your campaign X is live.” Internal: failed publish with reason code. Cooldown so nobody is spammed. |
| P3.9 | Public no-login links | Client approves a packet or leaves a revision without an account. Makoro’s public complaint link, used for the happy path. |

**Exit test:** Publishing a packet updates inventory, sends WhatsApp, and (if billed-on-delivery) creates a draft receivable. A client can approve from a phone with no login.

**Reuse:** existing calendar states, approvals, WhatsApp *channel*, email notifications, short links.

---

### Phase 4 — Agency finance (not a second Tally)

**Goal:** Enough finance that an agency owner sees net position without exporting to a CA. Stop before we become an accounting product.

| ID | Capability | Acceptance |
| --- | --- | --- |
| P4.1 | Receivables | Raised from go-live, monthly retainer, or manual. One-click PDF. Record collection. |
| P4.2 | Payables | Freelancer bills and (optional) ad-spend recoup against the campaign. |
| P4.3 | Aging | Buckets 0–30 / 31–60 / 61–90 / 90+. |
| P4.4 | Overdue reminders | WhatsApp/email with cooldown. |
| P4.5 | Net position | Cash (manual or Razorpay/Stripe) + AR − AP − committed ad spend. |
| P4.6 | Accounting sync | Two-way Zoho Books first (Makoro already proved this path); then Tally / QuickBooks. Stripe + Razorpay stay the payment rails. |

**Exit test:** A go-live can create an invoice; aging is visible; a Zoho Books sync round-trips one invoice and one payment. We do *not* ship GST returns, TDS, or full ledger in this phase.

**Boundary:** Comet already attributes *revenue from ads*. Finance attributes *revenue from clients*. Both must appear on the same Intelligence screen without being the same object.

---

### Phase 5 — Marketing floor (quality + channel health)

**Goal:** The factory-floor module, rewritten for channels and brand safety.

| ID | Capability | Acceptance |
| --- | --- | --- |
| P5.1 | Quality issues | Severity (S1–S4), linked to post/ad/client. States: Open → Triaged → Fixed → Closed. |
| P5.2 | Public report link | Client or viewer reports a brand/legal issue without login. |
| P5.3 | Channel registry | Every connected account: last successful publish, token expiry, last API error, spend status. |
| P5.4 | Preventive jobs | Alerts 7 days before token expiry; CAPI failure rate > threshold; pixel silent for 24h. |
| P5.5 | Reason codes | Failed publish / ad rejected / account disabled — coded, not free text only. |
| P5.6 | Channel load | Safe cadence vs scheduled volume. Analog of machine load. |

**Exit test:** A silent pixel and an expiring Instagram token both appear on Intelligence as interventions, not as log lines.

---

### Phase 6 — Intelligence layer (the North-star)

**Goal:** Pass Makoro’s test, rewritten for marketing:

> Can Postix tell you, right now, whether this week’s plan will hit target — and if not, which specific intervention will fix it?

| ID | Capability | What it synthesizes |
| --- | --- | --- |
| P6.1 | Unified signal ingest | Briefs, inventory, publish outcomes, inbox SLA, ad spend, Comet events, AR/AP, channel health |
| P6.2 | Will-we-hit | Target (leads/revenue/ROAS/posts) vs forecast from live pipeline + current ROAS |
| P6.3 | Intervention cards | “Pause ad set B (−₹12k/day, 0.4x ROAS).” “Approve 6 LinkedIn drafts or the feed goes dark Friday.” “Call Client X — ₹2.4L, 47 days.” |
| P6.4 | Velocity / yield | Brief→live cycle time; approval yield; publish success; creative win rate |
| P6.5 | Inventory health | Stale assets, pipeline days-of-cover per channel |
| P6.6 | Financial clarity | Net position, aging, cash cycle, attributed vs invoiced (the reconciliation that agencies skip) |
| P6.7 | Copilot on Intelligence | Ask/Guide sit on the same layer. Do can execute an intervention with Confirm. |

**Exit test:** In a seeded agency workspace, Intelligence names the correct intervention in 8/10 scripted scenarios (dark channel, dead ad set, overdue invoice, missing pixel, empty pipeline).

**Reuse:** Comet MTA, Ads Manager pause/budget, analytics, email digests. This phase *connects* them; it does not replace Comet.

---

### Phase 7 — Surfaces, pricing, and GTM (Makoro’s physical + commercial edge)

| ID | Work | Notes |
| --- | --- | --- |
| P7.1 | **Android app** | Approve packets, inbox reply, pause ads, confirm Copilot, see Intelligence. iOS after Android is stable. |
| P7.2 | **Free forever cloud** | 10 posts / 1 channel / 1,000 Comet events, no card, no expiry. Makoro’s best acquisition loop. |
| P7.3 | **Named onboarding** | Every paid workspace gets a human who imports history and sits on a call. |
| P7.4 | **Partner program** | Certified agencies resell / implement Postix. Selective, trained, backed. |
| P7.5 | **Custom implementations** | SSO, custom approval graphs, warehouse, client portals. Discovery → prototype → go-live. |
| P7.6 | **Trust pack** | Backups, encryption, status page (already exists), DPA, region options. Say it as loudly as Makoro does. |
| P7.7 | **Migration importers** | Buffer, Later, Hootsuite, Meta Business Suite, Google Ads, Shopify/Stripe for Comet. |

**Exit test:** A new agency can start free the same day, hit the cap, upgrade, and have a human import their last 90 days of posts and one ad account.

---

## 7. Module backlog (build list)

Use this as the product catalog once Phase 0 is done. Names are Postix-native.

| Module | Phase | One-line job |
| --- | --- | --- |
| **Partners** | 0 | Clients, brands, vendors in one directory |
| **Copilot** | 1 | Do / Ask / Learn / Guide with confirm-before-write |
| **Fulfillment** | 2 | Briefs, campaign BOM, shortfall, vendor POs |
| **Studio** | 2 | Work orders, batch composer, run-sheet PDF |
| **Inventory** | 3 | Creative register, usage ledger, pipeline alerts |
| **Publish** | 3 | Packets, status machine, WhatsApp, public approve |
| **Finance** | 4 | AR/AP, aging, net position, Zoho/Tally/QB sync |
| **Floor** | 5 | Quality issues, channel health, preventive alerts |
| **Intelligence** | 6 | Will-we-hit + intervention cards |
| **Mobile** | 7 | Android-first operator surface |
| **Access** | 0–7 | RBAC, multi-brand switch, CSV, audit, free tier |

Existing modules map in rather than getting replaced:

| Existing Postix | Becomes |
| --- | --- |
| Calendar + Campaigns + Evergreen + A/B | Studio + Publish |
| Approvals + Inbox | Publish + Floor (inbox SLA is a quality signal) |
| Media library | Inventory |
| AI Agent + MCP + CLI | Copilot |
| Comet + Ads Manager + Analytics | Intelligence (plus Finance for client cash) |
| Team / white-label / webhooks / API | Access + GTM |

---

## 8. Suggested sequencing of first ten ships

If engineering capacity is one squad, ship in this order. Each item is independently demoable.

1. **Confirm-before-write** on Agent/MCP (P0.5 + P1.1) — cheapest credibility win
2. **Clients & brands** (P0.2) — unblocks everything agency-shaped
3. **CSV everywhere** (P0.4) — owners notice immediately
4. **Campaign BOM + shortfall** (P2.2–P2.3) — the “wow, it told me what to make” moment, Makoro’s Materials tab
5. **Pipeline low-stock alerts** (P3.3) — daily habit
6. **Public client approve link** (P3.9) — kills the “please check WhatsApp” loop
7. **Operational WhatsApp** (P3.8) — go-live and failed-publish
8. **Aging + overdue reminders** (P4.3–P4.4) — Makoro’s ₹12L receivables story, for agencies
9. **Channel preventive health** (P5.3–P5.4) — tokens and silent pixels
10. **Will-we-hit + one intervention card** (P6.2–P6.3) — the Intelligence promise, even if only three signals

Items 1–3 are Phase 0/1. Items 4–7 are the core OS. Items 8–10 are why someone pays Pro.

---

## 9. Pricing implications (steal the loop, not the rupee amounts)

Makoro’s commercial design is better than most SaaS in this segment:

| Makoro move | Postix move |
| --- | --- |
| Free forever, 10 entries/section, no card | Free forever: 10 posts, 1 channel, 1k Comet events |
| All modules visible on free | All modules visible; caps, not feature gates, on the core loop |
| Paid = uncap + WhatsApp/SMS + multi-unit + priority | Paid = uncap + WhatsApp ops + multi-brand + Intelligence interventions |
| Dedicated human on every paid unit | Named onboarding on Pro/Agency |
| No auto-renewal, Razorpay, upgrade anytime | Keep Razorpay; be equally explicit about renewal |
| Custom 4–8 week implementations | Agency/enterprise SOW, not a generic “contact sales” |

Keep Social + Comet as the commercial split (create/publish vs measure/optimize). New modules (Fulfillment, Finance, Floor) should ride inside Social or Agency, not become a third SKU until they have independent willingness to pay.

---

## 10. Success metrics

### Product

| Metric | Why |
| --- | --- |
| % of publishes that originated from a brief (not a blank composer) | Measures whether the OS loop is real |
| Median brief → live cycle time | Makoro’s production velocity |
| Pipeline days-of-cover per channel | Inventory health |
| Confirm rate of Copilot drafts (and revert rate after confirm) | Copilot quality |
| % of go-lives that notify a client automatically | Dispatch loop |
| Attributed revenue vs invoiced revenue reconciliation rate | Intelligence honesty |
| Intervention-card accept rate | Whether Intelligence is a decision layer |

### Business

| Metric | Why |
| --- | --- |
| Free-to-paid conversion after hitting a cap | Makoro’s free-plan engine |
| Time-to-first-publish for a new workspace | “Set up in 3 minutes” claim |
| Paid workspaces with a completed onboarding call | Human edge |
| Net revenue retention on Agency | Whether Finance + Intelligence expand the account |

---

## 11. Risks and how we contain them

| Risk | Containment |
| --- | --- |
| Building a second ERP and losing Buffer/Cometly buyers | Phase gates; the “do not build” list is binding |
| Copilot that publishes without consent | Confirm token on every mutating tool; no silent MCP writes |
| Finance that traps us in GST/TDS forever | Zoho/Tally sync; we are not the system of record for tax |
| Intelligence that is a prettier dashboard | Exit test is intervention accuracy, not chart count |
| Android before the OS loop exists | Mobile is Phase 7; PWA of Copilot + approve links ships earlier |
| Scope explosion via “custom” | Custom is configuration + integrations, not a fork per customer |

---

## 12. Competitive stance after this roadmap

| Competitor | They have | Postix after this roadmap |
| --- | --- | --- |
| Postiz / Buffer / Later | Scheduling | Scheduling **plus** brief→BOM→inventory→publish→invoice→attribution |
| Cometly / Triple Whale / Hyros | Attribution | Attribution **plus** the operating system that creates the traffic |
| Hootsuite / Sprout | Teams + inbox | Teams + inbox **plus** confirm-copilot and will-we-hit |
| Agency CRMs (e.g. many India tools) | Clients + invoices | Clients + invoices **plus** the actual work and the ROAS |

The wedge stays the same as today’s homepage: *create, publish, advertise, and attribute in one system.* The roadmap makes that sentence operationally true, the way Makoro made “order to cash” operationally true.

---

## 13. Decision record

| Decision | Choice | Reason |
| --- | --- | --- |
| Pivot Postix into manufacturing ERP? | **No** | Different buyer, different compliance, different sales motion |
| Copy Makoro module names? | **No** | Use Partners, Fulfillment, Studio, Inventory, Publish, Finance, Floor, Intelligence, Copilot |
| Copilot autonomy? | **Human confirm on every write** | Makoro’s best trust mechanic; also required for ads spend |
| Full accounting? | **No — AR/AP + sync** | Stay the marketing OS; let Zoho/Tally/QB be the books |
| Free tier style? | **Caps, not feature gates** | Lets people feel the whole OS, then pay to uncap |
| Mobile first platform? | **Public links + PWA now; Android in Phase 7** | Approvals cannot wait for a store listing |

---

## 14. Sources

Primary product pages reviewed 13 August 2026:

- https://makoro.co/ and feature pages: fulfillment, production, inventory, dispatch, finance, workforce, factory-floor
- https://makoro.co/about, /enterprise, /partners
- https://makoro.co/blog/what-manufacturing-intelligence-actually-means
- https://makoro.co/blog/ai-on-the-shop-floor-hype-vs-reality
- https://makoro.co/blog/best-factory-management-software-2026
- https://postix.in/ , `/llms.txt`, `/product.json`, `/ai.txt`, `/pricing`, `/docs` index

---

## 15. One-page brief for the team

**Steal from Makoro:** connected loop, confirm-copilot, one-action fan-out, shortfall views, aging, public no-login links, WhatsApp as operations, CSV everywhere, free-with-caps, a human on paid.

**Do not steal:** machines, wages, e-way bills, spare parts, physical stock.

**Build next:** confirm-before-write, clients/brands, CSV, campaign BOM, pipeline alerts, public approve, ops WhatsApp, aging, channel health, will-we-hit.

**North star:** *Tell the owner, right now, whether this week hits target — and the one action that fixes it.*
