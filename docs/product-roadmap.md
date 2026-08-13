# Postix Product Roadmap

**Product:** [Postix](https://postix.in/) — The AI Marketing Operating System  
**Audience:** Product, engineering, design, GTM  
**Date:** 13 August 2026  
**Status:** Active backlog — new features to add on postix.in

This is the working product roadmap. It lists **new capabilities we do not have yet** (or have only in a thin form) and the order we should ship them.

Related reading: [Makoro feature-transfer analysis](./POSTIX_PRODUCT_ROADMAP.md) (why these features, and what we will not copy from factory ERP).

---

## How to read this

| Column | Meaning |
| --- | --- |
| **Now** | Build next. Unblocks the rest of the OS. |
| **Next** | Build once Now is in production. |
| **Later** | Real features, sequenced after the core loop works. |
| **Won’t** | Out of category. Do not add to Postix. |

Every feature has an ID (`PX-###`), owner surface, and an acceptance test. A feature is not done until the acceptance test is true in the hosted product at postix.in (and in the self-host binary).

---

## 1. What Postix already ships

Do not rebuild these. New work should *connect* them.

| Area | Already live |
| --- | --- |
| Scheduling | 30+ networks, unified calendar, campaigns, evergreen, A/B captions |
| AI | In-app Agent (text, image, UGC video), BYOK, MCP, CLI |
| Approvals | Member draft → admin approve/reject |
| Attribution | Comet pixel, 8 MTA models, CAPI (Meta/Google/TikTok/LinkedIn) |
| Ads | Comet Ads Manager — attributed ROAS, pause, budget |
| Team | Orgs, invites, white-label / brand kit |
| Platform | REST API, OAuth, webhooks, n8n/Make, audit trail, Razorpay, self-host Go binary |
| Other | Inbox, RSS, plugs, short links, email digests, affiliate, status page, demo |

**The gap:** Postix can create, publish, and attribute. It cannot yet run the agency as a connected operating system — briefs, asset shortfall, client approvals without a login, collections, channel health, or “will this week hit target?”

---

## 2. North star

> Tell the owner, right now, whether this week hits target — and the one action that fixes it.

Everything below either feeds that sentence or is a habit that makes people stay (approve, get paid, keep channels alive).

---

## 3. Now / Next / Later

```
NOW                          NEXT                         LATER
─────────────────────────    ─────────────────────────    ─────────────────────────
Confirm-before-write Agent   Campaign BOM + shortfall     Agency finance (AR/AP)
Clients & brands             Creative inventory           Zoho / Tally / QB sync
RBAC v2                      Publish packets              Android app
CSV export everywhere        Public client approve link   Free-forever cloud tier
Voice + Hinglish Copilot     Ops WhatsApp notifications   Partner / reseller program
Pipeline low-stock alerts    Quality + channel health     Custom implementations
                             Will-we-hit Intelligence     Warehouse / SSO pack
```

---

## 4. Feature catalog

### NOW — Foundation and Copilot

These ship first. Without them, later modules are demos.

---

#### PX-001 — Confirm-before-write on every Agent / MCP / CLI mutation

**Problem:** The Agent and MCP can draft posts, but writes are not a single, reviewable contract. Operators do not trust autonomy around publishing or ad spend.

**What to build**

- Every mutating tool returns a **draft card** (object type, fields, side effects) plus a `confirm` token.
- Nothing is persisted until the user taps **Confirm** (or sends the token).
- Confirm and reject both land in Audit.
- Applies to: create/schedule post, change status, pause ad, change budget, record payment.

**Acceptance**

- “Schedule this on Instagram tomorrow 11am” shows a card; the calendar does not change until Confirm.
- An MCP client cannot publish without a confirm step.
- Audit shows actor, draft, confirm/reject, timestamp.

**Surface:** Agent, MCP, CLI, API  
**Depends on:** existing Agent + audit

---

#### PX-002 — Clients and brands as first-class objects

**Problem:** Postix is org- and channel-centric. Agencies think **client → brand → channels**.

**What to build**

- `Client` (company, billing contact, WhatsApp, timezone).
- `Brand` under a client (logo, kit, tone, default UTMs).
- Channels, campaigns, posts, ads, and invoices belong to a brand.
- One-click workspace switcher: Client / Brand.

**Acceptance**

- Create Client “Acme”, brand “Acme Care”; connect Instagram to that brand only.
- Calendar and Comet can filter by client or brand.
- Existing orgs migrate: current workspace becomes one client + one brand.

**Surface:** new Partners module, switcher in app chrome  
**Depends on:** nothing

---

#### PX-003 — Role-based access v2

**Problem:** Invites exist; job-shaped roles do not.

**Roles to ship**

| Role | Can |
| --- | --- |
| Owner | Everything, billing |
| Admin | Settings, members, channels |
| Strategist | Briefs, campaigns, calendar, Intelligence |
| Creator | Drafts, media, work orders — cannot publish or pause ads |
| Media buyer | Ads Manager, CAPI, audiences — cannot edit brand kit |
| Client | Read + approve assigned brands only |
| Accountant | Invoices, aging, exports — no publishing |

**Acceptance**

- A Creator cannot publish or change ad budget.
- A Client sees only their brands and the public approve queue.
- Role is enforced on API and MCP, not only UI.

**Surface:** Team settings  
**Depends on:** PX-002

---

#### PX-004 — CSV export on every list

**Problem:** Owners still live in Excel. Missing export kills trust and blocks migration.

**What to build**

- Export CSV on: posts, campaigns, media, clients, brands, channels, approvals, Comet events (sampled), invoices (when finance ships).
- Column set = what’s on the table, plus IDs.
- Import CSV for posts and clients (minimum).

**Acceptance**

- Every major list has **Export CSV**.
- A 500-row post export opens cleanly in Sheets.
- Post import creates drafts, not live publishes.

**Surface:** all list views  
**Depends on:** nothing

---

#### PX-005 — Copilot modes: Do, Ask, Learn, Guide

**Problem:** The Agent generates content. It does not run the OS.

**What to build**

| Mode | Job | Example |
| --- | --- | --- |
| **Do** | Draft a write | “Schedule the Diwali offer on IG + LinkedIn Tuesday 11am, UTM `diwali26`.” |
| **Ask** | Read live data | “Which client’s ROAS fell this week?” |
| **Learn** | Answer from `/docs` | “Do you support TikTok CAPI?” — or “I don’t know.” |
| **Guide** | Steps + deep link | “How do I connect a LinkedIn Page?” → button to that screen |

Also: **voice input** and **Hinglish** on the same box (“Instagram pe yeh post kal subah 10 baje daal do”).

**Acceptance**

- Ten scripted utterances (5 EN, 5 Hinglish) complete without leaving chat.
- Learn answers cite a docs URL or refuse.
- Guide opens the correct settings screen.
- All Do paths go through PX-001.

**Surface:** Agent (rebranded Copilot in UI)  
**Depends on:** PX-001, PX-002

---

#### PX-006 — Pipeline low-stock alerts

**Problem:** Feeds go dark because nobody counted approved posts.

**What to build**

- Per brand + channel: count of approved/scheduled items in the next 7 / 14 / 30 days.
- Alert when below a threshold (default 3 in 14 days).
- Digest in-app + email; WhatsApp in PX-014.

**Acceptance**

- Empty LinkedIn queue for 14 days shows a red alert on the home/Intelligence strip.
- Alert clears when three posts are approved into that window.

**Surface:** home, calendar, notifications  
**Depends on:** calendar states, PX-002

---

### NEXT — The operating loop

Ship these after Now is live. This is the product people will pay for.

---

#### PX-007 — Client briefs (demand object)

**Problem:** Work starts as a WhatsApp paragraph. Nothing in Postix is the source of demand.

**What to build**

- Brief: client, brand, deliverables (line items), networks, rate or package, go-live date, notes.
- Status: Draft → Confirmed → In production → Live → Closed.
- Value-by-client rollup.

**Acceptance**

- Confirming a brief is the demand signal later modules plan against.
- Closed brief cannot accept new work orders without a reopen.

**Surface:** Fulfillment  
**Depends on:** PX-002

---

#### PX-008 — Campaign BOM and shortfall

**Problem:** Teams discover missing landing pages and pixels on go-live day.

**What to build**

- BOM template per campaign type (always-on social, launch, performance).
- Required items: copy, creative, landing URL, pixel, CAPI event, UTM, audience, approval.
- **Shortfall view:** open briefs vs available assets vs live pixels.

**Acceptance**

- Opening Fulfillment shows: “4 open briefs · missing 11 assets · 2 pixels not firing.”
- Marking an item received removes it from shortfall.

**Surface:** Fulfillment  
**Depends on:** PX-007, media library

---

#### PX-009 — Creative work orders + spreadsheet batch composer

**Problem:** Multi-network variants are re-typed one post at a time.

**What to build**

- Work order linked to a brief: outputs (posts/ads) + consumed brand assets.
- Spreadsheet composer: one row per post; fill-down brand, campaign, UTM, offer.
- Inline quick-add of an offer or asset (no page switch).
- Gate: cannot mark Ready if BOM or approval is missing.

**Acceptance**

- One batch (one offer) creates N network-native drafts in one confirm.
- Ready-to-publish is blocked if pixel or approval is missing.

**Surface:** Studio  
**Depends on:** PX-007, PX-008, PX-001

---

#### PX-010 — Daily run-sheet PDF

**Problem:** Studio and client workshops still need a paper/PDF of “what goes out today.”

**What to build**

- One-click PDF: date, brand, posts, networks, UTMs, approver, status.
- A4, printable, brand kit header.

**Acceptance**

- From today’s calendar, **Download run sheet** produces a readable A4 PDF in one click.

**Surface:** Studio, Calendar  
**Depends on:** PX-009, brand kit

---

#### PX-011 — Creative inventory (asset ledger)

**Problem:** Media library is a folder. It is not stock.

**What to build**

- Every asset: id, client, brand, campaign, cost, usage count, last used, expiry, status (active/retired).
- Movement history: created → used in post → used in ad → revised → retired (reason required).
- Stale report: unused 60+ days; landing URLs in live ads that 404.

**Acceptance**

- Publishing a post increments usage and writes a movement row.
- Deleting an asset requires a reason, visible in history.

**Surface:** Inventory (upgrade of Media)  
**Depends on:** PX-002, PX-004, existing media

---

#### PX-012 — Publish packets and status machine

**Problem:** A post has states. A *go-live* (the thing the client cares about) does not.

**What to build**

- Publish packet: brand, items, networks, UTMs, CAPI event, window.
- Status: Draft → Approved → Scheduled → Publishing → Live → Attributed.
- One action on **Go live**: mark items used (PX-011), notify client (PX-014), optionally raise a draft invoice (PX-016).

**Acceptance**

- Moving a packet to Live updates inventory and writes three events (publish, inventory, notify).
- Failed publish sets reason code, does not mark Live.

**Surface:** Publish (upgrade of Calendar)  
**Depends on:** PX-009, PX-011, event bus

---

#### PX-013 — Public client approve / revise link

**Problem:** Clients will not create another login. Approvals die in WhatsApp.

**What to build**

- Tokenized public URL, no account required.
- Client sees packets/posts, Approve or Request changes (comment + optional screenshot).
- Expiry and revoke. Brand-kit chrome (white-label).

**Acceptance**

- Client on mobile, no login, approves three posts; they move to Approved.
- Link can be revoked; old URL 404s.

**Surface:** Publish, white-label  
**Depends on:** PX-003 (Client role), PX-012

---

#### PX-014 — Operational WhatsApp (not just a publish channel)

**Problem:** WhatsApp is a network to post *to*. It is not how the OS talks to people.

**What to build**

- Templates: packet approved, campaign live, publish failed, pipeline low, invoice overdue (PX-018).
- Cooldown per template per recipient (no spam).
- Internal vs client audiences.
- Uses WhatsApp Cloud API already in providers.

**Acceptance**

- Go-live sends the client one WhatsApp with campaign name + public report link.
- Failed publish notifies the Creator once; retry does not resend within cooldown.

**Surface:** notifications settings  
**Depends on:** PX-012, existing WhatsApp provider, email notifications

---

#### PX-015 — Will-we-hit Intelligence (v1)

**Problem:** Analytics and Comet explain yesterday. Owners need an intervention today.

**What to build**

- Target per brand/week: posts live, leads, revenue, or ROAS.
- Forecast from pipeline (scheduled + current ROAS + pixel health).
- **Intervention cards** (start with three):
  1. Channel going dark (from PX-006)
  2. Ad set below ROAS floor (from Ads Manager)
  3. Silent pixel / CAPI failures
- Copilot Ask/Guide sit on this layer. Do can execute a card via PX-001.

**Acceptance**

- Seeded workspace: 8/10 scripted scenarios name the correct intervention.
- Accepting “Pause ad set B” goes through confirm and pauses in Ads Manager.

**Surface:** Intelligence home (new), feeds Comet  
**Depends on:** PX-006, Comet, Ads Manager, PX-001

---

### LATER — Money, floor, mobile, GTM

Do not start these until the Next loop is used by real workspaces.

---

#### PX-016 — Receivables and one-click PDF invoice

- Raise from go-live, monthly retainer, or manual.
- Professional PDF. Record collection.
- **Not** a full GST/TDS suite.

**Acceptance:** A Live packet with “bill on delivery” creates a Draft invoice; PDF downloads in one click.

---

#### PX-017 — Payables (freelancer + ad recoup)

- Vendor bills against a campaign/brief.
- Optional ad-spend recoup lines.

**Acceptance:** Net position = cash + AR − AP − committed ad spend (cash can be manual in v1).

---

#### PX-018 — Aging and overdue reminders

- Buckets 0–30 / 31–60 / 61–90 / 90+.
- WhatsApp/email with cooldown (PX-014).

**Acceptance:** A 47-day open invoice appears in 31–60 and can send one reminder.

---

#### PX-019 — Accounting sync (Zoho Books first)

- Two-way: push invoices/payments, pull status.
- Then Tally / QuickBooks.
- Stripe + Razorpay stay the payment rails.

**Acceptance:** One invoice and one payment round-trip to Zoho Books.

---

#### PX-020 — Quality issues and public report link

- S1–S4 issues linked to post/ad/client.
- Public “report a brand issue” link (no login).
- States: Open → Triaged → Fixed → Closed.

**Acceptance:** A client submits S2 via public link; it appears on the brand Floor and in Intelligence.

---

#### PX-021 — Channel and ad-account health

- Registry: last successful publish, token expiry, last API error, spend status.
- Preventive alerts: token < 7 days, pixel silent 24h, CAPI error rate over threshold.
- Reason codes on failed publish / ad reject / account disable.
- Channel load: scheduled volume vs safe cadence.

**Acceptance:** An Instagram token expiring in 5 days and a silent pixel both appear as intervention cards.

---

#### PX-022 — Vendor / freelancer POs

- PO against a brief (design, video, UGC, media buy).
- Received when assets land in inventory.

**Acceptance:** Shortfall drops when a PO is marked received and files are attached.

---

#### PX-023 — Android app

- Approve packets, inbox reply, pause ads, confirm Copilot drafts, see Intelligence.
- iOS after Android is stable.
- Until then: PWA of Copilot + public approve links (PX-013).

**Acceptance:** A media buyer can pause an ad and a client can approve a packet from Android.

---

#### PX-024 — Free-forever cloud tier

- No card, no expiry.
- Caps (not feature gates): 10 posts, 1 channel, 1,000 Comet events, 1 brand.
- All modules visible.

**Acceptance:** A new user completes first publish on free; hitting the cap explains upgrade without hiding Intelligence.

---

#### PX-025 — Named onboarding and importers

- Every paid workspace gets a human import call.
- Importers: Buffer, Later, Hootsuite, Meta Business Suite, Google Ads; Shopify/Stripe for Comet.

**Acceptance:** Last 90 days of posts + one ad account imported in one onboarding session.

---

#### PX-026 — Partner program and custom implementations

- Certified agencies resell / implement Postix.
- Custom: SSO, approval graphs, warehouse, client portals.
- Discovery → prototype → go-live. Configuration, not a fork per customer.

**Acceptance:** Partner docs + application form live; one custom SOW template in repo.

---

## 5. Won’t build (keep Postix a marketing OS)

- Physical inventory, FIFO, GRNs, e-way bills, vehicles, drivers
- Shop-floor attendance, overtime, wages, payroll
- Machine registry, spare parts, equipment maintenance
- Full GST returns / TDS / general ledger (sync out instead)
- Industry packs for factories (textiles, pharma, plastics)

If a customer asks, the answer is: that is a different product.

---

## 6. Suggested first ten ships

One squad, one after another. Each is independently demoable on postix.in.

| # | ID | Ship | Demo in one sentence |
| --- | --- | --- | --- |
| 1 | PX-001 | Confirm-before-write | Agent shows a card; calendar does not move until Confirm |
| 2 | PX-002 | Clients & brands | Switch “Acme Care” and only that brand’s channels show |
| 3 | PX-004 | CSV everywhere | Export posts; open in Sheets |
| 4 | PX-003 | RBAC v2 | Creator cannot pause ads |
| 5 | PX-005 | Copilot four modes | Hinglish schedule + “how do I connect LinkedIn?” deep link |
| 6 | PX-006 | Pipeline alerts | LinkedIn red on home when queue < 3 |
| 7 | PX-008 | Campaign BOM | Shortfall: missing pixel + 2 creatives |
| 8 | PX-013 | Public approve | Client approves on phone, no login |
| 9 | PX-014 | Ops WhatsApp | Client gets “your campaign is live” |
| 10 | PX-015 | Will-we-hit v1 | Card: “Pause ad set B — 0.4x ROAS” |

PX-007 and PX-009–012 fill the loop immediately after 7–10.

---

## 7. Module map (what the app should look like)

| New module | Features | Replaces / absorbs |
| --- | --- | --- |
| **Partners** | PX-002, PX-003, PX-022 | Thin org list |
| **Copilot** | PX-001, PX-005 | AI Agent (same engine, new contract) |
| **Fulfillment** | PX-007, PX-008, PX-022 | New |
| **Studio** | PX-009, PX-010 | Campaigns + composer |
| **Inventory** | PX-011 | Media library |
| **Publish** | PX-012, PX-013, PX-014 | Calendar + approvals |
| **Finance** | PX-016–019 | New (Later) |
| **Floor** | PX-020, PX-021 | New (Later) |
| **Intelligence** | PX-006, PX-015 | Home + Comet + analytics |
| **Access** | PX-004, PX-024–026 | Billing, team, GTM |

Existing Comet, Ads Manager, Inbox, MCP, API, and white-label stay. New modules plug into them; they are not rewritten.

---

## 8. Event bus (required platform work)

Add a typed event bus so one action can update three modules. Minimum events:

```
client.created
brand.created
brief.confirmed
brief.closed
work_order.ready
asset.used
asset.retired
packet.approved
packet.live
packet.failed
invoice.raised
invoice.paid
channel.unhealthy
pixel.silent
ad.paused
```

Rules:

- Writes go through the bus; modules subscribe.
- Copilot Do emits the same events as the UI.
- Intelligence reads the bus; it does not scrape screens.

This is not a user-facing feature. It is the reason PX-012 and PX-015 can be true.

---

## 9. Pricing changes that go with the features

| Plan | Change |
| --- | --- |
| Free forever (PX-024) | Caps, not feature gates. All modules visible. |
| Social | Uncap posts/channels + Copilot + Fulfillment/Studio/Publish |
| Social + Comet | Current split stays: create/publish vs measure/optimize |
| Agency / Pro | Multi-brand, ops WhatsApp, Intelligence interventions, named onboarding |
| Enterprise | SSO, warehouse, custom approval graph, partner-led rollout |

Do not add a third SKU for Fulfillment/Finance until those modules have independent willingness to pay.

---

## 10. Success metrics

| Metric | Target once Next is live |
| --- | --- |
| % of publishes that started from a brief | > 50% of paid workspaces |
| Pipeline days-of-cover (median brand) | ≥ 14 days |
| Copilot confirm rate | > 60%; revert after confirm < 10% |
| Public-link approvals | > 40% of client approvals |
| Go-lives with automatic client notify | > 80% |
| Intervention-card accept rate | > 25% |
| Free → paid after hitting a cap | Track weekly; no vanity signups |

---

## 11. Definition of done (every feature)

1. Works on hosted postix.in and in the self-host binary.
2. API + MCP coverage if the feature mutates data.
3. CSV export if it is a list.
4. Audit row on create/update/delete.
5. Docs page under `/docs` and a line in `llms.txt` / `product.json`.
6. Role check (PX-003) on UI and API.
7. Acceptance test in this file is green.

---

## 12. One-page backlog

**Now:** PX-001 Confirm · PX-002 Clients/brands · PX-003 RBAC · PX-004 CSV · PX-005 Copilot modes · PX-006 Pipeline alerts  

**Next:** PX-007 Briefs · PX-008 BOM/shortfall · PX-009 Work orders · PX-010 Run sheet · PX-011 Inventory · PX-012 Packets · PX-013 Public approve · PX-014 WhatsApp ops · PX-015 Intelligence v1  

**Later:** PX-016–019 Finance · PX-020–021 Floor · PX-022 Vendor POs · PX-023 Android · PX-024 Free tier · PX-025 Onboarding/importers · PX-026 Partners/custom  

**Won’t:** factory ERP, payroll, e-way bills, full accounting.

**Start Monday:** PX-001, then PX-002, then PX-004.
