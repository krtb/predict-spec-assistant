# Subscription Churn Model Specification

Customer: Northwind Software (fictional example)
Product line in scope: Atlas (project management)
Data sources: Salesforce (CRM) and a product-analytics platform
Entity: one row per renewal Opportunity in Salesforce

> Note: this is a synthetic specification created for a portfolio/learning
> project. Names, products, and figures are fictional and do not represent
> any real customer.

## Filters

Only the following records are in scope for the model:

- Account brand must equal Atlas
- Opportunity type must contain 'Renewal'

These filters exclude Northwind's other product lines (Beacon, Cobalt) and
all non-renewal opportunity types.

## Metric

The model classifies every in-scope opportunity into one of three states.

### Success (Churn)

In the platform's terminology the target event is called "Success". For this
model, Success means the customer churned. Conditions:

- Account brand equals Atlas
- Opportunity type contains 'Renewal'
- Opportunity Stage equals Closed Lost

### Fail (Renewal)

"Fail" means the customer renewed (the non-event the model learns as safe).
Conditions:

- Account brand equals Atlas
- Opportunity type contains 'Renewal'
- Opportunity Stage equals Closed Won

### Audience

The Audience is the set of live, open opportunities the model scores in
production. Conditions:

- Account brand equals Atlas
- Opportunity type contains 'Renewal'
- Opportunity Stage is NOT Closed Lost and NOT Closed Won
- Days to next renewal is greater than 30
- Days to next renewal is less than 180

The 30 to 180 day window is the actionable range: far enough out for a CSM
to intervene, close enough that the prediction is relevant.

## Reference Date

The reference date is the cutoff date used to time-travel each record.

- Success and Fail records: a random date between (Close date minus 180 days)
  and (Close date minus 30 days)
- Audience records: the current date

Random selection within the window prevents the model from learning patterns
tied to a fixed point in the renewal cycle.

## Aggregation

Product usage data is aggregated into time-windowed cohort features.

- Lag windows: Last 1 month, Last 1-2 months, Last 3-4 months,
  Last 10-11 months
- Regular cohorts plus ratios between adjacent windows (a declining ratio is
  a churn signal)

Account-level fields brought in as features:

- Days to next renewal: current date minus the Account's next renewal date
- Active Subscriptions ARR Rollup: segmented into buckets, not used as a raw
  number
- Planned ATR (Available to Renew): segmented into buckets

ARR and ATR are bucketed because large customers behave differently from
small ones; bucketing lets the model learn size-based patterns.

## Identity Mapping

The product-analytics platform and Salesforce do not share a common account
identifier. Northwind uses a custom Salesforce object called PlatformAssets,
a child object of Account, which holds the mapping between Salesforce Account
ID and the analytics platform's Account ID. Joining usage data to a
Salesforce Account must go through PlatformAssets as a mapping table.

## Additional Rules (from working sessions)

- Exclude "Frozen" opportunities; they are bad data.
- Limit training data to Q2 2025 (April 2025) or later.
- Include amendments as churn when amendment type has Ending ARR equal to 0.
- Three separate models are planned, one per product line: Atlas
  (Enterprise), Beacon (Professional), Cobalt (Spend). Atlas is built first.
