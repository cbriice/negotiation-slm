# Info
In this file: 
- Control function mathematical definition
- Definitions for each argument subfunction
- Definitions for parameters

- Control features - details to track within control function

## Subfunction inputs

Important notation: period in variable name - equivalent to a math subscript. to indicate indexed variables in a math equation. i.e. P.k:= price at turn k.
- if a variable has an incremented input then assume the function itself also increments
These variables are being defined so that you can understand the control function equation for the negotiation sims
Parent control function defined later

### Parameters
`beta_b`, `beta_s`: parameter defined by personality preset of agent
`ALPHA`: global simulation parameter to normalize A(p) (defined later)
`EPSILON`: global threshold parameter
`w_g`, `w_z`, `w_x`: float weighting parameters for control function. packaged as dict `weights` with keys wg, wz, wx

### Turns
`r`: current turn index
`R`: max turns (parameter)
`t.k`: r / R normalized turn index

## Extra items (price external)
`n_b.k`: number of extra for buyer offer. turn indexed because this can change with diff offers
`n_s.k`: number of extra for seller offer
`l.k`: number of xtra items included in current proposed offer


### Prices
`Ps`: seller minimum acceptable
`Pb`: buyer max acceptable
`P.k`: price at turn k. initialized as seller list, changes as buyer proposes/seller asserts

`P_s` = Ps + beta_s * Ps: seller adjusted min given personality preset 
`P_b` = Pb + beta_b * Pb: buyer adjusted max given personality preset
`delta_P` = abs(P_s - P_b): gap between adjusted buyer max/seller min
`mstar_s` = max(0, P_k - P_s): gap of current price and seller min
`mstar_b` = max(0, P_b - P_k): gap of current price and buyer max
`m_s` = mstar_s / delta_P if delta_P > EPSILON else -1000: -1000 a placeholder indicator of equivalent prices. normalizes mstar to gap
`m_b` = mstar_b / delta_P if delta_P > EPSILON else -1000

---
## Subfunction definitions

`g_s` = max(0, m_s): price gap for seller. seller would rather price be a little above minimum than close to minimum
`g_b` = max(0m, m_b): price gap for buyer. vice versa logic

`z_s` = t.k * max(0, Ps - P.k) / Ps: price range indicator - 0 if within seller accept
`z_b` = t.k * max(0, P.k - Pb) / Pb: 0 if within buyer accept

`A` = (exp(ALPHA * t.k) - 1) / (exp(ALPHA) - 1): exponential scaling of round pressure

`x` = 1 - l.k / (n_s.k + n_b.k) if (n_s.k + n_b.k) > 0 else None: if there are extra items, find % left out

---
# Control Function Definition
Whole function updates every turn k. Each argument besides weights updates for new turn
Buyer and seller both have their own control functions 
Mathematically defined

`V_s(A, g_s, z_s, x, weights)` = A * (weights.wz * z_s + weights.wg * g_s) + weights.wx * x

`V_b(A, g_b, z_b, x, weights)` = A * (weights.wz * z_b + weights.wg * g_b) + weights.wx * x

`V_max` = weights.wz + weights.wg + weights.wx

---
# Action Determining Functions
Information contained in the control function is used to determine choice of prompt for the agents participating in negotiation. Values are collected and input into a softmax normalization which stochastically determines action choice, weighted by contextual negotiation data

## Control features
Important features of `V` which are tracked for determining negotiation actions

`s` = 1 - `V` / `V_max`: completion percentage. successful negotiation minimizes `V` and maximizes `s`

`cg_s` = A * weights.wg * g_s 
`cg_b` = A * weights.wg * g_b

`cz_s` = A * weights.wz * z_s
`cz_b` = A * weights.wz * z_b

`cx`: weights.wx * x

`A` as defined before

## Action functions 
q-functions which compute values that are input into softmax
Buyer/seller ambiguous definitions. use buyer/seller specific variables in practice

**Parameters**: 
- `ha1`, `ha2`, `hz`, `hc`, `hl`: weighting parameters for action functions
- `eps`: local threshold parameter

**Compensate price**: `q_compensate` = `cz` * (1 + `ha1` * A)
**Fine tune price**: `q_finetune` = `cg` * `hz` * `c_z`

**Extra item**: `q_extra` = `cx`

**Close**: `q_close` = `hc` * (3 - `cg` - `cz` - `cx`) / 3 + `ha2` * `A`

**Clarify**: `q_clarify` = `hl` * (|`s.k` - `s.k-1`| < `eps`): expression in parentheses is a boolean {0, 1}. agent would potentially want clarification if no progression made in negotiation
