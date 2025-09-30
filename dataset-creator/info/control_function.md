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
`DELTA`: >=1 parameter weighing extras by progress
`GAMMA`: <=0.5 parameter to weigh extras more earlier in negotiation
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

`A_k` = (exp(ALPHA * t.k) - 1) / (exp(ALPHA) - 1): exponential scaling of round pressure

`x` = (1 - l.k / (n_s.k + n_b.k)) * (1 - s)^DELTA if (n_s.k + n_b.k) > 0 else 0: if there are extra items, find % left out

---
# Control Function Definition
Whole function updates every turn k. Each argument besides weights updates for new turn
Buyer and seller both have their own control functions 
Mathematically defined

`V_s(A_k, g_s, z_s, x, weights)` = A_k * (weights.wz * z_s + weights.wg * g_s) + weights.wx * x

`V_b(A_k, g_b, z_b, x, weights)` = A_k * (weights.wz * z_b + weights.wg * g_b) + weights.wx * x

`V_max` = A_k(weights.wz + weights.wg) + weights.wx

---
# Action Determining Functions
Information contained in the control function is used to determine choice of prompt for the agents participating in negotiation. Values are collected and input into a softmax normalization which stochastically determines action choice, weighted by contextual negotiation data

## Control features
Important features of `V_k` which are tracked for determining negotiation actions

`s` = 1 - `V_k` / `V_max`: completion percentage. successful negotiation minimizes `V` and maximizes `s`
`s_c`: threshold parameter for validity of `q_close` trigger. Should be >= 0.6

`cg_s` = A_k * weights.wg * g_s 
`cg_b` = A_k * weights.wg * g_b

`cz_s` = A_k * weights.wz * z_s
`cz_b` = A_k * weights.wz * z_b

`cx`: weights.wx * x

`A_k` as defined before

## Action functions 
q-functions which compute values that are input into softmax
Buyer/seller ambiguous definitions. use buyer/seller specific variables in practice

**Parameters**: 
- `ha1`, `ha2`, `hz`, `hc`, `hc2`, `hl`: weighting parameters for action functions
- `eps`: local threshold parameter

**Compensate price**: `q_compensate` = `cz` * (1 + `ha1` * min(`A_k`, 1 - `A_k`))
**Fine tune price**: `q_finetune` = `hz` * sqrt(`cz`*`cg`)

**Extra item**: `q_extra` = `cx` * `A_k`^`GAMMA`

**Close**: `q_close` = `hc` * `s` + `hc2` * BOOL(|`s_k` - `s_k-1` < `eps`) * BOOL(`s_k` > `s_c`)

**Clarify**: `q_clarify` = `hl` * (1 - `s`)

Softmax is computed for each item over the set {`q_compensate`, `q_finetune`, `q_extra`, `q_close`, `q_clarify`} and then next action is based on most likely value.
- Softmax temperature should be set to a constant value T = 1.5. 
