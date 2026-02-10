# Multi-Response Constraint Testing Guide

Use the file: `examples/test_multi_response_data.xlsx`

**Data ranges:**
- Tm: 43.73 to 55.49
- Aggregation: 3.56 to 37.36
- Activity: 23.55 to 75.54

---

## TEST 1: Valid Two-Response Setup ✅
**Should work without errors**

### Settings:
- Response 1: **Tm** → Direction: **Maximize**, Min: **50**
- Response 2: **Aggregation** → Direction: **Minimize**, Max: **20**

### Expected Result:
- ✅ Analysis completes successfully
- Shows best experiment and BO suggestions
- Shows Pareto frontier with trade-offs between Tm and Aggregation

---

## TEST 2: Single Contradiction (Tm only) ❌
**Should show error popup and stop**

### Settings:
- Response 1: **Tm** → Direction: **Minimize**, Min: **60**
- Response 2: **Aggregation** → Direction: **Minimize** (no constraints)

### Expected Result:
- ❌ Error popup appears saying:
  - "MINIMIZE Tm"
  - "Constraint: Tm >= 60"
  - "Your data range: 43.73 to 55.49"
  - "This is contradictory..."
- Analysis stops, Recommendations tab shows error message
- No "Analysis Complete" popup

### Why it's a contradiction:
- You want to MINIMIZE Tm (get lowest value)
- But constraint requires Tm >= 60 (higher than all your data!)
- That's impossible and contradictory

---

## TEST 3: Single Contradiction (Aggregation only) ❌
**Should show error popup and stop**

### Settings:
- Response 1: **Tm** → Direction: **Maximize**, Min: **50** (valid)
- Response 2: **Aggregation** → Direction: **Maximize**, Max: **2**

### Expected Result:
- ❌ Error popup appears saying:
  - "MAXIMIZE Aggregation"
  - "Constraint: Aggregation <= 2"
  - "Your data range: 3.56 to 37.36"
  - "This is contradictory..."
- Analysis stops immediately

### Why it's a contradiction:
- You want to MAXIMIZE Aggregation (get highest value)
- But constraint requires Aggregation <= 2 (lower than all your data!)
- That's impossible and contradictory

---

## TEST 4: Both Responses Have Contradictions ❌❌
**Should show error popup listing BOTH contradictions**

### Settings:
- Response 1: **Tm** → Direction: **Minimize**, Min: **60**
- Response 2: **Aggregation** → Direction: **Maximize**, Max: **2**

### Expected Result:
- ❌ Error popup appears saying:
  - "Found 2 logical contradictions:"
  - "1. Tm: MINIMIZE + Min>=60 (data: 43.73-55.49)"
  - "2. Aggregation: MAXIMIZE + Max<=2 (data: 3.56-37.36)"
- Analysis stops, won't continue with contradictory settings

---

## TEST 5: Three Responses, One Contradiction (Activity) ❌
**Should detect the Activity contradiction**

### Settings:
- Response 1: **Tm** → Direction: **Maximize**, Min: **50** (valid)
- Response 2: **Aggregation** → Direction: **Minimize**, Max: **20** (valid)
- Response 3: **Activity** → Direction: **Minimize**, Min: **80**

### Expected Result:
- ❌ Error popup appears saying:
  - "MINIMIZE Activity"
  - "Constraint: Activity >= 80"
  - "Your data range: 23.55 to 75.54"
  - "This is contradictory..."

### Why it's a contradiction:
- Activity max is 75.54, but you require >= 80
- You want to minimize but require a value higher than your best data
- Contradiction!

---

## TEST 6: Three Responses, ALL Have Contradictions ❌❌❌
**Should show error listing all 3 contradictions**

### Settings:
- Response 1: **Tm** → Direction: **Minimize**, Min: **60**
- Response 2: **Aggregation** → Direction: **Maximize**, Max: **2**
- Response 3: **Activity** → Direction: **Minimize**, Min: **80**

### Expected Result:
- ❌ Error popup appears saying:
  - "Found 3 logical contradictions:"
  - Lists all three with their data ranges
- Analysis cannot continue

---

## TEST 7: Valid Three-Response Setup ✅
**Should work perfectly with all constraints**

### Settings:
- Response 1: **Tm** → Direction: **Maximize**, Min: **50**
- Response 2: **Aggregation** → Direction: **Minimize**, Max: **20**
- Response 3: **Activity** → Direction: **Maximize**, Min: **60**

### Expected Result:
- ✅ Analysis completes successfully
- Shows Pareto frontier with 3-objective trade-offs
- All constraints are compatible with optimization directions

---

## TEST 8: Warning Case (Not Contradiction) ⚠️
**Should show warning but continue analysis**

### Settings:
- Response 1: **Tm** → Direction: **Maximize**, Min: **60**
- Response 2: **Aggregation** → Direction: **Minimize** (no constraints)

### Expected Result:
- ⚠️ Warning shown: "No experiments meet the constraint Tm >= 60"
- Analysis CONTINUES without the constraint
- Shows best experiment WITHOUT constraint applied
- This is NOT a contradiction because:
  - Direction is Maximize (want high Tm)
  - Constraint is Min=60 (want high Tm)
  - They're compatible, just no data meets it yet

---

## Quick Test Checklist

Run these in order and check off:

- [ ] TEST 1: Valid 2-response → ✅ Works
- [ ] TEST 2: Tm contradiction → ❌ Error popup
- [ ] TEST 3: Aggregation contradiction → ❌ Error popup
- [ ] TEST 4: Both contradictions → ❌ Shows both
- [ ] TEST 5: Activity contradiction → ❌ Error popup
- [ ] TEST 6: All 3 contradictions → ❌ Shows all 3
- [ ] TEST 7: Valid 3-response → ✅ Works
- [ ] TEST 8: Warning case → ⚠️ Warns but continues

---

## What to Look For

### ✅ Valid Cases Should Show:
- Analysis completes normally
- Best observed experiment section
- BO suggestions (5 experiments)
- Pareto frontier (for multi-objective)
- "Analysis Complete" popup at the end

### ❌ Contradiction Cases Should Show:
1. Error popup immediately (before analysis finishes)
2. Clear explanation of the contradiction
3. Suggestions to fix it
4. NO "Analysis Complete" popup after
5. Error message in Recommendations tab

### ⚠️ Warning Cases Should Show:
- Warning message in Recommendations tab
- "Showing best experiment WITHOUT constraint applied"
- Analysis continues and completes
- "Analysis Complete" popup still shows

---

## Tips for Testing

1. **Load the data file first**: `examples/test_multi_response_data.xlsx`

2. **Select responses**: Check the boxes for the responses you want to test

3. **Set directions**: Use the dropdowns next to each response

4. **Set constraints**: Enter Min/Max values in the constraint fields

5. **Click "Analyze Data"** and watch for:
   - Immediate error popups (contradictions)
   - Warnings in Recommendations tab
   - Success completion

6. **Reset between tests**: Clear constraints or reload the file

---

## Expected Behavior Summary

| Test | Responses | Result | Popup Type |
|------|-----------|--------|------------|
| 1 | 2 valid | Success | None (success at end) |
| 2 | 1 contradiction | Error | Contradiction error |
| 3 | 1 contradiction | Error | Contradiction error |
| 4 | 2 contradictions | Error | Multiple contradictions |
| 5 | 1 contradiction | Error | Contradiction error |
| 6 | 3 contradictions | Error | Multiple contradictions |
| 7 | 3 valid | Success | None (success at end) |
| 8 | 1 warning | Success with warning | None (success at end) |

If all tests behave as expected, the multi-response contradiction detection is working correctly! ✅
