
---

## 🚨 CRITICAL: Commands in README vs Actual CLI

### Issue: Many README Commands Don't Exist in CLI

**Problem**: User tries commands from README that aren't implemented

**Example from line 998**:
```bash
neurinspectre adversarial-ts-inverse --target-gradients federated_grads.npy --reconstruction-quality high
```

**Error**: `argument command: invalid choice: 'adversarial-ts-inverse'`

### Commands That DON'T Exist (from README):

❌ **Not Implemented**:
- `adversarial-ts-inverse`
- `adversarial-concretizer`
- `adversarial-attention`
- `evasion-transport`
- `evasion-demarking`
- `binary-analysis`
- `detect-backdoors`
- `analyze-attack-vectors`
- `security-monitor`
- `behavioral-analysis`
- `integrated-security-scan`
- `corner-case-detection`
- `security-indicators`
- `correlate-patterns`
- `forensic-binary`
- `generate-threat-assessment`
- `recommend-countermeasures`

### Commands That DO Exist:

✅ **Actual Working Commands**:
```bash
neurinspectre obfuscated-gradient create
neurinspectre math spectral
neurinspectre dashboard
neurinspectre comprehensive-scan
neurinspectre adversarial-detect
neurinspectre adversarial-ednn
neurinspectre evasion-detect
neurinspectre anomaly
neurinspectre attack-graph
neurinspectre gradient_inversion
neurinspectre activation_steganography
neurinspectre subnetwork_hijack
```

### Solution:

**For Implementation Checklist**: Already added note explaining commands are conceptual

**For README Command Examples**: Need to either:
1. Mark clearly as "Future/Planned Features"
2. Replace with actual working commands
3. Remove non-existent commands

**Recommendation**: Update README to use only actual commands or clearly mark planned features

---

**Last Updated**: December 7, 2025, 9:45 PM
