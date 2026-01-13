# Experiments A & B Comparison Report
==================================================

## Executive Summary

This report compares two text classification experiments on the IMDB sentiment analysis dataset:
- **Experiment A**: Frozen language model backbone with classification head
- **Experiment B**: End-to-end RNN classifier with Word2Vec embeddings

## Key Findings

**Experiment B significantly outperforms Experiment A:**
- **Accuracy**: 85.0% vs 66.8% (+27.1 percentage points)
- **F1-Score**: 85.0% vs 66.7% (+27.4 percentage points)
- **ROC-AUC**: 93.0% vs N/A (not measured in A)

## Detailed Performance Metrics

| Metric | Experiment A | Experiment B | Improvement |
|--------|-------------|--------------|-------------|
| Accuracy | 66.8% | 85.0% | +27.1% |
| Macro Avg F1 | 66.7% | 85.0% | +27.4% |
| Macro Avg Precision | 67.2% | 85.2% | +26.9% |
| Macro Avg Recall | 66.8% | 85.0% | +27.1% |

## Per-Class Performance Analysis

### Negative Class

- **Experiment A**: Precision: 69.5%, Recall: 60.1%, F1: 64.4%
- **Experiment B**: Precision: 82.3%, Recall: 89.0%, F1: 85.6%

### Positive Class

- **Experiment A**: Precision: 64.8%, Recall: 73.6%, F1: 68.9%
- **Experiment B**: Precision: 88.1%, Recall: 80.9%, F1: 84.3%

## Conclusions

1. **Performance Superiority**: Experiment B demonstrates clear superiority across all metrics.
2. **Transfer Learning Limitations**: The frozen language model approach shows limited adaptation.

## Recommendations

1. **Implement Enhanced Error Analysis**: Both experiments would benefit from systematic error analysis
2. **Consider Hybrid Approaches**: Explore combining the strengths of both approaches
3. **Domain-Specific Fine-tuning**: Consider fine-tuning Experiment B on domain-specific data
4. **Ensemble Methods**: Investigate ensemble approaches combining both models