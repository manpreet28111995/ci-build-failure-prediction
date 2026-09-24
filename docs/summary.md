# Cross project summary (rolling origin, ten primary projects)

| Scorer | Mean AUROC | 95% CI | Median | Projects > 0.65 | Mean lift | Norm. AUPRC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous outcome | 0.627 | [0.596, 0.659] | 0.620 | 3/10 | 1.46 | 0.195 |
| Last five rate | 0.623 | [0.591, 0.657] | 0.614 | 3/10 | 1.41 | 0.187 |
| History LR | 0.630 | [0.598, 0.663] | 0.624 | 3/10 | 1.48 | 0.209 |
| Autoencoder | 0.517 | [0.497, 0.535] | 0.515 | 0/10 | 1.15 | 0.052 |
| Isolation forest | 0.548 | [0.521, 0.575] | 0.538 | 0/10 | 1.25 | 0.073 |
| PCA reconstruction | 0.523 | [0.508, 0.537] | 0.523 | 0/10 | 1.12 | 0.037 |
| Random forest | 0.598 | [0.566, 0.631] | 0.590 | 2/10 | 1.37 | 0.173 |
| Gradient boosting | 0.577 | [0.549, 0.607] | 0.576 | 1/10 | 1.31 | 0.135 |

## Per project mean AUROC

| Project | Prev. | PREV | LAST5 | HIST-LR | AE | IF | PCA | RF | GB |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jruby/jruby (primary) | 0.678 | 0.675 | 0.668 | 0.684 | 0.457 | 0.495 | 0.499 | 0.691 | 0.664 |
| apache/jackrabbit-oak (primary) | 0.670 | 0.705 | 0.705 | 0.712 | 0.529 | 0.610 | 0.560 | 0.657 | 0.639 |
| gradle/gradle (primary) | 0.087 | 0.615 | 0.605 | 0.616 | 0.572 | 0.616 | 0.540 | 0.573 | 0.562 |
| opf/openproject (primary) | 0.314 | 0.593 | 0.589 | 0.593 | 0.523 | 0.539 | 0.508 | 0.530 | 0.527 |
| rails/rails (primary) | 0.272 | 0.625 | 0.622 | 0.632 | 0.522 | 0.575 | 0.546 | 0.568 | 0.517 |
| ruby/ruby (primary) | 0.440 | 0.717 | 0.718 | 0.713 | 0.509 | 0.581 | 0.539 | 0.630 | 0.601 |
| Graylog2/graylog2-server (primary) | 0.075 | 0.583 | 0.580 | 0.604 | 0.488 | 0.506 | 0.516 | 0.590 | 0.588 |
| cloudfoundry/cloud_controller_ng (primary) | 0.189 | 0.574 | 0.567 | 0.556 | 0.502 | 0.486 | 0.477 | 0.516 | 0.524 |
| facebook/presto (primary) | 0.557 | 0.629 | 0.629 | 0.632 | 0.553 | 0.538 | 0.530 | 0.639 | 0.588 |
| github/linguist (primary) | 0.234 | 0.555 | 0.547 | 0.559 | 0.509 | 0.532 | 0.513 | 0.590 | 0.565 |
| openSUSE/open-build-service (robustness) | 0.257 | 0.740 | 0.724 | 0.736 | 0.574 | 0.620 | 0.525 | 0.722 | 0.701 |
| mitchellh/vagrant (robustness) | 0.129 | 0.716 | 0.703 | 0.734 | 0.630 | 0.680 | 0.589 | 0.684 | 0.615 |
| rubinius/rubinius (robustness) | 0.390 | 0.534 | 0.525 | 0.532 | 0.549 | 0.555 | 0.537 | 0.544 | 0.556 |
| orbeon/orbeon-forms (robustness) | 0.734 | 0.762 | 0.757 | 0.775 | 0.509 | 0.616 | 0.535 | 0.754 | 0.756 |

## Holm tallies (wins/losses/ties) vs LAST5 and vs RF, AUROC then AUPRC

- vs LAST5, auroc: PREV 2/0/8, HIST-LR 1/0/9, AE 0/5/5, IF 0/3/7, PCA 0/5/5, RF 0/3/7, GB 0/3/7
- vs LAST5, auprc: PREV 2/0/8, HIST-LR 2/0/8, AE 0/3/7, IF 0/3/7, PCA 0/4/6, RF 1/1/8, GB 0/2/8
- vs RF, auroc: PREV 3/0/7, LAST5 3/0/7, HIST-LR 3/0/7, AE 0/3/7, IF 0/2/8, PCA 0/3/7, GB 0/2/8
- vs RF, auprc: PREV 1/1/8, LAST5 1/1/8, HIST-LR 1/0/9, AE 0/4/6, IF 0/2/8, PCA 0/3/7, GB 0/2/8
