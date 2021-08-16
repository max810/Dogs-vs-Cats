from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    files = [
        # sorted acc desc
        'results/resnext50_colab_test_preds_PREDS_ONLY.csv',
        'results/mobilenetv2_colab_test_preds_PREDS_ONLY.csv',
        'results/imagenet_default_baseline_test_preds_PREDS_ONLY.csv',
        'results/mobilenetv2_local_test_preds_PREDS_ONLY.csv',
    ]

    results = [(Path(file).name.split('.')[0], pd.read_csv(file)) for file in files]
    res_df = pd.DataFrame()
    res_df['filename'] = results[0][1]['filename']

    names = []
    for name, df in results:
        res_df[name] = df['pred']
        names.append(name)

    # majority voting
    # for i in range(5, 0, -1):
    preds = res_df[names]
    majority = preds.mean(axis=1).round().astype('int')
    res_df['ensemble_pred'] = majority  # if multiple modes, pick the first one
    final = res_df.loc[:, ['filename', 'ensemble_pred']]
    from natsort import natsort_keygen

    final.sort_values(
        by='filename',
        key=natsort_keygen(),
        inplace=True
    )
    final.to_csv('FINAL_PRED.csv', index=False, header=False)
# TOP 5 ensemble test accuracy: 0.9896
# TOP 4 ensemble test accuracy: 0.9932
# TOP 3 ensemble test accuracy: 0.9912
# TOP 2 ensemble test accuracy: 0.9896
# TOP 1 ensemble test accuracy: 0.9900
