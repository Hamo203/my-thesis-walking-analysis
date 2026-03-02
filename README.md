# my-thesis-walking-analysis

卒論用のコードがまとまっているリポジトリになっています．
歩行データベース2019に関するものは.gitignoreになっています．

\vscode\machinelearning
機械学習の段階で使ったコードです．実行環境はGoogle colab なので勝手が違うかもしれません．

\vscode\calcacc.py
1個のファイルに対して、Conventional特徴量抽出を行う

\vscode\calcacc_conventional.py
フォルダ内のファイルに対して、Conventional特徴量抽出を行う

\vscode\calcacc_multidomain.py
フォルダ内のファイルに対して、Multidomain特徴量抽出を行う

\vscode\kakunin_tsfresh.py
特徴量抽出関数を実行して、最初のファイルの加速度グラフとRHSイベントをプロットする

\vscode\merge_features.py
SACR, RANK, RANK2の特徴量ファイルと、関節角度・vGRFのQC後のファイル統合を行う

\vscode\rename_addsuffix.py
加速度特徴量ファイルの列名に "_マーカー名" のサフィックスを付ける

\vscode\to_compare_filter.py
20Hzローパスフィルタ前後の加速度の周波数スペクトルを比較するコード

\vscode\usethis_kosuu.py
use_thisになっている行数の確認