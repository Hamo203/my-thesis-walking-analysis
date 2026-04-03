# my-thesis-walking-analysis

卒論用のコードがまとまっているリポジトリになっています．
歩行データベース2019に関するものは.gitignoreになっています．

# 作業ファイル
vscode/machinelearning
機械学習の段階で使ったコードです．実行環境はGoogle colab なので勝手が違うかもしれません．

## 準備
MATLAB/exportC3Dopensim.m
c3d -> trc,motにエクスポートした

MATLAB/exportC3Dopensim2.m
フォルダに入ってるc3d全部 -> trc,motにエクスポートした
OpenSimの仕様に合わせて回転させようとしている
1．X軸を中心にマイナス90度回転。
2. 次に（回転後の）Y軸を中心にプラス90度回転。
→（通常 Y が上、X が進行方向）に合わせようとした

vscode/position_c3d.py
マーカーの位置データを取得する例

## 特徴量抽出
MATLAB/generate_vGRF.m
.motからvGRFを取得

MATLAB/trc_mot_QC_kai_4.m
①.trcファイルから得たマーカ座標から膝角度を算出
②vGRFと膝角度についてQCをおこなう

vscode/calacc.py
1個のc3dファイルに対して、Conventional特徴量抽出を行う

vscode/calcacc_conventional.py
フォルダ内のc3dファイルに対して、Conventional特徴量抽出を行う

vscode/calcacc_multidomain.py
フォルダ内のc3dファイルに対して、Multidomain特徴量抽出を行う



## 特徴量用ファイル作成
MATLAB/generate_marker.m
マーカー抽出用 -> trcファイル中身確認

vscode/merge_features.py
SACR, RANK, RANK2の特徴量ファイルと、関節角度・vGRFのQC後のファイル統合を行う

vscode/rename_addsuffix.py
加速度特徴量ファイルの列名に "_マーカー名" のサフィックスを付ける


## 確認
MATLAB/make_graph.m
膝角度peek,ROM,vgrfの分布と分位数をプロットする

vscode/kakunin_tsfresh.py
特徴量抽出関数を実行して、最初のファイルの加速度グラフとRHSイベントをプロットする

vscode/to_compare_filter.py
20Hzローパスフィルタ前後の加速度の周波数スペクトルを比較するコード

vscode/usethis_kosuu.py
use_thisになっている行数の確認
