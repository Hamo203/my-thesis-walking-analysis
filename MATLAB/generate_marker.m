%マーカー抽出用
clear; clc;
import java.io.*;

%trcファイルの読み込み
filePath = "markers.trc";
trc = readtable(filePath, 'FileType', 'text');

%変数名
vars = trc.Properties.VariableNames;

% 1. ファイルを開く
fileID = fopen('variable_names_trc.txt', 'w');

% 2. セル配列の内容を一行ずつ書き込む
% '%s\n' は文字列として出力し、改行することを示します。
% vars{:} はカンマ区切りリスト展開で、セル配列の全要素をfprintfに渡します。
fprintf(fileID, '%s\n', vars{:});

% 3. ファイルを閉じる
fclose(fileID);

% (オプション) ファイルの内容を確認する（MATLABのコマンドウィンドウに出力）
type 'variable_names_trc.txt'