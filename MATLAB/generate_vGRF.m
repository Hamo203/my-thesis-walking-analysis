% .motからvGRF 取得用コード
% 10Hz のローパスフィルタをかける必要がある
clear; clc;
import java.io.*;

BW_default = 70; % 体重デフォルト [kg]
fs = 1000; % サンプリング周波数 [Hz] 床反力計 1000Hz
cutoff = 10; % カットオフ周波数 [Hz]

filePath = "file.mot";
mot = readtable(filePath, 'FileType', 'text');

%被験者情報の取得 
[~, basename, ~] = fileparts(filePath); % basename =  number_number_sex_age_height_weight_BL.cmo+LorRnumber_marker
tokens = regexp(basename, '[0-9]+', 'match');
BW = BW_default;
subID = basename; 
nums = cellfun(@str2double, tokens);
candidates = nums(nums>=30 & nums<=200);
if ~isempty(candidates)
    BW = candidates(end); % 最後の値を採用
end

%変数名
vars = mot.Properties.VariableNames;
%変数名を取得して、全部小文字にそろえる
vars_lower = lower(vars);

% 垂直成分（vy）列の抽出
isFy = contains(vars_lower, 'ground_force_') & contains(vars_lower, '_vy');
fy_cols = vars(isFy);

if isempty(fy_cols)
    error('垂直成分 (ground_force_*_vy) の列が見つかりません。');
end

% 確認用
fprintf('Found vertical GRF columns:\n');
disp(fy_cols')

% 垂直成分データの取得
vGRF_all = mot{:, fy_cols}; 

% Butterworthローパスフィルタの設計
[b, a] = butter(4, cutoff / (fs / 2), 'low');
% 各プレートを個別にフィルタ処理
vGRF_all_filt = filtfilt(b, a, vGRF_all);
% 全プレートの最大値を1波形として採用
vGRF = max(vGRF_all_filt, [], 2);
%vGRF = max(vGRF_all, [], 2);  % 各行で最大値（6プレート中）

% 体重にかかる静止時の重力（N）で正規化
vGRF_norm = vGRF / (BW * 9.81); 

plot(vGRF_norm, 'LineWidth', 1.5)
ylabel('vGRF (×BW)')
xlabel('Frame')
title('Vertical GRF (Normalized)')
grid on

if max(vGRF_norm) > 1.3
    label_force = "HighLoad";
else
    label_force = "Normal";
end
fprintf('Processing %s (subject %s, BW%d kg, label_force= %s)\n', basename, subID, BW, label_force);
