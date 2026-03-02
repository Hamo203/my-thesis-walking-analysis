%フォルダに入ってるc3d全部 -> trc,motにエクスポートした
import org.opensim.modeling.*

%% ユーザ設定
useCenterOfPressureAsMomentsPoint = 1;  % 1 = COP
convertLengthUnits = 1;  % 1 = mm->m に変換

%% 入力フォルダ選択（C3D が入っているフォルダ）
inputDir = uigetdir(pwd, 'Select input folder containing C3D files');
if inputDir == 0
    error('No folder selected.');
end

%% 出力フォルダ選択
outputDir = uigetdir(pwd, 'Select output folder for TRC/MOT files');
if outputDir == 0
    error('No output folder selected.');
end

%% 全 C3D ファイルを取得
files = dir(fullfile(inputDir, '*.c3d'));
if isempty(files)
    error('No C3D files found in the selected input folder.');
end

%% 処理ループ
for k = 1:length(files)
    filename = files(k).name;
    c3dpath = fullfile(inputDir, filename);
    
    % C3D 読み込み
    % mot のラベルは変更されている
    c3d = osimC3D(c3dpath, useCenterOfPressureAsMomentsPoint);
    
    %% 基本情報
    nTrajectories = c3d.getNumTrajectories();
    rMarkers = c3d.getRate_marker();
    nForces = c3d.getNumForces();
    rForces = c3d.getRate_force();
    t0 = c3d.getStartTime();
    tn = c3d.getEndTime();

    % 回転（必要に応じて調整）
    % 軸の順序が最終的に (Y, Z, X) に見える
    c3d.rotateData('x', -90);
    c3d.rotateData('y', 90);
    
    

    %% テーブル / 構造体取得
    markerTable = c3d.getTable_markers();
    forceTable  = c3d.getTable_forces();
    [markerStruct, forceStruct] = c3d.getAsStructs();

    % 単位変換
    if convertLengthUnits
        c3d.convertMillimeters2Meters();
    end

    % 出力ファイル名決定（拡張子除去）
    [~, basename, ~] = fileparts(filename); % ★ここ修正

    % 出力パス（ここが変更ポイント）
    markersFilename = fullfile(outputDir, strcat(basename, '_markers.trc'));

    if useCenterOfPressureAsMomentsPoint == 0
        forcesFilename = fullfile(outputDir, strcat(basename,'_forces_EC.mot'));
    else
        forcesFilename = fullfile(outputDir, strcat(basename,'_forces_COP.mot'));
    end

    % 書き出し
    c3d.writeTRC(markersFilename);
    c3d.writeMOT(forcesFilename);

    fprintf('Wrote %s and %s\n', markersFilename, forcesFilename);
end

fprintf('\n All C3D files have been exported!\n');
