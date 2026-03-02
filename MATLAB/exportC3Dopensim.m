%c3d -> trc,motにエクスポートした

import org.opensim.modeling.*

%% ユーザ設定（ここを変えてください）
useCenterOfPressureAsMomentsPoint = 1;  % 1 = COP をモーメント基準点にする, 0 = electrical center
convertLengthUnits = 1;  % 1 = mm->m に変換する

%% C3D 選択
[filename, path] = uigetfile('*.c3d','Select a C3D file');
if isequal(filename,0)
    error('No file selected.');
end
c3dpath = fullfile(path, filename);

%% C3D 読み込み（ForceLocation を必ず渡す）
c3d = osimC3D(c3dpath, useCenterOfPressureAsMomentsPoint);

%% 基本情報
nTrajectories = c3d.getNumTrajectories();
rMarkers = c3d.getRate_marker();
nForces = c3d.getNumForces();
rForces = c3d.getRate_force();
t0 = c3d.getStartTime();
tn = c3d.getEndTime();

%% 必要なら回転（例: X 軸 -90deg）
c3d.rotateData('x', -90);
c3d.rotateData('y', 90);

%% テーブル / 構造体取得
markerTable = c3d.getTable_markers();
forceTable  = c3d.getTable_forces();
[markerStruct, forceStruct] = c3d.getAsStructs();

%% 単位変換（フラグに基づく）
if convertLengthUnits
    c3d.convertMillimeters2Meters();
end

%% 出力ファイル名決定
basename = strtok(filename, '.');
markersFilename = strcat(basename, '_markers.trc');
if useCenterOfPressureAsMomentsPoint == 0
    forcesFilename = strcat(basename, '_forces_EC.mot');
else
    forcesFilename = strcat(basename, '_forces_COP.mot');
end

%% 書き出し
c3d.writeTRC(markersFilename);
c3d.writeMOT(forcesFilename);

fprintf('Wrote %s and %s\n', markersFilename, forcesFilename);
